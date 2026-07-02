"""
Postprocessing Pipeline for ML Detection Ensemble.
Handles NMS, filtering, and coordinate transformations.
"""

import logging
from dataclasses import replace

import torch

from core.data_types import BoundingBox, Detection

logger = logging.getLogger("wildcams")


class PostprocessingPipeline:
    """Handles all postprocessing operations for ML detection."""

    def __init__(self):
        pass

    def apply_advanced_nms(self, detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
        """
        Apply advanced Non-Maximum Suppression across ensemble.

        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for NMS

        Returns:
            Filtered detections after NMS
        """
        if len(detections) <= 1:
            return detections

        # Convert to format needed for NMS
        import torchvision.ops as ops

        boxes = []
        scores = []

        for det in detections:
            boxes.append([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
            scores.append(det.confidence)

        if not boxes:
            return []

        # Convert to tensors
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        # Apply NMS
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)

        # Return filtered detections
        filtered_detections = []
        for idx in keep_indices:
            filtered_detections.append(detections[idx.item()])

        logger.info(
            f"🧹 ADVANCED NMS: {len(detections)} → {len(filtered_detections)} detections (removed {len(detections) - len(filtered_detections)} duplicates)"
        )

        return filtered_detections

    def filter_by_confidence(self, detections: list[Detection], threshold: float) -> list[Detection]:
        """
        Filter detections by confidence threshold.

        Args:
            detections: List of Detection objects
            threshold: Minimum confidence threshold

        Returns:
            Filtered detections
        """
        filtered = [det for det in detections if det.confidence >= threshold]

        if len(filtered) != len(detections):
            logger.debug(
                f"Confidence filtering: {len(detections)} → {len(filtered)} detections (threshold: {threshold})"
            )

        return filtered

    def filter_by_area(
        self, detections: list[Detection], min_area: int = 100, max_area: int | None = None
    ) -> list[Detection]:
        """
        Filter detections by bounding box area.

        Args:
            detections: List of Detection objects
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area (None for no limit)

        Returns:
            Filtered detections
        """
        filtered = []

        for det in detections:
            area = det.bbox.area

            if area >= min_area:
                if max_area is None or area <= max_area:
                    filtered.append(det)

        if len(filtered) != len(detections):
            logger.debug(
                f"Area filtering: {len(detections)} → {len(filtered)} detections (min: {min_area}, max: {max_area})"
            )

        return filtered

    def convert_coordinates(
        self, detections: list[Detection], source_size: tuple[int, int], target_size: tuple[int, int]
    ) -> list[Detection]:
        """
        Convert bounding box coordinates between different image sizes.

        Args:
            detections: List of Detection objects
            source_size: (width, height) of source image
            target_size: (width, height) of target image

        Returns:
            Detections with converted coordinates
        """
        if source_size == target_size:
            return detections

        source_w, source_h = source_size
        target_w, target_h = target_size

        # A degenerate (zero-sized) source has no meaningful scale factor; leave boxes as-is.
        if source_w <= 0 or source_h <= 0:
            return detections

        scale_x = target_w / source_w
        scale_y = target_h / source_h

        converted = []
        for det in detections:
            b = det.bbox
            new_bbox = BoundingBox(b.x1 * scale_x, b.y1 * scale_y, b.x2 * scale_x, b.y2 * scale_y)
            converted.append(replace(det, bbox=new_bbox))

        return converted

    def normalize_coordinates(self, detections: list[Detection], image_size: tuple[int, int]) -> list[Detection]:
        """
        Normalize bounding box coordinates to [0, 1] range.

        Args:
            detections: List of Detection objects
            image_size: (width, height) of image

        Returns:
            Detections with normalized coordinates
        """
        width, height = image_size

        # A degenerate (zero-sized) image cannot be normalized against; leave boxes as-is.
        if width <= 0 or height <= 0:
            return detections

        normalized = []
        for det in detections:
            b = det.bbox
            norm_bbox = BoundingBox(b.x1 / width, b.y1 / height, b.x2 / width, b.y2 / height)
            normalized.append(replace(det, bbox=norm_bbox))

        return normalized

    def denormalize_coordinates(self, detections: list[Detection], image_size: tuple[int, int]) -> list[Detection]:
        """
        Denormalize bounding box coordinates from [0, 1] range to pixel coordinates.

        Args:
            detections: List of Detection objects
            image_size: (width, height) of image

        Returns:
            Detections with pixel coordinates
        """
        width, height = image_size

        denormalized = []
        for det in detections:
            b = det.bbox
            pixel_bbox = BoundingBox(b.x1 * width, b.y1 * height, b.x2 * width, b.y2 * height)
            denormalized.append(replace(det, bbox=pixel_bbox))

        return denormalized

    def clip_to_image(self, detections: list[Detection], image_size: tuple[int, int]) -> list[Detection]:
        """
        Clip bounding boxes to image boundaries.

        Args:
            detections: List of Detection objects
            image_size: (width, height) of image

        Returns:
            Detections with clipped coordinates
        """
        width, height = image_size

        clipped = []
        for det in detections:
            b = det.bbox
            cx1 = max(0, min(b.x1, width))
            cy1 = max(0, min(b.y1, height))
            cx2 = max(0, min(b.x2, width))
            cy2 = max(0, min(b.y2, height))

            # Skip invalid boxes
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            clipped.append(replace(det, bbox=BoundingBox(cx1, cy1, cx2, cy2)))

        return clipped

    def merge_detections(self, detection_groups: list[list[Detection]]) -> list[Detection]:
        """
        Merge detections from multiple sources (e.g., TTA, multi-scale).

        Args:
            detection_groups: List of detection lists to merge

        Returns:
            Merged detections
        """
        merged = []

        for group in detection_groups:
            merged.extend(group)

        return merged

    def calculate_iou(self, bbox1: list[float], bbox2: list[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Calculate areas
        intersection_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate union
        union_area = bbox1_area + bbox2_area - intersection_area

        # Avoid division by zero
        if union_area <= 0:
            return 0.0

        return intersection_area / union_area
