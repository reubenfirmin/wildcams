"""
Postprocessing Pipeline for ML Detection Ensemble.
Handles NMS, filtering, and coordinate transformations.
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger('wildcams')

class PostprocessingPipeline:
    """Handles all postprocessing operations for ML detection."""
    
    def __init__(self):
        pass
    
    def apply_advanced_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply advanced Non-Maximum Suppression across ensemble.
        
        Args:
            detections: List of detection dictionaries
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
        sources = []
        
        for det in detections:
            boxes.append(det['bbox'])
            scores.append(det['confidence'])
            sources.append(det['source'])
        
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
        
        logger.info(f"🧹 ADVANCED NMS: {len(detections)} → {len(filtered_detections)} detections (removed {len(detections) - len(filtered_detections)} duplicates)")
        
        return filtered_detections
    
    def filter_by_confidence(self, detections: List[Dict], threshold: float) -> List[Dict]:
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: List of detection dictionaries
            threshold: Minimum confidence threshold
            
        Returns:
            Filtered detections
        """
        filtered = [det for det in detections if det['confidence'] >= threshold]
        
        if len(filtered) != len(detections):
            logger.debug(f"Confidence filtering: {len(detections)} → {len(filtered)} detections (threshold: {threshold})")
        
        return filtered
    
    def filter_by_area(self, detections: List[Dict], min_area: int = 100, max_area: int = None) -> List[Dict]:
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of detection dictionaries
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area (None for no limit)
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for det in detections:
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            if area >= min_area:
                if max_area is None or area <= max_area:
                    filtered.append(det)
        
        if len(filtered) != len(detections):
            logger.debug(f"Area filtering: {len(detections)} → {len(filtered)} detections (min: {min_area}, max: {max_area})")
        
        return filtered
    
    def convert_coordinates(self, detections: List[Dict], source_size: Tuple[int, int], 
                           target_size: Tuple[int, int]) -> List[Dict]:
        """
        Convert bounding box coordinates between different image sizes.
        
        Args:
            detections: List of detection dictionaries
            source_size: (width, height) of source image
            target_size: (width, height) of target image
            
        Returns:
            Detections with converted coordinates
        """
        if source_size == target_size:
            return detections
        
        source_w, source_h = source_size
        target_w, target_h = target_size
        
        scale_x = target_w / source_w
        scale_y = target_h / source_h
        
        converted = []
        for det in detections.copy():
            bbox = det['bbox']
            
            # Scale coordinates
            new_bbox = [
                bbox[0] * scale_x,  # x1
                bbox[1] * scale_y,  # y1
                bbox[2] * scale_x,  # x2
                bbox[3] * scale_y   # y2
            ]
            
            # Update detection
            det_copy = det.copy()
            det_copy['bbox'] = new_bbox
            converted.append(det_copy)
        
        return converted
    
    def normalize_coordinates(self, detections: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        """
        Normalize bounding box coordinates to [0, 1] range.
        
        Args:
            detections: List of detection dictionaries
            image_size: (width, height) of image
            
        Returns:
            Detections with normalized coordinates
        """
        width, height = image_size
        
        normalized = []
        for det in detections.copy():
            bbox = det['bbox']
            
            # Normalize coordinates
            norm_bbox = [
                bbox[0] / width,   # x1
                bbox[1] / height,  # y1
                bbox[2] / width,   # x2
                bbox[3] / height   # y2
            ]
            
            # Update detection
            det_copy = det.copy()
            det_copy['bbox'] = norm_bbox
            normalized.append(det_copy)
        
        return normalized
    
    def denormalize_coordinates(self, detections: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        """
        Denormalize bounding box coordinates from [0, 1] range to pixel coordinates.
        
        Args:
            detections: List of detection dictionaries
            image_size: (width, height) of image
            
        Returns:
            Detections with pixel coordinates
        """
        width, height = image_size
        
        denormalized = []
        for det in detections.copy():
            bbox = det['bbox']
            
            # Denormalize coordinates
            pixel_bbox = [
                bbox[0] * width,   # x1
                bbox[1] * height,  # y1
                bbox[2] * width,   # x2
                bbox[3] * height   # y2
            ]
            
            # Update detection
            det_copy = det.copy()
            det_copy['bbox'] = pixel_bbox
            denormalized.append(det_copy)
        
        return denormalized
    
    def clip_to_image(self, detections: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        """
        Clip bounding boxes to image boundaries.
        
        Args:
            detections: List of detection dictionaries
            image_size: (width, height) of image
            
        Returns:
            Detections with clipped coordinates
        """
        width, height = image_size
        
        clipped = []
        for det in detections.copy():
            bbox = det['bbox']
            
            # Clip coordinates
            clipped_bbox = [
                max(0, min(bbox[0], width)),   # x1
                max(0, min(bbox[1], height)),  # y1
                max(0, min(bbox[2], width)),   # x2
                max(0, min(bbox[3], height))   # y2
            ]
            
            # Skip invalid boxes
            if clipped_bbox[2] <= clipped_bbox[0] or clipped_bbox[3] <= clipped_bbox[1]:
                continue
            
            # Update detection
            det_copy = det.copy()
            det_copy['bbox'] = clipped_bbox
            clipped.append(det_copy)
        
        return clipped
    
    def merge_detections(self, detection_groups: List[List[Dict]]) -> List[Dict]:
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
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
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