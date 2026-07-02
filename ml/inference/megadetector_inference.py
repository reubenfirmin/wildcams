"""
MegaDetector Inference Engine for ML Detection Ensemble.
Handles MegaDetector v6 model variants.
"""

import cv2
import logging
from typing import List
import numpy as np

# Import constants
from ..constants import MODEL_DETECTION_THRESHOLD
from core.data_types import Detection, BoundingBox

logger = logging.getLogger('wildcams')

class MegaDetectorInferenceEngine:
    """Handles inference for MegaDetector v6 model variants."""
    
    def __init__(self, model_manager):
        """
        Initialize with model manager.
        
        Args:
            model_manager: ModelManager instance with loaded MegaDetector models
        """
        self.model_manager = model_manager
    
    def run_detection(self, model_name: str, frame: np.ndarray, config, full_frame: np.ndarray = None, 
                     timestamp_seconds: float = 0.0, frame_idx: int = 0) -> List[Detection]:
        """
        Run MegaDetector detection on a frame.
        
        Args:
            model_name: Name of MegaDetector model to use
            frame: Input frame (ignored for MegaDetector, uses full_frame)
            config: Processing configuration
            full_frame: Full frame for detection (required for MegaDetector)
            timestamp_seconds: Timestamp of the frame in seconds
            frame_idx: Index of the frame in the video
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if not self.model_manager.is_megadetector_model(model_name):
            logger.error(f"{model_name} is not a MegaDetector model")
            return detections
        
        if full_frame is None:
            logger.error(f"MegaDetector model {model_name} requires full_frame")
            return detections
        
        md_model = self.model_manager.get_model(model_name)
        if md_model is None:
            logger.error(f"MegaDetector model {model_name} not available")
            return detections
        
        # Convert BGR to RGB for MegaDetector
        rgb_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
        
        # Get model-specific threshold
        threshold = self.model_manager.get_model_threshold(model_name, config)
        
        # Run MegaDetector inference
        results = md_model.single_image_detection(
            rgb_frame,
            det_conf_thres=threshold
        )
        
        # Process MegaDetector results
        if results and 'detections' in results:
            for detection in results['detections']:
                # Handle case where detection might be a tuple instead of dict
                if isinstance(detection, tuple):
                    # Parse tuple format: (bbox_array, ?, confidence, class_id, ?, metadata)
                    if len(detection) >= 4:
                        bbox_array = detection[0]
                        confidence = float(detection[2])
                        class_id = int(detection[3])
                        
                        # Convert numpy array bbox to list
                        if hasattr(bbox_array, 'tolist'):
                            bbox = bbox_array.tolist()
                        else:
                            bbox = list(bbox_array)
                        
                        # Map class ID to category (MegaDetector standard classes)
                        class_mapping = {0: 'empty', 1: 'animal', 2: 'person', 3: 'vehicle'}
                        category = class_mapping.get(class_id, 'unknown')
                        
                        logger.debug(f"MegaDetector tuple parsed: bbox={bbox}, conf={confidence:.3f}, class={category}")
                    else:
                        logger.warning(f"MegaDetector returned malformed tuple: {detection}")
                        continue
                else:
                    # Standard dict format
                    bbox = detection.get('bbox', [])
                    confidence = detection.get('conf', 0.0)
                    category = detection.get('category', 'unknown')
                
                if len(bbox) == 4 and confidence > 0:
                    # MegaDetector returns normalized coordinates, convert to pixels
                    h, w = full_frame.shape[:2]
                    x1, y1, width, height = bbox
                    x2 = x1 + width
                    y2 = y1 + height
                    
                    # Convert to absolute coordinates
                    pixel_bbox = [
                        x1 * w,  # x1
                        y1 * h,  # y1
                        x2 * w,  # x2
                        y2 * h   # y2
                    ]
                    
                    detections.append(Detection(
                        confidence=float(confidence),
                        bbox=BoundingBox(pixel_bbox[0], pixel_bbox[1], pixel_bbox[2], pixel_bbox[3]),
                        source=model_name,
                        class_name=category,
                        timestamp=timestamp_seconds,
                        frame_idx=frame_idx,
                    ))
        
        return detections
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported MegaDetector model names."""
        megadetector_variants = [
            'MDV6-yolov9-e', 'MDV6-yolov9-c',
            'MDV6-yolov10-e', 'MDV6-yolov10-c',
            'MDV6-rtdetr-c'
        ]
        
        # Return only models that are actually loaded
        available = []
        for variant in megadetector_variants:
            if self.model_manager.get_model(variant) is not None:
                available.append(variant)
        
        return available
    
    def supports_crops(self) -> bool:
        """Check if this inference engine supports crop-based detection."""
        return False  # MegaDetector is full-frame only
    
    def supports_full_frame(self) -> bool:
        """Check if this inference engine supports full-frame detection."""
        return True
    
    def check_extended_class_correlations(self, detections: List[Detection], yolo_detections: List[Detection], timestamp: float) -> None:
        """Check for correlations between MegaDetector extended classes and YOLO detections."""
        megadetector_extended = [d for d in detections if d.source.startswith('MDV6-') and getattr(d, 'raw_class_id', None)]
        
        if not megadetector_extended or not yolo_detections:
            return
            
        # Check for bbox overlaps (IoU > 0.3 indicates correlation)
        for md_det in megadetector_extended:
            md_bbox = [md_det.bbox.x1, md_det.bbox.y1, md_det.bbox.x2, md_det.bbox.y2]
            md_class = md_det.class_name
            md_conf = md_det.confidence

            for yolo_det in yolo_detections:
                yolo_bbox = [yolo_det.bbox.x1, yolo_det.bbox.y1, yolo_det.bbox.x2, yolo_det.bbox.y2]
                yolo_source = yolo_det.source
                yolo_conf = yolo_det.confidence
                
                iou = self._calculate_iou(md_bbox, yolo_bbox)
                if iou > 0.3:  # Significant overlap
                    logger.warning(f"🚨🚨🚨 CORRELATION DETECTED 🚨🚨🚨")
                    logger.warning(f"⚡ MegaDetector extended class {md_class} (conf={md_conf:.3f}) correlates with {yolo_source} (conf={yolo_conf:.3f})")
                    logger.warning(f"📍 IoU={iou:.3f}, timestamp={timestamp:.2f}s")
                    logger.warning(f"📊 MegaDetector bbox: {[round(x,1) for x in md_bbox]}")
                    logger.warning(f"📊 YOLO bbox: {[round(x,1) for x in yolo_bbox]}")
                    logger.warning(f"🔥 CONSIDER ADDING CLASS {md_class} TO WHITELIST!")
                    logger.warning(f"🚨🚨🚨 END CORRELATION 🚨🚨🚨")
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area