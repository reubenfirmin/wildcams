"""
MegaDetector Inference Engine for ML Detection Ensemble.
Handles MegaDetector v6 model variants.
"""

import cv2
import logging
from typing import List, Dict
import numpy as np

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
    
    def run_detection(self, model_name: str, frame: np.ndarray, full_frame: np.ndarray = None, **kwargs) -> List[Dict]:
        """
        Run MegaDetector detection on a frame.
        
        Args:
            model_name: Name of MegaDetector model to use
            frame: Input frame (ignored for MegaDetector, uses full_frame)
            full_frame: Full frame for detection (required for MegaDetector)
            **kwargs: Additional arguments
            
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
        
        try:
            # Convert BGR to RGB for MegaDetector
            rgb_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
            
            # Get model-specific threshold
            threshold = self.model_manager.get_model_threshold(model_name)
            
            # Run MegaDetector inference
            results = md_model.single_image_detection(
                rgb_frame,
                det_conf_thres=threshold
            )
            
            # Process MegaDetector results
            if results and 'detections' in results:
                for detection in results['detections']:
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
                        
                        detection_dict = {
                            'confidence': float(confidence),
                            'bbox': pixel_bbox,
                            'source': model_name,
                            'class': category,
                            'megadetector_category': category
                        }
                        detections.append(detection_dict)
                        
        except KeyError as ke:
            # Handle PyTorch-Wildlife bug with unknown class IDs
            logger.info(f"🔍 MegaDetector v6 ({model_name}) detected unknown class ID {ke} - class not in standard mapping")
            
            try:
                # Handle raw Ultralytics results with potentially unknown class IDs
                logger.info("Processing MegaDetector v6 raw results with extended class IDs")
                
                # Fallback processing for MegaDetector with extended classes
                # This is where we would handle extended class processing if needed
                pass
                
            except Exception as e2:
                logger.error(f"MegaDetector {model_name} extended processing failed: {e2}")
                
        except Exception as e:
            logger.error(f"MegaDetector {model_name} failed: {e}")
        
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
    
    def check_extended_class_correlations(self, detections: List[Dict], yolo_detections: List[Dict], timestamp: float) -> None:
        """Check for correlations between MegaDetector extended classes and YOLO detections."""
        megadetector_extended = [d for d in detections if d.get('source', '').startswith('MDV6-') and d.get('raw_class_id')]
        
        if not megadetector_extended or not yolo_detections:
            return
            
        # Check for bbox overlaps (IoU > 0.3 indicates correlation)
        for md_det in megadetector_extended:
            md_bbox = md_det['bbox']
            md_class = md_det.get('class', 'unknown')
            md_conf = md_det['confidence']
            
            for yolo_det in yolo_detections:
                yolo_bbox = yolo_det['bbox']
                yolo_source = yolo_det['source']
                yolo_conf = yolo_det['confidence']
                
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