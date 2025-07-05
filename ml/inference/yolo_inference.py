"""
YOLO Inference Engine for ML Detection Ensemble.
Handles all YOLO model variants (v8, v10, v12).
"""

import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger('wildcams')

# Import constants
from ..constants import MODEL_DETECTION_THRESHOLD

class YOLOInferenceEngine:
    """Handles inference for YOLO model variants."""
    
    def __init__(self, model_manager):
        """
        Initialize with model manager.
        
        Args:
            model_manager: ModelManager instance with loaded YOLO models
        """
        self.model_manager = model_manager
    
    def run_detection(self, model_name: str, frame: np.ndarray, config, **kwargs) -> List[Dict]:
        """
        Run YOLO detection on a frame.
        
        Args:
            model_name: Name of YOLO model to use
            frame: Input frame
            **kwargs: Additional arguments (ignored for YOLO)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if not self.model_manager.is_yolo_model(model_name):
            logger.error(f"{model_name} is not a YOLO model")
            return detections
        
        detector = self.model_manager.get_model(model_name)
        if detector is None:
            logger.error(f"YOLO model {model_name} not available")
            return detections
        
        results = detector(frame, conf=MODEL_DETECTION_THRESHOLD, verbose=False)
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf)
                    bbox = box.xyxy.tolist()[0]
                    
                    detection = {
                        'confidence': confidence,
                        'bbox': bbox,
                        'source': model_name,
                        'class': 'animal'  # YOLO models detect generic animals
                    }
                    detections.append(detection)
        
        return detections
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported YOLO model names."""
        yolo_variants = [
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x',
            'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x'
        ]
        
        # Return only models that are actually loaded
        available = []
        for variant in yolo_variants:
            if self.model_manager.get_model(variant) is not None:
                available.append(variant)
        
        return available
    
    def supports_crops(self) -> bool:
        """Check if this inference engine supports crop-based detection."""
        return True
    
    def supports_full_frame(self) -> bool:
        """Check if this inference engine supports full-frame detection."""
        return True