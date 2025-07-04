"""
RT-DETR Inference Engine for ML Detection Ensemble.
Handles RT-DETR model variants (full-frame only).
"""

import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger('wildcams')

# Model detection threshold - minimal value to see ALL detections before ensemble filtering
MODEL_DETECTION_THRESHOLD = 0.001

class RTDETRInferenceEngine:
    """Handles inference for RT-DETR model variants."""
    
    def __init__(self, model_manager):
        """
        Initialize with model manager.
        
        Args:
            model_manager: ModelManager instance with loaded RT-DETR models
        """
        self.model_manager = model_manager
    
    def run_detection(self, model_name: str, frame: np.ndarray, full_frame: np.ndarray = None, **kwargs) -> List[Dict]:
        """
        Run RT-DETR detection on a frame.
        
        Args:
            model_name: Name of RT-DETR model to use
            frame: Input frame (ignored for RT-DETR, uses full_frame)
            full_frame: Full frame for detection (required for RT-DETR)
            **kwargs: Additional arguments
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if not self.model_manager.is_rtdetr_model(model_name):
            logger.error(f"{model_name} is not an RT-DETR model")
            return detections
        
        if full_frame is None:
            logger.error(f"RT-DETR model {model_name} requires full_frame")
            return detections
        
        rtdetr_model = self.model_manager.get_model(model_name)
        if rtdetr_model is None:
            logger.error(f"RT-DETR model {model_name} not available")
            return detections
        
        try:
            results = rtdetr_model(full_frame, conf=MODEL_DETECTION_THRESHOLD, verbose=False)
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf)
                        bbox = box.xyxy.tolist()[0]
                        
                        detection = {
                            'confidence': confidence,
                            'bbox': bbox,
                            'source': f'rtdetr_{model_name}',
                            'class': 'animal'  # RT-DETR models detect generic animals
                        }
                        detections.append(detection)
                        
        except Exception as e:
            logger.error(f"{model_name} RT-DETR model failed: {e}")
        
        return detections
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported RT-DETR model names."""
        rtdetr_variants = ['rtdetr-l', 'rtdetr-x']
        
        # Return only models that are actually loaded
        available = []
        for variant in rtdetr_variants:
            if self.model_manager.get_model(variant) is not None:
                available.append(variant)
        
        return available
    
    def supports_crops(self) -> bool:
        """Check if this inference engine supports crop-based detection."""
        return False  # RT-DETR is full-frame only
    
    def supports_full_frame(self) -> bool:
        """Check if this inference engine supports full-frame detection."""
        return True