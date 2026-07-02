"""
YOLO Inference Engine for ML Detection Ensemble.
Handles all YOLO model variants (v8, v10, v12).
"""

import logging
from typing import List
import numpy as np

logger = logging.getLogger('wildcams')

# Import constants
from ..constants import MODEL_DETECTION_THRESHOLD
from core.data_types import Detection, BoundingBox

class YOLOInferenceEngine:
    """Handles inference for YOLO model variants."""
    
    def __init__(self, model_manager):
        """
        Initialize with model manager.
        
        Args:
            model_manager: ModelManager instance with loaded YOLO models
        """
        self.model_manager = model_manager
    
    def run_detection(self, model_name: str, frame: np.ndarray, config,
                     timestamp_seconds: float = 0.0, frame_idx: int = 0) -> List[Detection]:
        """
        Run YOLO detection on a frame.
        
        Args:
            model_name: Name of YOLO model to use
            frame: Input frame
            config: Processing configuration
            timestamp_seconds: Timestamp of the frame in seconds
            frame_idx: Index of the frame in the video
            
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

                    detections.append(Detection(
                        confidence=confidence,
                        bbox=BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3]),
                        source=model_name,
                        class_name='animal',  # YOLO models detect generic animals
                        timestamp=timestamp_seconds,
                        frame_idx=frame_idx,
                    ))
        
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