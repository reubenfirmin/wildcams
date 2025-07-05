"""
Ensemble Coordinator for ML Detection Ensemble.
Orchestrates inference across multiple model types and coordinates results.
"""

import logging
from typing import List, Dict, Optional
import numpy as np

from .yolo_inference import YOLOInferenceEngine
from .rtdetr_inference import RTDETRInferenceEngine
from .megadetector_inference import MegaDetectorInferenceEngine
from ..preprocessing import PreprocessingPipeline
from ..postprocessing import PostprocessingPipeline

logger = logging.getLogger('wildcams')

class EnsembleCoordinator:
    """Coordinates inference across all model types in the ensemble."""
    
    def __init__(self, model_manager, enable_tta: bool = True, enable_multiscale: bool = True):
        """
        Initialize ensemble coordinator.
        
        Args:
            model_manager: ModelManager instance with loaded models
            enable_tta: Enable Test-Time Augmentation
            enable_multiscale: Enable multi-scale detection
        """
        self.model_manager = model_manager
        
        # Initialize inference engines
        self.yolo_engine = YOLOInferenceEngine(model_manager)
        self.rtdetr_engine = RTDETRInferenceEngine(model_manager)
        self.megadetector_engine = MegaDetectorInferenceEngine(model_manager)
        
        # Initialize preprocessing and postprocessing pipelines
        self.preprocessor = PreprocessingPipeline(enable_tta=enable_tta, enable_multiscale=enable_multiscale)
        self.postprocessor = PostprocessingPipeline()
        
        # Get available models
        self.ensemble_models = model_manager.ensemble_models
    
    def run_single_model_detection(self, model_name: str, frame: np.ndarray, config,
                                 timestamp_seconds: float = 0.0, frame_idx: int = 0, 
                                 full_frame: np.ndarray = None, 
                                 accepted_rtdetr_overlap: float = 0.5) -> List[Dict]:
        """
        Run detection on a single model and return results.
        
        Args:
            model_name: Name of model to use
            frame: Input frame (for crop-based models)
            timestamp_seconds: Timestamp of frame
            frame_idx: Frame index
            full_frame: Full frame (for full-frame only models)
            accepted_rtdetr_overlap: Overlap threshold for RT-DETR
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Route to appropriate inference engine
        if self.model_manager.is_yolo_model(model_name):
            detections = self.yolo_engine.run_detection(
                model_name=model_name,
                frame=frame,
                config=config,
                timestamp_seconds=timestamp_seconds,
                frame_idx=frame_idx
            )
        elif self.model_manager.is_rtdetr_model(model_name):
            detections = self.rtdetr_engine.run_detection(
                model_name=model_name,
                frame=frame,
                config=config,
                full_frame=full_frame,
                timestamp_seconds=timestamp_seconds,
                frame_idx=frame_idx
            )
        elif self.model_manager.is_megadetector_model(model_name):
            detections = self.megadetector_engine.run_detection(
                model_name=model_name,
                frame=frame,
                config=config,
                full_frame=full_frame,
                timestamp_seconds=timestamp_seconds,
                frame_idx=frame_idx
            )
        else:
            logger.error(f"Unknown model type for {model_name}")
        
        return detections
    
    def run_ensemble_detection(self, frame: np.ndarray, config, timestamp_seconds: float = 0.0, 
                             frame_idx: int = 0, full_frame: np.ndarray = None) -> List[Dict]:
        """
        Run detection across all models in the ensemble.
        
        Args:
            frame: Input frame
            timestamp_seconds: Timestamp of frame
            frame_idx: Frame index
            full_frame: Full frame for full-frame only models
            
        Returns:
            List of all detections from ensemble
        """
        all_detections = []
        
        for model_name in self.ensemble_models:
            model_detections = self.run_single_model_detection(
                model_name=model_name,
                frame=frame,
                config=config,
                timestamp_seconds=timestamp_seconds,
                frame_idx=frame_idx,
                full_frame=full_frame
            )
            all_detections.extend(model_detections)
        
        return all_detections
    
    def run_enhanced_preprocessing(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection with enhanced preprocessing (TTA, multi-scale).
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections from enhanced preprocessing
        """
        all_detections = []
        
        # Apply TTA transforms
        tta_frames = self.preprocessor.apply_tta_transforms(frame)
        
        for transformed_frame, transform_name in tta_frames:
            # Apply multi-scale detection
            multiscale_frames = self.preprocessor.apply_multiscale_detection(transformed_frame)
            
            for scaled_frame, scale_factor, scale_name in multiscale_frames:
                # Run ensemble on this processed frame
                detections = self.run_ensemble_detection(
                    frame=scaled_frame,
                    full_frame=scaled_frame  # Use scaled frame as full frame too
                )
                
                # Adjust coordinates back to original scale
                if scale_factor != 1.0:
                    for det in detections:
                        bbox = det['bbox']
                        det['bbox'] = [coord / scale_factor for coord in bbox]
                
                # Add processing metadata
                for det in detections:
                    det['transform'] = transform_name
                    det['scale'] = scale_name
                
                all_detections.extend(detections)
        
        return all_detections
    
    def run_multiscale_analysis(self, frame: np.ndarray, timestamp_seconds: float = 0.0) -> List[Dict]:
        """
        Run detection with multi-scale analysis.
        
        Args:
            frame: Input frame
            timestamp_seconds: Timestamp of frame
            
        Returns:
            List of detections from multi-scale analysis
        """
        detections = []
        
        # Get multi-scale frames
        multiscale_frames = self.preprocessor.apply_multiscale_detection(frame)
        
        for scaled_frame, scale_factor, scale_name in multiscale_frames:
            # Run ensemble detection on scaled frame
            scale_detections = self.run_ensemble_detection(
                frame=scaled_frame,
                timestamp_seconds=timestamp_seconds,
                full_frame=scaled_frame
            )
            
            # Adjust coordinates back to original scale
            if scale_factor != 1.0:
                scale_detections = self.postprocessor.convert_coordinates(
                    scale_detections,
                    source_size=(int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)),
                    target_size=(frame.shape[1], frame.shape[0])
                )
            
            # Add scale metadata
            for det in scale_detections:
                det['scale_factor'] = scale_factor
                det['scale_name'] = scale_name
            
            detections.extend(scale_detections)
        
        return detections
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models across all engines."""
        available = []
        available.extend(self.yolo_engine.get_supported_models())
        available.extend(self.rtdetr_engine.get_supported_models())
        available.extend(self.megadetector_engine.get_supported_models())
        return available
    
    def check_extended_class_correlations(self, detections: List[Dict], timestamp: float) -> None:
        """Check for correlations between MegaDetector extended classes and YOLO detections."""
        yolo_detections = [d for d in detections if any(d.get('source', '').startswith(model) for model in self.yolo_engine.get_supported_models())]
        
        if yolo_detections:
            self.megadetector_engine.check_extended_class_correlations(detections, yolo_detections, timestamp)