"""
Ensemble Wrapper for backward compatibility.
Provides the same interface as the original MLDetectionEnsemble.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

from .model_manager import ModelManager
from .feature_extractor import FeatureExtractor
from .inference.ensemble_coordinator import EnsembleCoordinator

logger = logging.getLogger('wildcams')

class MLDetectionEnsemble:
    """
    Backward-compatible wrapper for the refactored ML detection ensemble.
    Provides the same interface as the original monolithic class.
    """
    
    def __init__(self, confidence_threshold: float = 0.1, ensemble_models: List[str] = None, 
                 cache_dir: Optional[Path] = None):
        """
        Initialize the ensemble with the same interface as before.
        
        Args:
            confidence_threshold: Confidence threshold for detections
            ensemble_models: List of model names to use
            cache_dir: Cache directory for models
        """
        self.confidence_threshold = confidence_threshold
        self.ensemble_models = ensemble_models or []
        self.cache_dir = cache_dir or Path('./models_cache/torch')
        
        # Initialize components
        self.model_manager = ModelManager(self.ensemble_models, self.cache_dir)
        self.ensemble_coordinator = EnsembleCoordinator(self.model_manager)
        
        # Initialize feature extractor
        self.feature_extractor_component = None
        if self.model_manager.feature_extractor is not None:
            self.feature_extractor_component = FeatureExtractor(self.model_manager.feature_extractor)
        
        # Store model thresholds for backward compatibility
        self.model_thresholds = self.model_manager.model_thresholds
        
        # Detection scales for backward compatibility
        self.detection_scales = self.ensemble_coordinator.preprocessor.detection_scales
        
        # TTA settings for backward compatibility
        self.enable_tta = self.ensemble_coordinator.preprocessor.enable_tta
        self.tta_transforms = self.ensemble_coordinator.preprocessor.tta_transforms
    
    # Backward compatibility methods
    
    def run_single_model_detection(self, model_name: str, frame: np.ndarray, 
                                 timestamp_seconds: float = 0.0, frame_idx: int = 0, 
                                 full_frame: np.ndarray = None, 
                                 accepted_rtdetr_overlap: float = 0.5) -> List[Dict]:
        """Run detection on a single model - backward compatibility."""
        return self.ensemble_coordinator.run_single_model_detection(
            model_name, frame, timestamp_seconds, frame_idx, full_frame, accepted_rtdetr_overlap
        )
    
    def run_ensemble_detection(self, frame: np.ndarray, timestamp_seconds: float = 0.0, 
                             frame_idx: int = 0, full_frame: np.ndarray = None) -> List[Dict]:
        """Run ensemble detection - backward compatibility."""
        return self.ensemble_coordinator.run_ensemble_detection(
            frame, timestamp_seconds, frame_idx, full_frame
        )
    
    def apply_tta_transforms(self, frame: np.ndarray):
        """Apply TTA transforms - backward compatibility."""
        return self.ensemble_coordinator.preprocessor.apply_tta_transforms(frame)
    
    def apply_multiscale_detection(self, frame: np.ndarray):
        """Apply multiscale detection - backward compatibility."""
        return self.ensemble_coordinator.preprocessor.apply_multiscale_detection(frame)
    
    def apply_advanced_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply advanced NMS - backward compatibility."""
        return self.ensemble_coordinator.postprocessor.apply_advanced_nms(detections, iou_threshold)
    
    def run_enhanced_preprocessing(self, frame: np.ndarray) -> List[Dict]:
        """Run enhanced preprocessing - backward compatibility."""
        return self.ensemble_coordinator.run_enhanced_preprocessing(frame)
    
    def run_multiscale_analysis(self, frame: np.ndarray, timestamp_seconds: float = 0.0) -> List[Dict]:
        """Run multiscale analysis - backward compatibility."""
        return self.ensemble_coordinator.run_multiscale_analysis(frame, timestamp_seconds)
    
    def extract_features(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract features - backward compatibility."""
        if self.feature_extractor_component is None:
            return None
        return self.feature_extractor_component.extract_features(frame, bbox)
    
    def _check_extended_class_correlations(self, detections: List[Dict], timestamp: float) -> None:
        """Check extended class correlations - backward compatibility."""
        self.ensemble_coordinator.check_extended_class_correlations(detections, timestamp)
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU - backward compatibility."""
        return self.ensemble_coordinator.postprocessor.calculate_iou(bbox1, bbox2)
    
    # Property accessors for backward compatibility
    
    @property
    def yolo_detectors(self):
        """Access to YOLO detectors - backward compatibility."""
        return self.model_manager.yolo_detectors
    
    @property
    def rtdetr_models(self):
        """Access to RT-DETR models - backward compatibility."""
        return self.model_manager.rtdetr_models
    
    @property
    def megadetector_variants(self):
        """Access to MegaDetector variants - backward compatibility."""
        return self.model_manager.megadetector_variants
    
    @property
    def feature_extractor(self):
        """Access to feature extractor - backward compatibility."""
        return self.model_manager.feature_extractor