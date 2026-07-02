"""
Ensemble Wrapper for backward compatibility.
Provides the same interface as the original MLDetectionEnsemble.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.data_types import Detection

from .inference.ensemble_coordinator import EnsembleCoordinator
from .model_manager import ModelManager

logger = logging.getLogger("wildcams")


class MLDetectionEnsemble:
    """
    Backward-compatible wrapper for the refactored ML detection ensemble.
    Provides the same interface as the original monolithic class.
    """

    def __init__(
        self, confidence_threshold: float = 0.1, ensemble_models: list[str] | None = None, cache_dir: Path | None = None
    ):
        """
        Initialize the ensemble with the same interface as before.

        Args:
            confidence_threshold: Confidence threshold for detections
            ensemble_models: List of model names to use
            cache_dir: Cache directory for models
        """
        self.confidence_threshold = confidence_threshold
        self.ensemble_models = ensemble_models or []
        self.cache_dir = cache_dir or Path("./models_cache/torch")

        # Initialize components
        self.model_manager = ModelManager(self.ensemble_models, self.cache_dir)
        self.ensemble_coordinator = EnsembleCoordinator(self.model_manager)

        # Use global confidence threshold only
        self.confidence_threshold = confidence_threshold

        # Detection scales for backward compatibility
        self.detection_scales = self.ensemble_coordinator.preprocessor.detection_scales

        # TTA settings for backward compatibility
        self.enable_tta = self.ensemble_coordinator.preprocessor.enable_tta
        self.tta_transforms = self.ensemble_coordinator.preprocessor.tta_transforms

    # Backward compatibility methods

    def run_single_model_detection(
        self,
        model_name: str,
        frame: np.ndarray,
        config,
        timestamp_seconds: float = 0.0,
        frame_idx: int = 0,
        full_frame: np.ndarray | None = None,
        accepted_rtdetr_overlap: float = 0.5,
    ) -> list[Detection]:
        """Run detection on a single model - backward compatibility."""
        return self.ensemble_coordinator.run_single_model_detection(
            model_name, frame, config, timestamp_seconds, frame_idx, full_frame, accepted_rtdetr_overlap
        )

    def _calculate_iou(self, bbox1: list[float], bbox2: list[float]) -> float:
        """Calculate IoU - backward compatibility."""
        return self.ensemble_coordinator.postprocessor.calculate_iou(bbox1, bbox2)

    # Property accessors for backward compatibility

    @property
    def yolo_detectors(self) -> dict[str, Any]:
        """Access to YOLO detectors - backward compatibility."""
        return self.model_manager.yolo_detectors

    @property
    def rtdetr_models(self) -> dict[str, Any]:
        """Access to RT-DETR models - backward compatibility."""
        return self.model_manager.rtdetr_models

    @property
    def megadetector_variants(self) -> dict[str, Any]:
        """Access to MegaDetector variants - backward compatibility."""
        return self.model_manager.megadetector_variants
