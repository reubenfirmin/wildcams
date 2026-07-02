"""
Model Manager for ML Detection Ensemble.
Handles loading, caching, and management of all ML models.
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Set torch cache directory BEFORE importing torch
def setup_torch_cache(cache_dir: Path):
    """Setup torch cache directory."""
    cache_dir = cache_dir.absolute()
    cache_dir.mkdir(exist_ok=True)
    os.environ['TORCH_HOME'] = str(cache_dir)

# Import torch after cache setup
import torch

from ultralytics import YOLO

# PyTorch-Wildlife imports
try:
    from PytorchWildlife.models import detection as pw_detection
    PYTORCH_WILDLIFE_AVAILABLE = True
except ImportError:
    PYTORCH_WILDLIFE_AVAILABLE = False

# Import ResNet for feature extraction

logger = logging.getLogger('wildcams')

class ModelManager:
    """Manages loading and caching of all ML models in the ensemble."""
    
    def __init__(self, ensemble_models: List[str], cache_dir: Optional[Path] = None):
        self.ensemble_models = ensemble_models
        self.cache_dir = cache_dir or Path('./models_cache/torch')
        
        # Setup torch cache
        setup_torch_cache(self.cache_dir)
        torch.hub.set_dir(str(self.cache_dir))
        
        # Model storage
        self.yolo_detectors: Dict[str, Optional[YOLO]] = {}
        self.rtdetr_models: Dict[str, Optional[object]] = {}
        self.megadetector_variants: Dict[str, Optional[object]] = {}

        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models based on ensemble configuration."""
        try:
            logger.info(f"🤖 Initializing ML detection ensemble with models: {self.ensemble_models}")
            logger.info(f"📦 Model cache directory: {self.cache_dir}")
            logger.info(f"🔧 TORCH_HOME: {os.environ.get('TORCH_HOME', 'not set')}")
            logger.info(f"🔧 torch.hub.get_dir(): {torch.hub.get_dir()}")
            
            self._load_yolo_models()
            self._load_rtdetr_models()
            self._load_megadetector_models()

        except Exception as e:
            logger.error(f"❌ Failed to initialize ML ensemble: {e}")
    
    def _load_yolo_models(self):
        """Load YOLO model variants."""
        all_yolo_variants = [
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x',
            'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x'
        ]
        
        for variant in all_yolo_variants:
            if variant in self.ensemble_models:
                try:
                    logger.info(f"Loading {variant.upper()} detector...")
                    self.yolo_detectors[variant] = YOLO(f'{variant}.pt')
                    logger.info(f"✅ {variant.upper()} detector loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load {variant.upper()} detector: {e}")
                    self.yolo_detectors[variant] = None
            else:
                logger.debug(f"⏭️ Skipping {variant.upper()} (not in ensemble configuration)")
    
    def _load_rtdetr_models(self):
        """Load RT-DETR model variants."""
        rtdetr_variants = ['rtdetr-l', 'rtdetr-x']
        
        for variant in rtdetr_variants:
            if variant in self.ensemble_models:
                try:
                    logger.info(f"🔬 Loading RT-DETR model: {variant}")
                    from ultralytics import RTDETR
                    self.rtdetr_models[variant] = RTDETR(f'{variant}.pt')
                    logger.info(f"✅ {variant.upper()} RT-DETR loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load RT-DETR {variant}: {e}")
                    self.rtdetr_models[variant] = None
            else:
                logger.debug(f"⏭️ Skipping {variant.upper()} RT-DETR (not in ensemble configuration)")
    
    def _load_megadetector_models(self):
        """Load MegaDetector model variants."""
        megadetector_variants_in_ensemble = [
            model for model in self.ensemble_models 
            if model.startswith('MDV6-') and 'rtdetr' not in model.lower()
        ]
        
        if megadetector_variants_in_ensemble and PYTORCH_WILDLIFE_AVAILABLE:
            for variant in megadetector_variants_in_ensemble:
                try:
                    logger.info(f"🦎 Loading MegaDetector v6 variant: {variant}")
                    
                    # Load model - PyTorch-Wildlife should use environment cache settings
                    md_model = pw_detection.MegaDetectorV6(
                        version=variant, 
                        pretrained=True,
                        device='auto'  # Let PyTorch-Wildlife choose best device
                    )
                    
                    # Disable verbose to prevent KeyError with unknown class IDs
                    if hasattr(md_model, 'predictor') and hasattr(md_model.predictor, 'args'):
                        md_model.predictor.args.verbose = False
                    
                    self.megadetector_variants[variant] = md_model
                    logger.info(f"✅ MegaDetector v6 ({variant}) loaded successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load MegaDetector v6 variant {variant}: {e}")
                    self.megadetector_variants[variant] = None
            
            # Log cache status
            if self.cache_dir and self.megadetector_variants:
                logger.info(f"📦 Models cached to: {self.cache_dir / 'pytorch_wildlife'}")
                    
        elif megadetector_variants_in_ensemble:
            logger.warning("⚠️ PyTorch-Wildlife not available - MegaDetector variants disabled")
        else:
            logger.info("⏭️ Skipping MegaDetector variants (none in ensemble configuration)")
    
    def get_model(self, model_name: str):
        """Get a specific model by name."""
        # Check YOLO models
        if model_name in self.yolo_detectors:
            return self.yolo_detectors[model_name]
        
        # Check RT-DETR models
        if model_name in self.rtdetr_models:
            return self.rtdetr_models[model_name]
        
        # Check MegaDetector variants
        if model_name in self.megadetector_variants:
            return self.megadetector_variants[model_name]
        
        return None
    
    def get_model_threshold(self, model_name: str, config) -> float:
        """Get the confidence threshold for a specific model from config."""
        return config.confidence_threshold
    
    def get_available_models(self) -> List[str]:
        """Get list of successfully loaded models."""
        available = []
        
        for name, model in self.yolo_detectors.items():
            if model is not None:
                available.append(name)
        
        for name, model in self.rtdetr_models.items():
            if model is not None:
                available.append(name)
        
        for name, model in self.megadetector_variants.items():
            if model is not None:
                available.append(name)
        
        return available
    
    def is_yolo_model(self, model_name: str) -> bool:
        """Check if model is a YOLO variant."""
        return model_name in self.yolo_detectors
    
    def is_rtdetr_model(self, model_name: str) -> bool:
        """Check if model is an RT-DETR variant."""
        return model_name in self.rtdetr_models
    
    def is_megadetector_model(self, model_name: str) -> bool:
        """Check if model is a MegaDetector variant."""
        return model_name in self.megadetector_variants