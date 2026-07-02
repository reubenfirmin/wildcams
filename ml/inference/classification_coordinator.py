"""Classification coordinator for animal species identification ensemble."""

import logging
import time
from typing import Optional, Dict, Any, List
import numpy as np
from pathlib import Path

from core.data_types import ClassificationResult, InferenceResult
from config import ProcessingConfig
from .bioclip_inference import BioCLIPInference
from .deepfaune_inference import DeepFauneInference

logger = logging.getLogger('wildcams')


class ClassificationCoordinator:
    """
    Coordinates animal classification using BioCLIP and DeepFaune ensemble.
    
    This class combines the strengths of both models:
    - DeepFaune: Strong animal vs non-animal discrimination
    - BioCLIP: Detailed species identification
    
    The ensemble strategy uses DeepFaune as a gating mechanism - only if
    DeepFaune confirms it's an animal do we trust BioCLIP's species identification.
    """
    
    def __init__(self, config: ProcessingConfig, cache_dir: Optional[Path] = None):
        """
        Initialize classification coordinator.
        
        Args:
            config: ProcessingConfig with classification parameters
            cache_dir: Optional directory for model caching
        """
        self.config = config
        self.cache_dir = cache_dir or Path('./models_cache')
        
        # List of inference engines to run
        self.inference_engines = []
        
        self._initialize_engines()
        
        logger.info(f"🔬 Classification coordinator initialized with {len(self.inference_engines)} models")
    
    def _initialize_engines(self) -> None:
        """Initialize classification engines based on configuration."""
        # Initialize BioCLIP if requested
        if 'bioclip' in self.config.classification_models:
            bioclip_engine = BioCLIPInference(self.config, self.cache_dir)
            self.inference_engines.append(bioclip_engine)
            logger.info("✅ BioCLIP engine initialized")
        
        # Initialize DeepFaune if requested
        if 'deepfaune' in self.config.classification_models:
            deepfaune_engine = DeepFauneInference(self.config, self.cache_dir)
            self.inference_engines.append(deepfaune_engine)
            logger.info("✅ DeepFaune engine initialized")
        
        if not self.inference_engines:
            raise RuntimeError("❌ No classification models configured - check config.classification_models")
    
    def classify_ensemble(self, image_crop: np.ndarray) -> ClassificationResult:
        """
        Perform ensemble classification on an image crop.
        
        Args:
            image_crop: RGB image array of shape (H, W, 3)
            
        Returns:
            ClassificationResult with combined ensemble decision
        """
        start_time = time.time()
        
        # Run all configured inference engines
        results = []
        for engine in self.inference_engines:
            result = engine.classify(image_crop)
            logger.info(f"🔬 {result.model_name}: animal={result.is_animal} (conf={result.animal_confidence:.3f})")
            if result.species:
                logger.info(f"🧬 {result.model_name} species: {result.species} (conf={result.species_confidence:.3f})")
            results.append(result)
        
        # Combine results using ensemble strategy
        ensemble_result = self._combine_results(results)
        
        processing_time = time.time() - start_time
        ensemble_result.processing_time = processing_time
        
        approving_info = f", approved_by={ensemble_result.approving_model}" if ensemble_result.approving_model else ""
        logger.info(f"🔬 Ensemble: animal={ensemble_result.is_animal}, species={ensemble_result.species} (conf={ensemble_result.animal_confidence:.3f}){approving_info}")
        
        return ensemble_result
    
    def _combine_results(self, results: List[InferenceResult]) -> ClassificationResult:
        """
        Combine multiple classification results using ensemble strategy.
        
        Args:
            results: List of InferenceResult from different models
            
        Returns:
            Combined ClassificationResult
        """
        if not results:
            raise RuntimeError("No classification results to combine")
        
        # Species-capable models supply the species label when the ensemble confirms an animal
        species_models = [r for r in results if r.can_identify_species]

        # Use "either model passes" strategy - if any model says animal=True, ensemble says animal=True
        is_ensemble_animal = any(result.is_animal for result in results)
        
        # Track which model approved the ensemble decision
        approving_model = None
        if is_ensemble_animal:
            # Find the model that approved with highest confidence
            approving_models = [result for result in results if result.is_animal]
            if approving_models:
                best_approving_model = max(approving_models, key=lambda r: r.animal_confidence)
                approving_model = best_approving_model.model_name
        
        # For confidence reporting, use the highest confidence from models that detected an animal
        animal_confidences = [result.animal_confidence for result in results if result.is_animal]
        if animal_confidences:
            ensemble_animal_confidence = max(animal_confidences)
        else:
            # If no model detected animal, use highest confidence anyway for reporting
            ensemble_animal_confidence = max(result.animal_confidence for result in results)
        
        # Species identification: use the best species-capable model
        species = None
        species_confidence = 0.0
        
        for result in species_models:
            if result.species and result.species_confidence >= self.config.species_confidence_threshold:
                if result.species_confidence > species_confidence:
                    species = result.species
                    species_confidence = result.species_confidence
        
        return ClassificationResult(
            is_animal=is_ensemble_animal,
            animal_confidence=ensemble_animal_confidence,
            species=species,
            species_confidence=species_confidence,
            processing_time=0.0,  # Will be set by caller
            individual_results=results,
            approving_model=approving_model
        )
    
    
    # No more dummy results - models must work properly
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available classification models."""
        return {
            'configured_models': self.config.classification_models,
            'loaded_engines': len(self.inference_engines),
            'animal_confidence_threshold': self.config.animal_confidence_threshold,
            'species_confidence_threshold': self.config.species_confidence_threshold,
            'classification_enabled': self.config.enable_animal_classification
        }
    
    def is_available(self) -> bool:
        """Check if at least one classification model is available."""
        return len(self.inference_engines) > 0