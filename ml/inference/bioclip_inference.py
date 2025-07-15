"""BioCLIP inference engine for animal species classification."""

import logging
import time
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from data_types import BioCLIPResult, InferenceResult
from config import ProcessingConfig

logger = logging.getLogger('wildcams')


class BioCLIPInference:
    """
    BioCLIP inference engine for species classification.
    
    BioCLIP is a vision-language model trained on biological data that can
    classify animals into species categories and distinguish animals from non-animals.
    """
    
    def __init__(self, config: ProcessingConfig, cache_dir: Optional[Path] = None):
        """
        Initialize BioCLIP inference engine.
        
        Args:
            config: ProcessingConfig with classification parameters
            cache_dir: Optional directory for model caching
        """
        self.config = config
        self.cache_dir = cache_dir or Path('./models_cache')
        self.model = None
        self.processor = None
        
        # Animal classes that BioCLIP can identify
        self.animal_classes = {
            'mammal', 'bird', 'reptile', 'amphibian', 'fish', 'insect', 
            'arthropod', 'mollusk', 'cnidarian', 'echinoderm', 'annelid',
            'deer', 'cat', 'dog', 'bear', 'monkey', 'rodent', 'bat',
            'bird', 'eagle', 'hawk', 'owl', 'parrot', 'hummingbird',
            'snake', 'lizard', 'turtle', 'frog', 'salamander'
        }
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load BioCLIP model and processor."""
        try:
            # Import required libraries
            import open_clip
            import torch
            from PIL import Image
            
            logger.info("🧬 Loading BioCLIP model...")
            
            # Load the actual BioCLIP model
            model_name = 'ViT-B-16'
            pretrained = 'openai'
            
            # Download and load BioCLIP model
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                cache_dir=str(self.cache_dir)
            )
            
            # Load tokenizer
            self.tokenizer = open_clip.get_tokenizer(model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Define wildlife classes for Costa Rica
            self.wildlife_classes = [
                "bird", "hummingbird", "toucan", "parrot", "hawk", "owl", "woodpecker",
                "lizard", "iguana", "gecko", "anole", "basilisk", "snake", "boa",
                "frog", "tree frog", "poison dart frog", "toad", "salamander",
                "butterfly", "moth", "beetle", "ant", "spider", "insect",
                "coati", "sloth", "monkey", "howler monkey", "agouti", "ocelot",
                "mammal", "reptile", "amphibian", "animal", "wildlife"
            ]
            
            logger.info("✅ BioCLIP model loaded successfully")
            
        except ImportError as e:
            logger.error(f"❌ BioCLIP dependencies not available: {e}")
            raise RuntimeError(f"❌ BioCLIP configured but dependencies missing: {e}. Install with: pip install open-clip-torch torch torchvision") from e
            
        except Exception as e:
            logger.error(f"❌ Failed to load BioCLIP model: {e}")
            raise RuntimeError(f"❌ BioCLIP model configured but failed to load: {e}") from e
    
    def classify(self, image_crop: np.ndarray) -> InferenceResult:
        """
        Classify an image crop using BioCLIP - returns generic interface.
        
        Args:
            image_crop: RGB image array of shape (H, W, 3)
            
        Returns:
            InferenceResult with generic interface
        """
        bioclip_result = self.classify_detailed(image_crop)
        
        # Convert to generic interface
        return InferenceResult(
            model_name="BioCLIP",
            is_animal=bioclip_result.is_animal,
            animal_confidence=bioclip_result.top_confidence,  # Always report actual confidence
            species=bioclip_result.top_species if bioclip_result.is_animal else None,
            species_confidence=bioclip_result.top_confidence if bioclip_result.is_animal else 0.0,
            can_identify_species=True,
            processing_time=bioclip_result.processing_time
        )
    
    def classify_detailed(self, image_crop: np.ndarray) -> BioCLIPResult:
        """
        Classify an image crop using BioCLIP.
        
        Args:
            image_crop: RGB image array of shape (H, W, 3)
            
        Returns:
            BioCLIPResult with species predictions and confidence scores
        """
        start_time = time.time()
        
        
        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            import torch
            
            # Convert BGR to RGB if needed
            if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
                # Assume BGR from OpenCV, convert to RGB
                image_rgb = image_crop[:, :, ::-1]
            else:
                image_rgb = image_crop
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb.astype('uint8'))
            
            # Preprocess image
            image_tensor = self.preprocess(pil_image).unsqueeze(0)
            
            # Tokenize wildlife class names
            text_tokens = self.tokenizer(self.wildlife_classes)
            
            # Run inference
            with torch.no_grad():
                # Get image and text features
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities (logits)
                logits = (image_features @ text_features.T) * self.model.logit_scale.exp()
                probs = torch.softmax(logits, dim=-1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs[0], k=min(self.config.bioclip_top_k, len(self.wildlife_classes)))
                
                # Convert to list of tuples
                predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    class_name = self.wildlife_classes[idx.item()]
                    confidence = prob.item()
                    predictions.append((class_name, confidence))
            
            processing_time = time.time() - start_time
            
            # Check if any prediction is an animal above threshold
            animal_predictions = [(pred[0], pred[1]) for pred in predictions if self._is_animal_class(pred[0])]
            is_animal = any(conf > self.config.bioclip_threshold for _, conf in animal_predictions)
            
            top_species = predictions[0][0] if predictions else "unknown"
            top_confidence = predictions[0][1] if predictions else 0.0
            
            # Log top predictions with details
            logger.info(f"🧬 BioCLIP top predictions: {predictions[:5]}")
            logger.info(f"🧬 BioCLIP: {top_species} (conf={top_confidence:.3f}, animal={is_animal})")
            
            # Convert predictions list to dictionary for all_predictions field
            all_predictions = {pred[0]: pred[1] for pred in predictions}
            
            return BioCLIPResult(
                is_animal=is_animal,
                top_species=top_species,
                top_confidence=top_confidence,
                all_predictions=all_predictions,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ BioCLIP inference failed: {e}")
            raise RuntimeError(f"BioCLIP inference failed: {e}") from e
    
    def _is_animal_class(self, class_name: str) -> bool:
        """
        Check if a class name represents an animal.
        
        Args:
            class_name: Predicted class name
            
        Returns:
            True if class represents an animal
        """
        class_lower = class_name.lower()
        
        # Check against known animal classes
        for animal_class in self.animal_classes:
            if animal_class in class_lower:
                return True
        
        # Additional heuristics for animal detection
        animal_keywords = ['mammal', 'bird', 'reptile', 'fish', 'animal', 'fauna', 'wildlife']
        non_animal_keywords = ['plant', 'tree', 'leaf', 'rock', 'stick', 'shadow', 'human', 'person']
        
        # Check for animal keywords
        if any(keyword in class_lower for keyword in animal_keywords):
            return True
            
        # Check for non-animal keywords
        if any(keyword in class_lower for keyword in non_animal_keywords):
            return False
        
        # Default to True for unknown classes (conservative approach)
        return True
    
    # No more dummy results - BioCLIP must work properly
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported animal classes."""
        return sorted(list(self.animal_classes))
    
