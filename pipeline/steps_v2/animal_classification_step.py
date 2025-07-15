"""Animal classification step using BioCLIP and DeepFaune ensemble."""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2

from config import ProcessingConfig
from pipeline.step_interface_v2 import AnimalClassificationStep
from core.data_types import (
    AnimalClassificationResult, ClassifiedSequence
)
from core.data_types import (
    ValidationResult, ValidationSequence, Detection, StepTiming
)
from ml.inference.classification_coordinator import ClassificationCoordinator
from video_io import VideoReader

logger = logging.getLogger('wildcams')


class AnimalClassificationStepImpl(AnimalClassificationStep):
    """Step 4: Animal Classification implementation using BioCLIP + DeepFaune ensemble."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize animal classification step.
        
        Args:
            config: ProcessingConfig with classification parameters
        """
        self.config = config
        
        # Initialize classification coordinator
        self.classifier = ClassificationCoordinator(config)
        
        # Statistics tracking
        self.total_sequences_processed = 0
        self.sequences_confirmed_animal = 0
        self.sequences_filtered_non_animal = 0
        self.species_detected = set()
        
        logger.info(f"🔬 Animal Classification Step initialized")
        logger.info(f"📊 Classification enabled: {config.enable_animal_classification}")
        logger.info(f"🎯 Animal confidence threshold: {config.animal_confidence_threshold}")
        logger.info(f"🧬 Species confidence threshold: {config.species_confidence_threshold}")
        
    def process(self, video_path: Path, validation_result: ValidationResult, config: ProcessingConfig) -> AnimalClassificationResult:
        """
        Process validated sequences through animal classification.
        
        Args:
            video_path: Path to video file
            validation_result: Result from Step 3 (Full-frame validation)
            config: Processing configuration
            
        Returns:
            AnimalClassificationResult with species-identified animal sequences
        """
        start_time = time.time()
        
        logger.info(f"🔬 Step 4: Animal Classification")
        logger.info(f"📊 Input sequences: {len(validation_result.validated_sequences)}")
        
        # Check if classification is enabled
        if not config.enable_animal_classification:
            logger.info("⏭️ Animal classification disabled - passing through all sequences")
            return self._create_passthrough_result(validation_result, start_time)
        
        # Check if classifier is available
        if not self.classifier.is_available():
            logger.warning("⚠️ No classification models available - passing through all sequences")
            return self._create_passthrough_result(validation_result, start_time)
        
        # Process each validated sequence
        animal_sequences = []
        all_classified_sequences = []  # Store ALL classification results for reporting
        species_counts = {}
        
        video_reader = None
        try:
            # Open video for frame extraction
            video_reader = VideoReader(video_path)
            video_reader.open()
            
            for i, sequence in enumerate(validation_result.validated_sequences):
                logger.info(f"🔬 Classifying sequence {i+1}/{len(validation_result.validated_sequences)}")
                
                # Classify this sequence
                classified_sequence = self._classify_sequence(sequence, video_reader, video_path)
                
                # Store ALL classification results for reporting
                all_classified_sequences.append(classified_sequence)
                
                # Update statistics
                self.total_sequences_processed += 1
                
                if classified_sequence.is_confirmed_animal:
                    animal_sequences.append(classified_sequence)
                    self.sequences_confirmed_animal += 1
                    
                    # Track species
                    species = classified_sequence.classification.species
                    if species:
                        species_counts[species] = species_counts.get(species, 0) + 1
                        self.species_detected.add(species)
                        
                else:
                    self.sequences_filtered_non_animal += 1
                    logger.info(f"🚫 Sequence {i+1} filtered as non-animal (conf={classified_sequence.classification.animal_confidence:.3f})")
        
        finally:
            if video_reader:
                video_reader.close()
        
        processing_time = time.time() - start_time
        
        # Create result
        result = AnimalClassificationResult(
            input_sequences_count=len(validation_result.validated_sequences),
            animal_sequences=animal_sequences,
            filtered_sequences_count=self.sequences_filtered_non_animal,
            species_counts=species_counts,
            processing_time=processing_time,
            classification_enabled=True,
            all_classified_sequences=all_classified_sequences
        )
        
        logger.info(f"✅ Animal Classification completed: {len(animal_sequences)} animals confirmed, {self.sequences_filtered_non_animal} filtered")
        if species_counts:
            logger.info(f"🐾 Species detected: {', '.join(f'{species}({count})' for species, count in species_counts.items())}")
        
        return result
    
    def _classify_sequence(self, sequence: ValidationSequence, video_reader: VideoReader, video_path: Path) -> ClassifiedSequence:
        """
        Classify a single validation sequence.
        
        Args:
            sequence: ValidationSequence to classify
            video_reader: Open VideoReader for frame extraction
            
        Returns:
            ClassifiedSequence with classification results
        """
        # Extract the best detection from the sequence for classification
        best_detection = self._get_best_detection(sequence)
        
        if not best_detection:
            logger.warning("🔬 No valid detection found in sequence - marking as non-animal")
            return self._create_non_animal_sequence(sequence)
        
        # Extract image crop for classification
        image_crop = self._extract_detection_crop(best_detection, video_reader)
        
        if image_crop is None:
            logger.warning(f"🔬 Failed to extract crop for frame {best_detection.frame_idx} - marking as non-animal")
            return self._create_non_animal_sequence(sequence)
        
        # Debug: Log crop info and save crop for inspection
        bbox = best_detection.bbox
        orig_bbox_size = f"{bbox.x2-bbox.x1:.0f}x{bbox.y2-bbox.y1:.0f}"
        crop_info = f"crop_shape={image_crop.shape}, orig_bbox={orig_bbox_size}, bbox=({bbox.x1:.0f},{bbox.y1:.0f},{bbox.x2:.0f},{bbox.y2:.0f})"
        logger.info(f"🔬 Crop extracted: {crop_info}")
        
        # Save crop for debugging
        crop_path = f"debug_crops/{video_path.stem}_frame{best_detection.frame_idx}_seq{sequence.sequence_id}.jpg"
        import cv2
        import os
        os.makedirs("debug_crops", exist_ok=True)
        cv2.imwrite(crop_path, cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR))
        logger.info(f"🔬 Crop saved: {crop_path}")
        
        # Run classification ensemble
        classification_result = self.classifier.classify_ensemble(image_crop)
        
        # Add crop information to result
        classification_result.crop_path = crop_path
        classification_result.crop_info = crop_info
        
        # Determine if this sequence is confirmed as animal
        is_confirmed_animal = (
            classification_result.is_animal and 
            classification_result.animal_confidence >= self.config.animal_confidence_threshold
        )
        
        return ClassifiedSequence(
            sequence=sequence,
            classification=classification_result,
            is_confirmed_animal=is_confirmed_animal
        )
    
    def _get_best_detection(self, sequence: ValidationSequence) -> Optional[Detection]:
        """
        Get the best detection from a sequence for classification.
        
        Args:
            sequence: ValidationSequence
            
        Returns:
            Best Detection or None if no valid detections
        """
        if not sequence.detections:
            return None
        
        # Sort by confidence and return the highest confidence detection
        best_detection = max(sequence.detections, key=lambda d: d.confidence)
        return best_detection
    
    def _extract_detection_crop(self, detection: Detection, video_reader: VideoReader) -> Optional[np.ndarray]:
        """
        Extract image crop for a detection.
        
        Args:
            detection: Detection to extract crop for
            video_reader: Open VideoReader
            
        Returns:
            RGB image crop or None if extraction failed
        """
        try:
            # Read frame
            success, frame = video_reader.get_frame_at_index(detection.frame_idx)
            if not success or frame is None:
                logger.error(f"🔬 Failed to read frame {detection.frame_idx}")
                return None
            
            # Extract bounding box coordinates and calculate center
            bbox = detection.bbox
            center_x = (bbox.x1 + bbox.x2) / 2
            center_y = (bbox.y1 + bbox.y2) / 2
            orig_width = bbox.x2 - bbox.x1
            orig_height = bbox.y2 - bbox.y1
            
            # Step 1: Start with 240x240 area centered on bbox
            crop_size = 240
            
            # Step 2: Add 240 on each side (720x720 total)
            crop_size += 2 * 240  # 720x720
            
            # Step 3: If still smaller than bbox, expand to bbox size with 1:1 aspect ratio
            max_bbox_dim = max(orig_width, orig_height)
            if crop_size < max_bbox_dim:
                crop_size = max_bbox_dim
            
            # Calculate crop coordinates (square, centered on bbox center)
            half_size = crop_size / 2
            crop_x1 = center_x - half_size
            crop_y1 = center_y - half_size
            crop_x2 = center_x + half_size
            crop_y2 = center_y + half_size
            
            # Constrain to frame boundaries
            height, width = frame.shape[:2]
            x1 = max(0, int(crop_x1))
            y1 = max(0, int(crop_y1))
            x2 = min(width, int(crop_x2))
            y2 = min(height, int(crop_y2))
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            # Convert BGR to RGB (OpenCV uses BGR, models expect RGB)
            if len(crop.shape) == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Validate crop size
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                logger.warning(f"🔬 Crop too small: {crop.shape}")
                return None
            
            return crop
            
        except Exception as e:
            logger.error(f"🔬 Failed to extract crop: {e}")
            return None
    
    def _create_non_animal_sequence(self, sequence: ValidationSequence) -> ClassifiedSequence:
        """Create a ClassifiedSequence marked as non-animal."""
        from core.data_types import ClassificationResult
        
        failed_classification = ClassificationResult(
            is_animal=False,
            animal_confidence=0.0,
            species=None,
            species_confidence=0.0,
            processing_time=0.0,
            approving_model=None
        )
        
        return ClassifiedSequence(
            sequence=sequence,
            classification=failed_classification,
            is_confirmed_animal=False
        )
    
    def _create_passthrough_result(self, validation_result: ValidationResult, start_time: float) -> AnimalClassificationResult:
        """
        Create a passthrough result when classification is disabled or unavailable.
        
        Args:
            validation_result: Input validation result
            start_time: Processing start time
            
        Returns:
            AnimalClassificationResult with all sequences passed through
        """
        processing_time = time.time() - start_time
        
        # Convert all validated sequences to classified sequences (marked as animals)
        animal_sequences = []
        for sequence in validation_result.validated_sequences:
            from core.data_types import ClassificationResult
            
            passthrough_classification = ClassificationResult(
                is_animal=True,  # Pass through as animal
                animal_confidence=1.0,  # High confidence (passthrough)
                species=None,  # No species identification
                species_confidence=0.0,
                processing_time=0.0,
                approving_model="passthrough"
            )
            
            classified_sequence = ClassifiedSequence(
                sequence=sequence,
                classification=passthrough_classification,
                is_confirmed_animal=True
            )
            animal_sequences.append(classified_sequence)
        
        return AnimalClassificationResult(
            input_sequences_count=len(validation_result.validated_sequences),
            animal_sequences=animal_sequences,
            filtered_sequences_count=0,  # Nothing filtered in passthrough mode
            species_counts={},  # No species identification
            processing_time=processing_time,
            classification_enabled=False
        )
    
    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            'total_sequences_processed': self.total_sequences_processed,
            'sequences_confirmed_animal': self.sequences_confirmed_animal,
            'sequences_filtered_non_animal': self.sequences_filtered_non_animal,
            'unique_species_detected': len(self.species_detected)
        }