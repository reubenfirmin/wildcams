"""
Pipeline step interface using typed objects.

Clean typed system with proper data structures.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from config import ProcessingConfig
from core.data_types import (
    MotionDetectionResult, CameraHandlingResult, FullFrameValidationResult,
    VideoAnalysis
)
from core.data_types import ValidationResult
from core.data_types import AnimalClassificationResult

class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    @abstractmethod
    def get_step_name(self) -> str:
        """Get the name of this step."""
        pass
    
    @abstractmethod
    def process(self, video_path: Path, config: ProcessingConfig) -> 'StepResult':
        """Process the video through this step."""
        pass

# Union type for all possible step results
StepResult = MotionDetectionResult | CameraHandlingResult | FullFrameValidationResult | AnimalClassificationResult

class MotionDetectionStep(PipelineStep):
    """Motion detection step."""
    
    def get_step_name(self) -> str:
        return "motion_detection"
    
    @abstractmethod  
    def process(self, video_path: Path, config: ProcessingConfig) -> MotionDetectionResult:
        """Process motion detection."""
        pass

class CameraHandlingStep(PipelineStep):
    """Camera handling step."""
    
    def get_step_name(self) -> str:
        return "camera_handling"
    
    @abstractmethod
    def process(self, video_path: Path, config: ProcessingConfig, 
               motion_result: MotionDetectionResult) -> CameraHandlingResult:
        """Process camera handling filtering."""
        pass

class FullFrameValidationStep(PipelineStep):
    """Full frame validation step."""
    
    def get_step_name(self) -> str:
        return "fullframe_validation"
    
    @abstractmethod
    def process(self, video_path: Path, config: ProcessingConfig,
               motion_result: MotionDetectionResult,
               camera_result: CameraHandlingResult) -> FullFrameValidationResult:
        """Process full frame validation."""
        pass

class AnimalClassificationStep(PipelineStep):
    """Animal classification step."""
    
    def get_step_name(self) -> str:
        return "animal_classification"
    
    @abstractmethod
    def process(self, video_path: Path, validation_result: ValidationResult, 
               config: ProcessingConfig) -> AnimalClassificationResult:
        """Process animal classification."""
        pass

class PipelineOrchestrator:
    """Pipeline orchestrator using typed objects."""
    
    def __init__(self, 
                 motion_step: MotionDetectionStep,
                 camera_step: CameraHandlingStep, 
                 validation_step: FullFrameValidationStep,
                 classification_step: Optional[AnimalClassificationStep] = None):
        self.motion_step = motion_step
        self.camera_step = camera_step
        self.validation_step = validation_step
        self.classification_step = classification_step
    
    def process(self, video_path: Path, config: ProcessingConfig) -> VideoAnalysis:
        """Process video through entire typed pipeline."""
        # Step 1: Motion Detection
        motion_result = self.motion_step.process(video_path, config)
        
        if not motion_result.success:
            return self._create_failed_analysis(video_path, motion_result, None, None)
        
        if motion_result.early_exit:
            return self._create_early_exit_analysis(video_path, motion_result, None, None)
        
        # Step 2: Camera Handling
        camera_result = self.camera_step.process(video_path, config, motion_result)
        
        if not camera_result.success:
            return self._create_failed_analysis(video_path, motion_result, camera_result, None)
            
        if camera_result.early_exit:
            return self._create_early_exit_analysis(video_path, motion_result, camera_result, None)
        
        # Step 3: Full Frame Validation
        validation_result = self.validation_step.process(video_path, config, motion_result, camera_result)
        
        if not validation_result.success:
            return self._create_failed_analysis(video_path, motion_result, camera_result, validation_result)
        
        # Step 4: Animal Classification (optional)
        classification_result = None
        if self.classification_step and config.enable_animal_classification:
            from core.data_types import ValidationResult as ValidationResultData
            # Convert FullFrameValidationResult to ValidationResult for Step 4
            step3_validation = ValidationResultData(
                animals_detected=True,
                confidence=0.0,  # Will be recalculated
                ensemble_score=0.0,  # Will be recalculated  
                composite_score=0.0,  # Will be recalculated
                validated_sequences_count=len(validation_result.data.validated_sequences),
                best_detection=None,  # Will be set
                reason=None
            )
            # Add validated sequences manually since it's not in constructor
            step3_validation.validated_sequences = validation_result.data.validated_sequences
            
            classification_result = self.classification_step.process(video_path, step3_validation, config)
            
            # If classification filtered out all sequences, create failed analysis but KEEP classification_result
            if len(classification_result.animal_sequences) == 0:
                return self._create_classification_filtered_analysis(video_path, motion_result, camera_result, validation_result, classification_result)
        
        # Create complete analysis
        return self._create_successful_analysis(video_path, motion_result, camera_result, validation_result, classification_result)
    
    def _create_successful_analysis(self, 
                                   video_path: Path,
                                   motion_result: MotionDetectionResult,
                                   camera_result: CameraHandlingResult, 
                                   validation_result: FullFrameValidationResult,
                                   classification_result: Optional[AnimalClassificationResult] = None) -> VideoAnalysis:
        """Create successful video analysis."""
        from core.data_types import ValidationResult
        
        # Use Step 4 results if available, otherwise use Step 3 results
        if classification_result and classification_result.classification_enabled:
            # Step 4 ran - use its results
            if not classification_result.animal_sequences:
                # Step 4 filtered out all sequences - no animals
                return self._create_classification_filtered_analysis(video_path, motion_result, camera_result, validation_result, classification_result)
            
            # Use Step 4 confirmed sequences
            sequences_to_use = [cs.sequence for cs in classification_result.animal_sequences]
            best_sequence = max(sequences_to_use, key=lambda s: s.ensemble_score)
            
            # Add species information from Step 4
            species_info = classification_result.species_counts
            species_summary = ', '.join(f"{species}({count})" for species, count in species_info.items()) if species_info else None
            
        else:
            # No Step 4 or Step 4 disabled - use Step 3 results
            if not validation_result.data.validated_sequences:
                return self._create_failed_analysis(video_path, motion_result, camera_result, validation_result)
            
            sequences_to_use = validation_result.data.validated_sequences
            best_sequence = max(sequences_to_use, key=lambda s: s.ensemble_score)
            species_summary = None
        
        # Create validation result
        validation = ValidationResult(
            animals_detected=True,
            confidence=best_sequence.best_detection.confidence,
            ensemble_score=best_sequence.ensemble_score,
            composite_score=best_sequence.composite_score,
            validated_sequences_count=len(sequences_to_use),
            best_detection=best_sequence.best_detection
        )
        
        total_processing_time = (
            motion_result.metadata.timing.duration + 
            camera_result.metadata.timing.duration +
            validation_result.metadata.timing.duration
        )
        
        # Add Step 4 processing time if available
        if classification_result:
            total_processing_time += classification_result.processing_time
        
        return VideoAnalysis(
            video_path=video_path,
            video_name=video_path.name,
            processing_time=total_processing_time,
            validation_result=validation,
            motion_result=motion_result,
            camera_result=camera_result,
            fullframe_result=validation_result,
            species_summary=species_summary,
            classification_result=classification_result
        )
    
    def _create_failed_analysis(self,
                               video_path: Path,
                               motion_result: MotionDetectionResult,
                               camera_result: Optional[CameraHandlingResult],
                               validation_result: Optional[FullFrameValidationResult]) -> VideoAnalysis:
        """Create failed video analysis."""
        from core.data_types import ValidationResult, Detection, BoundingBox
        from core.data_types import create_empty_camera_result, create_empty_validation_result
        
        # Create empty detection for failed cases
        empty_detection = Detection(
            confidence=0.0,
            bbox=BoundingBox(0, 0, 0, 0),
            source='none',
            class_name='none'
        )
        
        validation = ValidationResult(
            animals_detected=False,
            confidence=0.0,
            ensemble_score=0.0,
            composite_score=0.0,
            validated_sequences_count=0,
            best_detection=empty_detection,
            reason='processing_failed'
        )
        
        return VideoAnalysis(
            video_path=video_path,
            video_name=video_path.name,
            processing_time=motion_result.metadata.timing.duration,
            validation_result=validation,
            motion_result=motion_result,
            camera_result=camera_result or create_empty_camera_result('not_processed'),
            fullframe_result=validation_result or create_empty_validation_result('not_processed')
        )
    
    def _create_early_exit_analysis(self,
                                   video_path: Path,
                                   motion_result: MotionDetectionResult,
                                   camera_result: Optional[CameraHandlingResult],
                                   validation_result: Optional[FullFrameValidationResult]) -> VideoAnalysis:
        """Create early exit video analysis."""
        from core.data_types import ValidationResult, Detection, BoundingBox
        from core.data_types import create_empty_camera_result, create_empty_validation_result
        
        # Create empty detection for early exit cases
        empty_detection = Detection(
            confidence=0.0,
            bbox=BoundingBox(0, 0, 0, 0),
            source='none',
            class_name='none'
        )
        
        # Determine reason for early exit
        if motion_result.early_exit:
            reason = motion_result.early_exit_reason
        elif camera_result and camera_result.early_exit:
            reason = camera_result.early_exit_reason
        else:
            reason = 'early_exit'
        
        validation = ValidationResult(
            animals_detected=False,
            confidence=0.0,
            ensemble_score=0.0,
            composite_score=0.0,
            validated_sequences_count=0,
            best_detection=empty_detection,
            reason=reason
        )
        
        return VideoAnalysis(
            video_path=video_path,
            video_name=video_path.name,
            processing_time=(motion_result.metadata.timing.duration + 
                           (camera_result.metadata.timing.duration if camera_result else 0.0)),
            validation_result=validation,
            motion_result=motion_result,
            camera_result=camera_result or create_empty_camera_result('early_exit'),
            fullframe_result=validation_result or create_empty_validation_result('early_exit')
        )
    
    def _create_classification_filtered_analysis(self,
                                               video_path: Path,
                                               motion_result: MotionDetectionResult,
                                               camera_result: CameraHandlingResult,
                                               validation_result: FullFrameValidationResult,
                                               classification_result: AnimalClassificationResult) -> VideoAnalysis:
        """Create analysis when Step 4 filtered out all sequences as non-animals."""
        from core.data_types import ValidationResult, Detection, BoundingBox
        
        # Create empty detection for filtered cases
        empty_detection = Detection(
            confidence=0.0,
            bbox=BoundingBox(0, 0, 0, 0),
            source='none',
            class_name='none'
        )
        
        validation = ValidationResult(
            animals_detected=False,
            confidence=0.0,
            ensemble_score=0.0,
            composite_score=0.0,
            validated_sequences_count=0,
            best_detection=empty_detection,
            reason='filtered_by_animal_classification'
        )
        
        total_processing_time = (
            motion_result.metadata.timing.duration +
            camera_result.metadata.timing.duration +
            validation_result.metadata.timing.duration +
            classification_result.processing_time
        )
        
        return VideoAnalysis(
            video_path=video_path,
            video_name=video_path.name,
            processing_time=total_processing_time,
            validation_result=validation,
            motion_result=motion_result,
            camera_result=camera_result,
            fullframe_result=validation_result,
            classification_result=classification_result  # KEEP the classification result!
        )