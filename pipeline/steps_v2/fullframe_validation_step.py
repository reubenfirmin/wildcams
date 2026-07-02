"""
Full-frame validation step using typed objects.

Runs ensemble ML validation on filtered motion tracks.
"""

import time
import logging
from pathlib import Path
from config import ProcessingConfig
from pipeline.fullframe_validator import FullFrameValidator
from pipeline.step_interface_v2 import FullFrameValidationStep
from core.data_types import (
    FullFrameValidationResult, FullFrameValidationMetadata,
    MotionDetectionResult, CameraHandlingResult, StepTiming,
    ValidationSequence, Track, Detection, BoundingBox, ModelContribution,
    create_empty_validation_result
)

logger = logging.getLogger('wildcams')

class FullFrameValidationStepImpl(FullFrameValidationStep):
    """Full-frame validation step implementation."""
    
    def __init__(self, config: ProcessingConfig, ml_ensemble):
        self.validator = FullFrameValidator(ml_ensemble)
        
    def process(self, video_path: Path, config: ProcessingConfig,
               motion_result: MotionDetectionResult,
               camera_result: CameraHandlingResult) -> FullFrameValidationResult:
        """Process full-frame validation with full typing."""
        start_time = time.time()
        
        if not camera_result.success:
            return create_empty_validation_result('camera_handling_failed')
        
        if camera_result.early_exit:
            return create_empty_validation_result('no_tracks_after_filtering')
        
        motion_tracks = camera_result.motion_tracks
        
        if not motion_tracks:
            return create_empty_validation_result('no_motion_tracks')
        
        # Run ensemble ML validation using typed motion tracks
        validation_results = self.validator.validate_motion_tracks(
            video_path, motion_tracks, config
        )
        
        # Validation results are already typed objects
        validated_sequences = validation_results
        
        # Extract model contributions from typed validation sequences
        model_contributions = self._extract_model_contributions_from_sequences(validated_sequences)
        
        end_time = time.time()
        
        # Create typed result
        return FullFrameValidationResult(
            success=True,
            validated_sequences=validated_sequences,
            metadata=FullFrameValidationMetadata(
                step_name='fullframe_validation',
                timing=StepTiming(start_time, end_time, end_time - start_time),
                tracks_evaluated=len(motion_tracks),
                tracks_passed=len([seq for seq in validated_sequences if seq.ensemble_score > 0]),
                sequences_validated=len(validated_sequences),
                model_contributions=model_contributions
            ),
            early_exit=len(validated_sequences) == 0,
            early_exit_reason='no_sequences_validated' if len(validated_sequences) == 0 else None
        )
    
    
    def _extract_model_contributions_from_sequences(self, validation_sequences: list[ValidationSequence]) -> list[ModelContribution]:
        """Extract model contributions from typed validation sequences."""
        contributions = {}
        
        for sequence in validation_sequences:
            # Track which models contributed to this sequence
            contributing_models = set()
            
            for detection in sequence.detections:
                model_name = detection.source
                contributing_models.add(model_name)
                
                if model_name not in contributions:
                    contributions[model_name] = ModelContribution(
                        model_name=model_name,
                        total_detections=0,
                        max_confidence=0.0,
                        contributing_tracks=0
                    )
                
                contributions[model_name].total_detections += 1
                contributions[model_name].max_confidence = max(
                    contributions[model_name].max_confidence,
                    detection.confidence
                )
            
            # Update contributing tracks count for each model that contributed to this sequence
            for model_name in contributing_models:
                contributions[model_name].contributing_tracks += 1
        
        return list(contributions.values())
    
