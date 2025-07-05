"""Full-frame validation step for wildlife video processing pipeline."""

import logging
from typing import Dict, Any

from ..step_interface import PipelineStep, StepInput, StepOutput
from ..fullframe_validator import FullFrameValidator
from config import ProcessingConfig

logger = logging.getLogger('wildcams')


class FullFrameValidationStep(PipelineStep):
    """Pipeline step for full-frame ML validation of motion tracks."""
    
    def __init__(self, config: ProcessingConfig, ml_ensemble):
        """
        Initialize full-frame validation step.
        
        Args:
            config: ProcessingConfig object with all parameters
            ml_ensemble: ML ensemble for running detections
        """
        super().__init__()
        
        # Initialize full-frame validator
        self.validator = FullFrameValidator(ml_ensemble)
        
        logger.info(f"🎯 Full-Frame Validation Step initialized")
    
    def validate_input(self, step_input: StepInput) -> bool:
        """Validate that input contains required data for full-frame validation."""
        if not step_input.video_path or not step_input.video_path.exists():
            logger.error(f"❌ Full-frame validation step: Invalid video path")
            return False
        
        if not step_input.motion_tracks:
            logger.warning(f"⚠️ Full-frame validation step: No motion tracks provided")
            return True  # Valid but will result in early exit
        
        return True
    
    def process(self, step_input: StepInput, config) -> StepOutput:
        """
        Process full-frame validation on motion tracks.
        
        Args:
            step_input: Input containing video path and filtered motion tracks
            
        Returns:
            StepOutput with validated sequences or early exit if no valid tracks
        """
        video_path = step_input.video_path
        motion_tracks = step_input.motion_tracks or []
        
        logger.info(f"🎯 STEP 3: Full-Frame Validation")
        logger.info(f"📹 Processing: {video_path.name}")
        
        # Handle empty motion tracks case
        if not motion_tracks:
            logger.info(f"⏹️ No motion tracks to validate - skipping full-frame analysis")
            return StepOutput(
                success=True,
                data={'validated_sequences': []},
                metadata={
                    'validated_sequences_count': 0,
                    'tracks_validated': 0,
                    'tracks_passed': 0
                },
                early_exit=True,
                early_exit_reason='No motion tracks provided'
            )
        
        # Run full-frame validation
        validated_sequences = self.validator.validate_motion_tracks(video_path, motion_tracks, config)
        
        # Check if any sequences passed validation
        if not validated_sequences:
            logger.info(f"⏹️ No sequences passed validation - early exit")
            return StepOutput(
                success=True,
                data={'validated_sequences': []},
                metadata={
                    'validated_sequences_count': 0,
                    'tracks_validated': len(motion_tracks),
                    'tracks_passed': 0
                },
                early_exit=True,
                early_exit_reason='No sequences passed validation'
            )
        
        logger.info(f"✅ Full-frame validation completed: {len(validated_sequences)} sequences validated from {len(motion_tracks)} tracks")
        
        # Get model contributions from validator
        model_contributions = self.validator.get_model_contributions()
        
        return StepOutput(
            success=True,
            data={'validated_sequences': validated_sequences},
            metadata={
                'validated_sequences_count': len(validated_sequences),
                'tracks_validated': len(motion_tracks),
                'model_contributions': model_contributions,
                'tracks_passed': len(validated_sequences)
            }
        )