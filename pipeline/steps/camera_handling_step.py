"""Camera handling filter step for wildlife video processing pipeline.

WARNING: This step contains dictionary violations and should be replaced
with the typed implementation in pipeline.typed_steps.
"""

import logging
from typing import Dict, Any

from ..step_interface import PipelineStep, StepInput, StepOutput
from ..camera_handling_filter import CameraHandlingFilter
from config import ProcessingConfig

logger = logging.getLogger('wildcams')


class CameraHandlingFilterStep(PipelineStep):
    """Pipeline step for filtering camera handling videos."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize camera handling filter step.
        
        Args:
        config: ProcessingConfig object with all parameters
        """
        super().__init__()
        
        # Initialize camera handling filter
        self.camera_filter = CameraHandlingFilter()
        
        logger.info(f"📹 Camera Handling Filter Step initialized")
    
    def validate_input(self, step_input: StepInput) -> bool:
        """Validate that input contains required data for camera handling filter."""
        if not step_input.video_path or not step_input.video_path.exists():
            logger.error(f"❌ Camera handling step: Invalid video path")
            return False
        
        if not step_input.motion_tracks:
            logger.warning(f"⚠️ Camera handling step: No motion tracks provided")
            return True  # Valid but will result in early exit
        
        return True
    
    def process(self, step_input: StepInput, config) -> StepOutput:
        """
        Process camera handling filtering on motion tracks.
        
        Args:
        step_input: Input containing video path and motion tracks
        
        Returns:
        StepOutput with filtered motion tracks or early exit if camera handling detected
        """
        video_path = step_input.video_path
        motion_tracks = step_input.motion_tracks or []
        
        logger.info(f"📹 STEP 2: Camera Handling Filter")
        logger.info(f"📹 Processing: {video_path.name}")
        
        # Handle empty motion tracks case
        if not motion_tracks:
            logger.info(f"⏹️ No motion tracks to filter - skipping camera handling detection")
            # DICTIONARY VIOLATIONS: Using {} for data and metadata
            # These violate Phase 5.0 no-dictionary rule
            return StepOutput(
                success=True,
                data={'motion_tracks': []},
                metadata={
                    'filtered_motion_tracks_count': 0,
                    'camera_handling_detected': False,
                    'composite_score': 0.0
                },
                early_exit=True,
                early_exit_reason='No motion tracks provided'
            )
        
        # Get initial track count from metadata
        initial_track_count = step_input.metadata.get('initial_track_count', len(motion_tracks))
        
        # Apply camera handling filter
        filtered_tracks = self.camera_filter.filter_motion_tracks_for_camera_handling(
            video_path, motion_tracks, config, initial_track_count
        )
        
        # Check if camera handling was detected (empty result)
        camera_handling_detected = len(filtered_tracks) == 0 and len(motion_tracks) > 0
        
        if camera_handling_detected:
            # Get rejection details
            rejection_reasons = self.camera_filter.get_rejection_reasons()
            composite_scores = self.camera_filter.get_composite_scores()
            
            logger.warning(f"⚠️ Camera handling detected - early exit")
            # DICTIONARY VIOLATIONS: Using {} for data and metadata
            # These violate Phase 5.0 no-dictionary rule
            return StepOutput(
                success=True,
                data={'motion_tracks': []},
                metadata={
                    'filtered_motion_tracks_count': 0,
                    'camera_handling_detected': True,
                    'composite_score': composite_scores.get(video_path.name, 0.0),
                    'rejection_reason': rejection_reasons.get(video_path.name, 'camera_handling')
                },
                early_exit=True,
                early_exit_reason='Camera handling detected'
            )
        
        logger.info(f"✅ Camera handling filter completed: {len(filtered_tracks)} tracks passed")
        
        # Get composite score for metadata
        composite_scores = self.camera_filter.get_composite_scores()
        composite_score = composite_scores.get(video_path.name, 0.0)
        
        # DICTIONARY VIOLATIONS: Using {} for data and metadata
        # These violate Phase 5.0 no-dictionary rule
        return StepOutput(
            success=True,
            data={'motion_tracks': filtered_tracks},
            metadata={
                'filtered_motion_tracks_count': len(filtered_tracks),
                'camera_handling_detected': False,
                'composite_score': composite_score
            }
        )
