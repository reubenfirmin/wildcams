"""Motion detection step for wildlife video processing pipeline."""

import logging
import cv2
from typing import Dict, Any
from pathlib import Path

from ..step_interface import PipelineStep, StepInput, StepOutput
from motion import MotionDetector, MotionTracker
from video_io import FrameExtractor
from config import ProcessingConfig

logger = logging.getLogger('wildcams')


class MotionDetectionStep(PipelineStep):
    """Pipeline step for motion detection and tracking."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize motion detection step.
        
        Args:
        config: ProcessingConfig object with all parameters
        """
        super().__init__()
        
        # Will initialize detector and tracker in process method with config
        
        logger.info(f"🔍 Motion Detection Step initialized")
    
    def validate_input(self, step_input: StepInput) -> bool:
        """Validate that input contains required data for motion detection."""
        if not step_input.video_path or not step_input.video_path.exists():
            logger.error(f"❌ Motion detection step: Invalid video path")
            return False
        return True
    
    def process(self, step_input: StepInput, config) -> StepOutput:
        """
        Process motion detection and tracking on video.
        
        Args:
        step_input: Input containing video path
        
        Returns:
        StepOutput with motion tracks or early exit if no motion
        """
        video_path = step_input.video_path
        
        logger.info(f"🔍 STEP 1: Motion Detection & Tracking")
        logger.info(f"📹 Processing: {video_path.name}")
        
        # Initialize motion detector and tracker with config
        motion_detector = MotionDetector(config)
        motion_tracker = MotionTracker(motion_detector, config)
        
        # Get video properties
        cap = motion_detector.open_video_stream(video_path)
        if not cap:
            return StepOutput(
                success=False,
                data={},
                metadata={'error': f'Could not open video: {video_path}'},
                early_exit=True,
                early_exit_reason='Failed to open video'
            )
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Check minimum motion threshold
        min_motion_threshold = config.min_motion_threshold
        if total_frames < min_motion_threshold:
            logger.info(f"⏹️ Video too short ({total_frames} < {min_motion_threshold} frames) - skipping")
            return StepOutput(
                success=True,
                data={'motion_tracks': []},
                metadata={
                    'fps': fps,
                    'total_frames': total_frames,
                    'motion_tracks_count': 0
                },
                early_exit=True,
                early_exit_reason=f'Video too short: {total_frames} frames'
            )
        
        # Find motion tracks
        motion_tracks = motion_tracker.find_consistent_motion_sequences_and_tracks(
            video_path, fps, total_frames, config
        )
        
        # Check if any valid motion tracks were found
        if not motion_tracks:
            logger.info(f"⏹️ No motion tracks found - early exit")
            return StepOutput(
                success=True,
                data={'motion_tracks': []},
                metadata={
                    'fps': fps,
                    'total_frames': total_frames,
                    'motion_tracks_count': 0,
                    'large_region_count': self.motion_tracker.get_large_region_count(),
                    'total_region_count': self.motion_tracker.get_total_region_count(),
                    'initial_track_count': self.motion_tracker.get_initial_track_count()
                },
                early_exit=True,
                early_exit_reason='No motion tracks found'
            )
        
        logger.info(f"✅ Motion detection completed: {len(motion_tracks)} tracks found")
        
        return StepOutput(
            success=True,
            data={'motion_tracks': motion_tracks},
            metadata={
                'fps': fps,
                'total_frames': total_frames,
                'motion_tracks_count': len(motion_tracks),
                'large_region_count': motion_tracker.get_large_region_count(),
                'total_region_count': motion_tracker.get_total_region_count(),
                'initial_track_count': motion_tracker.get_initial_track_count()
            }
        )
        
