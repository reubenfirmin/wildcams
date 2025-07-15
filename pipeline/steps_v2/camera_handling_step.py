"""
Camera handling step using typed objects.

Filters motion tracks to eliminate camera handling artifacts.
"""

import time
import logging
from pathlib import Path
from config import ProcessingConfig
from pipeline.camera_handling_filter import CameraHandlingFilter
from pipeline.step_interface_v2 import CameraHandlingStep
from core.data_types import (
    CameraHandlingResult, CameraHandlingData, CameraHandlingMetadata,
    MotionDetectionResult, StepTiming, MotionTrack,
    create_empty_camera_result
)

logger = logging.getLogger('wildcams')

class CameraHandlingStepImpl(CameraHandlingStep):
    """Camera handling filter step implementation."""
    
    def __init__(self, config: ProcessingConfig):
        self.camera_filter = CameraHandlingFilter()
        
    def process(self, video_path: Path, config: ProcessingConfig, 
               motion_result: MotionDetectionResult) -> CameraHandlingResult:
        """Process camera handling filtering with full typing."""
        start_time = time.time()
        
        if not motion_result.success:
            return create_empty_camera_result('motion_detection_failed')
        
        if motion_result.early_exit:
            return create_empty_camera_result('no_motion_tracks')
        
        input_tracks = motion_result.data.motion_tracks
        
        # Apply camera handling filter with typed objects
        filtered_tracks_data = self.camera_filter.filter_motion_tracks_for_camera_handling(
            video_path, input_tracks, config, len(input_tracks)
        )
        
        # filtered_tracks_data should already be typed objects
        filtered_tracks = filtered_tracks_data
        
        end_time = time.time()
        
        # Calculate composite motion score
        composite_score = self._calculate_composite_motion_score(filtered_tracks)
        
        # Determine if camera handling was detected
        camera_handling_detected = len(filtered_tracks) < len(input_tracks)
        
        # Create typed result
        return CameraHandlingResult(
            success=True,
            data=CameraHandlingData(motion_tracks=filtered_tracks),
            metadata=CameraHandlingMetadata(
                step_name='camera_handling',
                timing=StepTiming(start_time, end_time, end_time - start_time),
                tracks_input=len(input_tracks),
                tracks_filtered=len(filtered_tracks),
                camera_handling_detected=camera_handling_detected,
                composite_motion_score=composite_score
            ),
            early_exit=len(filtered_tracks) == 0,
            early_exit_reason='all_tracks_filtered' if len(filtered_tracks) == 0 else None
        )
    
    
    def _calculate_composite_motion_score(self, tracks: list[MotionTrack]) -> float:
        """Calculate composite motion score from tracks."""
        if not tracks:
            return 0.0
        
        total_area = sum(track.total_area for track in tracks)
        total_duration = sum(track.duration_seconds for track in tracks)
        
        # Simple composite score calculation
        return min(total_area * total_duration / 1000.0, 100.0)
    
