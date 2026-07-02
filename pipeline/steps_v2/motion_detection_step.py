"""
Motion detection step using typed objects.
"""

import time
import cv2
import logging
from pathlib import Path
from config import ProcessingConfig
from motion import MotionDetector, MotionTracker
from pipeline.step_interface_v2 import MotionDetectionStep
from core.data_types import (
    MotionDetectionResult, MotionDetectionMetadata,
    StepTiming, MotionTrack, MotionRegion, BoundingBox,
    create_empty_motion_result
)

logger = logging.getLogger('wildcams')

class MotionDetectionStepImpl(MotionDetectionStep):
    """Motion detection step implementation."""
    
    def __init__(self, config: ProcessingConfig):
        self.motion_detector = MotionDetector(config)
        self.motion_tracker = MotionTracker(self.motion_detector, config)
        
    def process(self, video_path: Path, config: ProcessingConfig) -> MotionDetectionResult:
        """Process motion detection with full typing."""
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return create_empty_motion_result(f'Could not open video: {video_path}')
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return create_empty_motion_result(f'Invalid video properties: fps={fps}, frames={total_frames}')
        
        # Find motion sequences and tracks
        motion_tracks_data = self.motion_tracker.find_consistent_motion_sequences_and_tracks(
            video_path, fps, total_frames, config
        )
        
        cap.release()
        
        # Convert legacy dict data to typed objects
        motion_tracks = self._convert_motion_tracks_to_typed(motion_tracks_data, fps)
        
        end_time = time.time()
        
        # Calculate total motion area
        total_area = sum(track.total_area for track in motion_tracks)
        
        # Create typed result
        return MotionDetectionResult(
            success=True,
            motion_tracks=motion_tracks,
            metadata=MotionDetectionMetadata(
                step_name='motion_detection',
                timing=StepTiming(start_time, end_time, end_time - start_time),
                tracks_found=len(motion_tracks),
                total_motion_area=total_area,
                motion_method=config.motion_method
            ),
            early_exit=len(motion_tracks) == 0,
            early_exit_reason='no_motion_detected' if len(motion_tracks) == 0 else None
        )
    
    def _convert_motion_tracks_to_typed(self, tracks_data: list, fps: float) -> list[MotionTrack]:
        """Convert legacy motion track dicts to typed objects."""
        typed_tracks = []
        
        if not tracks_data:
            return typed_tracks
        
        for track_id, track_dict in enumerate(tracks_data):
            # Handle case where track_dict might not be a dict
            if not isinstance(track_dict, dict):
                logger.warning(f"Expected dict for track {track_id}, got {type(track_dict)}. Skipping track.")
                continue
                
            # Convert motion regions - motion_regions is a list of lists of coordinate tuples
            regions = []
            frame_idx = track_dict.get('start_frame', 0)
            
            for frame_regions in track_dict.get('motion_regions', []):
                # frame_regions is a list of coordinate tuples for this frame
                for region_coords in frame_regions:
                    # region_coords is a tuple/list of (x1, y1, x2, y2)
                    x1, y1, x2, y2 = region_coords
                    bbox = BoundingBox(x1, y1, x2, y2)
                    
                    region = MotionRegion(
                        bbox=bbox,
                        area=(x2 - x1) * (y2 - y1),
                        frame_idx=frame_idx,
                        timestamp=frame_idx / fps if fps > 0 else 0.0
                    )
                    regions.append(region)
                
                frame_idx += 1
            
            # Calculate track properties
            start_frame = track_dict.get('start_frame', 0)
            end_frame = track_dict.get('end_frame', 0)
            duration = (end_frame - start_frame) / fps if fps > 0 else 0.0
            total_area = sum(region.area for region in regions)
            
            typed_track = MotionTrack(
                track_id=track_id,
                regions=regions,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_seconds=duration,
                total_area=total_area
            )
            typed_tracks.append(typed_track)
        
        return typed_tracks