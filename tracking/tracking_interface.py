"""
Abstract interface for temporal tracking systems.

Defines the contract that all tracking implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from .track_data import Detection, Track, TrackingInfo
from config import ProcessingConfig


class TemporalTracker(ABC):
    """
    Abstract base class for temporal tracking systems.
    
    Tracking systems maintain temporal consistency by linking detections
    across frames to form coherent tracks of individual animals.
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize tracker with configuration.
        
        Args:
            config: ProcessingConfig with tracking parameters
        """
        self.config = config
        self.tracks: List[Track] = []
        self.next_track_id = 1
        
    @abstractmethod
    def update_tracks(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        """
        Update tracks with new detections from current frame.
        
        Args:
            detections: List of detections from current frame
            frame_idx: Current frame index
            
        Returns:
            List of updated tracks (both active and completed)
        """
        pass
    
    @abstractmethod
    def finalize_tracks(self, total_frames: int, fps: float) -> TrackingInfo:
        """
        Finalize tracking and return complete tracking information.
        
        Args:
            total_frames: Total number of frames in video
            fps: Video frame rate
            
        Returns:
            TrackingInfo with all tracks and statistics
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset tracker state for new video."""
        pass
    
    def get_active_tracks(self) -> List[Track]:
        """Get currently active tracks."""
        return [track for track in self.tracks if self._is_track_active(track)]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get confirmed tracks only."""
        return [track for track in self.tracks if track.is_confirmed]
    
    def _is_track_active(self, track: Track) -> bool:
        """
        Check if track is still active (implementation dependent).
        
        Default implementation considers tracks active if they have detections.
        Subclasses can override for more sophisticated logic.
        """
        return len(track.detections) > 0
    
    def _create_new_track(self, detection: Detection) -> Track:
        """Create a new track from detection."""
        track = Track(
            track_id=self.next_track_id,
            detections=[detection],
            start_frame=detection.frame_idx,
            end_frame=detection.frame_idx,
            duration_seconds=0.0,
            is_confirmed=False
        )
        self.next_track_id += 1
        return track
    
    def _add_detection_to_track(self, track: Track, detection: Detection):
        """Add detection to existing track."""
        track.detections.append(detection)
        track.end_frame = max(track.end_frame, detection.frame_idx)
        track.duration_seconds = (track.end_frame - track.start_frame) / self.config.video_fps if hasattr(self.config, 'video_fps') else 0.0
        
        # Recalculate metrics
        track.__post_init__()
    
    def _should_confirm_track(self, track: Track) -> bool:
        """
        Determine if track should be confirmed.
        
        Default implementation confirms tracks with minimum detections.
        """
        min_detections = getattr(self.config, 'min_track_detections', 3)
        return len(track.detections) >= min_detections
    
    @property
    def tracking_method(self) -> str:
        """Return name of tracking method for identification."""
        return self.__class__.__name__