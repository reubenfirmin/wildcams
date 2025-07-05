"""
Data classes for tracking system.
Provides typed objects for detections, tracks, and tracking information.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class Detection:
    """
    A single detection from ML model at a specific frame.
    
    Used as input to tracking system and as building blocks for tracks.
    """
    frame_idx: int
    timestamp: float
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    source: str  # Model name that generated this detection
    
    # Additional metadata
    class_id: int = 1  # Default to "animal" class
    features: Optional[List[float]] = None  # Appearance features for DeepSORT
    motion_overlap: float = 0.0  # Overlap with motion regions
    
    def __post_init__(self):
        """Validate detection data."""
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 elements, got {len(self.bbox)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")


@dataclass  
class Track:
    """
    A temporal track consisting of multiple detections across frames.
    
    Represents a single animal being tracked through time.
    """
    track_id: int
    detections: List[Detection]
    start_frame: int
    end_frame: int
    duration_seconds: float
    
    # Track quality metrics
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    detection_density: float = 0.0  # detections per frame
    
    # Track metadata
    is_confirmed: bool = False  # Whether track has been confirmed by tracker
    last_seen_frame: int = 0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.detections:
            confidences = [d.confidence for d in self.detections]
            self.avg_confidence = sum(confidences) / len(confidences)
            self.max_confidence = max(confidences)
            
            frame_span = max(1, self.end_frame - self.start_frame + 1)
            self.detection_density = len(self.detections) / frame_span
            
            self.last_seen_frame = max(d.frame_idx for d in self.detections)
    
    @property
    def best_detection(self) -> Detection:
        """Get detection with highest confidence."""
        if not self.detections:
            raise ValueError("Track has no detections")
        return max(self.detections, key=lambda d: d.confidence)
    
    @property
    def frame_count(self) -> int:
        """Get number of frames spanned by track."""
        return self.end_frame - self.start_frame + 1


@dataclass
class TrackingInfo:
    """
    Information about tracking process for a video.
    
    Contains tracks and metadata about tracking performance.
    """
    video_path: Path
    tracks: List[Track]
    total_frames: int
    fps: float
    
    # Tracking statistics
    total_detections: int = 0
    confirmed_tracks: int = 0
    avg_track_length: float = 0.0
    tracking_method: str = "unknown"
    
    def __post_init__(self):
        """Calculate derived statistics."""
        if self.tracks:
            self.confirmed_tracks = sum(1 for t in self.tracks if t.is_confirmed)
            self.total_detections = sum(len(t.detections) for t in self.tracks)
            self.avg_track_length = sum(t.duration_seconds for t in self.tracks) / len(self.tracks)
    
    @property
    def longest_track(self) -> Optional[Track]:
        """Get track with longest duration."""
        if not self.tracks:
            return None
        return max(self.tracks, key=lambda t: t.duration_seconds)
    
    @property
    def best_track(self) -> Optional[Track]:
        """Get track with highest average confidence."""
        if not self.tracks:
            return None
        return max(self.tracks, key=lambda t: t.avg_confidence)