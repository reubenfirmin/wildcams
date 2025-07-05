"""
DeepSORT-based temporal tracker for wildlife video processing.

Provides appearance-based tracking with Kalman filtering for improved
temporal consistency in wildlife camera trap scenarios.
"""

import logging
import numpy as np
from typing import List, Dict, Optional
import cv2

from .tracking_interface import TemporalTracker
from .track_data import Detection, Track, TrackingInfo
from config import ProcessingConfig

try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

logger = logging.getLogger('wildcams')


class DeepSORTTracker(TemporalTracker):
    """
    DeepSORT-based tracker optimized for wildlife camera trap scenarios.
    
    Uses appearance features and Kalman filtering for robust tracking
    across frames with potential occlusions and lighting changes.
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize DeepSORT tracker with wildlife-optimized parameters.
        
        Args:
            config: ProcessingConfig with tracking parameters
        """
        super().__init__(config)
        
        if not DEEPSORT_AVAILABLE:
            raise ImportError("deep-sort-realtime not available. Install with: pip install deep-sort-realtime")
        
        # Extract tracking parameters from config with defaults
        max_age = getattr(config, 'tracking_max_age', 50)
        n_init = getattr(config, 'tracking_n_init', 3)
        max_iou_distance = getattr(config, 'tracking_max_iou_distance', 0.7)
        max_cosine_distance = getattr(config, 'tracking_max_cosine_distance', 0.4)
        nn_budget = getattr(config, 'tracking_nn_budget', 100)
        
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder="mobilenet",  # Lightweight embedder for speed
            half=True,  # Use half precision for speed
            bgr=True   # Input is BGR format
        )
        
        # Track management
        self.track_map = {}  # Maps DeepSORT track_id to our Track objects
        self.current_frame = None  # Store current frame for DeepSORT
        
        logger.info(f"🎯 DeepSORT tracker initialized:")
        logger.info(f"  📏 max_age={max_age}, n_init={n_init}")
        logger.info(f"  🎨 IoU distance={max_iou_distance}, cosine distance={max_cosine_distance}")
        logger.info(f"  💾 Neural network budget={nn_budget}")
    
    def update_tracks(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        """
        Update tracks with new detections using DeepSORT.
        
        Args:
            detections: List of Detection objects from current frame
            frame_idx: Current frame index
            
        Returns:
            List of updated Track objects
        """
        # Convert our Detection objects to DeepSORT format
        deepsort_detections = []
        detection_map = {}  # Map DeepSORT detection index to our Detection
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            # DeepSORT expects [left, top, width, height] format
            width = x2 - x1
            height = y2 - y1
            
            deepsort_det = ([x1, y1, width, height], detection.confidence, detection)
            deepsort_detections.append(deepsort_det)
            detection_map[i] = detection
        
        # Update DeepSORT tracker (requires frame for appearance features)
        if self.current_frame is not None:
            deepsort_tracks = self.tracker.update_tracks(deepsort_detections, frame=self.current_frame)
        else:
            # Fallback if no frame available
            deepsort_tracks = self.tracker.update_tracks(deepsort_detections)
        
        # Process DeepSORT results and update our Track objects
        active_track_ids = set()
        
        for deepsort_track in deepsort_tracks:
            if not deepsort_track.is_confirmed():
                continue
            
            track_id = deepsort_track.track_id
            active_track_ids.add(track_id)
            
            # Get the original detection from DeepSORT track
            if hasattr(deepsort_track, 'det') and deepsort_track.det is not None:
                original_detection = deepsort_track.det[2]  # Our Detection object
                
                # Update detection with tracking info
                original_detection.frame_idx = frame_idx
                
                # Get or create our Track object
                if track_id not in self.track_map:
                    # Create new track
                    track = Track(
                        track_id=track_id,
                        detections=[original_detection],
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                        duration_seconds=0.0,
                        is_confirmed=deepsort_track.hits >= self.tracker.n_init
                    )
                    self.track_map[track_id] = track
                    self.tracks.append(track)
                else:
                    # Update existing track
                    track = self.track_map[track_id]
                    self._add_detection_to_track(track, original_detection)
                    track.is_confirmed = deepsort_track.hits >= self.tracker.n_init
        
        # Clean up inactive tracks (not returned by DeepSORT)
        inactive_tracks = []
        for track_id, track in list(self.track_map.items()):
            if track_id not in active_track_ids:
                # Track is no longer active in DeepSORT
                inactive_tracks.append(track)
                # Keep track in self.tracks for final results, but remove from active map
                if track_id in self.track_map:
                    del self.track_map[track_id]
        
        logger.info(f"🎯 DeepSORT tracking (frame {frame_idx}):")
        logger.info(f"  📥 Input detections: {len(detections)}")
        logger.info(f"  🎯 Active tracks: {len(active_track_ids)}")
        logger.info(f"  ✅ Confirmed tracks: {len([t for t in self.tracks if t.is_confirmed])}")
        logger.info(f"  📊 Track IDs: {list(active_track_ids)}")
        
        return self.tracks
    
    def finalize_tracks(self, total_frames: int, fps: float) -> TrackingInfo:
        """
        Finalize tracking and return complete tracking information.
        
        Args:
            total_frames: Total number of frames in video
            fps: Video frame rate
            
        Returns:
            TrackingInfo with all tracks and statistics
        """
        # Update duration for all tracks
        for track in self.tracks:
            track.duration_seconds = (track.end_frame - track.start_frame) / fps
        
        # Filter tracks based on minimum requirements
        min_detections = getattr(self.config, 'min_track_detections', 3)
        min_duration = getattr(self.config, 'min_track_duration', 0.5)
        
        valid_tracks = [
            track for track in self.tracks
            if (len(track.detections) >= min_detections and
                track.duration_seconds >= min_duration)
        ]
        
        tracking_info = TrackingInfo(
            video_path=getattr(self.config, 'current_video_path', None),
            tracks=valid_tracks,
            total_frames=total_frames,
            fps=fps,
            tracking_method="DeepSORT"
        )
        
        logger.info(f"🎯 DeepSORT tracking finalized:")
        logger.info(f"  📊 Total tracks: {len(self.tracks)}")
        logger.info(f"  ✅ Valid tracks: {len(valid_tracks)}")
        logger.info(f"  📏 Min requirements: {min_detections} detections, {min_duration:.1f}s duration")
        
        return tracking_info
    
    def reset(self):
        """Reset tracker state for new video."""
        # Recreate DeepSORT tracker with same parameters
        self.tracker = DeepSort(
            max_age=self.tracker.max_age,
            n_init=self.tracker.n_init,
            max_iou_distance=self.tracker.max_iou_distance,
            max_cosine_distance=self.tracker.max_cosine_distance,
            nn_budget=self.tracker.nn_budget,
            embedder="mobilenet",
            half=True,
            bgr=True
        )
        
        # Reset our tracking state
        self.tracks = []
        self.track_map = {}
        self.next_track_id = 1
        self.current_frame = None
        
        logger.info("🔄 DeepSORT tracker reset for new video")
    
    def set_current_frame(self, frame: np.ndarray):
        """
        Set current frame for appearance feature extraction.
        
        Args:
            frame: Current video frame (BGR format)
        """
        self.current_frame = frame
    
    @property
    def tracking_method(self) -> str:
        """Return tracking method name."""
        return "DeepSORT"