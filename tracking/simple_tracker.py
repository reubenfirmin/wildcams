"""
Simple bounding box tracker for wildlife video processing.

Provides basic spatial tracking based on IoU overlap and distance
as a fallback when DeepSORT is not available.
"""

import logging
import math
from typing import List, Dict, Optional

from .tracking_interface import TemporalTracker
from .track_data import Detection, Track, TrackingInfo
from config import ProcessingConfig

logger = logging.getLogger('wildcams')


class SimpleBboxTracker(TemporalTracker):
    """
    Simple bounding box tracker using spatial proximity and IoU overlap.
    
    Tracks objects by associating detections based on:
    - Bounding box overlap (IoU)
    - Center-to-center distance
    - Size consistency
    
    Useful as a fallback when appearance-based tracking is not available.
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize simple tracker with spatial parameters.
        
        Args:
            config: ProcessingConfig with tracking parameters
        """
        super().__init__(config)
        
        # Extract tracking parameters with wildlife-optimized defaults
        self.max_distance = getattr(config, 'tracking_distance_threshold', 150.0)
        self.min_iou = getattr(config, 'tracking_min_iou', 0.1)
        self.max_age = getattr(config, 'tracking_max_age', 10)  # Frames without detection
        self.min_size_ratio = getattr(config, 'size_ratio_threshold', 0.5)
        
        # Track management
        self.active_tracks: List[Track] = []
        self.frame_idx = 0
        
        logger.info(f"🎯 Simple bbox tracker initialized:")
        logger.info(f"  📏 Max distance: {self.max_distance}px")
        logger.info(f"  📊 Min IoU: {self.min_iou}")
        logger.info(f"  ⏰ Max age: {self.max_age} frames")
        logger.info(f"  📐 Min size ratio: {self.min_size_ratio}")
    
    def update_tracks(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        """
        Update tracks with new detections using spatial association.
        
        Args:
            detections: List of Detection objects from current frame
            frame_idx: Current frame index
            
        Returns:
            List of updated Track objects
        """
        self.frame_idx = frame_idx
        
        # Associate detections with existing tracks
        used_detections = set()
        
        for track in self.active_tracks:
            best_detection = None
            best_score = 0.0
            best_idx = -1
            
            # Find best matching detection for this track
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                score = self._calculate_association_score(track, detection)
                if score > best_score and score > 0.0:
                    best_score = score
                    best_detection = detection
                    best_idx = i
            
            # Associate best detection with track
            if best_detection is not None:
                self._add_detection_to_track(track, best_detection)
                used_detections.add(best_idx)
                track.last_seen_frame = frame_idx
                
                # Confirm track if it has enough detections
                if not track.is_confirmed and self._should_confirm_track(track):
                    track.is_confirmed = True
                    logger.info(f"✅ Track {track.track_id} confirmed with {len(track.detections)} detections")
        
        # Create new tracks for unassociated detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                new_track = self._create_new_track(detection)
                new_track.last_seen_frame = frame_idx
                self.active_tracks.append(new_track)
                self.tracks.append(new_track)
                logger.info(f"🆕 Created new track {new_track.track_id}")
        
        # Remove old tracks that haven't been seen
        old_tracks = []
        for track in self.active_tracks:
            age = frame_idx - track.last_seen_frame
            if age > self.max_age:
                old_tracks.append(track)
                logger.info(f"⏰ Track {track.track_id} aged out after {age} frames")
        
        for old_track in old_tracks:
            self.active_tracks.remove(old_track)
        
        active_count = len(self.active_tracks)
        confirmed_count = len([t for t in self.active_tracks if t.is_confirmed])
        
        logger.info(f"🎯 Simple tracking (frame {frame_idx}):")
        logger.info(f"  📥 Input detections: {len(detections)}")
        logger.info(f"  🎯 Active tracks: {active_count}")
        logger.info(f"  ✅ Confirmed tracks: {confirmed_count}")
        logger.info(f"  🆕 New tracks: {len(detections) - len(used_detections)}")
        
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
            tracking_method="SimpleBbox"
        )
        
        logger.info(f"🎯 Simple tracking finalized:")
        logger.info(f"  📊 Total tracks: {len(self.tracks)}")
        logger.info(f"  ✅ Valid tracks: {len(valid_tracks)}")
        logger.info(f"  📏 Min requirements: {min_detections} detections, {min_duration:.1f}s duration")
        
        return tracking_info
    
    def reset(self):
        """Reset tracker state for new video."""
        self.tracks = []
        self.active_tracks = []
        self.next_track_id = 1
        self.frame_idx = 0
        
        logger.info("🔄 Simple bbox tracker reset for new video")
    
    def _calculate_association_score(self, track: Track, detection: Detection) -> float:
        """
        Calculate association score between track and detection.
        
        Args:
            track: Existing track
            detection: New detection
            
        Returns:
            Association score (0.0 = no match, 1.0 = perfect match)
        """
        if not track.detections:
            return 0.0
        
        # Use most recent detection from track
        last_detection = track.detections[-1]
        
        # Calculate IoU overlap
        iou = self._calculate_iou(last_detection.bbox, detection.bbox)
        if iou < self.min_iou:
            return 0.0
        
        # Calculate center distance
        last_center = self._get_bbox_center(last_detection.bbox)
        curr_center = self._get_bbox_center(detection.bbox)
        distance = math.sqrt(
            (last_center[0] - curr_center[0])**2 + 
            (last_center[1] - curr_center[1])**2
        )
        
        if distance > self.max_distance:
            return 0.0
        
        # Calculate size consistency
        last_area = self._get_bbox_area(last_detection.bbox)
        curr_area = self._get_bbox_area(detection.bbox)
        size_ratio = min(last_area, curr_area) / max(last_area, curr_area) if max(last_area, curr_area) > 0 else 0
        
        if size_ratio < self.min_size_ratio:
            return 0.0
        
        # Combine scores with weights
        iou_weight = 0.4
        distance_weight = 0.3
        size_weight = 0.3
        
        # Normalize distance score (closer = better)
        distance_score = max(0.0, 1.0 - (distance / self.max_distance))
        
        final_score = (iou * iou_weight + 
                      distance_score * distance_weight + 
                      size_ratio * size_weight)
        
        return final_score
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _get_bbox_center(self, bbox: List[float]) -> tuple:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _get_bbox_area(self, bbox: List[float]) -> float:
        """Get area of bounding box."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def tracking_method(self) -> str:
        """Return tracking method name."""
        return "SimpleBbox"