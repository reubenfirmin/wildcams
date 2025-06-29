#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "deep-sort-realtime>=1.3.2",
#   "numpy>=1.24.0",
#   "opencv-python>=4.8.0",
#   "torch>=2.0.0"
# ]
# ///
"""
Accuracy Enhancement 4: DeepSORT Integration for improved object tracking.
Provides appearance-based tracking with Kalman filtering for better temporal consistency.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2

try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("Warning: deep-sort-realtime not available. Install with: pip install deep-sort-realtime")

# Get logger
analysis_logger = logging.getLogger('wildcams')

class EnhancedDeepSortTracker:
    """
    Enhanced DeepSORT tracker optimized for wildlife camera trap scenarios.
    """
    
    def __init__(self, max_age: int = 50, n_init: int = 3, max_iou_distance: float = 0.7, 
                 max_cosine_distance: float = 0.4, nn_budget: int = 100):
        """
        Initialize DeepSORT tracker with wildlife-optimized parameters.
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            n_init: Number of consecutive detections before track is confirmed  
            max_iou_distance: Maximum IoU distance for track association
            max_cosine_distance: Maximum cosine distance for appearance matching
            nn_budget: Maximum size of appearance descriptor gallery
        """
        if not DEEPSORT_AVAILABLE:
            raise ImportError("deep-sort-realtime not available. Install with: pip install deep-sort-realtime")
        
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
        
        self.track_history = {}
        self.confirmed_tracks = set()
        
        analysis_logger.info(f"🎯 DeepSORT tracker initialized:")
        analysis_logger.info(f"  📏 max_age={max_age}, n_init={n_init}")
        analysis_logger.info(f"  🎨 IoU distance={max_iou_distance}, cosine distance={max_cosine_distance}")
        analysis_logger.info(f"  💾 Neural network budget={nn_budget}")
    
    def update_tracks(self, frame: np.ndarray, detections: List[Dict], 
                     frame_idx: int) -> List[Dict]:
        """
        Update tracks with new detections using DeepSORT.
        
        Args:
            frame: Current frame (BGR format)
            detections: List of detection dictionaries with bbox and confidence
            frame_idx: Current frame index
            
        Returns:
            List of tracked detections with track IDs
        """
        if not detections:
            # Update tracker with no detections
            tracks = self.tracker.update_tracks([], frame=frame)
            return []
        
        # Convert detections to DeepSORT format
        deepsort_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # DeepSORT expects [left, top, width, height] format
            width = x2 - x1
            height = y2 - y1
            
            deepsort_det = ([x1, y1, width, height], det['confidence'], det)
            deepsort_detections.append(deepsort_det)
        
        # Update tracks
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
        tracked_detections = []
        active_track_ids = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            active_track_ids.append(track_id)
            
            # Get the original detection data
            if hasattr(track, 'det') and track.det is not None:
                original_detection = track.det[2]  # Third element is our original detection dict
                
                # Add tracking information
                tracked_detection = original_detection.copy()
                tracked_detection.update({
                    'track_id': track_id,
                    'track_state': 'confirmed',
                    'time_since_update': track.time_since_update,
                    'hits': track.hits,
                    'hit_streak': track.hit_streak,
                    'age': track.age,
                    'frame_idx': frame_idx
                })
                
                tracked_detections.append(tracked_detection)
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append({
                    'frame_idx': frame_idx,
                    'bbox': tracked_detection['bbox'],
                    'confidence': tracked_detection['confidence'],
                    'timestamp': tracked_detection.get('timestamp', 0.0)
                })
                
                # Mark as confirmed track
                if track.hit_streak >= self.tracker.n_init:
                    self.confirmed_tracks.add(track_id)
        
        analysis_logger.info(f"🎯 DeepSORT tracking (frame {frame_idx}):")
        analysis_logger.info(f"  📥 Input detections: {len(detections)}")
        analysis_logger.info(f"  🎯 Active tracks: {len(active_track_ids)}")
        analysis_logger.info(f"  ✅ Confirmed tracks: {len([t for t in active_track_ids if t in self.confirmed_tracks])}")
        analysis_logger.info(f"  📊 Track IDs: {active_track_ids}")
        
        return tracked_detections
    
    def get_track_consistency_score(self, track_id: int, min_duration_frames: int = 30) -> float:
        """
        Calculate consistency score for a track based on temporal behavior.
        
        Args:
            track_id: Track ID to analyze
            min_duration_frames: Minimum frames for a valid track
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        if track_id not in self.track_history:
            return 0.0
        
        history = self.track_history[track_id]
        
        if len(history) < min_duration_frames:
            return 0.0
        
        # Calculate movement consistency
        if len(history) < 2:
            return 0.5
        
        # Analyze movement patterns
        movements = []
        for i in range(1, len(history)):
            prev_bbox = history[i-1]['bbox']
            curr_bbox = history[i]['bbox']
            
            prev_center = [(prev_bbox[0] + prev_bbox[2])/2, (prev_bbox[1] + prev_bbox[3])/2]
            curr_center = [(curr_bbox[0] + curr_bbox[2])/2, (curr_bbox[1] + curr_bbox[3])/2]
            
            movement = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            movements.append(movement)
        
        if not movements:
            return 0.5
        
        # Consistent movement should have low variance relative to mean
        mean_movement = np.mean(movements)
        std_movement = np.std(movements)
        
        if mean_movement == 0:
            consistency_score = 1.0  # No movement is perfectly consistent
        else:
            # Lower coefficient of variation = higher consistency
            cv = std_movement / mean_movement
            consistency_score = max(0.0, 1.0 - cv)
        
        # Boost score for longer tracks
        duration_bonus = min(0.2, len(history) / (min_duration_frames * 2))
        consistency_score += duration_bonus
        
        return min(1.0, consistency_score)
    
    def get_valid_tracks(self, min_duration_frames: int = 30, 
                        min_consistency_score: float = 0.6) -> List[Dict]:
        """
        Get tracks that meet validity criteria for wildlife detection.
        
        Args:
            min_duration_frames: Minimum frames for track validity
            min_consistency_score: Minimum consistency score
            
        Returns:
            List of valid track summaries
        """
        valid_tracks = []
        
        for track_id in self.confirmed_tracks:
            if track_id not in self.track_history:
                continue
            
            history = self.track_history[track_id]
            
            if len(history) < min_duration_frames:
                continue
            
            consistency_score = self.get_track_consistency_score(track_id, min_duration_frames)
            
            if consistency_score < min_consistency_score:
                continue
            
            # Calculate track statistics
            confidences = [h['confidence'] for h in history]
            timestamps = [h['timestamp'] for h in history]
            
            track_summary = {
                'track_id': track_id,
                'duration_frames': len(history),
                'duration_seconds': max(timestamps) - min(timestamps) if timestamps else 0.0,
                'consistency_score': consistency_score,
                'avg_confidence': np.mean(confidences),
                'max_confidence': max(confidences),
                'detection_count': len(history),
                'start_frame': history[0]['frame_idx'],
                'end_frame': history[-1]['frame_idx'],
                'best_detection': max(history, key=lambda h: h['confidence']),
                'history': history
            }
            
            valid_tracks.append(track_summary)
            
            analysis_logger.info(f"✅ Valid track {track_id}: {len(history)} frames, "
                               f"consistency={consistency_score:.3f}, "
                               f"avg_conf={np.mean(confidences):.3f}")
        
        # Sort by consistency score descending
        valid_tracks.sort(key=lambda t: t['consistency_score'], reverse=True)
        
        analysis_logger.info(f"🎯 Track validation results:")
        analysis_logger.info(f"  📊 Total tracks processed: {len(self.confirmed_tracks)}")
        analysis_logger.info(f"  ✅ Valid tracks found: {len(valid_tracks)}")
        analysis_logger.info(f"  📏 Min duration: {min_duration_frames} frames")
        analysis_logger.info(f"  🎯 Min consistency: {min_consistency_score}")
        
        return valid_tracks
    
    def reset(self):
        """Reset tracker state for new video."""
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
        self.track_history = {}
        self.confirmed_tracks = set()
        
        analysis_logger.info("🔄 DeepSORT tracker reset for new video")