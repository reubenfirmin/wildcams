#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "python-dotenv>=1.0.0",
#   "opencv-python>=4.8.0",
#   "ultralytics>=8.0.0",
#   "scikit-learn>=1.3.0",
#   "numpy>=1.24.0",
#   "pillow>=10.0.0",
#   "tqdm>=4.66.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
#   "transformers>=4.35.0",
#   "pybioclip>=0.1.0",
#   "pytorchwildlife>=1.0.0"
# ]
# ///
"""
Next Generation Wildlife Video Processor.
Combines motion detection with temporal consistency tracking and full-frame validation.
"""

import os
import sys
import cv2
import math
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from tqdm import tqdm
from dataclasses import dataclass

# Import base processor
from video_processor_base import VideoProcessorBase

# Global configuration object
@dataclass
class ProcessingConfig:
    """Global configuration for next-generation video processing."""
    # Video processing
    max_frames_per_video: int
    confidence_threshold: float
    
    # Validation thresholds
    megadetector_high_conf: float
    yolo_high_conf: float
    min_yolo_detections: int
    weak_evidence_threshold: float
    
    # Camera handling detection
    max_tracks_threshold: int
    max_long_tracks_threshold: int
    max_dense_tracks_threshold: int
    long_duration_threshold: float
    high_density_threshold: int
    composite_motion_threshold: int
    min_motion_threshold: int
    motion_frames_weight: float
    motion_regions_weight: float
    motion_tracks_weight: float
    large_region_multiplier: float
    
    # Motion detection
    motion_method: str
    motion_var_threshold: int
    filter_motion_var_threshold: int
    analysis_motion_var_threshold: int
    min_motion_area: int
    max_motion_area: int
    motion_history: int
    max_regions_per_frame: int
    min_region_width: int
    min_region_height: int
    max_aspect_ratio: float
    motion_margin: int
    
    # Temporal consistency
    min_track_duration: float
    motion_tracking_gap_seconds: float
    detection_validation_gap_seconds: float
    tracking_distance_threshold: float
    anchor_confidence_threshold: float
    min_track_frames: int
    
    # Step 4 validation
    max_validation_frames: int
    crop_weight: float
    fullframe_weight: float
    min_crop_size: int
    temporal_spread_seconds: float
    accepted_rtdetr_overlap: float
    
    # Missing parameters that need CLI args
    full_frame_validation_frames: int
    size_ratio_threshold: float
    track_search_seconds: float
    
    # Model configuration
    ensemble_models: List[str]

# Global config instance
config: ProcessingConfig = None

# Get loggers
logger = logging.getLogger('wildcams')

@dataclass
class Detection:
    """A single animal detection with metadata."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frame_idx: int
    timestamp: float
    source: str
    motion_area: Optional[float] = None

@dataclass
class TrackingInfo:
    """Tracking information for temporal consistency."""
    track_id: int
    detections: List[Detection]
    start_frame: int
    end_frame: int
    duration_seconds: float
    consistency_score: float
    
    def center(self, detection: Detection) -> Tuple[float, float]:
        """Get center point of detection bbox."""
        x1, y1, x2, y2 = detection.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def movement_distance(self) -> float:
        """Calculate total movement distance of track."""
        if len(self.detections) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.detections)):
            prev_center = self.center(self.detections[i-1])
            curr_center = self.center(self.detections[i])
            distance = math.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            total_distance += distance
        return total_distance

class NextGenVideoProcessor(VideoProcessorBase):
    """Next generation processor with temporal consistency and full-frame validation."""
    
    def __init__(self):
        super().__init__()
        
        # Override base class attributes with config values
        self.ensemble_models = config.ensemble_models
        
        # Create tracking subdirectory for .processed files
        self.tracking_dir = self.video_dir / '.tracking'
        self.tracking_dir.mkdir(exist_ok=True)
        
        # Motion detection configuration for filter (Step 2)
        self.filter_motion_config = {
            'method': config.motion_method,
            'var_threshold': config.filter_motion_var_threshold,
            'min_area': config.min_motion_area,
            'max_area': config.max_motion_area,
            'detect_shadows': True,
            'history': config.motion_history,
            'max_regions_per_frame': config.max_regions_per_frame,
            'min_region_width': config.min_region_width,
            'min_region_height': config.min_region_height,
            'max_aspect_ratio': config.max_aspect_ratio,
            'motion_margin': config.motion_margin
        }
        
        # Motion detection configuration for spatial analysis (Step 3)
        self.analysis_motion_config = {
            'method': config.motion_method,
            'var_threshold': config.analysis_motion_var_threshold,
            'min_area': config.min_motion_area,
            'max_area': config.max_motion_area,
            'detect_shadows': True,
            'history': config.motion_history,
            'max_regions_per_frame': config.max_regions_per_frame,
            'min_region_width': config.min_region_width,
            'min_region_height': config.min_region_height,
            'max_aspect_ratio': config.max_aspect_ratio,
            'motion_margin': config.motion_margin
        }
        
        # Primary motion config points to filter config for backward compatibility
        self.motion_config = self.filter_motion_config
        
        # Initialize motion detection algorithm
        self.bg_subtractor = None
        self.init_motion_detector()
        
        # Tracking state
        self.active_tracks = []
        
        # Create crop-only ensemble (exclude RT-DETR models)
        self.crop_ensemble_models = [model for model in self.ensemble_models 
                                    if 'rtdetr' not in model.lower()]
        
        logger.info(f"💡 ENSEMBLE SPLIT:")
        logger.info(f"  🔲 Crop analysis: {self.crop_ensemble_models}")
        logger.info(f"  🖼️ Full-frame analysis: {self.ensemble_models}")
        self.next_track_id = 0
        
        logger.info(f"🎯 Next Generation video processor initialized")
        logger.info(f"🕒 Temporal consistency: min {config.min_track_duration}s, motion gap {config.motion_tracking_gap_seconds}s, detection gap {config.detection_validation_gap_seconds}s")
        logger.info(f"🔍 Motion method: {self.motion_config['method']}")
        logger.info(f"🔀 Dual motion detection: filter_var_threshold={config.filter_motion_var_threshold}, analysis_var_threshold={config.analysis_motion_var_threshold}")
        logger.info(f"✅ Full-frame validation: {config.full_frame_validation_frames} consecutive frames")
    
    def init_motion_detector(self):
        """Initialize motion detection algorithm (for filter stage)."""
        if self.motion_config['method'] == 'MOG2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=self.motion_config['detect_shadows'],
                varThreshold=self.motion_config['var_threshold'],
                history=self.motion_config['history']
            )
        elif self.motion_config['method'] == 'KNN':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                detectShadows=self.motion_config['detect_shadows'],
                dist2Threshold=400,
                history=self.motion_config['history']
            )
        else:
            raise ValueError(f"Unknown motion detection method: {self.motion_config['method']}")
        
        logger.info(f"Motion detector initialized: {self.motion_config['method']}")
    
    def create_analysis_motion_detector(self):
        """Create separate motion detector for spatial analysis with stricter parameters."""
        if self.analysis_motion_config['method'] == 'MOG2':
            return cv2.createBackgroundSubtractorMOG2(
                detectShadows=self.analysis_motion_config['detect_shadows'],
                varThreshold=self.analysis_motion_config['var_threshold'],
                history=self.analysis_motion_config['history']
            )
        elif self.analysis_motion_config['method'] == 'KNN':
            return cv2.createBackgroundSubtractorKNN(
                detectShadows=self.analysis_motion_config['detect_shadows'],
                dist2Threshold=400,
                history=self.analysis_motion_config['history']
            )
        else:
            raise ValueError(f"Unknown motion detection method: {self.analysis_motion_config['method']}")
    
    def _detect_motion_regions_with_subtractor(self, frame: np.ndarray, bg_subtractor, use_config: str = 'analysis') -> List[Tuple[int, int, int, int]]:
        """Detect motion regions using a specific background subtractor and config."""
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Select appropriate config based on use case
        active_config = self.analysis_motion_config if use_config == 'analysis' else self.filter_motion_config
        
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < active_config['min_area'] or area > active_config['max_area']:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by dimensions and aspect ratio
            if (w < active_config['min_region_width'] or 
                h < active_config['min_region_height']):
                continue
            
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > active_config['max_aspect_ratio']:
                continue
            
            # Expand region with smart margin for better ML context
            base_margin = active_config['motion_margin']
            
            # Smart margin calculation: larger context for better crops
            percentage_margin_w = max(base_margin, w * 0.75)
            percentage_margin_h = max(base_margin, h * 0.75)
            
            # Ensure minimum crop size for ML effectiveness
            min_crop_width = 150
            min_crop_height = 150
            
            # Calculate expanded region
            frame_h, frame_w = frame.shape[:2]
            center_x, center_y = x + w // 2, y + h // 2
            
            # Initial expansion with smart margins
            x1_expanded = max(0, x - int(percentage_margin_w))
            y1_expanded = max(0, y - int(percentage_margin_h))
            x2_expanded = min(frame_w, x + w + int(percentage_margin_w))
            y2_expanded = min(frame_h, y + h + int(percentage_margin_h))
            
            # Ensure minimum crop size by expanding around center if needed
            current_width = x2_expanded - x1_expanded
            current_height = y2_expanded - y1_expanded
            
            if current_width < min_crop_width:
                expansion_needed = (min_crop_width - current_width) // 2
                x1_expanded = max(0, x1_expanded - expansion_needed)
                x2_expanded = min(frame_w, x2_expanded + expansion_needed)
                
            if current_height < min_crop_height:
                expansion_needed = (min_crop_height - current_height) // 2
                y1_expanded = max(0, y1_expanded - expansion_needed)
                y2_expanded = min(frame_h, y2_expanded + expansion_needed)
            
            motion_regions.append((x1_expanded, y1_expanded, x2_expanded, y2_expanded))
            
            # Limit number of regions to avoid processing too many
            if len(motion_regions) >= active_config['max_regions_per_frame']:
                break
        
        return motion_regions
    
    def _open_video_stream(self, video_path: Path) -> Optional[cv2.VideoCapture]:
        """Open video stream with fallback backends."""
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(str(video_path), backend)
            if cap.isOpened():
                return cap
            cap.release()
        
        logger.error(f"❌ Could not open video with any backend: {video_path}")
        return None
    
    def detect_motion_regions(self, frame: np.ndarray, use_config: str = 'filter') -> List[Tuple[int, int, int, int]]:
        """Detect motion regions in frame using background subtraction.
        
        Args:
            frame: Input frame
            use_config: 'filter' for lenient filtering config, 'analysis' for strict analysis config
        """
        if self.bg_subtractor is None:
            return []
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Select appropriate config based on use case
        active_config = self.analysis_motion_config if use_config == 'analysis' else self.filter_motion_config
        
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < active_config['min_area'] or area > active_config['max_area']:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by dimensions and aspect ratio
            if (w < active_config['min_region_width'] or 
                h < active_config['min_region_height']):
                continue
            
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > active_config['max_aspect_ratio']:
                continue
            
            # Expand region with smart margin for better ML context
            base_margin = active_config['motion_margin']  # Use appropriate config
            
            # Smart margin calculation: larger context for better crops
            # Use percentage-based expansion with minimum absolute margin
            percentage_margin_w = max(base_margin, w * 0.75)  # 75% of motion width or base margin
            percentage_margin_h = max(base_margin, h * 0.75)  # 75% of motion height or base margin
            
            # Ensure minimum crop size for ML effectiveness (at least 150x150)
            min_crop_width = 150
            min_crop_height = 150
            
            # Calculate expanded region
            frame_h, frame_w = frame.shape[:2]
            center_x, center_y = x + w // 2, y + h // 2
            
            # Initial expansion with smart margins
            x1_expanded = max(0, x - int(percentage_margin_w))
            y1_expanded = max(0, y - int(percentage_margin_h))
            x2_expanded = min(frame_w, x + w + int(percentage_margin_w))
            y2_expanded = min(frame_h, y + h + int(percentage_margin_h))
            
            # Ensure minimum crop dimensions by expanding from center if needed
            current_width = x2_expanded - x1_expanded
            current_height = y2_expanded - y1_expanded
            
            if current_width < min_crop_width:
                needed_expansion = (min_crop_width - current_width) // 2
                x1_expanded = max(0, x1_expanded - needed_expansion)
                x2_expanded = min(frame_w, x2_expanded + needed_expansion)
            
            if current_height < min_crop_height:
                needed_expansion = (min_crop_height - current_height) // 2
                y1_expanded = max(0, y1_expanded - needed_expansion)
                y2_expanded = min(frame_h, y2_expanded + needed_expansion)
            
            x1, y1, x2, y2 = x1_expanded, y1_expanded, x2_expanded, y2_expanded
            
            motion_regions.append((x1, y1, x2, y2))
            
            if len(motion_regions) >= self.motion_config['max_regions_per_frame']:
                break
        
        return motion_regions
    
    def run_ml_on_region(self, frame: np.ndarray, region: Tuple[int, int, int, int], 
                        frame_idx: int, timestamp: float) -> List[Detection]:
        """Run ML ensemble on motion region crop."""
        x1, y1, x2, y2 = region
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop = frame[y1:y2, x1:x2]
        
        # Run ML ensemble on crop
        crop_detections = self.ml_ensemble.run_ensemble_detection(crop, timestamp)
        
        # Convert crop coordinates to full frame coordinates
        detections = []
        for det in crop_detections:
            # Adjust bounding box coordinates
            crop_x1, crop_y1, crop_x2, crop_y2 = det['bbox']
            full_x1 = x1 + crop_x1
            full_y1 = y1 + crop_y1  
            full_x2 = x1 + crop_x2
            full_y2 = y1 + crop_y2
            
            detection = Detection(
                bbox=(full_x1, full_y1, full_x2, full_y2),
                confidence=det['confidence'],
                frame_idx=frame_idx,
                timestamp=timestamp,
                source=det['source'],
                motion_area=(x2-x1) * (y2-y1)
            )
            detections.append(detection)
        
        return detections
    
    def associate_detections_to_tracks(self, detections: List[Detection], fps: float) -> None:
        """Associate new detections to existing tracks or create new tracks."""
        unmatched_detections = list(detections)
        
        # Try to match detections to active tracks
        for track in self.active_tracks:
            if not track.detections:
                continue
                
            last_detection = track.detections[-1]
            last_center = track.center(last_detection)
            
            best_match = None
            best_distance = float('inf')
            
            for detection in unmatched_detections:
                curr_center = track.center(detection)
                distance = math.sqrt(
                    (curr_center[0] - last_center[0])**2 + 
                    (curr_center[1] - last_center[1])**2
                )
                
                if distance < config.tracking_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = detection
            
            if best_match:
                track.detections.append(best_match)
                track.end_frame = best_match.frame_idx
                track.duration_seconds = (track.end_frame - track.start_frame) / fps
                unmatched_detections.remove(best_match)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = TrackingInfo(
                track_id=self.next_track_id,
                detections=[detection],
                start_frame=detection.frame_idx,
                end_frame=detection.frame_idx,
                duration_seconds=0.0,
                consistency_score=1.0
            )
            self.active_tracks.append(new_track)
            self.next_track_id += 1
    
    def cleanup_old_tracks(self, current_frame: int) -> None:
        """Remove tracks that haven't been updated recently."""
        self.active_tracks = [
            track for track in self.active_tracks
            if (current_frame - track.end_frame) / fps <= config.motion_tracking_gap_seconds
        ]
    
    def get_valid_tracks(self) -> List[TrackingInfo]:
        """Get tracks that meet temporal consistency requirements."""
        valid_tracks = []
        for track in self.active_tracks:
            if (track.duration_seconds >= config.min_track_duration and 
                len(track.detections) >= 2):
                valid_tracks.append(track)
        return valid_tracks
    
    def find_consistent_motion_sequences_and_tracks(self, video_path: Path, fps: float, total_frames: int) -> List[Dict]:
        """STEP 1: Find sequences with consistent motion/tracking."""
        cap = self._open_video_stream(video_path)
        if not cap:
            return []
        
        # Reset motion detector for this video
        self.init_motion_detector()
        
        motion_sequences = []
        current_sequence = None
        
        logger.info(f"Analyzing motion across {total_frames} frames")
        logger.info(f"📊 Motion detection parameters:")
        logger.info(f"  🎯 Min motion area: {self.motion_config['min_area']} pixels")
        logger.info(f"  📈 Variance threshold: {self.motion_config['var_threshold']}")
        logger.info(f"  ⏱️ Min track duration: {config.min_track_duration}s")
        logger.info(f"  📦 Max regions per frame: {self.motion_config['max_regions_per_frame']}")
        
        frame_motion_count = 0
        total_motion_regions = 0
        motion_region_sizes = []
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            motion_regions = self.detect_motion_regions(frame)
            
            if motion_regions:
                frame_motion_count += 1
                total_motion_regions += len(motion_regions)
                
                # Track motion region sizes
                for region in motion_regions:
                    x1, y1, x2, y2 = region
                    area = (x2 - x1) * (y2 - y1)
                    motion_region_sizes.append(area)
                
                if frame_idx % 50 == 0:  # Log every 50 frames
                    logger.info(f"📊 Frame {frame_idx}: {len(motion_regions)} motion regions detected")
                if current_sequence is None:
                    # Start new sequence
                    current_sequence = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'start_timestamp': timestamp,
                        'end_timestamp': timestamp,
                        'motion_regions': [motion_regions],
                        'frames': [frame_idx]
                    }
                else:
                    # Continue current sequence
                    current_sequence['end_frame'] = frame_idx
                    current_sequence['end_timestamp'] = timestamp
                    current_sequence['motion_regions'].append(motion_regions)
                    current_sequence['frames'].append(frame_idx)
            else:
                # No motion - end current sequence if it meets duration requirement
                if current_sequence is not None:
                    duration = current_sequence['end_timestamp'] - current_sequence['start_timestamp']
                    if duration >= config.min_track_duration:
                        motion_sequences.append(current_sequence)
                        logger.info(f"Motion sequence found: frames {current_sequence['start_frame']}-{current_sequence['end_frame']} ({duration:.2f}s)")
                    current_sequence = None
        
        # Handle sequence that extends to end of video
        if current_sequence is not None:
            duration = current_sequence['end_timestamp'] - current_sequence['start_timestamp']
            if duration >= config.min_track_duration:
                motion_sequences.append(current_sequence)
                logger.info(f"Motion sequence found: frames {current_sequence['start_frame']}-{current_sequence['end_frame']} ({duration:.2f}s)")
        
        cap.release()
        
        # Add detailed motion detection summary
        logger.info(f"📊 MOTION DETECTION SUMMARY:")
        logger.info(f"  🎯 Total frames: {total_frames}")
        logger.info(f"  📈 Frames with motion: {frame_motion_count} ({100*frame_motion_count/total_frames:.1f}%)")
        logger.info(f"  📦 Total motion regions: {total_motion_regions}")
        if motion_region_sizes:
            avg_region_size = sum(motion_region_sizes) / len(motion_region_sizes)
            max_region_size = max(motion_region_sizes)
            min_region_size = min(motion_region_sizes)
            # Camera handling typically has many large regions vs animal movement with smaller focused regions
            # For 1600x900 videos, large regions would be >100k pixels (e.g. 400x250px or ~17% of frame)
            large_regions = len([size for size in motion_region_sizes if size > 100000])  # >400x250px
            
            # Store for use in composite score calculation
            self._large_region_count = large_regions
            self._total_region_count = len(motion_region_sizes)
            
            logger.info(f"  📏 Region sizes - Avg: {avg_region_size:.0f}px, Min: {min_region_size:.0f}px, Max: {max_region_size:.0f}px")
            logger.info(f"  📦 Large regions (>100k px): {large_regions}/{len(motion_region_sizes)} ({100*large_regions/len(motion_region_sizes):.1f}%)")
        else:
            # Store defaults if no motion regions found
            self._large_region_count = 0
            self._total_region_count = 0
        logger.info(f"  ⏱️ Motion sequences found: {len(motion_sequences)}")
        
        if frame_motion_count == 0:
            logger.warning(f"⚠️ NO MOTION DETECTED - Check parameters:")
            logger.warning(f"  🎯 Min area: {self.motion_config['min_area']} (try lower like 100-200)")
            logger.warning(f"  📈 Var threshold: {self.motion_config['var_threshold']} (try lower like 16-25)")
        elif len(motion_sequences) == 0:
            logger.warning(f"⚠️ MOTION DETECTED BUT NO SEQUENCES - Duration too strict:")
            logger.warning(f"  ⏱️ Min duration: {config.min_track_duration}s (try lower like 1.0s)")
        
        # Convert motion sequences to motion tracks for Step 2 filtering
        motion_tracks = []
        for i, sequence in enumerate(motion_sequences):
            duration = sequence['end_timestamp'] - sequence['start_timestamp']
            
            # Create motion track from sequence
            motion_track = {
                'track_id': i,
                'start_frame': sequence['start_frame'],
                'end_frame': sequence['end_frame'],
                'duration_seconds': duration,
                'frames': sequence['frames'],
                'motion_regions': sequence['motion_regions'],
                'detection_count': len(sequence['frames']),  # Number of frames with motion
                'avg_regions_per_frame': sum(len(regions) for regions in sequence['motion_regions']) / len(sequence['motion_regions']) if sequence['motion_regions'] else 0
            }
            motion_tracks.append(motion_track)
        
        return motion_tracks
    
    def filter_motion_tracks_for_camera_handling(self, video_path: Path, motion_tracks: List[Dict]) -> List[Dict]:
        """STEP 2: Filter motion tracks for camera handling detection using weighted composite motion score."""
        logger.info(f"[STEP2] {video_path.name}: Filtering {len(motion_tracks)} motion tracks for camera handling")
        
        # Calculate weighted composite motion score with large region multiplier
        total_motion_frames = sum(len(track['frames']) for track in motion_tracks)
        total_regions = sum(track.get('detection_count', len(track['frames'])) for track in motion_tracks)
        num_tracks = len(motion_tracks)
        
        # Calculate large region percentage from global motion_region_sizes
        # Note: This uses the global motion data calculated in find_consistent_motion_sequences_and_tracks
        large_region_count = getattr(self, '_large_region_count', 0)
        total_region_count = getattr(self, '_total_region_count', 0)
        large_region_percentage = large_region_count / total_region_count if total_region_count > 0 else 0.0
        
        # Weighted composite score: frames^a * regions^b * tracks^c * (1 + large_region_percentage * d)
        base_score = (total_motion_frames ** config.motion_frames_weight * 
                     total_regions ** config.motion_regions_weight * 
                     num_tracks ** config.motion_tracks_weight)
        
        large_region_multiplier = 1.0 + (large_region_percentage * config.large_region_multiplier)
        composite_score = base_score * large_region_multiplier
        
        threshold = config.composite_motion_threshold
        
        logger.info(f"[STEP2] {video_path.name}: Weighted composite score = {composite_score:.0f}")
        logger.info(f"  📊 Base: frames^{config.motion_frames_weight}={total_motion_frames}^{config.motion_frames_weight:.1f} * regions^{config.motion_regions_weight}={total_regions}^{config.motion_regions_weight:.1f} * tracks^{config.motion_tracks_weight}={num_tracks}^{config.motion_tracks_weight:.1f} = {base_score:.0f}")
        logger.info(f"  📊 Large regions: {large_region_count}/{total_region_count} ({large_region_percentage*100:.1f}%) * {config.large_region_multiplier} = {large_region_multiplier:.2f}x multiplier")
        
        # Store composite score for summary reporting
        self._composite_scores = getattr(self, '_composite_scores', {})
        self._composite_scores[video_path.name] = composite_score
        
        # Check for insufficient motion (static video)
        if composite_score < config.min_motion_threshold:
            logger.warning(f"⚠️  INSUFFICIENT MOTION: score={composite_score} < {config.min_motion_threshold}")
            # Store rejection reason for summary
            self._rejection_reasons = getattr(self, '_rejection_reasons', {})
            self._rejection_reasons[video_path.name] = f"insufficient_motion (score={composite_score:.0f})"
            return []
        
        # Check for excessive motion (camera handling)
        if composite_score > threshold:
            logger.warning(f"⚠️  CAMERA HANDLING: score={composite_score} > {threshold}")
            # Store rejection reason for summary
            self._rejection_reasons = getattr(self, '_rejection_reasons', {})
            self._rejection_reasons[video_path.name] = f"camera_handling (score={composite_score:.0f})"
            return []  # Early exit - skip expensive ML processing
        return motion_tracks
    
    def run_ml_on_motion_tracks(self, video_path: Path, motion_tracks: List[Dict]) -> List[Dict]:
        """STEP 3: Run crop-only ML analysis (YOLO + MegaDetector YOLO variants, NO RT-DETR)."""
        cap = self._open_video_stream(video_path)
        if not cap:
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        scored_tracks = []
        
        logger.info(f"[STEP3] Running ML on {len(motion_tracks)} motion tracks using {len(self.crop_ensemble_models)} models")
        
        for track in motion_tracks:
            track_detections = []
            track_scores = []
            
            # Sample representative frames from this motion track (max 5 per track)
            sample_indices = self._sample_frame_indices(track['frames'], max_samples=5)
            
            for sample_idx in sample_indices:
                frame_idx = track['frames'][sample_idx]
                motion_regions = track['motion_regions'][sample_idx]
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                timestamp = frame_idx / fps
                
                # Run ML on each motion region in this frame
                for region in motion_regions:
                    x1, y1, x2, y2 = region
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ensure valid crop dimensions and minimum size (updated for larger crops)
                    crop_width = x2 - x1
                    crop_height = y2 - y1
                    min_dim = 100  # Increased from 50 to match larger crop strategy
                    min_area = 15000  # 150x100 minimum for better ML context
                    if (x2 <= x1 or y2 <= y1 or 
                        crop_width < min_dim or crop_height < min_dim or
                        crop_width * crop_height < min_area):
                        continue
                        
                    crop = frame[y1:y2, x1:x2]
                    crop_area = crop_width * crop_height
                    
                    # Run crop-only ensemble (YOLO + MegaDetector YOLO variants, NO RT-DETR)
                    crop_detections = self._run_crop_ensemble(crop, timestamp, frame_idx, motion_region=region)
                    
                    for det in crop_detections:
                        # Transform coordinates back to full frame
                        crop_x1, crop_y1, crop_x2, crop_y2 = det['bbox']
                        full_bbox = [x1 + crop_x1, y1 + crop_y1, x1 + crop_x2, y1 + crop_y2]
                        
                        detection = {
                            'frame_idx': frame_idx,
                            'timestamp': timestamp,
                            'bbox': full_bbox,
                            'confidence': det['confidence'],
                            'source': f"crop_{det['source']}",
                            'crop_area': crop_area,
                            'crop_dimensions': f"{crop.shape[1]}x{crop.shape[0]}",
                            'track_id': track['track_id']
                        }
                        track_detections.append(detection)
                        track_scores.append(det['confidence'])
                    
                    # Log detection silently
            
            # Calculate track summary statistics
            if track_scores:
                max_score = max(track_scores)
                avg_score = sum(track_scores) / len(track_scores)
                detection_count = len(track_detections)
                
                # Calculate per-model contribution statistics
                model_contributions = {}
                for det in track_detections:
                    source = det['source']
                    if source not in model_contributions:
                        model_contributions[source] = {
                            'count': 0,
                            'max_conf': 0.0,
                            'total_conf': 0.0
                        }
                    model_contributions[source]['count'] += 1
                    model_contributions[source]['max_conf'] = max(model_contributions[source]['max_conf'], det['confidence'])
                    model_contributions[source]['total_conf'] += det['confidence']
                
                # Calculate average confidence per model
                for source in model_contributions:
                    model_contributions[source]['avg_conf'] = model_contributions[source]['total_conf'] / model_contributions[source]['count']
                
                track_summary = {
                    'track_id': track['track_id'],
                    'detections': track_detections,
                    'max_score': max_score,
                    'avg_score': avg_score,
                    'detection_count': detection_count,
                    'duration_seconds': track['duration_seconds'],
                    'frames_sampled': len(sample_indices),
                    'frames_total': len(track['frames']),
                    'model_contributions': model_contributions
                }
                scored_tracks.append(track_summary)
                
                logger.info(f"Track {track['track_id']}: {detection_count} detections, max_conf={max_score:.3f}")
            else:
                # Track had no detections
                logger.info(f"Track {track['track_id']}: No detections found")
        
        cap.release()
        
        # Step 3 complete
        return scored_tracks
    
    def _sample_frame_indices(self, frame_indices: List[int], max_samples: int = 5) -> List[int]:
        """Sample representative indices from frame list for motion track analysis."""
        if len(frame_indices) <= max_samples:
            return list(range(len(frame_indices)))
        
        # Temporal spread sampling: select frames evenly across the sequence
        step = len(frame_indices) / max_samples
        indices = [int(i * step) for i in range(max_samples)]
        
        # Ensure we always include the last frame if it's not already included
        if indices[-1] != len(frame_indices) - 1:
            indices[-1] = len(frame_indices) - 1
            
        return indices
    
    def _sample_crop_indices(self, frame_indices: List[int], max_samples: int = 5) -> List[int]:
        """Sample representative indices from frame list for crop analysis."""
        if len(frame_indices) <= max_samples:
            return list(range(len(frame_indices)))
        
        # Sample evenly distributed indices
        step = len(frame_indices) / max_samples
        return [int(i * step) for i in range(max_samples)]
    
    def _run_crop_ensemble(self, crop: np.ndarray, timestamp: float, frame_idx: int, motion_region: tuple = None) -> List[Dict]:
        """Run crop-only ensemble models (YOLO + MegaDetector YOLO variants, NO RT-DETR)."""
        all_detections = []
        
        for model_name in self.crop_ensemble_models:
            try:
                if (model_name.startswith('yolov8') or model_name.startswith('yolov10') or 
                    model_name.startswith('yolo12')):
                    # Run YOLO model directly
                    detections = self._run_yolo_on_crop(crop, model_name, timestamp, frame_idx, motion_region)
                elif model_name.startswith('MDV6-') and 'yolov' in model_name.lower():
                    # Run MegaDetector YOLO variant on crop
                    detections = self._run_megadetector_yolo_on_crop(crop, model_name, timestamp, frame_idx)
                else:
                    continue  # Skip non-YOLO models in crop analysis
                
                all_detections.extend(detections)
                
            except Exception as e:
                import traceback
                logger.error(f"Error running {model_name} on crop: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        return all_detections
    
    def _run_yolo_on_crop(self, crop: np.ndarray, model_name: str, timestamp: float, frame_idx: int, motion_region: tuple = None) -> List[Dict]:
        """Run YOLO model on crop using unified registry."""
        # Use unified YOLO detector registry
        detector = self.ml_ensemble.yolo_detectors.get(model_name)
        
        if detector is None:
            logger.warning(f"⚠️ {model_name} detector not found or not loaded")
            return []
        
        try:
            results = detector(crop, conf=config.confidence_threshold, verbose=False)
        except Exception as e:
            logger.error(f"❌ {model_name} detection failed on crop: {e}")
            return []
        
        detections = []
        if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf)
                if conf >= config.confidence_threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    detection = {
                        'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        'confidence': conf,
                        'source': model_name,
                        'timestamp': timestamp,
                        'frame_idx': frame_idx
                    }
                    detections.append(detection)
                    # Log individual YOLO crop detection with precise bbox
                    logger.info(f"🔲 {timestamp:.1f}s | {xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f} | {model_name} | CROP_DETECT | conf={conf:.3f}")
        
        # Log summary for this model - motion bbox and final count
        if motion_region:
            motion_bbox = motion_region
            logger.info(f"🔲 {timestamp:.1f}s | {motion_bbox[0]:.0f},{motion_bbox[1]:.0f},{motion_bbox[2]:.0f},{motion_bbox[3]:.0f} | {model_name} | CROP_RESULT | {len(detections)} detections")
        else:
            logger.info(f"🔲 {timestamp:.1f}s | {model_name} | CROP_RESULT | {len(detections)} detections")
        
        return detections
    
    def _run_megadetector_yolo_on_crop(self, crop: np.ndarray, model_name: str, timestamp: float, frame_idx: int) -> List[Dict]:
        """Run MegaDetector YOLO variant on crop (exclude RT-DETR)."""
        if 'rtdetr' in model_name.lower():
            return []  # Skip RT-DETR in crop analysis
        
        variant_model = self.ml_ensemble.megadetector_variants.get(model_name)
        if variant_model is None:
            return []
        
        try:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = variant_model.single_image_detection(
                rgb_crop, 
                det_conf_thres=config.confidence_threshold
            )
            
            detections = []
            if results and 'detections' in results:
                # Process MegaDetector results
                for det in results['detections']:
                    
                    # Handle different detection formats
                    if isinstance(det, dict):
                        # Dictionary format: {'conf': float, 'bbox': [...]}
                        conf = det.get('conf', 0.0)
                        bbox = det.get('bbox', None)
                        class_id = det.get('class_id', 1)
                    elif isinstance(det, (tuple, list)) and len(det) >= 3:
                        # MegaDetector tuple format: (bbox_array, None, confidence, class_id, None, {})
                        bbox = det[0]  # numpy array [x1, y1, x2, y2]
                        conf = det[2] if len(det) > 2 and det[2] is not None else 0.0
                        class_id = int(det[3]) if len(det) > 3 and det[3] is not None else 1
                    else:
                        print(f"DEBUG: Unexpected detection format: {type(det)} - {det}")
                        continue
                    
                    if conf >= config.confidence_threshold:
                        # Handle different bbox formats (bbox already extracted above)
                        if isinstance(bbox, dict):
                            # Convert dict format to list format [x1, y1, x2, y2]
                            bbox = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
                        elif isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
                            # Ensure bbox is a list format
                            bbox = list(bbox[:4])
                        elif hasattr(bbox, 'tolist'):
                            # Handle numpy arrays
                            bbox = bbox.tolist()[:4]
                        else:
                            # Log the unexpected format and skip
                            logger.warning(f"Unexpected bbox format from {model_name}: {type(bbox)} - {bbox}")
                            continue
                        # Log class correlation with motion tracks
                        logger.info(f"🔲 {timestamp:.1f}s | {bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f} | {model_name} | CROP_DETECT")
                        
                        detections.append({
                            'bbox': bbox,
                            'confidence': conf,
                            'class_id': class_id,
                            'source': model_name,
                            'timestamp': timestamp,
                            'frame_idx': frame_idx
                        })
            
            return detections
            
        except Exception as e:
            import traceback
            logger.error(f"Error running {model_name} on crop: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def run_full_frame_validation_on_scored_crops(self, video_path: Path, scored_crops: List[Dict]) -> List[Dict]:
        """STEP 4: Full-frame validation on N frames per high-scoring crop track."""
        cap = self._open_video_stream(video_path)
        if not cap:
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        validated_results = []
        
        # Sort scored crops by max score to prioritize best candidates
        scored_crops.sort(key=lambda x: x['max_score'], reverse=True)
        logger.info(f"[STEP4] Validating {len(scored_crops)} tracks with full-frame ensemble")
        
        # Limit validation to top tracks (validate all scored tracks)
        max_tracks_to_validate = len(scored_crops)
        
        # Collect all crop regions from all tracks for spatial correlation
        all_crop_regions = []
        for track in scored_crops:
            for detection in track['detections']:
                if 'bbox' in detection:
                    all_crop_regions.append(detection['bbox'])
        
        logger.info(f"[STEP4] Collected {len(all_crop_regions)} crop regions for spatial correlation: {all_crop_regions[:3]}{'...' if len(all_crop_regions) > 3 else ''}")
        
        for i, crop_track in enumerate(scored_crops[:max_tracks_to_validate]):
            # Validate track silently
            
            # Select best detection frames for full-frame validation
            track_detections = crop_track['detections']
            best_detections = sorted(track_detections, key=lambda x: x['confidence'], reverse=True)
            
            # Select frames to validate (spread across time + highest confidence)
            validation_frames = self._select_validation_frames(track_detections, max_frames=config.max_validation_frames)
            
            full_frame_scores = []
            frames_processed = 0
            
            # Initialize RT-DETR contribution tracking for this track
            rtdetr_contributions = {}
            
            for frame_idx in validation_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frames_processed += 1
                timestamp = frame_idx / fps
                
                # Run full ensemble (including RT-DETR) on full frame with crop region info for spatial correlation
                full_detections = self.ml_ensemble.run_ensemble_detection(
                    frame, timestamp, frame_idx, 
                    full_frame=frame,
                    crop_regions=all_crop_regions,
                    accepted_rtdetr_overlap=config.accepted_rtdetr_overlap
                )
                
                if full_detections:
                    max_full_conf = max(det['confidence'] for det in full_detections)
                    full_frame_scores.append(max_full_conf)
                    
                    # Track RT-DETR contributions from Step 4 full-frame validation
                    for det in full_detections:
                        source = det.get('source', 'unknown')
                        # Only track RT-DETR models (they have 'rtdetr' in their source name)
                        if 'rtdetr' in source.lower():
                            if source not in rtdetr_contributions:
                                rtdetr_contributions[source] = {
                                    'count': 0,
                                    'max_conf': 0.0,
                                    'total_conf': 0.0
                                }
                            rtdetr_contributions[source]['count'] += 1
                            rtdetr_contributions[source]['max_conf'] = max(rtdetr_contributions[source]['max_conf'], det['confidence'])
                            rtdetr_contributions[source]['total_conf'] += det['confidence']
                else:
                    # Apply heavy penalty for frames where ensemble found nothing
                    # Each zero frame gets a large negative impact
                    full_frame_scores.append(-0.3)  # Negative penalty for zero detection frames
            
            # Calculate combined score with penalties for zero-detection frames
            crop_score = crop_track['max_score']
            
            if frames_processed > 0:
                # Include penalties in the average - zeros become negative values
                avg_full_score = sum(full_frame_scores) / len(full_frame_scores)
                max_full_score = max(full_frame_scores) if full_frame_scores else 0.0
                
                # Apply additional penalty based on ratio of zero-detection frames
                zero_frames = sum(1 for score in full_frame_scores if score < 0)
                zero_ratio = zero_frames / len(full_frame_scores)
                
                # Progressive penalty: each zero frame reduces score exponentially
                zero_penalty = (1.0 - zero_ratio) ** 2  # Square the success ratio
                
                # Weighted combination with zero penalty
                base_combined = (config.crop_weight * crop_score + 
                               config.fullframe_weight * max(0, avg_full_score))  # Clamp negative avg to 0
                combined_score = base_combined * zero_penalty
                
                validation_passed = combined_score >= config.confidence_threshold
                
                # Detailed failure logging
                if not validation_passed:
                    if crop_score > 0.35:
                        logger.warning(f"📊 STRONG CROP FAILED VALIDATION: Track {crop_track['track_id']} - crop_score={crop_score:.3f} > 0.35 BUT combined={combined_score:.3f} < {config.confidence_threshold} (full-frame validation failed)")
                        logger.warning(f"   📈 Crop evidence: {len(track_detections)} detections, max_conf={max(det['confidence'] for det in track_detections):.3f}")
                        logger.warning(f"   📉 Full-frame failure: avg_score={avg_full_score:.3f}, zero_frames={zero_frames}/{len(full_frame_scores)}")
                    else:
                        logger.info(f"📊 WEAK CROP FAILED VALIDATION: Track {crop_track['track_id']} - crop_score={crop_score:.3f} <= 0.35 AND combined={combined_score:.3f} < {config.confidence_threshold}")
                
                validation_reason = f"combined={combined_score:.3f} (base={base_combined:.3f}*zero_penalty={zero_penalty:.3f}, zeros={zero_frames}/{len(full_frame_scores)})"
                
            else:
                # No frames processed
                avg_full_score = 0.0
                max_full_score = 0.0
                combined_score = 0.0
                validation_passed = False
                validation_reason = "no_frames_processed"
            
            # Calculate average confidence for RT-DETR contributions
            for source in rtdetr_contributions:
                if rtdetr_contributions[source]['count'] > 0:
                    rtdetr_contributions[source]['avg_conf'] = rtdetr_contributions[source]['total_conf'] / rtdetr_contributions[source]['count']
            
            validation_result = {
                'track_id': crop_track['track_id'],
                'crop_score': crop_score,
                'full_frame_avg_score': avg_full_score,
                'full_frame_max_score': max_full_score,
                'combined_score': combined_score,
                'crop_detections': len(track_detections),
                'validation_frames': len(validation_frames),
                'duration_seconds': crop_track['duration_seconds'],
                'best_detection': max(track_detections, key=lambda x: x['confidence']),
                'validation_passed': validation_passed,
                'rtdetr_contributions': rtdetr_contributions  # Add RT-DETR tracking from Step 4
            }
            
            validated_results.append(validation_result)
            logger.info(f"Track {crop_track['track_id']}: {validation_reason} ({'PASS' if validation_passed else 'FAIL'})")
        
        cap.release()
        
        # Filter to only validated results
        passed_results = [r for r in validated_results if r['validation_passed']]
        # Step 4 complete
        
        return passed_results
    
    def run_full_frame_analysis_on_motion_tracks(self, video_path: Path, motion_tracks: List[Dict]) -> List[Dict]:
        """
        NEW: Direct Step 2 → Step 4 connection.
        Run full-frame analysis on motion tracks with spatial overlap validation.
        
        Args:
            video_path: Path to video file
            motion_tracks: List of motion tracks from Step 1-2
            
        Returns:
            List of validated sequences with spatial overlap confirmation
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        validated_results = []
        failed_tracks_data = []  # Capture data from failed tracks for summary
        
        logger.info(f"[STEP3] Running full-frame analysis on {len(motion_tracks)} motion tracks")
        
        for track in motion_tracks:
            track_id = track['track_id']
            
            # Sample frames from motion track (temporal spread)
            sample_frames = self._sample_motion_track_frames(track, max_frames=config.max_validation_frames)
            
            # Re-run motion detection with stricter analysis parameters for spatial validation
            analysis_bg_subtractor = self.create_analysis_motion_detector()
            all_motion_regions = []
            
            for frame_idx in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Apply analysis motion detection with stricter parameters
                motion_regions = self._detect_motion_regions_with_subtractor(frame, analysis_bg_subtractor, use_config='analysis')
                all_motion_regions.extend(motion_regions)
            
            # Run full-frame analysis on sampled frames
            full_frame_detections = []
            model_contributions = {}
            
            for frame_idx in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                timestamp = frame_idx / fps
                
                # Log evaluation header with shared context
                logger.info(f"EVAL | {video_path.stem} | {timestamp:.2f}s | {frame_idx} | track_{track_id}")
                
                # Run each model individually and log evaluation as it occurs
                all_detections = []
                for model_name in config.ensemble_models:
                    model_detections = self.ml_ensemble.run_single_model_detection(
                        model_name, frame, timestamp, frame_idx,
                        full_frame=frame,
                        accepted_rtdetr_overlap=config.accepted_rtdetr_overlap
                    )
                    
                    if model_detections:
                        for det in model_detections:
                            det_bbox = det['bbox']
                            
                            # Calculate spatial overlap with motion regions
                            max_overlap = 0.0
                            best_motion_bbox = None
                            for motion_region in all_motion_regions:
                                overlap = self._calculate_bbox_overlap(motion_region, det_bbox)
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    best_motion_bbox = motion_region
                            
                            # Determine spatial validation
                            spatial_valid = max_overlap >= config.accepted_rtdetr_overlap
                            overall_score = det['confidence'] * max_overlap if spatial_valid else 0.0
                            
                            # Icon based on spatial validation
                            icon = "✅" if spatial_valid else "❌"
                            bbox_str = f"{det_bbox[0]:.0f},{det_bbox[1]:.0f},{det_bbox[2]:.0f},{det_bbox[3]:.0f}"
                            motion_bbox_str = f"{best_motion_bbox[0]:.0f},{best_motion_bbox[1]:.0f},{best_motion_bbox[2]:.0f},{best_motion_bbox[3]:.0f}" if best_motion_bbox else "none"
                            status = "spatial_valid" if spatial_valid else "spatial_invalid"
                            
                            logger.info(f"{icon} | {model_name} | {bbox_str} | {det['confidence']:.3f} | {max_overlap:.3f} | {motion_bbox_str} | {overall_score:.3f} | {status}")
                            
                            # Track model contributions for ensemble weighting
                            if model_name not in model_contributions:
                                model_contributions[model_name] = {
                                    'count': 0,
                                    'max_conf': 0.0,
                                    'total_conf': 0.0,
                                    'spatial_valid_count': 0,
                                    'total_score': 0.0
                                }
                            model_contributions[model_name]['count'] += 1
                            model_contributions[model_name]['max_conf'] = max(model_contributions[model_name]['max_conf'], det['confidence'])
                            model_contributions[model_name]['total_conf'] += det['confidence']
                            if spatial_valid:
                                model_contributions[model_name]['spatial_valid_count'] += 1
                                model_contributions[model_name]['total_score'] += overall_score
                                all_detections.append(det)
                    else:
                        # Model found no detections - add to model contributions with 0 score
                        logger.info(f"❌ | {model_name} | none | 0.000 | 0.000 | none | 0.000 | no_detection")
                        if model_name not in model_contributions:
                            model_contributions[model_name] = {
                                'count': 0,
                                'max_conf': 0.0,
                                'total_conf': 0.0,
                                'spatial_valid_count': 0,
                                'total_score': 0.0
                            }
                
                # Use all_detections for ensemble calculation instead of detections
                detections = all_detections
                
                # Ensemble weighting row: combine ALL model scores (including 0.000 scores)
                valid_detections = []
                total_ensemble_score = 0.0
                valid_model_count = 0
                
                # Calculate confidence-weighted ensemble score for THIS FRAME ONLY
                model_scores = {}
                model_weights = {}
                base_weight = 0.1  # Minimum weight for zero/low confidence models
                frame_valid_model_count = 0
                
                # Calculate individual model scores for current frame evaluations
                for model_name in config.ensemble_models:
                    # Get this frame's detection results for this model (handle source name variations)
                    if model_name.startswith('rtdetr-'):
                        # RT-DETR models have source like 'rtdetr_rtdetr-l' 
                        frame_detections = [det for det in detections if det.get('source') == f'rtdetr_{model_name}']
                    else:
                        frame_detections = [det for det in detections if det.get('source') == model_name]
                    
                    if frame_detections:
                        # Use the best detection from this model on this frame
                        best_det = max(frame_detections, key=lambda d: d['confidence'])
                        frame_conf = best_det['confidence']
                        
                        # Check if this detection passed spatial validation
                        spatial_valid = False
                        for motion_region in all_motion_regions:
                            overlap = self._calculate_bbox_overlap(motion_region, best_det['bbox'])
                            if overlap >= config.accepted_rtdetr_overlap:
                                spatial_valid = True
                                break
                        
                        if spatial_valid:
                            frame_valid_model_count += 1
                        else:
                            frame_conf = 0.0  # Zero out confidence if spatial validation fails
                    else:
                        frame_conf = 0.0
                    
                    model_scores[model_name] = frame_conf
                
                # Simple ensemble: sum of all spatially valid confidences, normalized by maximum possible
                # This favors multiple models contributing vs single model
                total_contributing_conf = sum(model_scores.values())
                max_possible_conf = len(config.ensemble_models) * 1.0  # Max if all models contribute 1.0
                total_ensemble_score = total_contributing_conf / max_possible_conf
                
                valid_model_count = frame_valid_model_count
                
                # Store valid detections with spatial overlap
                for det in detections:
                    det_bbox = det['bbox']
                    max_overlap = 0.0
                    for motion_region in all_motion_regions:
                        overlap = self._calculate_bbox_overlap(motion_region, det_bbox)
                        max_overlap = max(max_overlap, overlap)
                    
                    if max_overlap >= config.accepted_rtdetr_overlap:
                        det['motion_overlap'] = max_overlap
                        det['frame_idx'] = frame_idx  # Store frame index for temporal continuity validation
                        det['timestamp'] = timestamp  # Store timestamp for temporal continuity validation
                        valid_detections.append(det)
                
                # Log ensemble weighting header and row
                logger.info(f"ENSEMBLE | {video_path.stem} | {timestamp:.2f}s | {frame_idx} | track_{track_id}")
                ensemble_icon = "✅" if total_ensemble_score > 0 else "❌"
                logger.info(f"{ensemble_icon} | combined | valid_models={valid_model_count} | ensemble_score={total_ensemble_score:.3f} | valid_detections={len(valid_detections)}")
                
                full_frame_detections.extend(valid_detections)
            
            # Calculate average confidence for model contributions
            for source in model_contributions:
                if model_contributions[source]['count'] > 0:
                    model_contributions[source]['avg_conf'] = model_contributions[source]['total_conf'] / model_contributions[source]['count']
            
            # Determine if track passes validation
            if full_frame_detections:
                # Calculate ensemble-based combined score (sum of all spatially valid detections)
                summed_confidence = sum(d['confidence'] for d in full_frame_detections)
                avg_confidence = summed_confidence / len(full_frame_detections)
                max_confidence = max(d['confidence'] for d in full_frame_detections)
                
                # New scoring: use summed confidence normalized by track duration
                track_duration = track['duration_seconds']
                duration_normalized_score = summed_confidence / max(1.0, track_duration)
                
                # Use summed confidence as the combined score (not just max)
                combined_score = summed_confidence
                
                # Track validation requires confidence, minimum frames, AND temporal continuity
                confidence_passed = combined_score >= config.confidence_threshold
                frames_passed = len(full_frame_detections) >= config.min_track_frames
                
                # Check temporal continuity: gaps between validated detections must not exceed detection_validation_gap_seconds
                temporal_continuity_passed = True
                if len(full_frame_detections) > 1:
                    # Get timestamps of validated detections
                    validated_timestamps = sorted([det.get('timestamp', 0.0) for det in full_frame_detections])
                    
                    # Check time gaps between consecutive validated detections
                    for i in range(1, len(validated_timestamps)):
                        time_gap = validated_timestamps[i] - validated_timestamps[i-1]
                        logger.info(f"TEMPORAL_DEBUG | gap={time_gap:.3f}s | threshold={config.detection_validation_gap_seconds:.3f}s | timestamps={validated_timestamps}")
                        if time_gap > config.detection_validation_gap_seconds:  # Time gap > max allowed means temporal discontinuity
                            temporal_continuity_passed = False
                            logger.info(f"TEMPORAL_DEBUG | FAILED: gap {time_gap:.3f}s > threshold {config.detection_validation_gap_seconds:.3f}s")
                            break
                
                validation_passed = confidence_passed and frames_passed and temporal_continuity_passed
                
                # Find best detection for metadata
                best_detection = max(full_frame_detections, key=lambda d: d['confidence'])
                
                # Track evaluation header and row: log the combined results for the track
                total_models_with_detections = sum(1 for contrib in model_contributions.values() if contrib.get('spatial_valid_count', 0) > 0)
                total_spatial_valid_detections = sum(contrib.get('spatial_valid_count', 0) for contrib in model_contributions.values())
                
                logger.info(f"TRACK | {video_path.stem} | track_{track_id}")
                track_icon = "✅" if validation_passed else "❌"
                logger.info(f"{track_icon} | duration={track_duration:.2f}s | frames={len(sample_frames)} | detections={len(full_frame_detections)} | models_active={total_models_with_detections} | summed_conf={summed_confidence:.3f} | avg_conf={avg_confidence:.3f} | max_conf={max_confidence:.3f} | duration_norm={duration_normalized_score:.3f} | conf_pass={confidence_passed} | frames_pass={frames_passed} | temporal_pass={temporal_continuity_passed} | validated={validation_passed}")
                
                if validation_passed:
                    validated_result = {
                        'track_id': track_id,
                        'best_detection': {
                            'frame_idx': best_detection.get('frame_idx', sample_frames[0]),
                            'timestamp': best_detection.get('timestamp', sample_frames[0] / fps),
                            'bbox': best_detection['bbox'],
                            'confidence': best_detection['confidence'],
                            'source': f"fullframe_{best_detection['source']}",
                            'motion_overlap': best_detection.get('motion_overlap', 0.0)
                        },
                        'combined_score': combined_score,
                        'summed_confidence': summed_confidence,
                        'duration_normalized_score': duration_normalized_score,
                        'full_frame_avg_score': avg_confidence,
                        'full_frame_max_score': max_confidence,
                        'detection_count': len(full_frame_detections),
                        'validation_frames': len(sample_frames),
                        'duration_seconds': track['duration_seconds'],
                        'validation_passed': validation_passed,
                        'model_contributions': model_contributions
                    }
                    validated_results.append(validated_result)
                else:
                    # Capture failed track data for summary
                    failed_tracks_data.append({
                        'track_id': track_id,
                        'confidence': max_confidence,
                        'combined_score': combined_score,
                        'summed_confidence': summed_confidence,
                        'detections': len(full_frame_detections)
                    })
            else:
                # Track evaluation header and row for failed track
                track_duration = track['duration_seconds']
                logger.info(f"TRACK | {video_path.stem} | track_{track_id}")
                logger.info(f"❌ | duration={track_duration:.2f}s | frames={len(sample_frames)} | detections=0 | models_active=0 | summed_conf=0.000 | avg_conf=0.000 | max_conf=0.000 | duration_norm=0.000 | validated=false")
                
                # Capture failed track data for summary (no overlap)
                failed_tracks_data.append({
                    'track_id': track_id,
                    'confidence': 0.0,
                    'combined_score': 0.0,
                    'summed_confidence': 0.0,
                    'detections': 0
                })
        
        cap.release()
        
        # Filter to only validated results
        passed_results = [r for r in validated_results if r['validation_passed']]
        
        # Store failed track data for summary reporting
        if hasattr(self, '_failed_tracks_data'):
            self._failed_tracks_data.extend(failed_tracks_data)
        else:
            self._failed_tracks_data = failed_tracks_data
        
        return passed_results
    
    def _sample_motion_track_frames(self, track: Dict, max_frames: int = 5) -> List[int]:
        """Sample representative frames from a motion track."""
        frames = track['frames']
        if len(frames) <= max_frames:
            return frames
        
        # Sample evenly across the track duration
        step = len(frames) / max_frames
        return [frames[int(i * step)] for i in range(max_frames)]
    
    def _calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap percentage between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)
        
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0  # No overlap
        
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        if bbox2_area <= 0:
            return 0.0
        
        return intersection_area / bbox2_area
    
    def _select_validation_frames(self, detections: List[Dict], max_frames: int = 3) -> List[int]:
        """Select representative frames for validation (temporal spread + high confidence)."""
        if not detections:
            return []
        
        # Group by frame
        frame_groups = {}
        for det in detections:
            frame_idx = det['frame_idx']
            if frame_idx not in frame_groups:
                frame_groups[frame_idx] = []
            frame_groups[frame_idx].append(det)
        
        # Get best detection per frame
        frame_scores = {}
        for frame_idx, dets in frame_groups.items():
            frame_scores[frame_idx] = max(det['confidence'] for det in dets)
        
        # Select frames: prioritize high confidence but ensure temporal spread
        sorted_frames = sorted(frame_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_frames) <= max_frames:
            return [f[0] for f in sorted_frames]
        
        # Take top frame + spread remaining across time
        selected = [sorted_frames[0][0]]  # Highest confidence frame
        remaining_frames = [f[0] for f in sorted_frames[1:]]
        remaining_frames.sort()  # Sort by time
        
        # Add temporally spread frames
        if len(remaining_frames) > 0 and max_frames > 1:
            step = len(remaining_frames) / (max_frames - 1)
            for i in range(1, max_frames):
                idx = min(int((i-1) * step), len(remaining_frames) - 1)
                selected.append(remaining_frames[idx])
        
        return sorted(selected)
    
    def build_temporal_tracks(self, all_detections_by_frame: Dict, fps: float) -> List[Dict]:
        """Build temporal tracks using DeepSORT for consistency filtering."""
        if not all_detections_by_frame:
            return []
        
        # Try to import DeepSORT
        try:
            from deep_sort_realtime import DeepSort
            deepsort_available = True
            logger.info("🎯 DeepSORT available - using for temporal consistency tracking")
        except ImportError:
            deepsort_available = False
            logger.warning("⚠️ DeepSORT not available - falling back to simple bbox linking")
        
        if deepsort_available:
            return self._build_tracks_with_deepsort(all_detections_by_frame, fps)
        else:
            return self._build_tracks_simple_linking(all_detections_by_frame, fps)
    
    def _build_tracks_with_deepsort(self, all_detections_by_frame: Dict, fps: float) -> List[Dict]:
        """Build tracks using DeepSORT for robust temporal consistency."""
        from deep_sort_realtime import DeepSort
        
        # Initialize DeepSORT tracker
        tracker = DeepSort(
            max_age=int(config.motion_tracking_gap_seconds * fps),  # Convert seconds to frames
            n_init=config.min_track_frames,
            max_iou_distance=0.3,
            max_cosine_distance=0.2,
            nn_budget=100
        )
        
        logger.info(f"🎯 Initializing DeepSORT: max_age={int(config.motion_tracking_gap_seconds * fps)}frames, n_init={config.min_track_frames}")
        
        # Convert detections to DeepSORT format and track
        tracks_by_id = {}
        
        for frame_idx in sorted(all_detections_by_frame.keys()):
            frame_detections = all_detections_by_frame[frame_idx]
            
            # Convert to DeepSORT format: [x1, y1, x2, y2, confidence]
            deepsort_detections = []
            for det in frame_detections:
                bbox = det['bbox']
                confidence = det['confidence']
                deepsort_detections.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence])
            
            # Update tracker
            if deepsort_detections:
                tracks = tracker.update_tracks(deepsort_detections, frame=frame_idx)
                
                # Store track information
                for track in tracks:
                    if track.is_confirmed():
                        track_id = track.track_id
                        if track_id not in tracks_by_id:
                            tracks_by_id[track_id] = {
                                'track_id': track_id,
                                'frames': [],
                                'bboxes': [],
                                'confidences': [],
                                'timestamps': []
                            }
                        
                        # Get bbox from track
                        ltrb = track.to_ltrb()
                        tracks_by_id[track_id]['frames'].append(frame_idx)
                        tracks_by_id[track_id]['bboxes'].append([ltrb[0], ltrb[1], ltrb[2], ltrb[3]])
                        tracks_by_id[track_id]['confidences'].append(track.get_det_conf())
                        tracks_by_id[track_id]['timestamps'].append(frame_idx / fps)
        
        # Convert to consistent bbox sequences format
        consistent_sequences = []
        for track_id, track_data in tracks_by_id.items():
            if len(track_data['frames']) >= config.min_track_frames:
                duration = (max(track_data['frames']) - min(track_data['frames'])) / fps
                if duration >= config.min_track_duration:
                    bbox_track = {
                        'track_id': track_id,
                        'frames': track_data['frames'],
                        'bboxes': track_data['bboxes'],
                        'confidences': track_data['confidences'],
                        'start_frame': min(track_data['frames']),
                        'end_frame': max(track_data['frames'])
                    }
                    
                    consistent_sequences.append({
                        'bbox_track': bbox_track,
                        'duration_seconds': duration,
                        'detection_count': len(track_data['frames']),
                        'avg_confidence': sum(track_data['confidences']) / len(track_data['confidences']),
                        'max_confidence': max(track_data['confidences']),
                        'tracking_method': 'deepsort'
                    })
        
        logger.info(f"🎯 DeepSORT generated {len(consistent_sequences)} valid tracks from {len(tracks_by_id)} total tracks")
        return consistent_sequences
    
    def _build_tracks_simple_linking(self, all_detections_by_frame: Dict, fps: float) -> List[Dict]:
        """Fallback: Simple bbox linking without DeepSORT."""
        logger.info("🔗 Using simple bbox linking (DeepSORT fallback)")
        
        # Simple implementation - group nearby detections across frames
        tracks = []
        used_detections = set()
        track_id_counter = 0
        
        for start_frame in sorted(all_detections_by_frame.keys()):
            for start_det in all_detections_by_frame[start_frame]:
                det_id = (start_frame, id(start_det))
                if det_id in used_detections:
                    continue
                
                # Start new track
                track = {
                    'track_id': track_id_counter,
                    'frames': [start_frame],
                    'bboxes': [start_det['bbox']],
                    'confidences': [start_det['confidence']],
                    'start_frame': start_frame,
                    'end_frame': start_frame
                }
                used_detections.add(det_id)
                
                # Look for linked detections in subsequent frames
                current_bbox = start_det['bbox']
                for frame_idx in range(start_frame + 1, start_frame + int(config.motion_tracking_gap_seconds * fps * 3)):
                    if frame_idx not in all_detections_by_frame:
                        continue
                    
                    best_match = None
                    best_distance = float('inf')
                    
                    for det in all_detections_by_frame[frame_idx]:
                        det_id = (frame_idx, id(det))
                        if det_id in used_detections:
                            continue
                        
                        # Calculate center distance
                        curr_center = [(current_bbox[0] + current_bbox[2])/2, (current_bbox[1] + current_bbox[3])/2]
                        det_center = [(det['bbox'][0] + det['bbox'][2])/2, (det['bbox'][1] + det['bbox'][3])/2]
                        distance = ((curr_center[0] - det_center[0])**2 + (curr_center[1] - det_center[1])**2)**0.5
                        
                        if distance < config.tracking_distance_threshold and distance < best_distance:
                            best_match = det
                            best_distance = distance
                    
                    if best_match:
                        track['frames'].append(frame_idx)
                        track['bboxes'].append(best_match['bbox'])
                        track['confidences'].append(best_match['confidence'])
                        track['end_frame'] = frame_idx
                        used_detections.add((frame_idx, id(best_match)))
                        current_bbox = best_match['bbox']
                
                # Check if track meets minimum requirements
                duration = (track['end_frame'] - track['start_frame']) / fps
                if len(track['frames']) >= config.min_track_frames and duration >= config.min_track_duration:
                    tracks.append({
                        'bbox_track': track,
                        'duration_seconds': duration,
                        'detection_count': len(track['frames']),
                        'avg_confidence': sum(track['confidences']) / len(track['confidences']),
                        'max_confidence': max(track['confidences']),
                        'tracking_method': 'simple_linking'
                    })
                
                track_id_counter += 1
        
        logger.info(f"🔗 Simple linking generated {len(tracks)} valid tracks")
        return tracks
    
    def process_video_with_features(self, video_path: Path) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Process video with correct pipeline: motion tracking -> bbox extraction -> ML on crops -> full-frame validation."""
        import time
        start_time = time.time()
        
        # Clear failed tracks data for this video
        self._failed_tracks_data = []
        
        cap = self._open_video_stream(video_path)
        if not cap:
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info("=" * 80)
        logger.info(f"🎬 PROCESSING: {video_path.name} ({total_frames} frames, {total_frames/fps:.1f}s)")
        logger.info("=" * 80)
        
        # Create debug directory
        video_debug_dir = self.debug_dir / video_path.stem
        video_debug_dir.mkdir(exist_ok=True)
        
        # STEP 1: Find motion tracks (no ML)
        logger.info(f"🔍 [STEP 1/4] {video_path.name}: Motion detection + temporal tracking")
        motion_tracks = self.find_consistent_motion_sequences_and_tracks(video_path, fps, total_frames)
        logger.info(f"✅ [STEP 1/4] Found {len(motion_tracks)} motion tracks")
        
        if not motion_tracks:
            logger.warning(f"❌ [STEP 1/4] No motion detected - skipping video")
            processing_time = time.time() - start_time
            self._processing_times = getattr(self, '_processing_times', {})
            self._processing_times[video_path.name] = processing_time
            cap.release()
            return None, None
        
        # STEP 2: Filter motion tracks for camera handling (fast exit)
        logger.info(f"🔧 [STEP 2/4] {video_path.name}: Motion analysis filter")
        filtered_tracks = self.filter_motion_tracks_for_camera_handling(video_path, motion_tracks)
        
        if not filtered_tracks:
            logger.warning(f"❌ [STEP 2/4] Motion filter rejected video")
            processing_time = time.time() - start_time
            self._processing_times = getattr(self, '_processing_times', {})
            self._processing_times[video_path.name] = processing_time
            cap.release()
            return None, None
        else:
            logger.info(f"✅ [STEP 2/4] {len(filtered_tracks)} tracks passed filter")
        
        # STEP 3 REMOVED: Direct connection Step 2 → Step 4
        # Skip crop analysis entirely, go directly to full-frame validation
        
        # STEP 3: Full-frame analysis on motion tracks with spatial overlap validation
        logger.info(f"🎯 [STEP 3/4] {video_path.name}: Full-frame analysis on motion tracks")
        final_validated_sequences = self.run_full_frame_analysis_on_motion_tracks(video_path, filtered_tracks)
        
        # Note: RT-DETR contributions are now tracked directly in the full-frame analysis function
        # No need for separate integration since we're not using crop-based scoring
        
        if not final_validated_sequences:
            logger.warning(f"❌ [STEP 3/4] No tracks passed full-frame validation")
            logger.info(f"📊 MOTION TRACKS FAILED FULL-FRAME: {len(filtered_tracks)} motion tracks found no animals in full-frame analysis")
            logger.info(f"   📉 This suggests motion was from non-animal sources (wind, shadows, etc.)")
            
            processing_time = time.time() - start_time
            # Store rejection reason and timing for summary
            self._rejection_reasons = getattr(self, '_rejection_reasons', {})
            self._rejection_reasons[video_path.name] = "failed_full_frame_analysis"
            self._processing_times = getattr(self, '_processing_times', {})
            self._processing_times[video_path.name] = processing_time
            # Store motion track data for summary (full-frame analysis failed)
            self._failed_video_data = getattr(self, '_failed_video_data', {})
            
            # Get the best confidence from failed tracks if available
            failed_tracks = getattr(self, '_failed_tracks_data', [])
            best_conf = 'N/A'
            best_combined = 'N/A'
            if failed_tracks:
                # Find the best confidence from failed tracks
                best_track = max(failed_tracks, key=lambda x: x['confidence'] if x['confidence'] > 0 else 0)
                if best_track['confidence'] > 0:
                    best_conf = f"{best_track['confidence']:.3f}"
                    best_combined = f"{best_track['combined_score']:.3f}"
            
            self._failed_video_data[video_path.name] = {
                'motion_tracks': len(motion_tracks),
                'conf': best_conf,  # Best confidence from failed tracks
                'combined': best_combined,  # Best combined score from failed tracks  
                'tracks': len(filtered_tracks),
                'validated': 0
            }
            cap.release()
            return None, None
        else:
            logger.info(f"✅ [STEP 4/4] {len(final_validated_sequences)} tracks validated")
        
        # Final result summary
        best_sequence = max(final_validated_sequences, key=lambda s: s['combined_score'])
        processing_time = time.time() - start_time
        
        # Store processing time for summary
        self._processing_times = getattr(self, '_processing_times', {})
        self._processing_times[video_path.name] = processing_time
        
        # Model contributions are now tracked directly in the full-frame analysis function
        # Extract from validated sequences for summary
        self._model_contributions = getattr(self, '_model_contributions', {})
        video_model_stats = {}
        
        for sequence in final_validated_sequences:
            for model_name, stats in sequence.get('model_contributions', {}).items():
                if model_name not in video_model_stats:
                    video_model_stats[model_name] = {
                        'total_detections': 0,
                        'max_confidence': 0.0,
                        'contributing_tracks': 0
                    }
                video_model_stats[model_name]['total_detections'] += stats['count']
                video_model_stats[model_name]['max_confidence'] = max(video_model_stats[model_name]['max_confidence'], stats['max_conf'])
                video_model_stats[model_name]['contributing_tracks'] += 1
        
        self._model_contributions[video_path.name] = video_model_stats
        
        logger.info(f"🏆 COMPLETED: {video_path.name} - Animal detected (score: {best_sequence['combined_score']:.3f}) - {processing_time:.1f}s")
        logger.info("=" * 80)
        
        # Get video dimensions before releasing cap
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        
        # Create final analysis from best validated sequence
        best_sequence = max(final_validated_sequences, key=lambda s: s['combined_score'])
        best_detection = best_sequence['best_detection']
        
        # Save crops for validated sequences
        saved_crop_paths = self.save_validated_crops(video_path, final_validated_sequences)
        
        analysis = {
            'video_path': str(video_path),
            'animals_detected': ['animal'],
            'detection_count': sum(seq['detection_count'] for seq in final_validated_sequences),
            'frames_processed': total_frames,
            'motion_tracks': len(motion_tracks),
            'validated_sequences': len(final_validated_sequences),
            'temporal_consistency_duration': best_sequence['duration_seconds'],
            'best_detection_frame': best_detection['frame_idx'],
            'best_detection_timestamp': best_detection['timestamp'],
            'saved_crops': saved_crop_paths,
            'detection': {
                'confidence': best_detection['confidence'],
                'bbox': best_detection['bbox'],
                'area_ratio': self.calculate_area_ratio(best_detection['bbox'], frame_width, frame_height),
                'source': best_detection['source'],
                'full_frame_score': best_sequence['full_frame_avg_score'],
                'combined_score': best_sequence['combined_score']
            },
            'processing_mode': 'next_generation_3step_pipeline'
        }
        
        # Extract features from best detection
        features = self.extract_features_from_best_sequence(video_path, best_sequence)
        
        logger.info(f"=== COMPLETED: {video_path.name} - Animal detected (conf={best_detection['confidence']:.3f}, combined={best_sequence['combined_score']:.3f}) ===")
        return analysis, features
    
    def calculate_area_ratio(self, bbox: Tuple[int, int, int, int], frame_w: float, frame_h: float) -> float:
        """Calculate detection area ratio relative to frame."""
        x1, y1, x2, y2 = bbox
        detection_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_w * frame_h
        return detection_area / frame_area if frame_area > 0 else 0.0
    
    def extract_features_from_best_sequence(self, video_path: Path, best_sequence: Dict) -> Optional[np.ndarray]:
        """Extract ResNet features from best detection in sequence."""
        cap = self._open_video_stream(video_path)
        if not cap:
            return None
        
        best_detection = best_sequence['best_detection']
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_detection['frame_idx'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Crop detection region  
        best_detection = best_sequence['best_detection']
        x1, y1, x2, y2 = best_detection['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop = frame[y1:y2, x1:x2]
        
        # Extract features using ResNet
        return self.ml_ensemble.extract_features(frame, best_detection['bbox'])
    
    def process_all_videos(self, video_filter=None):
        """Process all videos using next-generation approach."""
        import time
        batch_start_time = time.time()
        
        logger.info("###############################################")
        logger.info("NEXT-GENERATION BATCH PROCESSING SESSION START")
        logger.info("###############################################")
        
        # Get videos to process
        videos_to_process = self.get_filtered_videos(video_filter)
        
        if not videos_to_process:
            if video_filter:
                logger.info(f"BATCH RESULT: No videos found matching filter: {video_filter}")
                logger.info(f"⚠️ No videos found matching filter: {video_filter}")
            else:
                logger.info("BATCH RESULT: No unprocessed videos found")
                logger.info("✅ No unprocessed videos found")
            return
        
        logger.info(f"Videos to process: {[v.name for v in videos_to_process]}")
        logger.info(f"🎬 Found {len(videos_to_process)} videos to process")
        
        # Clear previous session data
        self.all_features = []
        self.video_metadata = []
        all_analyses = []
        
        # Process each video
        for i, video_path in enumerate(videos_to_process):
            logger.info(f"Processing video {i+1}/{len(videos_to_process)}: {video_path.name}")
            try:
                analysis, features = self.process_video_with_features(video_path)
                if analysis:
                    logger.info(f"VIDEO SUCCESS: {video_path.name} - Animal detected with temporal consistency")
                    all_analyses.append(analysis)
                    self.save_analysis(analysis, video_path)
                    self.mark_video_processed(video_path, analysis, success=True)
                    
                    if features is not None:
                        self.all_features.append(features)
                        self.video_metadata.append(analysis)
                        logger.info(f"Features extracted: {len(features)} dimensions")
                else:
                    logger.info(f"VIDEO SKIPPED: {video_path.name} - No consistent animal movement detected")
                    logger.info(f"⚪ {video_path.name}: No consistent animal movement")
                    self.mark_video_processed(video_path, None, success=False)
                    
            except Exception as e:
                logger.error(f"VIDEO ERROR: {video_path.name} - {str(e)}")
                logger.error(f"❌ {video_path.name}: Processing failed - {str(e)}")
                # Don't mark failed videos as processed - they should be retried
        
        # Generate detailed final summary
        total_videos = len(videos_to_process)
        animal_videos = len(all_analyses)
        no_animal_videos = total_videos - animal_videos
        
        logger.info("=" * 80)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"🎬 Total videos processed: {total_videos}")
        logger.info(f"🦌 Videos with animals detected: {animal_videos}")
        logger.info(f"⚪ Videos with no animals detected: {no_animal_videos}")
        logger.info("")
        
        # Get stored data for comprehensive summary
        composite_scores = getattr(self, '_composite_scores', {})
        rejection_reasons = getattr(self, '_rejection_reasons', {})
        processing_times = getattr(self, '_processing_times', {})
        
        # Calculate total processing time
        batch_total_time = time.time() - batch_start_time
        
        if all_analyses:
            logger.info("🦌 VIDEOS WITH ANIMALS DETECTED:")
            for analysis in all_analyses:
                video_name = Path(analysis['video_path']).name
                confidence = analysis['detection']['confidence']
                combined_score = analysis['detection']['combined_score']
                motion_tracks = analysis['motion_tracks']
                validated_sequences = analysis['validated_sequences']
                composite_score = composite_scores.get(video_name, 'N/A')
                runtime = processing_times.get(video_name, 'N/A')
                runtime_str = f"{runtime:.1f}s" if isinstance(runtime, (int, float)) else runtime
                
                # Add time range information
                best_timestamp = analysis['best_detection_timestamp']
                duration = analysis['temporal_consistency_duration']
                time_range = f"{best_timestamp:.1f}s"
                if duration > 1.0:
                    time_range = f"{best_timestamp:.1f}-{best_timestamp + duration:.1f}s"
                
                logger.info(f"  ✅ {video_name}: time_range={time_range}, conf={confidence:.3f}, combined={combined_score:.3f}, tracks={motion_tracks}, validated={validated_sequences}, runtime={runtime_str}")
                
                # Show saved crop paths
                saved_crops = analysis.get('saved_crops', [])
                if saved_crops:
                    logger.info(f"     📷 Validated crops: {len(saved_crops)} saved")
                    for crop_path in saved_crops:
                        logger.info(f"       {crop_path}")
                else:
                    logger.info(f"     📷 No crops saved")
            logger.info("")
        
        # Log videos that had no animals (need to track these during processing)
        processed_names = {Path(analysis['video_path']).name for analysis in all_analyses}
        no_animal_names = [Path(video).name for video in videos_to_process if Path(video).name not in processed_names]
        
        if no_animal_names:
            failed_video_data = getattr(self, '_failed_video_data', {})
            logger.info("⚪ VIDEOS WITH NO ANIMALS DETECTED:")
            for video_name in no_animal_names:
                composite_score = composite_scores.get(video_name, 'N/A')
                rejection_reason = rejection_reasons.get(video_name, 'unknown_reason')
                runtime = processing_times.get(video_name, 'N/A')
                runtime_str = f"{runtime:.1f}s" if isinstance(runtime, (int, float)) else runtime
                
                # Get detailed failure data if available
                failure_data = failed_video_data.get(video_name, {})
                if failure_data:
                    conf = failure_data.get('conf', 'N/A')
                    combined = failure_data.get('combined', 'N/A')
                    tracks = failure_data.get('tracks', 'N/A')
                    validated = failure_data.get('validated', 'N/A')
                    motion_tracks = failure_data.get('motion_tracks', 'N/A')
                    
                    # Check for strong crop/zero full-frame condition
                    strong_crop_indicator = ""
                    if (conf != 'N/A' and isinstance(conf, str) and 
                        conf.replace('.', '').isdigit() and float(conf) > 0.35 and 
                        combined == 'N/A' and rejection_reason == 'failed_full_frame_analysis'):
                        strong_crop_indicator = " 🎯[STRONG_CROP+ZERO_FULLFRAME]"
                    
                    logger.info(f"  ⚪ {video_name}: motion_score={composite_score}, conf={conf}, combined={combined}, tracks={tracks}, validated={validated}, reason={rejection_reason}, runtime={runtime_str}{strong_crop_indicator}")
                else:
                    logger.info(f"  ⚪ {video_name}: motion_score={composite_score}, reason={rejection_reason}, runtime={runtime_str}")
            logger.info("")
        
        # Add timing summary
        if processing_times:
            total_video_time = sum(t for t in processing_times.values() if isinstance(t, (int, float)))
            avg_time = total_video_time / len(processing_times)
            logger.info(f"⏱️  TIMING SUMMARY:")
            logger.info(f"  📊 Total batch time: {batch_total_time:.1f}s")
            logger.info(f"  📊 Total video processing time: {total_video_time:.1f}s")
            logger.info(f"  📊 Average per video: {avg_time:.1f}s")
            logger.info(f"  📊 Videos processed: {len(processing_times)}")
            logger.info("")
        
        # Add model contribution analysis (always show, even for failed videos)
        model_contributions = getattr(self, '_model_contributions', {})
        logger.info("🤖 MODEL CONTRIBUTION ANALYSIS:")
        logger.info("   (YOLO vs MegaDetector performance comparison)")
        logger.info("")
        
        if model_contributions:
            # Aggregate model statistics across ALL videos with ML data (successful AND failed)
            all_models_stats = {}
            videos_with_ml_data = list(model_contributions.keys())
            videos_with_animals = [Path(analysis['video_path']).name for analysis in all_analyses]
            
            for video_name in videos_with_ml_data:
                video_stats = model_contributions.get(video_name, {})
                for model_name, stats in video_stats.items():
                    if model_name not in all_models_stats:
                        all_models_stats[model_name] = {
                            'total_detections': 0,
                            'videos_contributed': 0,
                            'max_confidence': 0.0,
                            'total_tracks': 0
                        }
                    all_models_stats[model_name]['total_detections'] += stats['total_detections']
                    all_models_stats[model_name]['videos_contributed'] += 1
                    all_models_stats[model_name]['max_confidence'] = max(all_models_stats[model_name]['max_confidence'], stats['max_confidence'])
                    all_models_stats[model_name]['total_tracks'] += stats['contributing_tracks']
            
            # Sort models by type (YOLO, RT-DETR, MegaDetector) and show statistics
            yolo_models = {k: v for k, v in all_models_stats.items() if not k.startswith('MDV6-') and not k.startswith('rtdetr_')}
            rtdetr_models = {k: v for k, v in all_models_stats.items() if k.startswith('rtdetr_')}
            md_models = {k: v for k, v in all_models_stats.items() if k.startswith('MDV6-')}
            
            logger.info(f"  📊 ANALYSIS COVERS: {len(videos_with_ml_data)} videos with ML data ({len(videos_with_animals)} successful, {len(videos_with_ml_data) - len(videos_with_animals)} failed validation)")
            logger.info("")
            
            if yolo_models:
                logger.info("  🎯 YOLO MODELS:")
                for model_name, stats in sorted(yolo_models.items()):
                    logger.info(f"    {model_name}: {stats['total_detections']} detections, {stats['videos_contributed']}/{len(videos_with_ml_data)} videos, max_conf={stats['max_confidence']:.3f}, tracks={stats['total_tracks']}")
                logger.info("")
            
            if rtdetr_models:
                logger.info("  🔬 RT-DETR MODELS:")
                for model_name, stats in sorted(rtdetr_models.items()):
                    logger.info(f"    {model_name}: {stats['total_detections']} detections, {stats['videos_contributed']}/{len(videos_with_ml_data)} videos, max_conf={stats['max_confidence']:.3f}, tracks={stats['total_tracks']}")
                logger.info("")
            
            if md_models:
                logger.info("  🦎 MEGADETECTOR MODELS:")
                for model_name, stats in sorted(md_models.items()):
                    logger.info(f"    {model_name}: {stats['total_detections']} detections, {stats['videos_contributed']}/{len(videos_with_ml_data)} videos, max_conf={stats['max_confidence']:.3f}, tracks={stats['total_tracks']}")
                logger.info("")
            
            # Per-video breakdown for detailed analysis - ALL videos with ML data
            logger.info("  📹 PER-VIDEO MODEL BREAKDOWN:")
            for video_name in sorted(videos_with_ml_data):
                video_stats = model_contributions.get(video_name, {})
                if video_stats:
                    # Indicate if video passed or failed validation
                    status = "PASS" if video_name in videos_with_animals else "FAIL"
                    logger.info(f"    {video_name} [{status}]:")
                    # Sort by detection count for easier comparison
                    sorted_models = sorted(video_stats.items(), key=lambda x: x[1]['total_detections'], reverse=True)
                    for model_name, stats in sorted_models:
                        model_type = "MD" if model_name.startswith('MDV6-') else "YOLO"
                        logger.info(f"      {model_type:<4} {model_name}: {stats['total_detections']} detections, max_conf={stats['max_confidence']:.3f}")
                else:
                    logger.info(f"    {video_name}: No model contribution data")
            logger.info("")
        else:
            logger.info("  ❌ No model contribution data collected")
            logger.info("     (No videos reached Step 3 ML analysis)")
            logger.info("")
            
        logger.info("📁 Analysis files saved to: /home/rfirmin/Videos/wildcams/analysis/")
        
        if all_analyses:
            logger.info(f"✅ Successfully processed {len(all_analyses)} videos with temporal consistency")
            
            # Run clustering if we have features
            if len(self.all_features) > 1:
                logger.info("🔗 Running similarity clustering...")
                # TODO: Implement clustering for next-gen pipeline
                # self.cluster_animal_videos(all_analyses)
        
        logger.info("=" * 80)
        
        logger.info("###############################################")
        logger.info("NEXT-GENERATION BATCH PROCESSING SESSION END")
        logger.info("###############################################")
    
    def get_unprocessed_videos(self) -> List[Path]:
        """Get list of videos that haven't been processed yet (using tracking directory)."""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
        unprocessed = []
        
        for video_file in self.video_dir.iterdir():
            if (video_file.is_file() and 
                video_file.suffix.lower() in video_extensions):
                
                # Check for .processed file in tracking directory
                processed_file = self.tracking_dir / f"{video_file.stem}.processed"
                if not processed_file.exists():
                    unprocessed.append(video_file)
        
        return sorted(unprocessed)
    
    def mark_video_processed(self, video_path: Path, analysis: Dict = None, success: bool = True):
        """Mark video as processed with detailed information in tracking directory."""
        processed_file = self.tracking_dir / f"{video_path.stem}.processed"
        
        # Create processed data
        processed_data = {
            'video_file': video_path.name,
            'processed_timestamp': __import__('datetime').datetime.now().isoformat(),
            'processing_status': 'success' if success else 'no_animals',
            'processor_version': 'next_generation_4step_pipeline',
        }
        
        if analysis and success:
            # Add detailed information about strongest detections
            processed_data.update({
                'animals_detected': True,
                'confidence': analysis['detection']['confidence'],
                'combined_score': analysis['detection']['combined_score'],
                'detection_count': analysis['detection_count'],
                'motion_tracks': analysis['motion_tracks'],
                'validated_sequences': analysis['validated_sequences'],
                'temporal_consistency_duration': analysis['temporal_consistency_duration'],
                'best_detection': {
                    'frame_idx': analysis['best_detection_frame'],
                    'timestamp': analysis['best_detection_timestamp'],
                    'bbox': analysis['detection']['bbox'],
                    'crop_score': analysis['detection'].get('crop_score', 0.0),  # Default for 3-step pipeline
                    'full_frame_score': analysis['detection']['full_frame_score'],
                    'source': analysis['detection']['source']
                }
            })
        else:
            processed_data.update({
                'animals_detected': False,
                'reason': 'no_consistent_animal_movement'
            })
        
        # Write processed file with JSON data
        import json
        with open(processed_file, 'w') as f:
            # Convert numpy types to JSON-serializable types
            processed_data_clean = self._convert_for_json(processed_data)
            json.dump(processed_data_clean, f, indent=2)
        
        logger.info(f"📝 Marked {video_path.name} as processed: {processed_file}")
    
    
    def build_temporal_tracks(self, all_detections_by_frame: Dict[int, List[Dict]], fps: float) -> List[Dict]:
        """Build temporal tracks using bidirectional linking from anchor points."""
        # 1. Identify anchor points (motion frames + high confidence detections)
        anchor_frames = []
        for frame_idx, detections in all_detections_by_frame.items():
            if any(det['confidence'] > config.anchor_confidence_threshold for det in detections):
                anchor_frames.append(frame_idx)
        
        logger.info(f"🔗 Building temporal tracks from {len(anchor_frames)} anchor frames")
        
        # 2. Build tracks from each anchor point
        temporal_tracks = []
        processed_frames = set()
        
        for anchor_frame in anchor_frames:
            if anchor_frame in processed_frames:
                continue
                
            track = self.build_track_from_anchor(anchor_frame, all_detections_by_frame, fps)
            if track and track['duration_seconds'] >= config.min_track_duration:
                temporal_tracks.append({
                    'motion_sequence_idx': len(temporal_tracks),
                    'bbox_track': track,
                    'duration_seconds': track['duration_seconds'],
                    'consistency_score': len(track['frames']) / max(1, track['end_frame'] - track['start_frame'])
                })
                
                # Mark frames as processed
                for frame_idx in track['frames']:
                    processed_frames.add(frame_idx)
                
                logger.info(f"🎯 Temporal track built: {len(track['frames'])} frames over {track['duration_seconds']:.2f}s")
        
        return temporal_tracks
    
    def save_validated_crops(self, video_path: Path, validated_sequences: List[Dict]) -> List[str]:
        """Save crops from validated sequences where full-frame analysis confirmed detection."""
        import cv2
        
        # Create crops directory
        crops_dir = self.output_dir / 'validated_crops' / video_path.stem
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        # Open video for crop extraction
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video for crop extraction: {video_path}")
            return saved_paths
        
        try:
            for seq_idx, sequence in enumerate(validated_sequences):
                # Get the best detection from this sequence for crop extraction
                best_detection = sequence['best_detection']
                frame_idx = best_detection['frame_idx']
                bbox = best_detection['bbox']
                timestamp = best_detection['timestamp']
                
                # Seek to the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Extract crop using bbox coordinates
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        
                        # Generate filename with timestamp and confidence
                        confidence = best_detection['confidence']
                        combined_score = sequence['combined_score']
                        filename = f"seq{seq_idx}_t{timestamp:.1f}s_conf{confidence:.3f}_combined{combined_score:.3f}.jpg"
                        crop_path = crops_dir / filename
                        
                        # Save crop
                        cv2.imwrite(str(crop_path), crop)
                        saved_paths.append(str(crop_path))
                        
                        logger.info(f"💾 Saved validated crop: {crop_path}")
                
        finally:
            cap.release()
        
        return saved_paths
    
    def build_track_from_anchor(self, anchor_frame: int, all_detections_by_frame: Dict[int, List[Dict]], fps: float) -> Optional[Dict]:
        """Build bidirectional track from anchor point."""
        if anchor_frame not in all_detections_by_frame:
            return None
            
        anchor_detection = self.get_best_detection(all_detections_by_frame[anchor_frame])
        if not anchor_detection:
            return None
            
        track_frames = [anchor_frame]
        track_bboxes = [anchor_detection['bbox']]
        
        # Search backwards (configurable seconds)
        for frame_idx in range(anchor_frame - 1, max(0, anchor_frame - int(config.track_search_seconds * fps)), -1):
            if frame_idx in all_detections_by_frame:
                best_det = self.get_best_detection(all_detections_by_frame[frame_idx])
                if best_det and self.is_same_animal(anchor_detection, best_det):
                    track_frames.insert(0, frame_idx)
                    track_bboxes.insert(0, best_det['bbox'])
                else:
                    break  # Lost the track
        
        # Search forwards (configurable seconds)  
        for frame_idx in range(anchor_frame + 1, anchor_frame + int(config.track_search_seconds * fps)):
            if frame_idx in all_detections_by_frame:
                best_det = self.get_best_detection(all_detections_by_frame[frame_idx])
                if best_det and self.is_same_animal(anchor_detection, best_det):
                    track_frames.append(frame_idx)
                    track_bboxes.append(best_det['bbox'])
                else:
                    break  # Lost the track
        
        if len(track_frames) < config.min_track_frames:
            return None
            
        return {
            'track_id': anchor_frame,  # Use anchor frame as ID
            'start_frame': min(track_frames),
            'end_frame': max(track_frames),
            'frames': track_frames,
            'bboxes': track_bboxes,
            'duration_seconds': (max(track_frames) - min(track_frames)) / fps,
            'anchor_frame': anchor_frame
        }
    
    def get_best_detection(self, detections: List[Dict]) -> Optional[Dict]:
        """Get highest confidence detection from a frame."""
        if not detections:
            return None
        return max(detections, key=lambda d: d['confidence'])
    
    def is_same_animal(self, det1: Dict, det2: Dict) -> bool:
        """Check if two detections represent the same animal."""
        # Center distance check
        center1 = self.get_bbox_center(det1['bbox'])
        center2 = self.get_bbox_center(det2['bbox'])
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Size consistency check
        area1 = self.get_bbox_area(det1['bbox'])
        area2 = self.get_bbox_area(det2['bbox'])
        size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        return distance <= config.tracking_distance_threshold and size_ratio >= config.size_ratio_threshold
    
    def get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_bbox_area(self, bbox: List[float]) -> float:
        """Get area of bounding box."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

def initialize_config_from_args(args) -> None:
    """Initialize global config from CLI arguments."""
    global config
    
    config = ProcessingConfig(
        # Video processing
        max_frames_per_video=args.max_frames,
        confidence_threshold=args.confidence_threshold,
        
        # Validation thresholds
        megadetector_high_conf=args.megadetector_high_conf,
        yolo_high_conf=args.yolo_high_conf,
        min_yolo_detections=args.min_yolo_detections,
        weak_evidence_threshold=args.weak_evidence_threshold,
        
        # Camera handling detection
        max_tracks_threshold=20,  # Legacy parameter
        max_long_tracks_threshold=10,  # Legacy parameter  
        max_dense_tracks_threshold=5,  # Legacy parameter
        long_duration_threshold=300.0,  # Legacy parameter
        high_density_threshold=200,  # Legacy parameter
        composite_motion_threshold=args.composite_motion_threshold,
        min_motion_threshold=args.min_motion_threshold,
        motion_frames_weight=args.motion_frames_weight,
        motion_regions_weight=args.motion_regions_weight,
        motion_tracks_weight=args.motion_tracks_weight,
        large_region_multiplier=args.large_region_multiplier,
        
        # Motion detection
        motion_method=args.motion_method,
        motion_var_threshold=args.motion_var_threshold,
        filter_motion_var_threshold=args.filter_motion_var_threshold or args.motion_var_threshold,
        analysis_motion_var_threshold=args.analysis_motion_var_threshold or args.motion_var_threshold,
        min_motion_area=args.min_motion_area,
        max_motion_area=args.max_motion_area,
        motion_history=args.motion_history,
        max_regions_per_frame=args.max_regions_per_frame,
        min_region_width=args.min_region_width,
        min_region_height=args.min_region_height,
        max_aspect_ratio=args.max_aspect_ratio,
        motion_margin=args.motion_margin,
        
        # Temporal consistency
        min_track_duration=args.min_track_duration,
        motion_tracking_gap_seconds=args.motion_tracking_gap_seconds,
        detection_validation_gap_seconds=args.detection_validation_gap_seconds,
        tracking_distance_threshold=args.tracking_distance_threshold,
        anchor_confidence_threshold=args.anchor_confidence_threshold,
        min_track_frames=args.min_track_frames,
        
        # Step 4 validation
        max_validation_frames=args.max_validation_frames,
        crop_weight=args.crop_weight,
        fullframe_weight=args.fullframe_weight,
        min_crop_size=args.min_crop_size,
        temporal_spread_seconds=args.temporal_spread_seconds,
        accepted_rtdetr_overlap=args.accepted_rtdetr_overlap,
        
        # Additional parameters
        full_frame_validation_frames=args.full_frame_validation_frames,
        size_ratio_threshold=args.size_ratio_threshold,
        track_search_seconds=args.track_search_seconds,
        
        # Model configuration
        ensemble_models=args.ensemble.split(','),
    )

def main():
    """Main entry point for next-generation processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Next Generation Wildlife Video Processor with Temporal Consistency')
    parser.add_argument('--videos', '-v', nargs='+', help='Optional list of video indices (e.g. 7 8 9) or names to process')
    
    # Add common arguments from base class
    VideoProcessorBase.setup_common_arguments(parser)
    
    # Add motion detection arguments (includes temporal consistency for Next-Gen)
    VideoProcessorBase.setup_motion_detection_arguments(parser)
    
    args = parser.parse_args()
    
    # Convert video arguments
    video_filter = None
    if args.videos:
        video_filter = []
        for video in args.videos:
            try:
                video_filter.append(int(video))
            except ValueError:
                video_filter.append(video)
    
    # Set environment variables from arguments (for base class compatibility)
    VideoProcessorBase.set_environment_from_args(args, include_motion=True)
    
    # Initialize global config from CLI arguments
    initialize_config_from_args(args)
    
    try:
        processor = NextGenVideoProcessor()
        
        # Log ALL parameters explicitly AFTER logger is initialized
        logger.info("================================================================================")
        logger.info("🎯 COMMAND PARAMETERS")
        logger.info("================================================================================")
        for attr_name in sorted(vars(args)):
            attr_value = getattr(args, attr_name)
            logger.info(f"{attr_name}: {attr_value}")
        logger.info("================================================================================")
        
        print(f"🎬 Starting Next Generation wildlife video processing...")
        print(f"📊 Mode: Motion detection + temporal consistency + full-frame validation")
        print(f"🕒 Temporal parameters: {args.min_track_duration}s duration, motion gap {args.motion_tracking_gap_seconds}s, detection gap {args.detection_validation_gap_seconds}s")
        
        logger.info(f"🎯 Processing strategy: Next Generation Temporal Consistency")
        logger.info(f"🕒 Min track duration: {args.min_track_duration}s")
        logger.info(f"✅ Full-frame validation frames: {args.full_frame_validation_frames}")
        
        processor.process_all_videos(video_filter=video_filter)
        
    except KeyboardInterrupt:
        print("🛑 Processing interrupted by user")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()