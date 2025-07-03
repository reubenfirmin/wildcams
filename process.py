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
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass

# Import base processor
from video_processor_base import VideoProcessorBase
from ml_detection import MODEL_DETECTION_THRESHOLD

# Global configuration object
@dataclass
class ProcessingConfig:
    """Global configuration for next-generation video processing."""
    # Video processing
    max_frames_per_video: int
    confidence_threshold: float
    
    # Camera handling detection
    composite_motion_threshold: int
    min_motion_threshold: int
    motion_frames_weight: float
    motion_regions_weight: float
    motion_tracks_weight: float
    large_region_multiplier: float
    
    # Motion detection
    motion_method: str
    motion_var_threshold: int
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
    min_consecutive_detection_seconds: float
    tracking_distance_threshold: float
    anchor_confidence_threshold: float
    min_track_frames: int
    
    # Step 3 validation
    max_validation_frames: int
    temporal_spread_seconds: float
    spatial_overlap_threshold: float
    
    # Track infilling parameters
    enable_track_infilling: bool
    infill_max_gap_seconds: float
    infill_max_distance_pixels: float
    infill_min_overlap_ratio: float
    
    # Debug parameters
    debug_show_spatially_invalid: bool
    
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

class NextGenVideoProcessor(VideoProcessorBase):
    """Next generation processor with temporal consistency and full-frame validation."""
    
    def __init__(self):
        super().__init__()
        
        # Override base class attributes with config values
        self.ensemble_models = config.ensemble_models
        
        # Create tracking subdirectory for .processed files
        self.tracking_dir = self.video_dir / '.tracking'
        self.tracking_dir.mkdir(exist_ok=True)
        
        # Motion detection configuration (single config for all steps)
        self.motion_config = {
            'method': config.motion_method,
            'var_threshold': config.motion_var_threshold,
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
        
        # Initialize motion detection algorithm
        self.bg_subtractor = None
        self.init_motion_detector()
        
        # All models used in full-frame analysis
        logger.info(f"💡 ENSEMBLE MODELS:")
        logger.info(f"  🖼️ Full-frame analysis: {self.ensemble_models}")
        
        logger.info(f"🎯 Next Generation video processor initialized")
        logger.info(f"🕒 Temporal consistency: min {config.min_track_duration}s, motion gap {config.motion_tracking_gap_seconds}s, min consecutive detection {config.min_consecutive_detection_seconds}s")
        logger.info(f"🔍 Motion method: {self.motion_config['method']}")
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
        if self.motion_config['method'] == 'MOG2':
            return cv2.createBackgroundSubtractorMOG2(
                detectShadows=self.motion_config['detect_shadows'],
                varThreshold=self.motion_config['var_threshold'],
                history=self.motion_config['history']
            )
        elif self.motion_config['method'] == 'KNN':
            return cv2.createBackgroundSubtractorKNN(
                detectShadows=self.motion_config['detect_shadows'],
                dist2Threshold=400,
                history=self.motion_config['history']
            )
        else:
            raise ValueError(f"Unknown motion detection method: {self.motion_config['method']}")
    
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
        
        # Use unified motion config (dual configs removed)
        active_config = self.motion_config
        
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
        
        # Use unified motion config (dual configs removed)
        active_config = self.motion_config
        
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
                # No motion - end current sequence (duration filtering happens after infilling)
                if current_sequence is not None:
                    duration = current_sequence['end_timestamp'] - current_sequence['start_timestamp']
                    motion_sequences.append(current_sequence)
                    logger.info(f"Motion sequence found: frames {current_sequence['start_frame']}-{current_sequence['end_frame']} ({duration:.2f}s)")
                    current_sequence = None
        
        # Handle sequence that extends to end of video (duration filtering happens after infilling)
        if current_sequence is not None:
            duration = current_sequence['end_timestamp'] - current_sequence['start_timestamp']
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
        
        # Store initial track count for camera handling filter
        self._initial_track_count = len(motion_tracks)
        
        # Apply track infilling if enabled
        if config.enable_track_infilling:
            motion_tracks = self._infill_motion_tracks(motion_tracks, fps)
        else:
            # Print track summary even without infilling
            self._print_track_summary(motion_tracks, fps)
        
        return motion_tracks
    
    def _infill_motion_tracks(self, motion_tracks: List[Dict], fps: float) -> List[Dict]:
        """Infill gaps between nearby motion tracks to create continuous tracks.
        
        Args:
            motion_tracks: List of motion tracks
            fps: Video frame rate
            
        Returns:
            List of motion tracks with gaps filled between compatible tracks
        """
        if not motion_tracks:
            return motion_tracks
        
        logger.info(f"[INFILL] Starting with {len(motion_tracks)} tracks, checking for infilling opportunities")
        
        # Sort tracks by start frame
        sorted_tracks = sorted(motion_tracks, key=lambda t: t['frames'][0])
        infilled_tracks = []
        used_track_ids = set()
        
        for i, track_a in enumerate(sorted_tracks):
            if track_a['track_id'] in used_track_ids:
                continue
                
            # Start with this track
            merged_track = track_a.copy()
            used_track_ids.add(track_a['track_id'])
            
            # Look for compatible tracks to merge
            changed = True
            while changed:
                changed = False
                
                for j, track_b in enumerate(sorted_tracks):
                    if track_b['track_id'] in used_track_ids:
                        continue
                    
                    # Check if tracks can be infilled
                    if self._can_infill_tracks(merged_track, track_b, fps):
                        logger.info(f"[INFILL] Merging track_{track_b['track_id']} into track_{merged_track['track_id']}")
                        merged_track = self._merge_tracks_with_infill(merged_track, track_b, fps)
                        used_track_ids.add(track_b['track_id'])
                        changed = True
                        break
            
            infilled_tracks.append(merged_track)
        
        logger.info(f"[INFILL] Result: {len(infilled_tracks)} tracks after infilling ({len(motion_tracks) - len(infilled_tracks)} tracks merged)")
        
        # Filter out tracks shorter than max(min_track_duration, min_consecutive_detection_seconds)
        min_required_duration = max(config.min_track_duration, config.min_consecutive_detection_seconds)
        filtered_tracks = [track for track in infilled_tracks if track['duration_seconds'] >= min_required_duration]
        
        removed_count = len(infilled_tracks) - len(filtered_tracks)
        if removed_count > 0:
            logger.info(f"[INFILL] Removed {removed_count} tracks shorter than {min_required_duration:.2f}s (max of min_track_duration={config.min_track_duration:.2f}s, min_consecutive_detection={config.min_consecutive_detection_seconds:.2f}s)")
        
        # Print detailed track summary
        self._print_track_summary(filtered_tracks, fps)
        
        return filtered_tracks
    
    def _can_infill_tracks(self, track_a: Dict, track_b: Dict, fps: float) -> bool:
        """Check if two tracks can be infilled based on spatial and temporal criteria."""
        # Get track boundaries
        frames_a = track_a['frames']
        frames_b = track_b['frames']
        
        end_frame_a = frames_a[-1]
        start_frame_b = frames_b[0]
        
        # Check temporal gap
        gap_frames = start_frame_b - end_frame_a
        gap_seconds = gap_frames / fps
        
        if gap_frames <= 0:  # Overlapping or adjacent tracks
            return False
        
        if gap_seconds > config.infill_max_gap_seconds:
            return False
        
        # Check spatial proximity using end bbox of track_a vs start bbox of track_b
        motion_regions_a = track_a.get('motion_regions', [])
        motion_regions_b = track_b.get('motion_regions', [])
        
        if not motion_regions_a or not motion_regions_b:
            return False
        
        # Get end bbox from track_a and start bbox from track_b
        end_region_a = motion_regions_a[-1][-1] if motion_regions_a[-1] else None
        start_region_b = motion_regions_b[0][0] if motion_regions_b[0] else None
        
        if not end_region_a or not start_region_b:
            return False
        
        # Calculate distance between track end and track start centers
        center_a = ((end_region_a[0] + end_region_a[2]) / 2, (end_region_a[1] + end_region_a[3]) / 2)
        center_b = ((start_region_b[0] + start_region_b[2]) / 2, (start_region_b[1] + start_region_b[3]) / 2)
        
        distance = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
        
        if distance > config.infill_max_distance_pixels:
            return False
        
        # Check bbox overlap ratio between end and start positions
        overlap = self._calculate_bbox_overlap(end_region_a, start_region_b)
        if overlap < config.infill_min_overlap_ratio:
            return False
        
        return True
    
    def _merge_tracks_with_infill(self, track_a: Dict, track_b: Dict, fps: float) -> Dict:
        """Merge two tracks with interpolated frames in the gap."""
        frames_a = track_a['frames']
        frames_b = track_b['frames']
        regions_a = track_a['motion_regions']
        regions_b = track_b['motion_regions']
        
        end_frame_a = frames_a[-1]
        start_frame_b = frames_b[0]
        
        # Create interpolated frames for the gap
        gap_frames = list(range(end_frame_a + 1, start_frame_b))
        
        # Interpolate regions for gap frames (simple: use last region from track_a)
        last_region_a = regions_a[-1][-1] if regions_a and regions_a[-1] else track_a.get('representative_region')
        first_region_b = regions_b[0][0] if regions_b and regions_b[0] else track_b.get('representative_region')
        
        # Simple interpolation: use last region from track_a for all gap frames
        gap_regions = [[last_region_a] for _ in gap_frames] if last_region_a else []
        
        # Merge the tracks
        merged_track = {
            'track_id': track_a['track_id'],  # Keep first track's ID
            'start_frame': frames_a[0],
            'end_frame': frames_b[-1],
            'frames': frames_a + gap_frames + frames_b,
            'motion_regions': regions_a + gap_regions + regions_b,
            'duration_seconds': (frames_b[-1] - frames_a[0]) / fps,
            'representative_region': track_a.get('representative_region'),
            'detection_count': len(frames_a) + len(gap_frames) + len(frames_b),
            'avg_regions_per_frame': track_a.get('avg_regions_per_frame', 0),
            'infilled_from': [track_a['track_id'], track_b['track_id']],
            'infill_gap_frames': len(gap_frames)
        }
        
        return merged_track
    
    def _print_track_summary(self, motion_tracks: List[Dict], fps: float) -> None:
        """Print a detailed summary of motion tracks."""
        if not motion_tracks:
            logger.info("📋 TRACK SUMMARY: No motion tracks found")
            return
        
        logger.info("📋 MOTION TRACK SUMMARY:")
        logger.info("================================================================================")
        
        for track in motion_tracks:
            track_id = track['track_id']
            frames = track['frames']
            start_frame, end_frame = frames[0], frames[-1]
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = track['duration_seconds']
            
            # Get start and end bbox from motion regions
            motion_regions = track.get('motion_regions', [])
            if motion_regions:
                # Get first and last motion regions
                start_region = motion_regions[0][0] if motion_regions[0] else None
                end_region = motion_regions[-1][-1] if motion_regions[-1] else None
                
                if start_region:
                    start_bbox_str = f"start_bbox:{start_region[0]:.0f},{start_region[1]:.0f},{start_region[2]:.0f},{start_region[3]:.0f}"
                    start_width = start_region[2] - start_region[0]
                    start_height = start_region[3] - start_region[1]
                    start_size_str = f"start_size:{start_width:.0f}x{start_height:.0f}"
                else:
                    start_bbox_str = "start_bbox:unknown"
                    start_size_str = "start_size:unknown"
                
                if end_region:
                    end_bbox_str = f"end_bbox:{end_region[0]:.0f},{end_region[1]:.0f},{end_region[2]:.0f},{end_region[3]:.0f}"
                    end_width = end_region[2] - end_region[0]
                    end_height = end_region[3] - end_region[1]
                    end_size_str = f"end_size:{end_width:.0f}x{end_height:.0f}"
                else:
                    end_bbox_str = "end_bbox:unknown"
                    end_size_str = "end_size:unknown"
                
                bbox_str = f"{start_bbox_str} | {end_bbox_str}"
                size_str = f"{start_size_str} | {end_size_str}"
            else:
                bbox_str = "start_bbox:unknown | end_bbox:unknown"
                size_str = "start_size:unknown | end_size:unknown"
            
            # Infill information if available
            infill_info = ""
            if 'infilled_from' in track:
                original_tracks = track['infilled_from']
                gap_frames = track.get('infill_gap_frames', 0)
                infill_info = f" | infilled:{len(original_tracks)}tracks,{gap_frames}gap_frames"
            
            logger.info(f"  🎯 track_{track_id}: frames:{start_frame}-{end_frame} ({len(frames)}frames) | "
                       f"time:{start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s) | {bbox_str} | {size_str}{infill_info}")
        
        logger.info("================================================================================")
        
        # Summary statistics
        total_frames = sum(len(track['frames']) for track in motion_tracks)
        avg_duration = sum(track['duration_seconds'] for track in motion_tracks) / len(motion_tracks)
        longest_track = max(motion_tracks, key=lambda t: t['duration_seconds'])
        
        logger.info(f"📊 SUMMARY: {len(motion_tracks)} tracks | {total_frames} total frames | "
                   f"avg_duration:{avg_duration:.2f}s | longest:{longest_track['duration_seconds']:.2f}s (track_{longest_track['track_id']})")
    
    def filter_motion_tracks_for_camera_handling(self, video_path: Path, motion_tracks: List[Dict]) -> List[Dict]:
        """STEP 2: Filter motion tracks for camera handling detection using weighted composite motion score."""
        logger.info(f"[STEP2] {video_path.name}: Filtering {len(motion_tracks)} motion tracks for camera handling")
        
        # Calculate motion density: frames × avg bbox area for each track
        total_motion_density = 0
        for track in motion_tracks:
            track_frames = len(track['frames'])
            
            # Calculate average bbox area for this track
            motion_regions = track.get('motion_regions', [])
            total_area = 0
            bbox_count = 0
            
            for frame_regions in motion_regions:
                for bbox in frame_regions:
                    if bbox:
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        area = width * height
                        total_area += area
                        bbox_count += 1
            
            avg_bbox_area = total_area / bbox_count if bbox_count > 0 else 0
            track_motion_density = track_frames * (avg_bbox_area / 1000)  # Normalize by 1000 pixels
            total_motion_density += track_motion_density
        
        num_tracks = len(motion_tracks)
        
        # Get track count before filtering for penalty calculation
        initial_track_count = getattr(self, '_initial_track_count', len(motion_tracks))
        filtering_penalty = 1.0 + ((initial_track_count - num_tracks) / max(1, initial_track_count)) * 2.0
        
        # Spatial clustering: group tracks by bbox overlap
        spatial_clusters = self._cluster_tracks_by_spatial_overlap(motion_tracks)
        effective_regions = len(spatial_clusters)
        
        # Log spatial clustering details
        logger.info(f"  📊 SPATIAL CLUSTERING DEBUG:")
        for i, cluster in enumerate(spatial_clusters):
            cluster_frames = sum(len(track['frames']) for track in cluster)
            track_ids = [track['track_id'] for track in cluster]
            logger.info(f"    Cluster {i}: tracks={track_ids}, total_frames={cluster_frames}")
            
            # Show bbox positions for debugging
            for track in cluster:
                motion_regions = track.get('motion_regions', [])
                if motion_regions and motion_regions[0]:
                    start_bbox = motion_regions[0][0]
                    logger.info(f"      track_{track['track_id']}: start_bbox={start_bbox}")
        
        # Calculate consistency penalty based on bbox variance within clusters
        consistency_penalty = self._calculate_bbox_consistency_penalty(spatial_clusters)
        
        # No large region calculation needed with spatial clustering
        large_region_percentage = 0.0
        large_region_multiplier = 1.0
        
        # INVERTED LOGIC: Camera handling detection
        # LOW scores = concentrated animal movement (good)
        # HIGH scores = dispersed camera handling (bad)
        
        # Calculate spatial dispersion: ratio of clusters to tracks
        # High dispersion (many clusters per track) indicates camera handling
        spatial_dispersion = effective_regions / max(1, num_tracks)
        
        # Invert motion density: low density = higher penalty
        # Camera handling has sparse, erratic movement
        motion_sparsity = 1.0 / max(1.0, total_motion_density / 1000)  # Normalize and invert
        
        # Calculate composite score for camera handling detection
        # Higher values indicate MORE camera handling characteristics
        base_score = (spatial_dispersion ** config.motion_regions_weight * 
                     motion_sparsity ** config.motion_frames_weight * 
                     num_tracks ** config.motion_tracks_weight)
        
        composite_score = base_score * consistency_penalty * filtering_penalty
        
        # Use CLI parameter for camera handling detection
        threshold = config.composite_motion_threshold
        
        logger.info(f"[STEP2] {video_path.name}: Camera handling score = {composite_score:.6f}")
        logger.info(f"  📊 Spatial dispersion: {effective_regions}/{num_tracks} = {spatial_dispersion:.3f}")
        logger.info(f"  📊 Motion density: {total_motion_density:.0f} → sparsity: {motion_sparsity:.6f}")
        logger.info(f"  📊 Base: dispersion^{config.motion_regions_weight:.1f}={spatial_dispersion:.3f}^{config.motion_regions_weight:.1f} * sparsity^{config.motion_frames_weight:.1f}={motion_sparsity:.6f}^{config.motion_frames_weight:.1f} * tracks^{config.motion_tracks_weight:.1f}={num_tracks}^{config.motion_tracks_weight:.1f} = {base_score:.6f}")
        logger.info(f"  📊 Penalties: consistency={consistency_penalty:.2f}x, filtering={filtering_penalty:.2f}x (initial_tracks={initial_track_count}→{num_tracks})")
        logger.info(f"  📊 Spatial clusters: {len(spatial_clusters)} effective regions from {num_tracks} tracks")
        
        # Store composite score for summary reporting
        self._composite_scores = getattr(self, '_composite_scores', {})
        self._composite_scores[video_path.name] = composite_score
        
        # Skip insufficient motion check - use only camera handling threshold
        # With inverted logic, we only need one threshold for camera handling detection
        
        # Check for excessive motion (camera handling)
        if composite_score > threshold:
            logger.warning(f"⚠️  CAMERA HANDLING: score={composite_score} > {threshold}")
            # Store rejection reason for summary
            self._rejection_reasons = getattr(self, '_rejection_reasons', {})
            self._rejection_reasons[video_path.name] = f"camera_handling (score={composite_score:.0f})"
            return []  # Early exit - skip expensive ML processing
        return motion_tracks
    
    def _cluster_tracks_by_spatial_overlap(self, motion_tracks: List[Dict]) -> List[List[Dict]]:
        """Group tracks into spatial clusters based on bbox overlap."""
        if not motion_tracks:
            return []
        
        clusters = []
        for track in motion_tracks:
            # Get representative bbox for this track (use start bbox)
            motion_regions = track.get('motion_regions', [])
            if not motion_regions or not motion_regions[0]:
                continue
            
            track_bbox = motion_regions[0][0] if motion_regions[0] else None
            if not track_bbox:
                continue
            
            # Find if this track overlaps with any existing cluster
            assigned = False
            for cluster in clusters:
                for cluster_track in cluster:
                    cluster_regions = cluster_track.get('motion_regions', [])
                    if cluster_regions and cluster_regions[0]:
                        cluster_bbox = cluster_regions[0][0]
                        if self._calculate_bbox_overlap(track_bbox, cluster_bbox) > 0.3:  # 30% overlap threshold
                            cluster.append(track)
                            assigned = True
                            break
                if assigned:
                    break
            
            # If no overlap found, create new cluster
            if not assigned:
                clusters.append([track])
        
        return clusters
    
    def _calculate_bbox_consistency_penalty(self, spatial_clusters: List[List[Dict]]) -> float:
        """Calculate consistency reward using logarithmic decay for repeated spatial regions."""
        if not spatial_clusters:
            return 1.0
        
        total_weight = 0.0
        
        logger.info(f"  📊 CONSISTENCY PENALTY DEBUG:")
        for i, cluster in enumerate(spatial_clusters):
            # Each spatial cluster starts with weight 1.0
            cluster_frames = sum(len(track['frames']) for track in cluster)
            
            if cluster_frames <= 1:
                # Single frame = full weight
                cluster_weight = 1.0
                total_weight += cluster_weight
                logger.info(f"    Cluster {i}: {cluster_frames} frames → weight={cluster_weight:.3f} (single frame)")
            else:
                # Logarithmic decay: consistent movement gets lower weight
                # log(frames) rewards staying in same spatial area longer
                cluster_weight = 1.0 / max(1.0, np.log(cluster_frames))
                total_weight += cluster_weight
                logger.info(f"    Cluster {i}: {cluster_frames} frames → log({cluster_frames})={np.log(cluster_frames):.3f} → weight={cluster_weight:.3f}")
        
        # Normalize by number of clusters to get average consistency
        avg_weight = total_weight / len(spatial_clusters) if spatial_clusters else 1.0
        
        # Convert to penalty: consistent movement (low avg_weight) = low penalty
        # Scale: 0.1 avg_weight → 1.0x penalty, 1.0 avg_weight → 5.0x penalty
        consistency_penalty = 1.0 + (avg_weight * 4.0)
        
        logger.info(f"    Total weight: {total_weight:.3f}, avg_weight: {avg_weight:.3f} → penalty: {consistency_penalty:.3f}")
        
        return min(consistency_penalty, 5.0)  # Cap at 5x penalty
    
    
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
    
    
    
    
    
    def run_full_frame_analysis_on_motion_tracks(self, video_path: Path, motion_tracks: List[Dict]) -> List[Dict]:
        """
        NEW: Direct Step 2 → Step 4 connection.
        Run full-frame analysis on motion tracks with spatial overlap validation.
        
        Args:
            video_path: Path to video file
            motion_tracks: List of motion tracks from Step 1-2 (basic motion tracks)
            
        Returns:
            List of validated sequences with spatial overlap confirmation
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        validated_results = []
        failed_tracks_data = []  # Capture data from failed tracks for summary
        
        logger.info(f"[STEP3] Running full-frame analysis on {len(motion_tracks)} motion tracks")
        
        # Convert basic motion tracks to extended bbox tracks for spatial validation
        extended_tracks = []
        for motion_track in motion_tracks:
            # Extract motion track data
            track_id = motion_track['track_id']
            motion_frames = motion_track['frames']
            motion_regions = motion_track.get('motion_regions', [])
            
            if not motion_frames:
                continue
            
            # Build full video coverage detections using motion regions as bboxes
            motion_start_frame = min(motion_frames)
            motion_end_frame = max(motion_frames)
            
            # Use motion regions or create default bboxes for motion frames
            full_detections = []
            
            # Get representative bbox from motion regions (use first available or default)
            if motion_regions and len(motion_regions) > 0:
                # Use first available motion region as representative bbox - convert tuple to list
                first_region = motion_regions[0][0] if motion_regions[0] else (100, 100, 200, 200)
                first_known_position = list(first_region)  # Convert tuple to list for .copy()
                last_region = motion_regions[-1][-1] if motion_regions[-1] else first_region
                last_known_position = list(last_region)   # Convert tuple to list for .copy()
            else:
                # Default bbox if no motion regions available
                first_known_position = [100, 100, 200, 200]
                last_known_position = first_known_position[:]  # Make a copy
            
            # Backfill: frame 0 to motion_start-1
            for frame_idx in range(0, motion_start_frame):
                full_detections.append({
                    'frame': frame_idx,
                    'bbox': first_known_position.copy(),
                    'timestamp': frame_idx / fps,
                    'motion_detected': False,
                    'fill_type': 'backfill'
                })
            
            # Motion frames: use actual motion regions or default
            for i, frame_idx in enumerate(motion_frames):
                if motion_regions and i < len(motion_regions) and motion_regions[i]:
                    # Use first region from this frame - convert tuple to list
                    bbox = list(motion_regions[i][0])
                else:
                    # Use default bbox
                    bbox = first_known_position.copy()
                    
                full_detections.append({
                    'frame': frame_idx,
                    'bbox': bbox,
                    'timestamp': frame_idx / fps,
                    'motion_detected': True,
                    'fill_type': 'motion'
                })
            
            # Forward-fill: motion_end+1 to video end
            for frame_idx in range(motion_end_frame + 1, total_frames):
                full_detections.append({
                    'frame': frame_idx,
                    'bbox': last_known_position.copy(),
                    'timestamp': frame_idx / fps,
                    'motion_detected': False,
                    'fill_type': 'forward_fill'
                })
            
            # Sort detections by frame
            full_detections.sort(key=lambda x: x['frame'])
            
            # Create extended bbox track structure
            bbox_track = {
                'track_id': track_id,
                'detections': full_detections,
                'start_frame': 0,
                'end_frame': total_frames - 1,
                'motion_start_frame': motion_start_frame,
                'motion_end_frame': motion_end_frame,
                'first_known_position': first_known_position,
                'last_known_position': last_known_position
            }
            
            # Create extended track with bbox_track structure
            extended_track = {
                'track_id': track_id,
                'bbox_track': bbox_track,
                'duration_seconds': motion_track['duration_seconds'],
                'detection_count': len(motion_frames),
                'total_coverage_frames': len(full_detections),
                'motion_frames': len(motion_frames),
                'original_motion_track': motion_track
            }
            
            extended_tracks.append(extended_track)
        
        logger.info(f"[STEP3] Built {len(extended_tracks)} extended tracks from {len(motion_tracks)} motion tracks")
        
        # Sample frames for analysis from all tracks (get unique frames)
        all_sample_frames = set()
        track_sample_frames = {}  # Store per-track sample frames
        
        for track in extended_tracks:
            sample_frames = self._sample_motion_track_frames(track, max_frames=config.max_validation_frames)
            track_sample_frames[track['track_id']] = sample_frames
            all_sample_frames.update(sample_frames)
        
        all_sample_frames = sorted(all_sample_frames)
        logger.info(f"[STEP3] Processing {len(all_sample_frames)} unique frames across {len(extended_tracks)} tracks")
        
        # Helper function to get track bbox for any track at any frame
        def get_track_bbox_for_frame(track_id: int, frame_idx: int) -> tuple:
            """Get bbox for specific track at specific frame, handling implicit fills."""
            track = next((t for t in extended_tracks if t['track_id'] == track_id), None)
            if not track:
                return None, None, None
            
            bbox_track = track['bbox_track']
            for det in bbox_track['detections']:
                if det['frame'] == frame_idx:
                    bbox = det['bbox']
                    fill_type = det['fill_type']
                    motion_detected = det['motion_detected']
                    return bbox, fill_type, motion_detected
            
            return None, None, None
        
        # Global tracking for all tracks
        track_detections = {track['track_id']: [] for track in extended_tracks}
        model_contributions = {}
        frame_results = []
        
        # PHASE 1: Frame-First Processing (your specified algorithm)
        # We iterate over frames to start because we never want to run detection over the same frame twice
        for frame_idx in all_sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            timestamp = frame_idx / fps
            logger.info(f"EVAL | {video_path.stem} | {timestamp:.2f}s | {frame_idx}")
            
            # Run each model against the frame
            for model_name in config.ensemble_models:
                model_detections = self.ml_ensemble.run_single_model_detection(
                    model_name, frame, timestamp, frame_idx,
                    full_frame=frame,
                    accepted_rtdetr_overlap=config.spatial_overlap_threshold
                )
                
                # Initialize model contributions if needed
                if model_name not in model_contributions:
                    model_contributions[model_name] = {
                        'count': 0, 'max_conf': 0.0, 'total_conf': 0.0,
                        'spatial_valid_count': 0, 'total_score': 0.0
                    }
                
                # Update global model contributions for this frame (once per model)
                if model_detections:
                    model_contributions[model_name]['count'] = len(model_detections)
                    model_contributions[model_name]['max_conf'] = max(det['confidence'] for det in model_detections)
                    model_contributions[model_name]['total_conf'] = sum(det['confidence'] for det in model_detections)
                
                # For each track, determine if there was an overlap above the threshold yielded by the detection
                # Follow algorithm: for track in tracks: check all detections against this track's bbox
                for track in extended_tracks:
                    track_id = track['track_id']
                    track_bbox, fill_type, motion_detected = get_track_bbox_for_frame(track_id, frame_idx)
                    
                    # Debug: Ensure we always process every model-track combination
                    logger.debug(f"Processing {model_name} against track_{track_id}: bbox={'present' if track_bbox else 'none'}")
                    
                    if track_bbox is not None:
                        overlap_type = 'explicit' if fill_type == 'motion' else f'implicit_{fill_type}'
                        motion_bbox_str = f"motn:{track_bbox[0]:.0f},{track_bbox[1]:.0f},{track_bbox[2]:.0f},{track_bbox[3]:.0f}"
                        
                        overlapping_count = 0
                        valid_detections = []
                        
                        if model_detections:
                            for det in model_detections:
                                det_bbox = det['bbox']
                                overlap = self._calculate_bbox_overlap(track_bbox, det_bbox)
                                
                                if overlap >= config.spatial_overlap_threshold:
                                    # Store valid detection for potential boosting
                                    valid_detections.append({
                                        'detection': det,
                                        'overlap': overlap,
                                        'bbox_str': f"bbox:{det_bbox[0]:.0f},{det_bbox[1]:.0f},{det_bbox[2]:.0f},{det_bbox[3]:.0f}"
                                    })
                                    overlapping_count += 1
                                    
                                elif overlap > 0.0:
                                    # Log detection with warning
                                    if config.debug_show_spatially_invalid:
                                        bbox_str = f"bbox:{det_bbox[0]:.0f},{det_bbox[1]:.0f},{det_bbox[2]:.0f},{det_bbox[3]:.0f}"
                                        logger.info(f"⚠️ | {model_name} | {bbox_str} | conf:{det['confidence']:.3f} | ovlp:{overlap:.3f} | {motion_bbox_str} | scor:0.000 | threshold_failed | {overlap_type} | track_{track_id}")
                                    overlapping_count += 1
                            
                            # Process valid detections - create ONE synthetic detection per model per track
                            if valid_detections:
                                # Find best detection by score (confidence * overlap) after consensus boosting
                                consensus_boost = 1.0 + 0.1 * (len(valid_detections) - 1)
                                
                                best_detection = None
                                best_score = 0.0
                                
                                for valid_det in valid_detections:
                                    det = valid_det['detection']
                                    overlap = valid_det['overlap']
                                    boosted_conf = min(1.0, det['confidence'] * consensus_boost)
                                    score = boosted_conf * overlap
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_detection = {
                                            'detection': det,
                                            'overlap': overlap,
                                            'bbox_str': valid_det['bbox_str'],
                                            'boosted_conf': boosted_conf,
                                            'score': score
                                        }
                                
                                # Log ONE synthetic detection representing the best from this model for this track
                                det = best_detection['detection']
                                overlap = best_detection['overlap']
                                bbox_str = best_detection['bbox_str']
                                boosted_conf = best_detection['boosted_conf']
                                overall_score = best_detection['score']
                                
                                consensus_note = f"consensus:{len(valid_detections)}" if len(valid_detections) > 1 else "single"
                                logger.info(f"✅ | {model_name} | {bbox_str} | conf:{det['confidence']:.3f}→{boosted_conf:.3f} | ovlp:{overlap:.3f} | {motion_bbox_str} | scor:{overall_score:.3f} | spatial_valid | {overlap_type} | track_{track_id} | {consensus_note}")
                                
                                model_contributions[model_name]['spatial_valid_count'] += 1
                                model_contributions[model_name]['total_score'] += overall_score
                                
                                # Store ONE synthetic detection for track evaluation (use boosted confidence)
                                det_copy = det.copy()
                                det_copy['source'] = model_name  # FIX: Add the source field for ensemble scoring
                                det_copy['confidence'] = boosted_conf
                                det_copy['original_confidence'] = det['confidence']
                                det_copy['consensus_boost'] = consensus_boost
                                det_copy['consensus_count'] = len(valid_detections)
                                det_copy['motion_overlap'] = overlap
                                det_copy['frame_idx'] = frame_idx
                                det_copy['timestamp'] = timestamp
                                det_copy['overlap_type'] = overlap_type
                                track_detections[track_id].append(det_copy)
                        
                        # If no detections overlapped with this track, log summary
                        if overlapping_count == 0:
                            if model_detections:
                                non_overlapping = len(model_detections)
                                logger.info(f"❌ | {model_name} | {non_overlapping}_detections | conf:0.000 | ovlp:0.000 | {motion_bbox_str} | scor:0.000 | no_overlap | {overlap_type} | track_{track_id}")
                            else:
                                logger.info(f"❌ | {model_name} | none | conf:0.000 | ovlp:0.000 | {motion_bbox_str} | scor:0.000 | no_detection | {overlap_type} | track_{track_id}")
                    else:
                        # Track has no bbox for this frame
                        if model_detections:
                            logger.info(f"❌ | {model_name} | {len(model_detections)}_detections | conf:0.000 | ovlp:0.000 | motn:none | scor:0.000 | no_track_bbox | track_{track_id}")
                        else:
                            logger.info(f"❌ | {model_name} | none | conf:0.000 | ovlp:0.000 | motn:none | scor:0.000 | no_detection | track_{track_id}")
            
            # Calculate ensemble score per track for this frame
            track_ensemble_results = []
            frame_valid = False
            
            for track in extended_tracks:
                track_id = track['track_id']
                
                # Get this frame's detections for this track
                track_frame_detections = [d for d in track_detections[track_id] if d.get('frame_idx') == frame_idx]
                
                # Calculate ensemble score for this track
                track_model_contributions = {}
                for model_name in config.ensemble_models:
                    # Find best detection from this model for this track on this frame
                    model_track_detections = [d for d in track_frame_detections if d.get('source') == model_name]
                    if model_track_detections:
                        best_detection = max(model_track_detections, key=lambda d: d['confidence'])
                        track_model_contributions[model_name] = best_detection['confidence']
                    else:
                        track_model_contributions[model_name] = 0.0
                
                # TODO: Pull out ensemble weighting parameters to CLI/config
                # Calculate track ensemble score with weighted contributions
                detecting_models = [m for m, conf in track_model_contributions.items() if conf > 0]
                non_detecting_models = [m for m, conf in track_model_contributions.items() if conf == 0]
                
                non_detecting_weight = 0.1   # TODO: Make configurable
                min_detecting_weight = 0.15  # TODO: Make configurable - floor for low-confidence detecting models
                
                track_ensemble_score = 0.0
                
                # Non-detecting models get 10% each (× 0 confidence = 0 contribution)
                for model_name in non_detecting_models:
                    track_ensemble_score += track_model_contributions[model_name] * non_detecting_weight
                
                # Detecting models: confidence-proportional with 15% floor
                if detecting_models:
                    # Reserve weight for non-detecting models and minimum weight for detecting models
                    reserved_weight = len(non_detecting_models) * non_detecting_weight
                    min_reserved_weight = len(detecting_models) * min_detecting_weight
                    available_weight = 1.0 - reserved_weight - min_reserved_weight
                    
                    if available_weight > 0:
                        total_detecting_conf = sum(track_model_contributions[m] for m in detecting_models)
                        for model_name in detecting_models:
                            model_conf = track_model_contributions[model_name]
                            
                            # Base weight (15% floor) + proportional share of remaining weight
                            base_weight = min_detecting_weight
                            if total_detecting_conf > 0:
                                conf_proportion = model_conf / total_detecting_conf
                                proportional_weight = available_weight * conf_proportion
                            else:
                                proportional_weight = available_weight / len(detecting_models)
                            
                            final_weight = base_weight + proportional_weight
                            track_ensemble_score += model_conf * final_weight
                    else:
                        # Fallback: just use minimum weights if not enough weight available
                        for model_name in detecting_models:
                            model_conf = track_model_contributions[model_name]
                            track_ensemble_score += model_conf * min_detecting_weight
                
                valid_models_for_track = [m for m, conf in track_model_contributions.items() if conf > 0]
                
                # Determine if this track passes ensemble threshold
                track_passed = track_ensemble_score >= config.confidence_threshold
                if track_passed:
                    frame_valid = True
                
                # Log ensemble result for this track
                if track_passed:
                    track_icon = "✅"
                    reason = "passed"
                elif len(valid_models_for_track) == 0:
                    track_icon = "❌"
                    reason = "no_valid_models"
                elif len(track_frame_detections) == 0:
                    track_icon = "❌"
                    reason = "no_detections"
                else:
                    track_icon = "❌"
                    reason = f"low_confidence ({track_ensemble_score:.3f}<{config.confidence_threshold})"
                
                logger.info(f"ENSEMBLE | {video_path.stem} | {timestamp:.2f}s | {frame_idx} | track_{track_id}")
                logger.info(f"{track_icon} | track_{track_id} | valid_models={len(valid_models_for_track)} | ensemble_score={track_ensemble_score:.3f} | detections={len(track_frame_detections)} | {reason}")
                
                track_ensemble_results.append({
                    'track_id': track_id,
                    'ensemble_score': track_ensemble_score,
                    'valid_models': len(valid_models_for_track),
                    'detections': len(track_frame_detections),
                    'passed': track_passed
                })
            
            # Log overall frame result
            frame_icon = "✅" if frame_valid else "❌"
            passed_tracks = [r for r in track_ensemble_results if r['passed']]
            frame_reason = f"{len(passed_tracks)}_tracks_passed" if frame_valid else "no_tracks_passed"
            
            logger.info(f"FRAME | {video_path.stem} | {timestamp:.2f}s | {frame_idx}")
            logger.info(f"{frame_icon} | frame_result | tracks_evaluated={len(track_ensemble_results)} | tracks_passed={len(passed_tracks)} | {frame_reason}")
            
            # Store frame results with track-based structure
            frame_results.append({
                'frame_idx': frame_idx,
                'track_results': track_ensemble_results,
                'frame_valid': frame_valid,
                'tracks_passed': len(passed_tracks),
                'tracks_evaluated': len(track_ensemble_results)
            })
        
        # PHASE 2: Track-Level Evaluation
        validated_results = []
        failed_tracks_data = []
        
        # Now collect all track statistics and evaluate against parameters to see if we have any viable tracks
        for track in extended_tracks:
            track_id = track['track_id']
            track_duration = track['duration_seconds']
            full_frame_detections = track_detections[track_id]
            
            # Calculate average confidence for model contributions for this track
            track_model_contributions = {}
            for source in model_contributions:
                if model_contributions[source]['count'] > 0:
                    track_model_contributions[source] = model_contributions[source].copy()
                    track_model_contributions[source]['avg_conf'] = model_contributions[source]['total_conf'] / model_contributions[source]['count']
            
            if full_frame_detections:
                # Calculate track-level statistics
                summed_confidence = sum(d['confidence'] for d in full_frame_detections)
                avg_confidence = summed_confidence / len(full_frame_detections)
                max_confidence = max(d['confidence'] for d in full_frame_detections)
                duration_normalized_score = summed_confidence / max(1.0, track_duration)
                combined_score = summed_confidence
                
                # Count successful frame evaluations for this track
                track_frames = track_sample_frames[track_id]
                passed_frames = 0
                total_frames_evaluated = len(track_frames)
                
                # Count frames where this specific track passed ensemble validation
                for fr in frame_results:
                    if fr['frame_idx'] in track_frames:
                        # Check if this specific track passed in this frame
                        track_result = next((tr for tr in fr['track_results'] if tr['track_id'] == track_id), None)
                        if track_result and track_result['passed']:
                            passed_frames += 1
                
                frame_success_rate = passed_frames / max(1, total_frames_evaluated)
                
                # Track validation criteria
                confidence_passed = combined_score >= config.confidence_threshold
                frames_passed = len(full_frame_detections) >= config.min_track_frames
                
                # Check minimum consecutive detection duration
                temporal_continuity_passed = True
                if passed_frames > 1:
                    # Get timestamps of frames that passed ensemble validation for this track
                    passed_frame_timestamps = []
                    for fr in frame_results:
                        if fr['frame_idx'] in track_frames:
                            track_result = next((tr for tr in fr['track_results'] if tr['track_id'] == track_id), None)
                            if track_result and track_result['passed']:
                                frame_timestamp = fr['frame_idx'] / fps
                                passed_frame_timestamps.append(frame_timestamp)
                    
                    # Find longest consecutive sequence of passing frames
                    if len(passed_frame_timestamps) > 1:
                        passed_frame_timestamps.sort()
                        max_consecutive_duration = 0.0
                        current_start = passed_frame_timestamps[0]
                        current_end = passed_frame_timestamps[0]
                        
                        for i in range(1, len(passed_frame_timestamps)):
                            # Check if this frame is consecutive (within reasonable sampling gap)
                            time_gap = passed_frame_timestamps[i] - passed_frame_timestamps[i-1]
                            max_expected_gap = track_duration / (len(track_frames) - 1) * 1.5  # Allow 50% tolerance
                            
                            if time_gap <= max_expected_gap:
                                # Consecutive - extend current sequence
                                current_end = passed_frame_timestamps[i]
                            else:
                                # Gap too large - end current sequence, start new one
                                consecutive_duration = current_end - current_start
                                max_consecutive_duration = max(max_consecutive_duration, consecutive_duration)
                                current_start = passed_frame_timestamps[i]
                                current_end = passed_frame_timestamps[i]
                        
                        # Check final sequence
                        consecutive_duration = current_end - current_start
                        max_consecutive_duration = max(max_consecutive_duration, consecutive_duration)
                        
                        # Validate minimum consecutive detection duration
                        temporal_continuity_passed = max_consecutive_duration >= config.min_consecutive_detection_seconds
                    else:
                        # Single frame passing - duration is 0
                        temporal_continuity_passed = config.min_consecutive_detection_seconds <= 0.0
                
                # Final track validation decision
                validation_passed = confidence_passed and frames_passed and temporal_continuity_passed
                
                # Find best detection for metadata
                best_detection = max(full_frame_detections, key=lambda d: d['confidence'])
                
                # Count contributing models
                total_models_with_detections = sum(1 for contrib in track_model_contributions.values() if contrib.get('spatial_valid_count', 0) > 0)
                
                # Log track evaluation
                logger.info(f"TRACK | {video_path.stem} | track_{track_id}")
                track_icon = "✅" if validation_passed else "❌"
                logger.info(f"{track_icon} | duration={track_duration:.2f}s | frames_evaluated={total_frames_evaluated} | frames_passed={passed_frames} | success_rate={frame_success_rate:.2f} | detections={len(full_frame_detections)} | models_active={total_models_with_detections} | summed_conf={summed_confidence:.3f} | avg_conf={avg_confidence:.3f} | max_conf={max_confidence:.3f} | duration_norm={duration_normalized_score:.3f} | conf_pass={confidence_passed} | frames_pass={frames_passed} | temporal_pass={temporal_continuity_passed} | validated={validation_passed}")
                
                if validation_passed:
                    validated_result = {
                        'track_id': track_id,
                        'best_detection': {
                            'frame_idx': best_detection.get('frame_idx', track_frames[0]),
                            'timestamp': best_detection.get('timestamp', track_frames[0] / fps),
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
                        'validation_frames': len(track_frames),
                        'duration_seconds': track['duration_seconds'],
                        'validation_passed': validation_passed,
                        'model_contributions': track_model_contributions,
                        'frame_success_rate': frame_success_rate
                    }
                    validated_results.append(validated_result)
                else:
                    # Capture failed track data for summary
                    failed_tracks_data.append({
                        'track_id': track_id,
                        'confidence': max_confidence,
                        'combined_score': combined_score,
                        'summed_confidence': summed_confidence,
                        'detections': len(full_frame_detections),
                        'frame_success_rate': frame_success_rate
                    })
            else:
                # Track evaluation for failed track
                logger.info(f"TRACK | {video_path.stem} | track_{track_id}")
                logger.info(f"❌ | duration={track_duration:.2f}s | frames=0 | detections=0 | models_active=0 | summed_conf=0.000 | avg_conf=0.000 | max_conf=0.000 | duration_norm=0.000 | validated=false")
                
                # Capture failed track data for summary
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
        """Sample representative frames from a motion track (new structure)."""
        bbox_track = track.get('bbox_track', track)  # Handle both old and new structure
        
        # For new extended track structure, only sample from motion frames
        if 'detections' in bbox_track:
            motion_detections = [det for det in bbox_track['detections'] if det['motion_detected']]
            frames = [det['frame'] for det in motion_detections]
        else:
            # Fallback for old structure
            frames = track.get('frames', [])
        
        if len(frames) <= max_frames:
            return frames
        
        # Sample evenly across the track duration 
        step = len(frames) / max_frames
        return [frames[int(i * step)] for i in range(max_frames)]
    
    def _calculate_bbox_overlap(self, motion_bbox: List[float], detection_bbox: List[float]) -> float:
        """Calculate IoU (Intersection over Union) between motion and detection bboxes.
        
        Args:
            motion_bbox: Motion region coordinates [x1, y1, x2, y2]  
            detection_bbox: ML detection coordinates [x1, y1, x2, y2]
            
        Returns:
            float: IoU score (0.0 to 1.0)
                  1.0 = perfect overlap
                  0.0 = no overlap
        """
        mx1, my1, mx2, my2 = motion_bbox
        dx1, dy1, dx2, dy2 = detection_bbox
        
        # Calculate intersection
        ix1 = max(mx1, dx1)
        iy1 = max(my1, dy1)
        ix2 = min(mx2, dx2)
        iy2 = min(my2, dy2)
        
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0  # No overlap
        
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        motion_area = (mx2 - mx1) * (my2 - my1)
        detection_area = (dx2 - dx1) * (dy2 - dy1)
        union_area = motion_area + detection_area - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        # IoU = intersection / union
        return intersection_area / union_area
    
    
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
            return self._build_tracks_with_deepsort(all_detections_by_frame, fps, total_frames)
        else:
            return self._build_tracks_simple_linking(all_detections_by_frame, fps, total_frames)
    
    def _build_tracks_with_deepsort(self, all_detections_by_frame: Dict, fps: float, total_frames: int) -> List[Dict]:
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
        
        # Convert to consistent bbox sequences format with full video extension
        consistent_sequences = []
        for track_id, track_data in tracks_by_id.items():
            if len(track_data['frames']) >= config.min_track_frames:
                duration = (max(track_data['frames']) - min(track_data['frames'])) / fps
                # Duration filtering happens after infilling
                # Motion detection boundaries
                motion_start_frame = min(track_data['frames'])
                motion_end_frame = max(track_data['frames'])
                first_known_position = track_data['bboxes'][0]  # First detected position
                last_known_position = track_data['bboxes'][-1]  # Last detected position
                
                # Create full video coverage detections
                full_detections = []
                
                # Backfill: frame 0 to motion_start-1 using first_known_position
                for frame_idx in range(0, motion_start_frame):
                    full_detections.append({
                        'frame': frame_idx,
                        'bbox': first_known_position.copy(),
                        'timestamp': frame_idx / fps,
                        'motion_detected': False,
                        'fill_type': 'backfill'
                    })
                
                # Actual motion detections
                for i, frame_idx in enumerate(track_data['frames']):
                    full_detections.append({
                        'frame': frame_idx,
                        'bbox': track_data['bboxes'][i].copy(),
                        'timestamp': track_data['timestamps'][i],
                        'motion_detected': True,
                        'fill_type': 'motion'
                        })
                
                # Forward-fill: motion_end+1 to video end using last_known_position
                for frame_idx in range(motion_end_frame + 1, total_frames):
                    full_detections.append({
                        'frame': frame_idx,
                        'bbox': last_known_position.copy(),
                        'timestamp': frame_idx / fps,
                        'motion_detected': False,
                        'fill_type': 'forward_fill'
                    })
                
                # Sort detections by frame
                full_detections.sort(key=lambda x: x['frame'])
                
                bbox_track = {
                    'track_id': track_id,
                    'detections': full_detections,
                    'start_frame': 0,  # Always video start
                    'end_frame': total_frames - 1,  # Always video end
                    'motion_start_frame': motion_start_frame,
                    'motion_end_frame': motion_end_frame,
                    'first_known_position': first_known_position,
                    'last_known_position': last_known_position
                }
                
                consistent_sequences.append({
                    'bbox_track': bbox_track,
                    'duration_seconds': duration,
                    'detection_count': len(track_data['frames']),
                    'total_coverage_frames': len(full_detections),
                    'motion_frames': len(track_data['frames']),
                    'backfill_frames': motion_start_frame,
                    'forward_fill_frames': total_frames - motion_end_frame - 1,
                    'avg_confidence': sum(track_data['confidences']) / len(track_data['confidences']),
                    'max_confidence': max(track_data['confidences']),
                    'tracking_method': 'deepsort_extended'
                })
        
        logger.info(f"🎯 DeepSORT generated {len(consistent_sequences)} valid tracks from {len(tracks_by_id)} total tracks")
        return consistent_sequences
    
    def _build_tracks_simple_linking(self, all_detections_by_frame: Dict, fps: float, total_frames: int) -> List[Dict]:
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
                if len(track['frames']) >= config.min_track_frames:
                    # Duration filtering happens after infilling
                    # Create extended track structure with backfill/forward-fill
                    motion_start_frame = track['start_frame']
                    motion_end_frame = track['end_frame']
                    first_known_position = track['bboxes'][0]
                    last_known_position = track['bboxes'][-1]
                    
                    # Create full video coverage detections
                    full_detections = []
                    
                    # Backfill: frame 0 to motion_start-1
                    for frame_idx in range(0, motion_start_frame):
                        full_detections.append({
                            'frame': frame_idx,
                            'bbox': first_known_position.copy(),
                            'timestamp': frame_idx / fps,
                            'motion_detected': False,
                            'fill_type': 'backfill'
                        })
                    
                    # Actual motion detections
                    for i, frame_idx in enumerate(track['frames']):
                        full_detections.append({
                            'frame': frame_idx,
                            'bbox': track['bboxes'][i].copy(),
                            'timestamp': frame_idx / fps,
                            'motion_detected': True,
                            'fill_type': 'motion'
                        })
                    
                    # Forward-fill: motion_end+1 to video end
                    for frame_idx in range(motion_end_frame + 1, total_frames):
                        full_detections.append({
                            'frame': frame_idx,
                            'bbox': last_known_position.copy(),
                            'timestamp': frame_idx / fps,
                            'motion_detected': False,
                            'fill_type': 'forward_fill'
                        })
                    
                    # Sort detections by frame
                    full_detections.sort(key=lambda x: x['frame'])
                    
                    extended_track = {
                        'track_id': track['track_id'],
                        'detections': full_detections,
                        'start_frame': 0,
                        'end_frame': total_frames - 1,
                        'motion_start_frame': motion_start_frame,
                        'motion_end_frame': motion_end_frame,
                        'first_known_position': first_known_position,
                        'last_known_position': last_known_position
                    }
                    
                    tracks.append({
                        'bbox_track': extended_track,
                        'duration_seconds': duration,
                        'detection_count': len(track['frames']),
                        'total_coverage_frames': len(full_detections),
                        'motion_frames': len(track['frames']),
                        'backfill_frames': motion_start_frame,
                        'forward_fill_frames': total_frames - motion_end_frame - 1,
                        'avg_confidence': sum(track['confidences']) / len(track['confidences']),
                        'max_confidence': max(track['confidences']),
                        'tracking_method': 'simple_linking_extended'
                    })
                
                track_id_counter += 1
        
        logger.info(f"🔗 Simple linking generated {len(tracks)} valid tracks")
        return tracks
    
    def process_video_with_features(self, video_path: Path) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Process video with direct pipeline: motion tracking -> full-frame analysis."""
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
        
        # No crop saving in direct pipeline
        
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
                import traceback
                logger.error(f"VIDEO ERROR: {video_path.name} - {str(e)}")
                logger.error(f"❌ {video_path.name}: Processing failed - {str(e)}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
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
                    
                    logger.info(f"  ⚪ {video_name}: motion_score={composite_score}, conf={conf}, combined={combined}, tracks={tracks}, validated={validated}, reason={rejection_reason}, runtime={runtime_str}")
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
            if track:  # Duration filtering happens after infilling
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
        
        # Camera handling detection
        composite_motion_threshold=args.composite_motion_threshold,
        min_motion_threshold=args.min_motion_threshold,
        motion_frames_weight=args.motion_frames_weight,
        motion_regions_weight=args.motion_regions_weight,
        motion_tracks_weight=args.motion_tracks_weight,
        large_region_multiplier=args.large_region_multiplier,
        
        # Motion detection
        motion_method=args.motion_method,
        motion_var_threshold=args.motion_var_threshold,
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
        min_consecutive_detection_seconds=args.min_consecutive_detection_seconds,
        tracking_distance_threshold=args.tracking_distance_threshold,
        anchor_confidence_threshold=args.anchor_confidence_threshold,
        min_track_frames=args.min_track_frames,
        
        # Step 3 validation
        max_validation_frames=args.max_validation_frames,
        temporal_spread_seconds=args.temporal_spread_seconds,
        spatial_overlap_threshold=args.spatial_overlap_threshold,
        
        # Track infilling parameters
        enable_track_infilling=args.enable_track_infilling,
        infill_max_gap_seconds=args.infill_max_gap_seconds,
        infill_max_distance_pixels=args.infill_max_distance_pixels,
        infill_min_overlap_ratio=args.infill_min_overlap_ratio,
        
        # Debug parameters
        debug_show_spatially_invalid=args.debug_show_spatially_invalid,
        
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
        print(f"🕒 Temporal parameters: {args.min_track_duration}s duration, motion gap {args.motion_tracking_gap_seconds}s, min consecutive detection {args.min_consecutive_detection_seconds}s")
        
        logger.info(f"🎯 Processing strategy: Next Generation Temporal Consistency")
        logger.info(f"🕒 Min track duration: {args.min_track_duration}s")
        logger.info(f"✅ Full-frame validation frames: {args.full_frame_validation_frames}")
        
        processor.process_all_videos(video_filter=video_filter)
        
    except KeyboardInterrupt:
        print("🛑 Processing interrupted by user")
    except Exception as e:
        import traceback
        print(f"❌ Processing failed: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()