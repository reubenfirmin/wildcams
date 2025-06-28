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
Motion detection wildlife video processor.
Uses motion detection to identify regions of interest before running ML models,
significantly improving performance and accuracy for camera trap footage.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# Import base processor
from video_processor_base import VideoProcessorBase

# Get loggers
logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger('analysis')

class MotionDetectionVideoProcessor(VideoProcessorBase):
    """Motion detection video processor using crop-based ML ensemble."""
    
    def __init__(self):
        super().__init__()
        
        # Motion detection configuration
        self.motion_config = {
            'method': os.getenv('MOTION_METHOD', 'MOG2'),  # 'MOG2', 'KNN', or 'frame_diff'
            'var_threshold': int(os.getenv('MOTION_VAR_THRESHOLD', '16')),
            'history': int(os.getenv('MOTION_HISTORY', '20')),
            'detect_shadows': os.getenv('MOTION_DETECT_SHADOWS', 'True').lower() == 'true',
            'min_motion_area': int(os.getenv('MIN_MOTION_AREA', '500')),
            'max_motion_area': int(os.getenv('MAX_MOTION_AREA', '100000')),
            'bbox_padding': float(os.getenv('MOTION_BBOX_PADDING', '0.2')),
            'min_fill_ratio': float(os.getenv('MIN_FILL_RATIO', '0.3')),
            'min_persistence': int(os.getenv('MOTION_MIN_PERSISTENCE', '3')),
            'motion_history_length': int(os.getenv('MOTION_HISTORY_LENGTH', '5')),
            'max_aspect_ratio': float(os.getenv('MAX_ASPECT_RATIO', '5.0')),
            'vegetation_zone_height': float(os.getenv('VEGETATION_ZONE_HEIGHT', '0.3')),
            'min_region_confidence': float(os.getenv('MIN_REGION_CONFIDENCE', '0.3')),
        }
        
        # Motion detection state
        self.motion_history = []
        self.previous_frame = None
        self.bg_subtractor = None
        
        # Initialize motion detector
        self.init_motion_detector()
        
        logger.info("🎯 Motion detection video processor initialized")
    
    def init_motion_detector(self):
        """Initialize motion detection background subtractor."""
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
        else:  # frame_diff
            self.bg_subtractor = None  # Will use frame differencing
        
        logger.info(f"🎯 Motion detector initialized: {self.motion_config['method']}")
    
    def detect_motion_regions(self, frame, frame_idx):
        """Detect regions with motion in the current frame."""
        motion_regions = []
        
        try:
            if self.motion_config['method'] == 'frame_diff':
                motion_mask = self._frame_difference_motion(frame)
            else:
                motion_mask = self._background_subtraction_motion(frame)
            
            if motion_mask is None:
                return motion_regions
            
            # Extract motion regions from mask
            raw_regions = self._extract_motion_regions(motion_mask)
            
            # Apply intelligent filtering
            filtered_regions = self._filter_motion_regions(raw_regions, frame.shape)
            
            # Track across frames for temporal consistency
            consistent_regions = self._track_motion_consistency(filtered_regions)
            
            # Add to motion history
            self.motion_history.append(consistent_regions)
            if len(self.motion_history) > self.motion_config['motion_history_length']:
                self.motion_history.pop(0)
            
            analysis_logger.info(f"Frame {frame_idx}: {len(raw_regions)} raw, {len(filtered_regions)} filtered, {len(consistent_regions)} consistent motion regions")
            
            return consistent_regions
            
        except Exception as e:
            analysis_logger.error(f"Motion detection failed for frame {frame_idx}: {e}")
            return motion_regions
    
    def _frame_difference_motion(self, frame):
        """Detect motion using frame differencing."""
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None
            
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.previous_frame, current_gray)
        _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        
        self.previous_frame = current_gray
        return motion_mask
    
    def _background_subtraction_motion(self, frame):
        """Detect motion using background subtraction."""
        if self.bg_subtractor is None:
            self.init_motion_detector()
            
        motion_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        
        return motion_mask
    
    def _extract_motion_regions(self, motion_mask):
        """Extract bounding boxes from motion mask."""
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.motion_config['min_motion_area'] <= area <= self.motion_config['max_motion_area']:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Expand box for context
                padding = self.motion_config['bbox_padding']
                x_pad = int(w * padding)
                y_pad = int(h * padding)
                
                motion_regions.append({
                    'bbox': [max(0, x-x_pad), max(0, y-y_pad), x+w+x_pad, y+h+y_pad],
                    'area': area,
                    'confidence': area / (w * h),  # Fill ratio
                    'contour_area': area,
                    'bbox_area': w * h
                })
        
        return motion_regions
    
    def _filter_motion_regions(self, regions, frame_shape):
        """Apply intelligent filtering to motion regions."""
        filtered = []
        
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            w, h = x2 - x1, y2 - y1
            
            if w <= 0 or h <= 0:
                continue
            
            # Filter by aspect ratio (remove vegetation sway)
            aspect_ratio = w / h
            if aspect_ratio > self.motion_config['max_aspect_ratio'] or aspect_ratio < (1/self.motion_config['max_aspect_ratio']):
                continue
            
            # Filter by position (skip vegetation zone)
            if y1 < frame_shape[0] * self.motion_config['vegetation_zone_height']:
                continue
            
            # Filter by confidence
            if region['confidence'] < self.motion_config['min_region_confidence']:
                continue
            
            filtered.append(region)
        
        return filtered
    
    def _track_motion_consistency(self, regions):
        """Track motion consistency across frames."""
        if len(self.motion_history) < self.motion_config['min_persistence']:
            return regions
        
        consistent = []
        for region in regions:
            persistence_count = 1  # Current frame
            
            # Check overlap with previous frames
            for past_frame_regions in self.motion_history[-self.motion_config['min_persistence']+1:]:
                for past_region in past_frame_regions:
                    if self._regions_overlap(region, past_region):
                        persistence_count += 1
                        break
            
            if persistence_count >= self.motion_config['min_persistence']:
                consistent.append(region)
        
        return consistent
    
    def _regions_overlap(self, region1, region2, threshold=0.3):
        """Check if two regions overlap significantly."""
        x1_1, y1_1, x2_1, y2_1 = region1['bbox']
        x1_2, y1_2, x2_2, y2_2 = region2['bbox']
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return False
        
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        union_area = region1['area'] + region2['area'] - intersection_area
        
        return (intersection_area / union_area) > threshold
    
    def enhanced_frame_analysis(self, frame: np.ndarray, frame_idx: int, video_debug_dir: Path, timestamp_seconds: float = None) -> List[Dict]:
        """Analyze a single frame with motion detection pre-filtering for focused ML processing."""
        analysis_logger.info(f"--- FRAME {frame_idx} ANALYSIS START ---")
        analysis_logger.info(f"PREPROCESSING_MODE: MOTION_DETECTION (process2.py)")
        analysis_logger.info(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        if timestamp_seconds is not None:
            analysis_logger.info(f"Video timestamp: {timestamp_seconds:.2f}s")
        
        detections = []
        
        # Step 1: Detect motion regions
        analysis_logger.info(f"MOTION_STEP_1: Detecting motion regions in frame {frame_idx}")
        motion_regions = self.detect_motion_regions(frame, frame_idx)
        
        if not motion_regions:
            # No significant motion - skip expensive ML processing but save debug frame
            analysis_logger.info(f"MOTION_RESULT: Frame {frame_idx} - No significant motion detected, skipping ML processing")
            
            # Save debug frame showing no motion
            debug_frame_path = video_debug_dir / f"frame_{frame_idx:04d}_no_motion.jpg"
            cv2.imwrite(str(debug_frame_path), frame)
            
            analysis_logger.info(f"--- FRAME {frame_idx} ANALYSIS END: 0 total detections ---")
            return detections
        
        # Step 2: Process each motion region with ML ensemble
        analysis_logger.info(f"MOTION_STEP_2: Processing {len(motion_regions)} motion regions with ML ensemble")
        total_region_detections = 0
        for i, region in enumerate(motion_regions):
            analysis_logger.info(f"MOTION_REGION: Frame {frame_idx} Region {i+1}/{len(motion_regions)} - area={region['area']}, confidence={region['confidence']:.3f}")
            
            # Extract and validate crop
            x1, y1, x2, y2 = region['bbox']
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                analysis_logger.warning(f"Invalid crop region: {region['bbox']}")
                continue
                
            cropped_frame = frame[y1:y2, x1:x2]
            
            if cropped_frame.size == 0:
                analysis_logger.warning(f"Empty crop for region {i+1}")
                continue
            
            # Save debug crop
            crop_debug_path = video_debug_dir / f"frame_{frame_idx:04d}_region_{i+1:02d}_crop.jpg"
            cv2.imwrite(str(crop_debug_path), cropped_frame)
            
            # Run ML ensemble on cropped region
            analysis_logger.info(f"CROP_ML: Running ML ensemble on motion crop {i+1}")
            region_detections = self.run_ml_ensemble_on_crop(cropped_frame, region['bbox'], frame_idx, i+1, timestamp_seconds)
            
            # Scale detections back to original frame coordinates
            analysis_logger.info(f"CROP_SCALE: Scaling {len(region_detections)} detections back to original coordinates")
            scaled_detections = self.scale_detections_to_original(region_detections, region['bbox'])
            detections.extend(scaled_detections)
            total_region_detections += len(scaled_detections)
        
        # Save debug frame with motion regions and detections
        self.save_debug_frame_with_annotations(frame, motion_regions, detections, video_debug_dir, frame_idx)
        
        analysis_logger.info(f"--- FRAME {frame_idx} ANALYSIS END: {total_region_detections} total detections from {len(motion_regions)} motion regions ---")
        return detections
    
    def run_ml_ensemble_on_crop(self, cropped_frame: np.ndarray, original_bbox: List[int], frame_idx: int, region_idx: int, timestamp_seconds: float = None) -> List[Dict]:
        """Run the 5-model ML ensemble on a cropped motion region using shared module."""
        analysis_logger.info(f"CROP_ENSEMBLE: Frame {frame_idx} Region {region_idx} - Running ML ensemble on crop {cropped_frame.shape}")
        
        # Use shared ML ensemble for consistent detection across process.py and process2.py
        detections = self.ml_ensemble.run_ensemble_detection(
            cropped_frame, timestamp_seconds, frame_idx
        )
        
        # Update source labels to indicate crop processing
        for detection in detections:
            original_source = detection.get('source', 'unknown')
            detection['source'] = f"{original_source}_crop"
            detection['region_idx'] = region_idx
        
        analysis_logger.info(f"CROP_RESULT: Frame {frame_idx} Region {region_idx} - ML ensemble found {len(detections)} detections on crop")
        return detections
    
    def scale_detections_to_original(self, detections: List[Dict], motion_bbox: List[int]) -> List[Dict]:
        """Scale detection coordinates from crop back to original frame coordinates."""
        x_offset, y_offset = motion_bbox[0], motion_bbox[1]
        
        scaled_detections = []
        for detection in detections:
            scaled_detection = detection.copy()
            
            # Scale bbox coordinates
            crop_bbox = detection['bbox']
            original_bbox = [
                crop_bbox[0] + x_offset,  # x1
                crop_bbox[1] + y_offset,  # y1
                crop_bbox[2] + x_offset,  # x2
                crop_bbox[3] + y_offset   # y2
            ]
            scaled_detection['bbox'] = original_bbox
            scaled_detection['motion_region_bbox'] = motion_bbox
            
            scaled_detections.append(scaled_detection)
        
        return scaled_detections
    
    def save_debug_frame_with_annotations(self, frame, motion_regions, detections, video_debug_dir, frame_idx):
        """Save debug frame with motion regions and detections annotated."""
        debug_frame = frame.copy()
        
        # Draw motion regions in blue
        for region in motion_regions:
            x1, y1, x2, y2 = [int(coord) for coord in region['bbox']]
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(debug_frame, f"Motion", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw detections in green
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            confidence = detection['confidence']
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"{confidence:.3f}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save annotated frame
        debug_path = video_debug_dir / f"frame_{frame_idx:04d}_annotated.jpg"
        cv2.imwrite(str(debug_path), debug_frame)
    
    def process_video_with_features(self, video_path: Path) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Process a single video with motion detection and extract features from the best detection."""
        analysis_logger.info(f"=== VIDEO PROCESSING START: {video_path.name} ===")
        
        # Reset motion detection state for each video
        self.motion_history = []
        self.previous_frame = None
        if self.bg_subtractor is not None:
            self.init_motion_detector()  # Reset background model
        
        # Extract frames from video
        frames, timestamps = self.extract_frames(video_path)
        if not frames:
            analysis_logger.error(f"No frames extracted from {video_path.name}")
            return None, None
        
        # Create debug directory for this video
        video_debug_dir = self.debug_dir / video_path.stem
        video_debug_dir.mkdir(exist_ok=True)
        
        # Process all frames and collect detections
        all_detections = []
        frames_processed = 0
        
        analysis_logger.info("=== MOTION-BASED ANIMAL DETECTION START ===")
        analysis_logger.info(f"Processing {len(frames)} frames with motion detection pre-filtering")
        
        for frame_idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            frame_detections = self.enhanced_frame_analysis(
                frame, frame_idx, video_debug_dir, timestamp
            )
            
            # Add frame index to each detection for scoring
            for detection in frame_detections:
                detection['frame_idx'] = frame_idx
                detection['timestamp'] = timestamp
            
            all_detections.extend(frame_detections)
            frames_processed += 1
        
        analysis_logger.info("=== MOTION-BASED ANIMAL DETECTION END ===")
        
        # Check for camera handling
        if self.detect_camera_handling(all_detections, frames_processed):
            analysis_logger.info("VIDEO RESULT: Rejected as camera handling")
            self.mark_as_processed(video_path)
            return None, None
        
        # Validate using ensemble logic
        if not self.ensemble_validation(all_detections):
            analysis_logger.info("VIDEO RESULT: Insufficient evidence for animal presence")
            self.mark_as_processed(video_path)
            return None, None
        
        # Find best detection for feature extraction
        best_detection, total_scored = self.score_and_find_best_detection(all_detections)
        
        if best_detection is None:
            analysis_logger.info("VIDEO RESULT: No suitable detection found for feature extraction")
            self.mark_as_processed(video_path)
            return None, None
        
        # Extract features from best detection
        best_frame_idx = best_detection['frame_idx']
        best_frame = frames[best_frame_idx]
        
        features = self.extract_features_from_detection(best_frame, best_detection)
        
        # Calculate detection statistics for the area ratio
        bbox = best_detection['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = best_frame.shape[0] * best_frame.shape[1]
        area_ratio = area / frame_area
        
        # Build analysis result
        analysis = {
            'video_path': video_path,
            'animals_detected': ['animal'],  # Generic classification
            'detection_count': len(all_detections),
            'frames_processed': frames_processed,
            'best_detection_frame': best_frame_idx,
            'best_detection_timestamp': best_detection['timestamp'],
            'detection': {
                'confidence': best_detection['confidence'],
                'bbox': best_detection['bbox'],
                'area_ratio': area_ratio,
                'source': best_detection.get('source', 'unknown')
            },
            'processing_mode': 'motion_detection'
        }
        
        analysis_logger.info(f"=== VIDEO PROCESSING END: {video_path.name} - SUCCESS ===")
        self.mark_as_processed(video_path)
        
        return analysis, features
    
    def process_all_videos(self, video_filter=None):
        """Process all videos using motion detection approach."""
        analysis_logger.info("###############################################")
        analysis_logger.info("BATCH PROCESSING SESSION START")
        analysis_logger.info("###############################################")
        
        # Get videos to process (handles filter logic)
        videos_to_process = self.get_filtered_videos(video_filter)
        
        if not videos_to_process:
            if video_filter:
                analysis_logger.info(f"BATCH RESULT: No videos found matching filter: {video_filter}")
                logger.info(f"⚠️ No videos found matching filter: {video_filter}")
            else:
                analysis_logger.info("BATCH RESULT: No unprocessed videos found")
                logger.info("✅ No unprocessed videos found")
            return
        
        analysis_logger.info(f"Videos to process: {[v.name for v in videos_to_process]}")
        logger.info(f"🎬 Found {len(videos_to_process)} videos to process")
        
        # Clear previous session data
        self.all_features = []
        self.video_metadata = []
        all_analyses = []
        
        # Process each video
        for video_path in tqdm(videos_to_process, desc="Processing videos"):
            try:
                analysis, features = self.process_video_with_features(video_path)
                if analysis:
                    analysis_logger.info(f"VIDEO SUCCESS: {video_path.name} - Animal detected")
                    all_analyses.append(analysis)
                    self.save_analysis(analysis, video_path)
                    
                    if features is not None:
                        self.all_features.append(features)
                        self.video_metadata.append(analysis)
                        analysis_logger.info(f"Features extracted: {len(features)} dimensions")
                    
                    logger.info(f"✅ Successfully processed {video_path.name}")
                else:
                    analysis_logger.info(f"VIDEO SKIPPED: {video_path.name} - No animals detected or camera handling")
                    logger.info(f"⏭️ Skipped {video_path.name} (no animals detected)")
                    
            except Exception as e:
                analysis_logger.error(f"VIDEO ERROR: {video_path.name} - {e}")
                logger.error(f"❌ Failed to process {video_path.name}: {e}")
        
        # Perform clustering if we have features
        if self.all_features:
            analysis_logger.info("=== CLUSTERING START ===")
            logger.info(f"🧬 Performing clustering analysis on {len(self.all_features)} videos...")
            
            clusters = self.cluster_animal_videos(self.video_metadata)
            self.save_clustering_results(clusters)
            
            analysis_logger.info(f"=== CLUSTERING END: Found {len(clusters)} clusters ===")
            logger.info(f"🎯 Clustering complete: {len(clusters)} animal groups identified")
        else:
            clusters = {}
            analysis_logger.info("No clustering performed - no valid animal features found")
            logger.warning("⚠️ No videos with valid animal features found for clustering")
        
        # Generate summary report
        self.generate_summary_report(all_analyses, clusters)
        
        analysis_logger.info("###############################################")
        analysis_logger.info("BATCH PROCESSING SESSION END")
        analysis_logger.info("###############################################")
        logger.info(f"🎉 Processing complete! Analyzed {len(all_analyses)} videos with animals")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Motion detection wildlife video processor')
    parser.add_argument('--videos', '-v', nargs='+', help='Optional list of video indices (e.g. 7 8 9) or names (e.g. IMG_0007.MP4) to process')
    args = parser.parse_args()
    
    # Convert video arguments to appropriate format
    video_filter = None
    if args.videos:
        video_filter = []
        for video in args.videos:
            try:
                # Try to parse as integer first
                video_filter.append(int(video))
            except ValueError:
                # If not an integer, treat as string
                video_filter.append(video)
    
    try:
        processor = MotionDetectionVideoProcessor()
        processor.process_all_videos(video_filter=video_filter)
    except KeyboardInterrupt:
        logger.info("🛑 Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()