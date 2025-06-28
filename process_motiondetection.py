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
Uses real-time motion detection to focus ML analysis on frames/regions with movement.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Import base processor
from video_processor_base import VideoProcessorBase

# Get loggers
logger = logging.getLogger('wildcams')
analysis_logger = logger

class MotionDetectionVideoProcessor(VideoProcessorBase):
    """Motion detection video processor that streams through video frames sequentially."""
    
    def __init__(self):
        super().__init__()
        
        # Motion detection configuration - tuned for camera trap videos
        self.motion_config = {
            'method': os.getenv('MOTION_METHOD', 'MOG2'),
            'var_threshold': int(os.getenv('MOTION_VAR_THRESHOLD', '32')),
            'min_area': int(os.getenv('MIN_MOTION_AREA', '2000')),
            'max_area': int(os.getenv('MAX_MOTION_AREA', '80000')),
            'detect_shadows': True,
            'history': int(os.getenv('MOTION_HISTORY', '100')),
            'max_regions_per_frame': int(os.getenv('MAX_REGIONS_PER_FRAME', '10')),
            'min_region_width': int(os.getenv('MIN_REGION_WIDTH', '30')),
            'min_region_height': int(os.getenv('MIN_REGION_HEIGHT', '30')),
            'max_aspect_ratio': float(os.getenv('MAX_ASPECT_RATIO', '5.0')),
            'motion_margin': int(os.getenv('MOTION_MARGIN', '30'))
        }
        
        # Initialize motion detection algorithm
        self.bg_subtractor = None
        self.init_motion_detector()
        
        logger.info(f"🎯 Motion detection video processor initialized")
        logger.info(f"🔍 Motion method: {self.motion_config['method']}")
        logger.info(f"🎚️ Motion thresholds: area {self.motion_config['min_area']}-{self.motion_config['max_area']}, variance {self.motion_config['var_threshold']}")
    
    def init_motion_detector(self):
        """Initialize motion detection algorithm."""
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
        
        analysis_logger.info(f"Motion detector initialized: {self.motion_config['method']}")
    
    def _open_video_stream(self, video_path: Path) -> Optional[cv2.VideoCapture]:
        """Open video stream with fallback backends."""
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(str(video_path), backend)
            if cap.isOpened():
                return cap
            cap.release()
        
        logger.error(f"❌ Could not open video with any backend: {video_path}")
        return None
    
    def _get_frame_at_index(self, video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame from the video."""
        cap = self._open_video_stream(video_path)
        if cap is None:
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    def detect_motion_regions(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """Detect significant motion regions suitable for camera trap wildlife."""
        if self.bg_subtractor is None:
            return []
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # More aggressive morphological operations to reduce noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove small noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small)
        # Connect nearby regions 
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)
        # Final erosion to remove remaining noise
        fg_mask = cv2.erode(fg_mask, kernel_small, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first) and limit to max regions
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[:self.motion_config['max_regions_per_frame']]
        
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - focus on significant motion only
            if area < self.motion_config['min_area'] or area > self.motion_config['max_area']:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Reject very thin or small regions (likely noise)
            if (w < self.motion_config['min_region_width'] or 
                h < self.motion_config['min_region_height'] or 
                w/h > self.motion_config['max_aspect_ratio'] or 
                h/w > self.motion_config['max_aspect_ratio']):
                continue
            
            # Expand bounding box for better ML detection context
            margin = self.motion_config['motion_margin']
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)
            
            motion_region = {
                'bbox': [x, y, x + w, y + h],
                'area': area,
                'contour': contour,
                'frame_idx': frame_idx
            }
            motion_regions.append(motion_region)
            
            logger.debug(f"Frame {frame_idx}: Motion region area={area}, bbox=[{x},{y},{x+w},{y+h}]")
        
        return motion_regions
    
    def run_ml_ensemble_on_crop(self, frame: np.ndarray, motion_region: Dict, frame_idx: int, region_idx: int, timestamp: float) -> List[Dict]:
        """Run ML ensemble on a motion-detected crop."""
        bbox = motion_region['bbox']
        x1, y1, x2, y2 = bbox
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        
        analysis_logger.info(f"Frame {frame_idx}, Region {region_idx}: Running ML ensemble on crop {crop.shape}")
        logger.info(f"🔬 Analyzing motion region {region_idx+1} ({crop.shape[1]}x{crop.shape[0]}px)")
        
        # Run ensemble detection on the crop
        crop_detections = self.ml_ensemble.run_ensemble_detection(crop, timestamp, frame_idx)
        
        # Convert crop-relative coordinates to frame-absolute coordinates
        frame_detections = []
        for detection in crop_detections:
            crop_bbox = detection['bbox']
            # Convert crop coordinates to frame coordinates
            frame_bbox = [
                crop_bbox[0] + x1,  # x1
                crop_bbox[1] + y1,  # y1
                crop_bbox[2] + x1,  # x2
                crop_bbox[3] + y1   # y2
            ]
            
            frame_detection = detection.copy()
            frame_detection['bbox'] = frame_bbox
            frame_detection['motion_region'] = region_idx
            frame_detection['motion_area'] = motion_region['area']
            
            frame_detections.append(frame_detection)
            
            analysis_logger.info(f"Motion crop detection: conf={detection['confidence']:.4f}, source={detection.get('source', 'unknown')}")
        
        logger.info(f"✅ Region {region_idx+1}: {len(frame_detections)} detections found")
        return frame_detections
    
    def _save_debug_frame_with_regions(self, frame: np.ndarray, motion_regions: List[Dict], detections: List[Dict], output_path: Path):
        """Save debug frame with motion regions and detections highlighted."""
        debug_frame = frame.copy()
        
        # Draw motion regions in blue
        for region in motion_regions:
            bbox = region['bbox']
            cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(debug_frame, f"Motion: {region['area']}", (bbox[0], bbox[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw animal detections in green
        for detection in detections:
            bbox = detection['bbox']
            cv2.rectangle(debug_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Animal: {detection['confidence']:.2f}", 
                       (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(str(output_path), debug_frame)
    
    def process_video_with_features(self, video_path: Path) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Process a single video using real-time motion detection streaming."""
        analysis_logger.info(f"=== VIDEO PROCESSING START: {video_path.name} ===")
        analysis_logger.info(f"PREPROCESSING_MODE: MOTION_DETECTION (streaming)")
        
        # Open video stream
        cap = self._open_video_stream(video_path)
        if cap is None:
            return None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"🎬 Processing {video_path.name} ({duration:.1f}s, {total_frames} frames)")
        logger.info(f"🎯 Streaming motion detection - analyzing every {self.frame_skip} frames")
        
        # Create debug directory
        video_debug_dir = self.debug_dir / video_path.stem
        video_debug_dir.mkdir(exist_ok=True)
        
        # Reset motion detection for this video
        self.init_motion_detector()
        
        # Process video frame by frame
        all_detections = []
        frames_processed = 0
        frames_with_motion = 0
        frames_with_detections = 0
        saved_debug_frames = 0
        
        analysis_logger.info("=== MOTION DETECTION STREAMING START ===")
        
        frame_idx = 0
        progress_bar = tqdm(total=total_frames//self.frame_skip, desc="Motion Detection", unit="frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for performance
                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue
                
                timestamp_seconds = frame_idx / fps if fps > 0 else 0
                frames_processed += 1
                progress_bar.update(1)
                
                # Detect motion regions
                motion_regions = self.detect_motion_regions(frame, frame_idx)
                
                if not motion_regions:
                    analysis_logger.debug(f"Frame {frame_idx}: No motion detected")
                    frame_idx += 1
                    continue
                
                frames_with_motion += 1
                analysis_logger.info(f"Frame {frame_idx}: Motion detected - {len(motion_regions)} regions")
                
                # Run ML analysis on motion regions
                frame_detections = []
                for region_idx, region in enumerate(motion_regions):
                    crop_detections = self.run_ml_ensemble_on_crop(
                        frame, region, frame_idx, region_idx, timestamp_seconds
                    )
                    frame_detections.extend(crop_detections)
                
                if frame_detections:
                    frames_with_detections += 1
                    
                    # Save debug frame for first few detections
                    if saved_debug_frames < 10:
                        debug_path = video_debug_dir / f"detection_frame_{frame_idx:04d}.jpg"
                        self._save_debug_frame_with_regions(frame, motion_regions, frame_detections, debug_path)
                        saved_debug_frames += 1
                
                # Add metadata to detections
                for detection in frame_detections:
                    detection['frame_idx'] = frame_idx
                    detection['timestamp'] = timestamp_seconds
                
                all_detections.extend(frame_detections)
                frame_idx += 1
                
        finally:
            progress_bar.close()
            cap.release()
        
        analysis_logger.info("=== MOTION DETECTION STREAMING END ===")
        logger.info(f"📊 Motion summary: {frames_with_motion}/{frames_processed} frames had motion")
        logger.info(f"📊 Detection summary: {frames_with_detections} frames had animal detections")
        logger.info(f"🛠️ Debug frames saved: {saved_debug_frames} to {video_debug_dir}")
        
        # Validate detections
        if self.detect_camera_handling(all_detections, frames_processed):
            analysis_logger.info("VIDEO RESULT: Rejected as camera handling")
            logger.info(f"⏭️ Skipped {video_path.name} (camera handling detected)")
            self.mark_as_processed(video_path)
            return None, None
        
        if not self.ensemble_validation(all_detections):
            analysis_logger.info("VIDEO RESULT: Insufficient evidence for animal presence")
            logger.info(f"⏭️ Skipped {video_path.name} (no animals detected)")
            self.mark_as_processed(video_path)
            return None, None
        
        # Find best detection
        best_detection, total_scored = self.score_and_find_best_detection(all_detections)
        if best_detection is None:
            analysis_logger.info("VIDEO RESULT: No suitable detection for feature extraction")
            self.mark_as_processed(video_path)
            return None, None
        
        # Get best frame and extract features
        best_frame = self._get_frame_at_index(video_path, best_detection['frame_idx'])
        if best_frame is None:
            analysis_logger.error("Failed to retrieve best frame")
            self.mark_as_processed(video_path)
            return None, None
        
        features = self.extract_features_from_detection(best_frame, best_detection)
        
        # Build analysis result
        bbox = best_detection['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = best_frame.shape[0] * best_frame.shape[1]
        area_ratio = area / frame_area
        
        analysis = {
            'video_path': video_path,
            'animals_detected': ['animal'],
            'detection_count': len(all_detections),
            'frames_processed': frames_processed,
            'frames_with_motion': frames_with_motion,
            'frames_with_detections': frames_with_detections,
            'motion_efficiency': frames_with_motion / frames_processed if frames_processed > 0 else 0,
            'best_detection_frame': best_detection['frame_idx'],
            'best_detection_timestamp': best_detection['timestamp'],
            'detection': {
                'confidence': best_detection['confidence'],
                'bbox': best_detection['bbox'],
                'area_ratio': area_ratio,
                'source': best_detection.get('source', 'unknown'),
                'motion_area': best_detection.get('motion_area', 0)
            },
            'processing_mode': 'motion_detection'
        }
        
        analysis_logger.info(f"=== VIDEO PROCESSING END: {video_path.name} - SUCCESS ===")
        logger.info(f"✅ Successfully processed {video_path.name}")
        self.mark_as_processed(video_path)
        
        return analysis, features
    
    def process_all_videos(self, video_filter=None):
        """Process all videos using motion detection approach."""
        analysis_logger.info("###############################################")
        analysis_logger.info("BATCH PROCESSING SESSION START - MOTION DETECTION")
        analysis_logger.info("###############################################")
        
        # Get videos to process
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
        logger.info(f"🎬 Found {len(videos_to_process)} videos to process with motion detection")
        
        # Clear previous session data
        self.all_features = []
        self.video_metadata = []
        all_analyses = []
        
        # Process each video
        for i, video_path in enumerate(videos_to_process):
            analysis_logger.info(f"Processing video {i+1}/{len(videos_to_process)}: {video_path.name}")
            logger.info(f"🎯 Processing video {i+1}/{len(videos_to_process)}: {video_path.name}")
            
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
                    
                    logger.info(f"✅ {video_path.name}: {analysis['detection_count']} detections, {analysis['frames_with_motion']} motion frames")
                else:
                    analysis_logger.info(f"VIDEO SKIPPED: {video_path.name} - No animals detected")
                    
            except Exception as e:
                analysis_logger.error(f"VIDEO ERROR: {video_path.name} - {e}")
                logger.error(f"❌ Failed to process {video_path.name}: {e}")
        
        # Perform clustering
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
        analysis_logger.info("BATCH PROCESSING SESSION END - MOTION DETECTION")
        analysis_logger.info("###############################################")
        logger.info(f"🎉 Motion detection processing complete! Analyzed {len(all_analyses)} videos with animals")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Motion detection wildlife video processor')
    parser.add_argument('--videos', '-v', nargs='+', help='Optional list of video indices (e.g. 7 8 9) or names (e.g. IMG_0007.MP4) to process')
    
    # Add common arguments from base class
    MotionDetectionVideoProcessor.setup_common_arguments(parser)
    
    # Add motion detection specific arguments
    MotionDetectionVideoProcessor.setup_motion_detection_arguments(parser)
    
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
    
    # Set environment variables from parsed arguments (including motion detection)
    MotionDetectionVideoProcessor.set_environment_from_args(args, include_motion=True)
    
    try:
        processor = MotionDetectionVideoProcessor()
        logger.info(f"🎯 MegaDetector version: {args.megadetector_version}")
        logger.info(f"🤖 Ensemble models: {args.ensemble}")
        processor.process_all_videos(video_filter=video_filter)
    except KeyboardInterrupt:
        logger.info("🛑 Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()