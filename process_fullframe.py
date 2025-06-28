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
Full-frame wildlife video processor.
Uses comprehensive ML ensemble on entire frames for maximum detection accuracy.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Import base processor
from video_processor_base import VideoProcessorBase

# Get loggers
logger = logging.getLogger('wildcams')
analysis_logger = logger

class FullFrameVideoProcessor(VideoProcessorBase):
    """Full-frame video processor using complete ML ensemble on entire frames."""
    
    def __init__(self):
        super().__init__()
        logger.info("🖼️ Full-frame video processor initialized")
    
    def enhanced_frame_analysis(self, frame: np.ndarray, frame_idx: int, video_debug_dir: Path, timestamp_seconds: float = None) -> List[Dict]:
        """Analyze a single frame using the full ML ensemble on the complete frame."""
        analysis_logger.info(f"--- FRAME {frame_idx} ANALYSIS START ---")
        analysis_logger.info(f"PREPROCESSING_MODE: FULL_FRAME (process.py)")
        analysis_logger.info(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        if timestamp_seconds is not None:
            analysis_logger.info(f"Video timestamp: {timestamp_seconds:.2f}s")
        
        detections = []
        
        # 1. Run full 5-model ensemble detection
        ensemble_detections = self.ml_ensemble.run_ensemble_detection(
            frame, timestamp_seconds, frame_idx
        )
        detections.extend(ensemble_detections)
        
        # 2. Run enhanced preprocessing
        enhanced_detections = self.ml_ensemble.run_enhanced_preprocessing(
            frame, timestamp_seconds
        )
        detections.extend(enhanced_detections)
        
        # 3. Run multi-scale analysis
        multiscale_detections = self.ml_ensemble.run_multiscale_analysis(
            frame, timestamp_seconds
        )
        detections.extend(multiscale_detections)
        
        analysis_logger.info(f"--- FRAME {frame_idx} ANALYSIS END: {len(detections)} total detections ---")
        return detections
    
    def process_video_with_features(self, video_path: Path) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Process a single video and extract features from the best detection."""
        analysis_logger.info(f"=== VIDEO PROCESSING START: {video_path.name} ===")
        
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
        
        analysis_logger.info("=== BEST ANIMAL FRAME SEARCH START ===")
        analysis_logger.info(f"Processing {len(frames)} frames")
        
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
        
        analysis_logger.info("=== BEST ANIMAL FRAME SEARCH END ===")
        
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
            'processing_mode': 'full_frame'
        }
        
        analysis_logger.info(f"=== VIDEO PROCESSING END: {video_path.name} - SUCCESS ===")
        self.mark_as_processed(video_path)
        
        return analysis, features
    
    def process_all_videos(self, video_filter=None):
        """Process all videos using full-frame approach."""
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
        for i, video_path in enumerate(videos_to_process):
            analysis_logger.info(f"Processing video {i+1}/{len(videos_to_process)}: {video_path.name}")
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
    
    parser = argparse.ArgumentParser(description='Full-frame wildlife video processor')
    parser.add_argument('--videos', '-v', nargs='+', help='Optional list of video indices (e.g. 7 8 9) or names (e.g. IMG_0007.MP4) to process')
    
    # Add common arguments from base class
    FullFrameVideoProcessor.setup_common_arguments(parser)
    
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
    
    # Set environment variables from parsed arguments
    FullFrameVideoProcessor.set_environment_from_args(args)
    
    try:
        processor = FullFrameVideoProcessor()
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