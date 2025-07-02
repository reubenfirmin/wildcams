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
#   "torchvision>=0.15.0"
# ]
# ///
"""
Base video processor with common functionality shared between process.py and process2.py.
Contains video I/O, analysis management, clustering, and reporting functions.
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Import shared ML detection module
from ml_detection import MLDetectionEnsemble

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get loggers
logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger('analysis')

class VideoProcessorBase:
    """Base class for wildlife video processing with common functionality."""
    
    def __init__(self):
        # Directory setup
        self.video_dir = Path(os.getenv('VIDEO_DIR', './videos'))
        self.output_dir = self.video_dir / 'analysis'
        self.debug_dir = self.output_dir / 'debug_frames'
        self.logs_dir = Path('./logs')
        self.models_cache_dir = Path('./models_cache')
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.models_cache_dir.mkdir(exist_ok=True)
        
        # Setup logging for this session
        self.setup_logging()
        
        # Clean up debug directory from previous runs
        if self.debug_dir.exists():
            import shutil
            shutil.rmtree(self.debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        
        # Processing parameters - HIGH QUALITY for wildlife detection
        self.frame_skip = int(os.getenv('FRAME_SKIP', '15'))  
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.1'))
        self.max_frames_per_video = int(os.getenv('MAX_FRAMES_PER_VIDEO', '20'))
        self.clustering_eps = float(os.getenv('CLUSTERING_EPS', '0.3'))
        self.min_samples = int(os.getenv('MIN_SAMPLES', '2'))
        
        # Model configuration parameters
        self.ensemble_models = os.getenv('ENSEMBLE_MODELS').split(',')
        
        # Animal validation thresholds
        self.megadetector_high_conf = float(os.getenv('MEGADETECTOR_HIGH_CONFIDENCE', '0.3'))
        self.yolo_high_conf = float(os.getenv('YOLO_HIGH_CONFIDENCE', '0.4'))
        self.min_yolo_detections = int(os.getenv('MIN_YOLO_DETECTIONS', '3'))
        self.weak_evidence_threshold = float(os.getenv('WEAK_EVIDENCE_THRESHOLD', '0.25'))
        self.wildlife_model_confidence = float(os.getenv('WILDLIFE_MODEL_CONFIDENCE', '0.2'))
        
        # Camera handling detection thresholds
        self.detection_density_threshold = float(os.getenv('DETECTION_DENSITY_THRESHOLD', '15.0'))
        self.low_confidence_ratio_threshold = float(os.getenv('LOW_CONFIDENCE_RATIO_THRESHOLD', '0.7'))
        self.low_confidence_cutoff = float(os.getenv('LOW_CONFIDENCE_CUTOFF', '0.2'))
        
        # Store all extracted features for clustering
        self.all_features = []
        self.video_metadata = []
        
        # Initialize ML detection ensemble after all parameters are set
        self.ml_ensemble = MLDetectionEnsemble(
            confidence_threshold=self.confidence_threshold,
            ensemble_models=self.ensemble_models,
            cache_dir=self.models_cache_dir
        )
        
        # Log model configuration after everything is initialized
        self._log_model_configuration()
        
        logger.info(f"🎬 Video processor base initialized")
        logger.info(f"📁 Video directory: {self.video_dir}")
        logger.info(f"📊 Analysis output: {self.output_dir}")
        logger.info(f"📋 Logs directory: {self.logs_dir}")
    
    @staticmethod
    def setup_common_arguments(parser):
        """Add common arguments to an argument parser."""
        # Model configuration  
        parser.add_argument('--ensemble', '-e', default='yolov8x,yolov8m,MDV6-yolov10-e,MDV6-rtdetr-c',
                           help='Comma-separated list of models to use in ensemble. Available: yolov8x,yolov8m,yolov8n,yolov10n,yolov10s,yolov10m,yolov10b,yolov10l,yolov10x,yolo12n,yolo12s,yolo12m,yolo12l,yolo12x,MDV6-yolov9-c,MDV6-yolov9-e,MDV6-yolov10-c,MDV6-yolov10-e,MDV6-rtdetr-c (default: yolov8x,yolov8m,MDV6-yolov10-e,MDV6-rtdetr-c)')
        
        # Processing parameters
        parser.add_argument('--confidence-threshold', '--conf', type=float, default=0.25,
                           help='Confidence threshold for detections (default: 0.25)')
        parser.add_argument('--max-frames', type=int, default=20,
                           help='Maximum frames to extract per video (default: 20)')
        parser.add_argument('--frame-skip', type=int, default=15,
                           help='Frame skip for video processing (default: 15)')
        
        # Validation thresholds
        parser.add_argument('--megadetector-high-conf', type=float, default=0.3,
                           help='High confidence threshold for MegaDetector (default: 0.3)')
        parser.add_argument('--yolo-high-conf', type=float, default=0.4,
                           help='High confidence threshold for YOLO models (default: 0.4)')
        parser.add_argument('--min-yolo-detections', type=int, default=3,
                           help='Minimum YOLO detections for validation (default: 3)')
        parser.add_argument('--weak-evidence-threshold', type=float, default=0.25,
                           help='Threshold for weak evidence validation (default: 0.25)')
        parser.add_argument('--wildlife-model-confidence', type=float, default=0.2,
                           help='Confidence threshold for wildlife-specific models (default: 0.2)')
        
        # Camera handling detection
        parser.add_argument('--detection-density-threshold', type=float, default=15.0,
                           help='Detection density threshold for camera handling detection (default: 15.0)')
        parser.add_argument('--composite-motion-threshold', type=int, default=3000000,
                           help='Composite motion threshold for camera handling (default: 3000000)')
        parser.add_argument('--min-motion-threshold', type=int, default=100,
                           help='Minimum motion threshold to avoid processing static videos (default: 100)')
        parser.add_argument('--motion-frames-weight', type=float, default=1.2,
                           help='Weight exponent for motion frames in composite score (default: 1.2)')
        parser.add_argument('--motion-regions-weight', type=float, default=1.1,
                           help='Weight exponent for motion regions in composite score (default: 1.1)')
        parser.add_argument('--motion-tracks-weight', type=float, default=1.0,
                           help='Weight exponent for motion tracks in composite score (default: 1.0)')
        parser.add_argument('--large-region-multiplier', type=float, default=15.0,
                           help='Multiplier for large region percentage in composite score (default: 15.0)')
        parser.add_argument('--low-confidence-ratio-threshold', type=float, default=0.7,
                           help='Low confidence ratio threshold for camera handling (default: 0.7)')
        parser.add_argument('--low-confidence-cutoff', type=float, default=0.2,
                           help='Low confidence cutoff for camera handling detection (default: 0.2)')
        
        # Clustering parameters
        parser.add_argument('--clustering-eps', type=float, default=0.3,
                           help='DBSCAN eps parameter for clustering (default: 0.3)')
        parser.add_argument('--min-samples', type=int, default=2,
                           help='DBSCAN min_samples parameter for clustering (default: 2)')
    
    @staticmethod
    def setup_motion_detection_arguments(parser):
        """Add motion detection specific arguments to an argument parser."""
        parser.add_argument('--motion-method', choices=['MOG2', 'KNN'], default='MOG2',
                           help='Motion detection method (default: MOG2)')
        parser.add_argument('--motion-var-threshold', type=int, default=32,
                           help='Motion detection variance threshold - higher = less sensitive (default: 32)')
        parser.add_argument('--filter-motion-var-threshold', type=int, default=None,
                           help='Lenient variance threshold for Step 2 motion filter (default: same as motion-var-threshold)')
        parser.add_argument('--analysis-motion-var-threshold', type=int, default=None,
                           help='Strict variance threshold for Step 3 spatial analysis (default: same as motion-var-threshold)')
        parser.add_argument('--min-motion-area', type=int, default=300,
                           help='Minimum motion area threshold in pixels (default: 2000)')
        parser.add_argument('--max-motion-area', type=int, default=80000,
                           help='Maximum motion area threshold in pixels (default: 80000)')
        parser.add_argument('--motion-history', type=int, default=100,
                           help='Motion detection background history frames (default: 100)')
        parser.add_argument('--max-regions-per-frame', type=int, default=10,
                           help='Maximum motion regions to process per frame (default: 10)')
        parser.add_argument('--min-region-width', type=int, default=30,
                           help='Minimum motion region width in pixels (default: 30)')
        parser.add_argument('--min-region-height', type=int, default=30,
                           help='Minimum motion region height in pixels (default: 30)')
        parser.add_argument('--max-aspect-ratio', type=float, default=5.0,
                           help='Maximum width/height aspect ratio for motion regions (default: 5.0)')
        parser.add_argument('--motion-margin', type=int, default=30,
                           help='Margin to expand motion regions for ML context (default: 30)')
        
        # Temporal consistency arguments (for Next-Gen processor)
        parser.add_argument('--min-track-duration', type=float, default=0.1,
                           help='Minimum track duration in seconds (default: 0.1)')
        parser.add_argument('--motion-tracking-gap-seconds', type=float, default=1.0,
                           help='Maximum time gap for motion track linking in seconds (default: 1.0)')
        parser.add_argument('--detection-validation-gap-seconds', type=float, default=0.3,
                           help='Maximum time gap between ML detections for validation in seconds (default: 0.3)')
        parser.add_argument('--tracking-distance-threshold', type=float, default=150.0,
                           help='Maximum distance for tracking association in pixels (default: 150.0)')
        parser.add_argument('--full-frame-validation-frames', type=int, default=5,
                           help='Consecutive frames needed for full-frame validation (default: 5)')
        parser.add_argument('--anchor-confidence-threshold', type=float, default=0.5,
                           help='Minimum confidence for anchor point detection (default: 0.5)')
        parser.add_argument('--min-track-frames', type=int, default=1,
                           help='Minimum frames required for valid track (default: 1)')
        parser.add_argument('--track-search-seconds', type=float, default=2.0,
                           help='Seconds to search backwards/forwards from anchor (default: 2.0)')
        parser.add_argument('--size-ratio-threshold', type=float, default=0.3,
                           help='Minimum size ratio for same animal detection (default: 0.3)')
        
        # Step 4 full-frame validation parameters
        parser.add_argument('--max-validation-frames', type=int, default=5,
                           help='Maximum frames to validate with full ensemble (default: 5)')
        parser.add_argument('--crop-weight', type=float, default=0.6,
                           help='Weight for crop-based ML scores (default: 0.6)')
        parser.add_argument('--fullframe-weight', type=float, default=0.4,
                           help='Weight for full-frame ML scores (default: 0.4)')
        parser.add_argument('--min-crop-size', type=int, default=900,
                           help='Minimum crop area in pixels for ML analysis (default: 900)')
        parser.add_argument('--temporal-spread-seconds', type=float, default=2.0,
                           help='Minimum seconds between selected validation frames (default: 2.0)')
        parser.add_argument('--accepted-rtdetr-overlap', type=float, default=0.5,
                           help='Minimum overlap threshold for accepting RT-DETR extended class detections (default: 0.5)')
    
    @staticmethod
    def set_environment_from_args(args, include_motion=False):
        """Set environment variables from parsed arguments."""
        import os
        os.environ['ENSEMBLE_MODELS'] = args.ensemble
        os.environ['CONFIDENCE_THRESHOLD'] = str(args.confidence_threshold)
        os.environ['MAX_FRAMES_PER_VIDEO'] = str(args.max_frames)
        os.environ['FRAME_SKIP'] = str(args.frame_skip)
        os.environ['MEGADETECTOR_HIGH_CONFIDENCE'] = str(args.megadetector_high_conf)
        os.environ['YOLO_HIGH_CONFIDENCE'] = str(args.yolo_high_conf)
        os.environ['MIN_YOLO_DETECTIONS'] = str(args.min_yolo_detections)
        os.environ['WEAK_EVIDENCE_THRESHOLD'] = str(args.weak_evidence_threshold)
        os.environ['WILDLIFE_MODEL_CONFIDENCE'] = str(args.wildlife_model_confidence)
        os.environ['DETECTION_DENSITY_THRESHOLD'] = str(args.detection_density_threshold)
        os.environ['COMPOSITE_MOTION_THRESHOLD'] = str(args.composite_motion_threshold)
        os.environ['MIN_MOTION_THRESHOLD'] = str(args.min_motion_threshold)
        os.environ['MOTION_FRAMES_WEIGHT'] = str(args.motion_frames_weight)
        os.environ['MOTION_REGIONS_WEIGHT'] = str(args.motion_regions_weight)
        os.environ['MOTION_TRACKS_WEIGHT'] = str(args.motion_tracks_weight)
        os.environ['LARGE_REGION_MULTIPLIER'] = str(args.large_region_multiplier)
        os.environ['LOW_CONFIDENCE_RATIO_THRESHOLD'] = str(args.low_confidence_ratio_threshold)
        os.environ['LOW_CONFIDENCE_CUTOFF'] = str(args.low_confidence_cutoff)
        os.environ['CLUSTERING_EPS'] = str(args.clustering_eps)
        os.environ['MIN_SAMPLES'] = str(args.min_samples)
        
        # Temporal consistency parameters
        os.environ['MIN_TRACK_DURATION'] = str(args.min_track_duration)
        os.environ['MOTION_TRACKING_GAP_SECONDS'] = str(args.motion_tracking_gap_seconds)
        os.environ['DETECTION_VALIDATION_GAP_SECONDS'] = str(args.detection_validation_gap_seconds)
        os.environ['TRACKING_DISTANCE_THRESHOLD'] = str(args.tracking_distance_threshold)
        os.environ['FULL_FRAME_VALIDATION_FRAMES'] = str(args.full_frame_validation_frames)
        os.environ['ANCHOR_CONFIDENCE_THRESHOLD'] = str(args.anchor_confidence_threshold)
        os.environ['MIN_TRACK_FRAMES'] = str(args.min_track_frames)
        os.environ['TRACK_SEARCH_SECONDS'] = str(args.track_search_seconds)
        os.environ['SIZE_RATIO_THRESHOLD'] = str(args.size_ratio_threshold)
        
        # Step 4 validation parameters
        os.environ['MAX_VALIDATION_FRAMES'] = str(args.max_validation_frames)
        os.environ['CROP_WEIGHT'] = str(args.crop_weight)
        os.environ['FULLFRAME_WEIGHT'] = str(args.fullframe_weight)
        os.environ['MIN_CROP_SIZE'] = str(args.min_crop_size)
        os.environ['TEMPORAL_SPREAD_SECONDS'] = str(args.temporal_spread_seconds)
        
        # Motion detection specific parameters
        if include_motion:
            os.environ['MOTION_METHOD'] = args.motion_method
            os.environ['MOTION_VAR_THRESHOLD'] = str(args.motion_var_threshold)
            # Set dual motion thresholds - use main threshold as default if not specified
            os.environ['FILTER_MOTION_VAR_THRESHOLD'] = str(args.filter_motion_var_threshold or args.motion_var_threshold)
            os.environ['ANALYSIS_MOTION_VAR_THRESHOLD'] = str(args.analysis_motion_var_threshold or args.motion_var_threshold)
            os.environ['MIN_MOTION_AREA'] = str(args.min_motion_area)
            os.environ['MAX_MOTION_AREA'] = str(args.max_motion_area)
            os.environ['MOTION_HISTORY'] = str(args.motion_history)
            os.environ['MAX_REGIONS_PER_FRAME'] = str(args.max_regions_per_frame)
            os.environ['MIN_REGION_WIDTH'] = str(args.min_region_width)
            os.environ['MIN_REGION_HEIGHT'] = str(args.min_region_height)
            os.environ['MAX_ASPECT_RATIO'] = str(args.max_aspect_ratio)
            os.environ['MOTION_MARGIN'] = str(args.motion_margin)
            
            # Temporal consistency parameters
            os.environ['MIN_TRACK_DURATION'] = str(args.min_track_duration)
            os.environ['MOTION_TRACKING_GAP_SECONDS'] = str(args.motion_tracking_gap_seconds)
            os.environ['DETECTION_VALIDATION_GAP_SECONDS'] = str(args.detection_validation_gap_seconds)
            os.environ['TRACKING_DISTANCE_THRESHOLD'] = str(args.tracking_distance_threshold)
            os.environ['FULL_FRAME_VALIDATION_FRAMES'] = str(args.full_frame_validation_frames)
    
    def setup_logging(self):
        """Setup single logger to file and console."""
        # Create timestamp for this session
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup single logger
        global logger, analysis_logger
        logger = logging.getLogger('wildcams')
        analysis_logger = logger  # Use same logger
        
        # Remove any existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create log file handler
        log_file = self.logs_dir / f'wildcams_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        logger.setLevel(logging.DEBUG)
        
        # Log the setup
        logger.info(f"📋 Logging initialized - session {timestamp}")
        logger.info(f"📋 Log file: {log_file}")
    
    def _log_model_configuration(self):
        """Log the model configuration."""
        logger.info("=" * 80)
        logger.info("🎯 MODEL CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Ensemble Models: {', '.join(self.ensemble_models)}")
        logger.info(f"Confidence Threshold: {self.confidence_threshold}")
        logger.info("=" * 80)
    
    def get_unprocessed_videos(self) -> List[Path]:
        """Get list of videos that haven't been processed yet."""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
        unprocessed = []
        
        for video_file in self.video_dir.iterdir():
            if (video_file.is_file() and 
                video_file.suffix.lower() in video_extensions and
                not (video_file.with_suffix('.processed')).exists()):
                unprocessed.append(video_file)
        
        return sorted(unprocessed)
    
    def get_filtered_videos(self, video_filter: List) -> List[Path]:
        """Get filtered videos, ignoring .processed status when filter is provided."""
        if video_filter:
            analysis_logger.info("OVERRIDE: Video filter provided - ignoring .processed files and forcing reprocessing")
            logger.info("🔄 Filter provided - forcing reprocessing of specified videos")
            
            # Get all videos in directory, not just unprocessed ones
            all_videos = list(self.video_dir.glob("*.MP4"))
            filtered_videos = []
            
            for filter_item in video_filter:
                if isinstance(filter_item, int):
                    # Filter by video index (e.g. 7 for IMG_0007.MP4)
                    video_name = f"IMG_{filter_item:04d}.MP4"
                    matching = [v for v in all_videos if v.name == video_name]
                    filtered_videos.extend(matching)
                elif isinstance(filter_item, str):
                    # Filter by exact video name
                    matching = [v for v in all_videos if v.name == filter_item]
                    filtered_videos.extend(matching)
            
            return filtered_videos
        else:
            # No filter provided - use normal unprocessed video logic
            return self.get_unprocessed_videos()
    
    def extract_frames(self, video_path: Path) -> Tuple[List[np.ndarray], List[float]]:
        """Extract frames evenly distributed throughout the video for analysis."""
        frames = []
        timestamps = []
        
        # Try different video reading backends for corrupted files
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(str(video_path), backend)
            
            if cap.isOpened():
                break
            cap.release()
        else:
            logger.error(f"❌ Could not open video with any backend: {video_path}")
            return frames, timestamps
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        if total_frames <= 0:
            logger.error(f"❌ Invalid frame count for {video_path.name}")
            cap.release()
            return frames, timestamps
        
        # Calculate frame indices to sample evenly throughout the video
        frames_to_extract = min(self.max_frames_per_video, total_frames)
        if frames_to_extract == total_frames:
            # Extract every frame if video is short
            target_frame_indices = list(range(total_frames))
        else:
            # Calculate evenly spaced frame indices
            step = total_frames / frames_to_extract
            target_frame_indices = [int(i * step) for i in range(frames_to_extract)]
        
        logger.info(f"🎬 Processing {video_path.name} ({duration:.1f}s, {total_frames} frames)")
        logger.info(f"📍 Sampling {len(target_frame_indices)} frames evenly throughout video")
        
        # Extract frames at target indices
        for target_idx in target_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None and frame.size > 0:
                # Calculate timestamp for this frame
                timestamp_seconds = target_idx / fps if fps > 0 else 0
                frames.append(frame)
                timestamps.append(timestamp_seconds)
                analysis_logger.info(f"Extracted frame {len(frames)-1}: index={target_idx}, timestamp={timestamp_seconds:.2f}s")
            else:
                logger.debug(f"⚠️ Failed to read frame {target_idx}")
        
        cap.release()
        
        # Save frames to debug directory
        if frames:
            video_debug_dir = self.debug_dir / video_path.stem
            video_debug_dir.mkdir(exist_ok=True)
            
            logger.info(f"💾 Saving {len(frames)} frames to debug directory...")
            for idx, frame in enumerate(frames):
                frame_path = video_debug_dir / f"frame_{idx:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
            
            logger.info(f"📸 Successfully extracted {len(frames)} frames")
            logger.info(f"🛠️ Debug frames saved to: {video_debug_dir}")
        
        return frames, timestamps
    
    def detect_camera_handling(self, all_detections: List[Dict], frames_processed: int) -> bool:
        """
        Detect if this appears to be camera handling/equipment rather than animals.
        Returns True if likely camera handling (should be rejected).
        """
        if not all_detections:
            return False
        
        total_detections = len(all_detections)
        detection_density = total_detections / frames_processed
        
        # Count low confidence detections
        low_conf_detections = [d for d in all_detections if d['confidence'] < self.low_confidence_cutoff]
        low_conf_ratio = len(low_conf_detections) / total_detections
        
        analysis_logger.info(f"CAMERA HANDLING CHECK:")
        analysis_logger.info(f"  Detection density: {detection_density:.2f} (threshold: {self.detection_density_threshold})")
        analysis_logger.info(f"  Low confidence ratio: {low_conf_ratio:.3f} (threshold: {self.low_confidence_ratio_threshold})")
        analysis_logger.info(f"  Total detections: {total_detections}, Frames: {frames_processed}")
        
        # High detection density + lots of low confidence = likely camera handling
        if (detection_density > self.detection_density_threshold and 
            low_conf_ratio > self.low_confidence_ratio_threshold):
            analysis_logger.info("CAMERA HANDLING DETECTED: High density + low confidence mass")
            return True
        
        # Very high detection density alone is suspicious 
        if detection_density > self.detection_density_threshold * 2:
            analysis_logger.info("CAMERA HANDLING DETECTED: Extremely high detection density")
            return True
        
        return False
    
    def ensemble_validation(self, all_detections: List[Dict]) -> bool:
        """
        Validate detections using ensemble logic across multiple models.
        Returns True if evidence supports real animal presence.
        """
        if not all_detections:
            analysis_logger.info("ENSEMBLE VALIDATION: No detections - rejected")
            return False
        
        # Count detections by source
        source_counts = {}
        high_conf_detections = []
        
        for detection in all_detections:
            source = detection.get('source', 'unknown')
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
            
            # Track high confidence detections
            conf = detection['confidence']
            if source.startswith('megadetector') and conf >= self.megadetector_high_conf:
                high_conf_detections.append(detection)
            elif source.startswith(('primary', 'backup')) and conf >= self.yolo_high_conf:
                high_conf_detections.append(detection)
            elif source.startswith(('deepfaune', 'wildlife')) and conf >= self.wildlife_model_confidence:
                high_conf_detections.append(detection)
        
        analysis_logger.info(f"ENSEMBLE VALIDATION: Source counts: {source_counts}")
        analysis_logger.info(f"ENSEMBLE VALIDATION: High confidence detections: {len(high_conf_detections)}")
        
        # Strong evidence: Multiple models agree OR high confidence detections
        strong_evidence = len(source_counts) >= 2 or len(high_conf_detections) >= 2
        
        # Moderate evidence: YOLO models with sufficient detections
        yolo_detections = sum(source_counts.get(s, 0) for s in source_counts if s.startswith(('primary', 'backup')))
        moderate_evidence = yolo_detections >= self.min_yolo_detections
        
        # Weak evidence: Any wildlife-specific model detection
        wildlife_evidence = any(s.startswith(('megadetector', 'deepfaune', 'wildlife')) for s in source_counts)
        
        if strong_evidence:
            analysis_logger.info("ENSEMBLE VALIDATION: STRONG evidence - multiple models agree")
            return True
        elif moderate_evidence:
            analysis_logger.info("ENSEMBLE VALIDATION: MODERATE evidence - sufficient YOLO detections")
            return True
        elif wildlife_evidence:
            # Check if weak evidence meets threshold
            avg_confidence = np.mean([d['confidence'] for d in all_detections])
            if avg_confidence >= self.weak_evidence_threshold:
                analysis_logger.info(f"ENSEMBLE VALIDATION: WEAK evidence accepted - avg conf {avg_confidence:.3f} >= {self.weak_evidence_threshold}")
                return True
            else:
                analysis_logger.info(f"ENSEMBLE VALIDATION: WEAK evidence rejected - avg conf {avg_confidence:.3f} < {self.weak_evidence_threshold}")
                return False
        else:
            analysis_logger.info("ENSEMBLE VALIDATION: Insufficient evidence - rejected")
            return False
    
    def score_and_find_best_detection(self, all_detections: List[Dict]) -> Tuple[Optional[Dict], int]:
        """Score all detections and find the best one for feature extraction."""
        if not all_detections:
            return None, 0
        
        best_detection = None
        best_score = 0.0
        total_scored = 0
        
        analysis_logger.info("=== DETECTION SCORING START ===")
        
        for detection in all_detections:
            # Calculate bounding box area and size ratio
            bbox = detection['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = 1600 * 900  # Typical frame size
            size_ratio = area / frame_area
            
            # Score based on confidence, size, and source reliability
            base_score = detection['confidence'] * size_ratio
            
            # Apply source-specific multipliers
            source = detection.get('source', 'unknown')
            if source.startswith('megadetector'):
                multiplier = 1.5  # Wildlife-specific model bonus
            elif source.startswith('deepfaune'):
                multiplier = 1.4  # Specialized wildlife model
            elif source.startswith('primary'):
                multiplier = 1.2  # Primary model bonus
            elif source.startswith('backup'):
                multiplier = 1.1  # Backup model
            else:
                multiplier = 1.0  # Default
            
            final_score = base_score * multiplier
            
            # Enhanced detection gets additional boost
            if 'enhanced' in source:
                final_score *= 1.2
            
            analysis_logger.info(f"SCORING: frame_{detection.get('frame_idx', '?')}, source={source}, conf={detection['confidence']:.4f}, area={area}, size_ratio={size_ratio:.4f}, multiplier={multiplier}, score={final_score:.6f}")
            
            if final_score > best_score:
                analysis_logger.info(f"NEW BEST: previous_score={best_score:.6f}, new_score={final_score:.6f}, frame={detection.get('frame_idx', '?')}, source={source}")
                best_score = final_score
                best_detection = detection
            
            total_scored += 1
        
        analysis_logger.info(f"=== DETECTION SCORING END: {total_scored} detections scored, best_score={best_score:.6f} ===")
        return best_detection, total_scored
    
    def extract_features_from_detection(self, frame: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """Extract ResNet18 features from the best detection."""
        try:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                analysis_logger.warning(f"Invalid detection bbox after clipping: {bbox}")
                return None
            
            # Extract region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                analysis_logger.warning("Empty crop region")
                return None
            
            # Use ML ensemble feature extraction
            features = self.ml_ensemble.extract_features(frame, bbox)
            
            if features is not None:
                analysis_logger.info(f"FEATURE EXTRACTION: Successfully extracted {len(features)} features from detection")
                return features
            else:
                analysis_logger.warning("FEATURE EXTRACTION: Failed to extract features")
                return None
                
        except Exception as e:
            analysis_logger.error(f"FEATURE EXTRACTION: Error extracting features: {e}")
            return None
    
    def cluster_animal_videos(self, video_metadata: List[Dict]) -> Dict:
        """Cluster videos with similar animal features."""
        if len(self.all_features) < 2:
            analysis_logger.info("CLUSTERING: Not enough features for clustering")
            return {}
        
        analysis_logger.info(f"CLUSTERING: Starting with {len(self.all_features)} feature vectors")
        
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(self.all_features)
            
            # Apply PCA for dimensionality reduction if needed
            if features_scaled.shape[1] > 50:
                pca = PCA(n_components=50)
                features_scaled = pca.fit_transform(features_scaled)
                analysis_logger.info(f"CLUSTERING: Applied PCA, reduced to {features_scaled.shape[1]} dimensions")
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.min_samples)
            cluster_labels = clustering.fit_predict(features_scaled)
            
            analysis_logger.info(f"CLUSTERING: Found {len(set(cluster_labels))} clusters (including noise)")
            
            # Group videos by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'video_metadata': video_metadata[i],
                    'features': self.all_features[i]
                })
            
            # Format results
            result = {}
            for cluster_id, cluster_videos in clusters.items():
                if cluster_id == -1:
                    continue  # Skip noise
                
                result[f'cluster_{cluster_id}'] = {
                    'cluster_id': cluster_id,
                    'size': len(cluster_videos),
                    'videos': [v['video_metadata'] for v in cluster_videos],
                    'avg_confidence': np.mean([v['video_metadata']['detection']['confidence'] for v in cluster_videos]),
                    'avg_animal_size': np.mean([v['video_metadata']['detection']['area_ratio'] for v in cluster_videos])
                }
            
            if -1 in clusters:
                result['unclustered'] = {
                    'cluster_id': -1,
                    'size': len(clusters[-1]),
                    'videos': [v['video_metadata'] for v in clusters[-1]],
                    'description': 'Videos that did not fit into any cluster'
                }
            
            return result
            
        except Exception as e:
            analysis_logger.error(f"CLUSTERING: Failed with error: {e}")
            return {}
    
    def save_clustering_results(self, clusters: Dict):
        """Save clustering results and feature data."""
        try:
            # Save clusters
            clusters_file = self.output_dir / 'clusters.json'
            with open(clusters_file, 'w') as f:
                json.dump(clusters, f, indent=2)
            
            # Save features for future analysis
            features_file = self.output_dir / 'features.pkl'
            with open(features_file, 'wb') as f:
                pickle.dump({
                    'features': self.all_features,
                    'metadata': self.video_metadata
                }, f)
            
            logger.info(f"💾 Clustering results saved to {clusters_file}")
            logger.debug(f"💾 Features saved to {features_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save clustering results: {e}")
    
    def _debug_numpy_objects(self, obj, path=""):
        """Debug helper to find numpy objects in nested data."""
        if hasattr(obj, 'dtype'):  # numpy array/scalar
            logger.error(f"Found numpy object at {path}: type={type(obj)}, dtype={obj.dtype}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self._debug_numpy_objects(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                self._debug_numpy_objects(item, f"{path}[{i}]")

    def _convert_for_json(self, obj):
        """Convert numpy types and other non-JSON types for serialization."""
        import numpy as np
        
        # Check if object has numpy module in its type string - catch all numpy types
        obj_type_str = str(type(obj))
        if 'numpy' in obj_type_str:
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                # Last resort for numpy objects
                return float(obj)
        
        # Handle specific numpy types
        if isinstance(obj, (np.integer, np.floating, np.bool_, np.complexfloating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):  # Other numpy-like objects
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
        elif hasattr(obj, 'item') and hasattr(obj, '__float__'):  # numpy scalars without dtype
            try:
                return obj.item()
            except:
                return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj

    def save_analysis(self, analysis: Dict, video_path: Path):
        """Save analysis results to JSON file."""
        analysis_file = self.output_dir / f"{video_path.stem}_analysis.json"
        
        try:
            # Convert all non-JSON types for serialization
            analysis_copy = self._convert_for_json(analysis)
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_copy, f, indent=2)
            logger.debug(f"💾 Saved analysis to {analysis_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save analysis: {e}")
            # Debug: Try to find the problematic data
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            # Try to identify the numpy objects
            self._debug_numpy_objects(analysis)
    
    def mark_as_processed(self, video_path: Path):
        """Create .processed marker file."""
        marker_file = video_path.with_suffix('.processed')
        try:
            marker_file.write_text(datetime.now().isoformat())
            logger.debug(f"✅ Marked {video_path.name} as processed")
        except Exception as e:
            logger.error(f"❌ Failed to create marker file: {e}")
    
    def generate_summary_report(self, all_analyses: List[Dict], clusters: Dict):
        """Generate a summary report of all processed videos."""
        report_file = self.output_dir / 'processing_summary.json'
        
        # Aggregate statistics
        total_videos = len(all_analyses)
        videos_with_animals = len([a for a in all_analyses if a.get('animals_detected')])
        all_animals = set()
        total_detections = 0
        
        for analysis in all_analyses:
            if analysis.get('animals_detected'):
                all_animals.update(analysis['animals_detected'])
                total_detections += analysis.get('detection_count', 0)
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_videos_processed': total_videos,
            'videos_with_animals': videos_with_animals,
            'unique_animal_types': list(all_animals),
            'total_animal_detections': total_detections,
            'clusters': clusters,
            'processing_stats': {
                'frame_skip': self.frame_skip,
                'confidence_threshold': self.confidence_threshold,
                'max_frames_per_video': self.max_frames_per_video
            }
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"📊 Summary report saved to {report_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save summary report: {e}")