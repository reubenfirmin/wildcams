"""Configuration manager for wildlife video processing."""

import argparse
from typing import Optional, List, Union
from .processing_config import ProcessingConfig
from argparse import Namespace
from constants import DEFAULT_ENSEMBLE_MODELS, DEFAULT_VIDEO_DIR
from data_types import ConfigurationUpdate


class ConfigurationManager:
    """Manages configuration loading directly from CLI arguments."""
    
    def __init__(self):
        self._config: Optional[ProcessingConfig] = None
    
    def setup_common_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add common arguments to an argument parser."""
        # Video selection
        parser.add_argument('--videos', '-v', nargs='+', 
                           help='Optional list of video indices (e.g. 7 8 9) or names to process')
        
        # Model configuration  
        parser.add_argument('--ensemble', '-e', default=','.join(DEFAULT_ENSEMBLE_MODELS),
                           help='Comma-separated list of models to use in ensemble. Available: yolov8x,yolov8m,yolov8n,yolov10n,yolov10s,yolov10m,yolov10b,yolov10l,yolov10x,yolo12n,yolo12s,yolo12m,yolo12l,yolo12x,MDV6-yolov9-c,MDV6-yolov9-e,MDV6-yolov10-c,MDV6-yolov10-e,MDV6-rtdetr-c,rtdetr-l (default: yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l)')
        
        # Processing parameters
        parser.add_argument('--video-dir', default=DEFAULT_VIDEO_DIR,
                           help='Directory containing videos to process (default: ./videos)')
        parser.add_argument('--confidence-threshold', '--conf', type=float, default=0.8,
                           help='Confidence threshold for detections (default: 0.8)')
        parser.add_argument('--max-frames', type=int, default=20,
                           help='Maximum frames to extract per video (default: 20)')
        
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
        parser.add_argument('--composite-motion-threshold', type=float, default=0.5,
                           help='Camera handling detection threshold - higher values indicate camera handling (default: 0.5)')
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
        
        # Temporal continuity
        parser.add_argument('--confidence-bridge-threshold', type=float, default=0.6,
                           help='Threshold for bridging medium-confidence frames between high-confidence ones (default: 0.6)')
    
    def setup_motion_detection_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add motion detection specific arguments to an argument parser."""
        parser.add_argument('--motion-method', choices=['MOG2', 'KNN'], default='MOG2',
                           help='Motion detection method (default: MOG2)')
        parser.add_argument('--motion-var-threshold', type=int, default=32,
                           help='Motion detection variance threshold - higher = less sensitive (default: 32)')
        parser.add_argument('--min-motion-area', type=int, default=300,
                           help='Minimum motion area threshold in pixels (default: 300)')
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
        parser.add_argument('--min-consecutive-detection-seconds', type=float, default=0.2,
                           help='Minimum duration of consecutive detection required for validation (seconds)')
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
        parser.add_argument('--temporal-spread-seconds', type=float, default=2.0,
                           help='Minimum seconds between selected validation frames (default: 2.0)')
        parser.add_argument('--spatial-overlap-threshold', type=float, default=0.1,
                           help='Minimum spatial overlap threshold between detections and motion regions (default: 0.1)')
        
        # Track infilling parameters
        parser.add_argument('--enable-track-infilling', action='store_true', default=True,
                           help='Enable track infilling to connect nearby tracks (default: True)')
        parser.add_argument('--infill-max-gap-seconds', type=float, default=0.7,
                           help='Maximum time gap in seconds to allow infilling between tracks (default: 0.7)')
        parser.add_argument('--infill-max-distance-pixels', type=float, default=350.0,
                           help='Maximum spatial distance in pixels to allow infilling between tracks (default: 350.0)')
        parser.add_argument('--infill-min-overlap-ratio', type=float, default=0.3,
                           help='Minimum bbox overlap ratio to consider tracks for infilling (default: 0.3)')
        
        # Debug parameters
        parser.add_argument('--debug-show-spatially-invalid', action='store_true',
                           help='Show spatially invalid detections in logs (default: False)')
    
    def setup_tracking_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add tracking system specific arguments to an argument parser."""
        parser.add_argument('--tracking-method', choices=['auto', 'deepsort', 'simple'], default='auto',
                           help='Tracking method to use (default: auto)')
        parser.add_argument('--tracking-max-age', type=int, default=50,
                           help='Maximum frames to keep track alive without detection (default: 50)')
        parser.add_argument('--tracking-n-init', type=int, default=3,
                           help='Number of consecutive detections before track confirmation (default: 3)')
        parser.add_argument('--tracking-max-iou-distance', type=float, default=0.7,
                           help='Maximum IoU distance for track association (default: 0.7)')
        parser.add_argument('--tracking-max-cosine-distance', type=float, default=0.4,
                           help='Maximum cosine distance for appearance matching (default: 0.4)')
        parser.add_argument('--tracking-nn-budget', type=int, default=100,
                           help='Maximum size of appearance descriptor gallery (default: 100)')
        parser.add_argument('--tracking-min-iou', type=float, default=0.1,
                           help='Minimum IoU for simple tracker association (default: 0.1)')
        parser.add_argument('--min-track-detections', type=int, default=3,
                           help='Minimum detections required for valid track (default: 3)')
    
    def load_from_cli_args(self, args, include_motion: bool = True, include_tracking: bool = True) -> None:
        """Load configuration directly from command line arguments."""
        if isinstance(args, list):
            # If args is a list of strings, parse them
            parser = argparse.ArgumentParser(description="Wildlife Video Processing")
            
            # Add common arguments
            self.setup_common_arguments(parser)
            
            # Add motion detection arguments if requested
            if include_motion:
                self.setup_motion_detection_arguments(parser)
                
            # Add tracking arguments if requested
            if include_tracking:
                self.setup_tracking_arguments(parser)
            
            # Parse arguments
            parsed_args = parser.parse_args(args)
        else:
            # If args is already a parsed Namespace object, use it directly
            parsed_args = args
        
        # Create configuration directly from parsed arguments
        self._config = self._create_config_from_args(parsed_args)
    
    def _create_config_from_args(self, args: Namespace) -> ProcessingConfig:
        """Create ProcessingConfig directly from parsed CLI arguments."""
        return ProcessingConfig(
            # Video processing
            video_dir=args.video_dir,
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
            ensemble_models=[model.strip() for model in args.ensemble.split(',') if model.strip()],
            
            # Animal validation thresholds  
            megadetector_high_confidence=args.megadetector_high_conf,
            yolo_high_confidence=args.yolo_high_conf,
            min_yolo_detections=args.min_yolo_detections,
            weak_evidence_threshold=args.weak_evidence_threshold,
            wildlife_model_confidence=args.wildlife_model_confidence,
            
            # Camera handling detection thresholds
            detection_density_threshold=args.detection_density_threshold,
            low_confidence_ratio_threshold=args.low_confidence_ratio_threshold,
            low_confidence_cutoff=args.low_confidence_cutoff,
            
            # Clustering parameters
            clustering_eps=args.clustering_eps,
            min_samples=args.min_samples,
            
            # Temporal continuity
            confidence_bridge_threshold=args.confidence_bridge_threshold,
            
            # Tracking system configuration
            tracking_method=getattr(args, 'tracking_method', 'auto'),
            tracking_max_age=getattr(args, 'tracking_max_age', 50),
            tracking_n_init=getattr(args, 'tracking_n_init', 3),
            tracking_max_iou_distance=getattr(args, 'tracking_max_iou_distance', 0.7),
            tracking_max_cosine_distance=getattr(args, 'tracking_max_cosine_distance', 0.4),
            tracking_nn_budget=getattr(args, 'tracking_nn_budget', 100),
            tracking_min_iou=getattr(args, 'tracking_min_iou', 0.1),
            min_track_detections=getattr(args, 'min_track_detections', 3)
        )
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get the current processing configuration."""
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load_from_cli_args() first.")
        return self._config
    
    def update_config(self, update: ConfigurationUpdate) -> None:
        """Update specific configuration parameters using typed object."""
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load_from_cli_args() first.")
        
        # Create new config with updated values
        config_dict = self._config.__dict__.copy()
        config_dict.update(update.to_dict())
        self._config = ProcessingConfig(**config_dict)
    
    def parse_video_filter(self, args: Namespace) -> Optional[List[Union[int, str]]]:
        """Parse video filter from command line arguments."""
        video_filter = None
        if hasattr(args, 'videos') and args.videos:
            video_filter = []
            for video in args.videos:
                try:
                    video_filter.append(int(video))  # Try to convert to int (for indices)
                except ValueError:
                    video_filter.append(video)      # Keep as string (for filenames)
        return video_filter