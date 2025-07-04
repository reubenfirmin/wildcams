"""Processing configuration dataclass for wildlife video processing."""

from dataclasses import dataclass
from typing import List


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