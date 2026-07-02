"""Processing configuration dataclass for wildlife video processing."""

from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Global configuration for next-generation video processing."""

    # Video processing
    video_dir: str
    max_frames_per_video: int
    confidence_threshold: float

    # Camera handling detection
    composite_motion_threshold: float
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
    ensemble_models: list[str]

    # Animal validation thresholds
    megadetector_high_confidence: float
    yolo_high_confidence: float
    min_yolo_detections: int
    weak_evidence_threshold: float
    wildlife_model_confidence: float

    # Camera handling detection thresholds
    detection_density_threshold: float
    low_confidence_ratio_threshold: float
    low_confidence_cutoff: float

    # Temporal continuity
    confidence_bridge_threshold: float

    # Step 4: Animal classification
    enable_animal_classification: bool
    animal_confidence_threshold: float
    species_confidence_threshold: float
    classification_models: list[str]
    bioclip_top_k: int
    bioclip_threshold: float
    deepfaune_threshold: float

    # Step 3 composite-score tuning (defaults preserve historical hardcoded behavior)
    default_fps: float
    consensus_boost_per_detection: float
    composite_temporal_multiplier_cap: float
    composite_consensus_boost_per_model: float
    composite_motion_multiplier_base: float
    composite_motion_multiplier_span: float
    composite_duration_bonus_base: float
    composite_duration_bonus_cap: float
    composite_duration_bonus_divisor: float

    # Step 3 temporal continuity check (opt-in; default off preserves prior always-pass behavior)
    enable_temporal_continuity_check: bool
    temporal_continuity_max_gap_seconds: float

    # Step 3 per-frame pass gate. A frame "passes" for a track when that single frame's
    # summed detection confidence clears this. Historically this reused confidence_threshold
    # (the whole-track budget), making the per-frame gate the binding constraint on
    # low-confidence footage. Defaults to confidence_threshold to preserve prior behavior;
    # lower it to admit tracks whose confidence is spread across frames rather than concentrated.
    frame_pass_confidence_threshold: float
