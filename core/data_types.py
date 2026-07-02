"""
Complete typed data structures for wildlife video processing.

This file contains EVERY data structure used in the system.
NO dictionaries are allowed except for pure algorithm use.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from core.constants import PROCESSING_STATUS, PROCESSOR_VERSIONS

# =============================================================================
# CORE DETECTION TYPES
# =============================================================================

@dataclass
class BoundingBox:
    """Represents a bounding box with pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property 
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

@dataclass
class Detection:
    """Represents a single detection from ML model."""
    confidence: float
    bbox: BoundingBox
    source: str
    class_name: str
    timestamp: float = 0.0
    frame_idx: int = 0

@dataclass
class ScoredDetection:
    """A Detection plus the per-frame scoring fields the validator attaches.

    Replaces the ad-hoc dict the pipeline used to build mid-scoring; `detection`
    is the raw model output, the rest are computed during spatial validation.
    """
    detection: Detection
    boosted_confidence: float
    motion_overlap: float
    overlap_type: str
    consensus_boost: float = 1.0
    consensus_count: int = 1

@dataclass
class Track:
    """Represents a temporal track of detections."""
    track_id: int
    detections: List[Detection]
    start_frame: int
    end_frame: int
    duration_seconds: float
    confidence_scores: List[float]
    bbox_sequence: List[BoundingBox]

@dataclass
class MotionRegion:
    """Represents a motion region."""
    bbox: BoundingBox
    area: float
    frame_idx: int
    timestamp: float

@dataclass
class MotionTrack:
    """Represents a motion track."""
    track_id: int
    regions: List[MotionRegion]
    start_frame: int
    end_frame: int
    duration_seconds: float
    total_area: float

# =============================================================================
# SCORING AND ANALYSIS TYPES
# =============================================================================

@dataclass
class CompositeScore:
    """Represents a composite score calculation result."""
    final_score: float
    base_score: float
    temporal_density: float
    consensus_models: int
    motion_alignment: float
    duration_seconds: float
    temporal_multiplier: float
    consensus_multiplier: float
    motion_multiplier: float
    duration_bonus: float

    @classmethod
    def empty(cls) -> "CompositeScore":
        return cls(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

@dataclass
class ModelContribution:
    """Represents a model's contribution to ensemble detection."""
    model_name: str
    total_detections: int
    max_confidence: float
    contributing_tracks: int

@dataclass
class ValidationResult:
    """Result of animal validation process."""
    animals_detected: bool
    confidence: float
    ensemble_score: float
    composite_score: float
    validated_sequences_count: int
    best_detection: Detection
    reason: Optional[str] = None

# =============================================================================
# CLASSIFICATION TYPES
# =============================================================================

@dataclass
class InferenceResult:
    """Generic interface for all inference results."""
    model_name: str
    is_animal: bool
    animal_confidence: float
    species: Optional[str]
    species_confidence: float
    can_identify_species: bool  # True for BioCLIP, False for DeepFaune
    processing_time: float
    all_predictions: Optional[Dict[str, float]] = None  # species -> prob, when the model exposes it

@dataclass
class ClassificationResult:
    """Combined result from classification ensemble."""
    is_animal: bool
    animal_confidence: float
    species: Optional[str]
    species_confidence: float
    processing_time: float
    individual_results: List[InferenceResult] = field(default_factory=list)
    crop_path: Optional[str] = None
    crop_info: Optional[str] = None
    approving_model: Optional[str] = None  # Which model approved this as animal

@dataclass
class ClassifiedSequence:
    """Sequence with classification results."""
    sequence: 'ValidationSequence'  # Forward reference
    classification: ClassificationResult
    is_confirmed_animal: bool

@dataclass
class AnimalClassificationResult:
    """Result from animal classification step."""
    input_sequences_count: int
    animal_sequences: List[ClassifiedSequence]
    filtered_sequences_count: int
    species_counts: Dict[str, int]
    processing_time: float
    classification_enabled: bool
    all_classified_sequences: List[ClassifiedSequence] = field(default_factory=list)  # ALL attempts including filtered

# =============================================================================
# PIPELINE STEP TYPES
# =============================================================================

@dataclass
class StepTiming:
    """Timing information for pipeline steps."""
    start_time: float
    end_time: float
    duration: float

@dataclass(kw_only=True)
class StepResult:
    """Common fields shared by every pipeline-step result.

    kw_only so subclasses can add non-default fields without dataclass
    field-ordering conflicts; all construction sites use keyword arguments.
    """
    success: bool
    early_exit: bool = False
    early_exit_reason: Optional[str] = None

@dataclass
class MotionDetectionMetadata:
    """Metadata for motion detection step."""
    step_name: str
    timing: StepTiming
    tracks_found: int
    total_motion_area: float
    motion_method: str
    error: Optional[str] = None

@dataclass(kw_only=True)
class MotionDetectionResult(StepResult):
    """Complete result from motion detection step."""
    motion_tracks: List[MotionTrack]
    metadata: MotionDetectionMetadata

@dataclass
class CameraHandlingMetadata:
    """Metadata for camera handling filter step."""
    step_name: str
    timing: StepTiming
    tracks_input: int
    tracks_filtered: int
    camera_handling_detected: bool
    composite_motion_score: float
    error: Optional[str] = None

@dataclass(kw_only=True)
class CameraHandlingResult(StepResult):
    """Complete result from camera handling filter step."""
    motion_tracks: List[MotionTrack]
    metadata: CameraHandlingMetadata

@dataclass
class ValidationSequence:
    """A validated sequence of detections."""
    sequence_id: int
    track: Track
    detections: List[Detection]
    ensemble_score: float
    composite_score: float
    best_detection: Detection
    frame_range: tuple[int, int]
    duration_seconds: float

@dataclass
class FullFrameValidationMetadata:
    """Metadata for full-frame validation step."""
    step_name: str
    timing: StepTiming
    tracks_evaluated: int
    tracks_passed: int
    sequences_validated: int
    model_contributions: List[ModelContribution]
    error: Optional[str] = None

@dataclass(kw_only=True)
class FullFrameValidationResult(StepResult):
    """Complete result from full-frame validation step."""
    validated_sequences: List[ValidationSequence]
    metadata: FullFrameValidationMetadata

# =============================================================================
# VIDEO ANALYSIS TYPES
# =============================================================================

@dataclass
class VideoAnalysis:
    """Complete analysis results for a single video."""
    video_path: Path
    video_name: str
    processing_time: float
    validation_result: ValidationResult
    motion_result: MotionDetectionResult
    camera_result: CameraHandlingResult
    fullframe_result: FullFrameValidationResult
    species_summary: Optional[str] = None  # Step 4: Species information
    classification_result: Optional[AnimalClassificationResult] = None  # Step 4: Full results

@dataclass
class ProcessedVideoRecord:
    """Record of processed video for tracking."""
    video_file: str
    processed_timestamp: datetime
    processing_status: str
    processor_version: str
    animals_detected: bool = False
    confidence: Optional[float] = None
    ensemble_score: Optional[float] = None
    processing_time: Optional[float] = None
    best_detection: Optional[Detection] = None
    reason: Optional[str] = None

# =============================================================================
# SESSION AND BATCH PROCESSING TYPES
# =============================================================================

@dataclass
class TimingStatistics:
    """Timing statistics for processing session."""
    total_batch_time: float
    total_video_time: float
    average_per_video: float
    videos_processed: int

@dataclass
class ProcessingSessionData:
    """Internal session tracking data."""
    videos_to_process: List[str] = field(default_factory=list)
    total_videos: int = 0
    processing_times: List[tuple[str, float]] = field(default_factory=list)
    composite_scores: List[tuple[str, float]] = field(default_factory=list)
    rejection_reasons: List[tuple[str, str]] = field(default_factory=list)
    model_contributions: List[tuple[str, List[ModelContribution]]] = field(default_factory=list)

@dataclass
class SessionSummary:
    """Summary of processing session."""
    total_videos: int
    successful_videos: int
    failed_videos: int
    timing_stats: TimingStatistics
    analyses: List[VideoAnalysis]
    failed_video_paths: List[Path]

@dataclass
class BatchProcessingResult:
    """Result of batch video processing operation."""
    success: bool
    summary: Optional[SessionSummary] = None
    videos_processed: int = 0
    reason: Optional[str] = None

# =============================================================================
# ERROR AND STATUS TYPES
# =============================================================================

@dataclass
class ProcessingStatus:
    """Current status of processing operation."""
    videos_total: int
    videos_processed: int
    videos_successful: int
    videos_failed: int
    current_video: Optional[str] = None
    elapsed_time: float = 0.0
    estimated_remaining: Optional[float] = None

# =============================================================================
# CONFIGURATION TYPES
# =============================================================================

@dataclass
class ConfigurationUpdate:
    """Typed object for configuration updates."""
    confidence_threshold: Optional[float] = None
    motion_method: Optional[str] = None
    tracking_method: Optional[str] = None
    ensemble_models: Optional[List[str]] = None
    motion_var_threshold: Optional[int] = None
    min_motion_area: Optional[int] = None
    max_motion_area: Optional[int] = None
    spatial_overlap_threshold: Optional[float] = None
    composite_motion_threshold: Optional[float] = None

# =============================================================================
# UTILITY FUNCTIONS FOR CREATION
# =============================================================================

def create_processing_record(
    video_path: Path,
    analysis: Optional[VideoAnalysis] = None,
    success: bool = True,
    reason: Optional[str] = None
) -> ProcessedVideoRecord:
    """Create a ProcessedVideoRecord from analysis results."""
    from core.constants import CURRENT_PROCESSOR_VERSION
    
    if analysis and success:
        return ProcessedVideoRecord(
            video_file=video_path.name,
            processed_timestamp=datetime.now(),
            processing_status=PROCESSING_STATUS['success'],
            processor_version=CURRENT_PROCESSOR_VERSION,
            animals_detected=True,
            confidence=analysis.validation_result.confidence,
            ensemble_score=analysis.validation_result.ensemble_score,
            processing_time=analysis.processing_time,
            best_detection=analysis.validation_result.best_detection
        )
    else:
        return ProcessedVideoRecord(
            video_file=video_path.name,
            processed_timestamp=datetime.now(),
            processing_status=PROCESSING_STATUS['no_animals'] if not success else PROCESSING_STATUS['error'],
            processor_version=CURRENT_PROCESSOR_VERSION,
            animals_detected=False,
            reason=reason or 'no_consistent_animal_movement'
        )

def create_empty_motion_result(error: str) -> MotionDetectionResult:
    """Create an empty motion detection result for errors."""
    return MotionDetectionResult(
        success=False,
        motion_tracks=[],
        metadata=MotionDetectionMetadata(
            step_name='motion_detection',
            timing=StepTiming(0.0, 0.0, 0.0),
            tracks_found=0,
            total_motion_area=0.0,
            motion_method='unknown',
            error=error
        )
    )

def create_empty_camera_result(error: str) -> CameraHandlingResult:
    """Create an empty camera handling result for errors."""
    return CameraHandlingResult(
        success=False,
        motion_tracks=[],
        metadata=CameraHandlingMetadata(
            step_name='camera_handling',
            timing=StepTiming(0.0, 0.0, 0.0),
            tracks_input=0,
            tracks_filtered=0,
            camera_handling_detected=False,
            composite_motion_score=0.0,
            error=error
        )
    )

def create_empty_validation_result(error: str) -> FullFrameValidationResult:
    """Create an empty validation result for errors."""
    return FullFrameValidationResult(
        success=False,
        validated_sequences=[],
        metadata=FullFrameValidationMetadata(
            step_name='fullframe_validation',
            timing=StepTiming(0.0, 0.0, 0.0),
            tracks_evaluated=0,
            tracks_passed=0,
            sequences_validated=0,
            model_contributions=[],
            error=error
        )
    )