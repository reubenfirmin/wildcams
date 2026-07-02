"""
Constants for wildlife video processing system.

Centralizes all magic strings, model names, file paths, and configuration
constants to follow DRY principle and eliminate magic strings throughout codebase.
"""

from typing import Final, List, Set, Dict, Tuple

# =============================================================================
# MODEL CONSTANTS
# =============================================================================

# Default ensemble models for wildlife detection
DEFAULT_ENSEMBLE_MODELS: Final[List[str]] = [
    'yolo12x', 
    'yolo12m', 
    'MDV6-yolov10-e', 
    'rtdetr-l'
]

# Available YOLO model variants
YOLO_VARIANTS: Final[List[str]] = [
    'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
    'yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x',
    'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x'
]

# MegaDetector model variants
MEGADETECTOR_VARIANTS: Final[List[str]] = [
    'MDV6-yolov9-c', 
    'MDV6-yolov9-e',
    'MDV6-yolov10-c', 
    'MDV6-yolov10-e',
    'MDV6-rtdetr-c'
]

# RT-DETR model variants
RTDETR_VARIANTS: Final[List[str]] = [
    'rtdetr-l',
    'rtdetr-x'
]

# All available models combined
ALL_AVAILABLE_MODELS: Final[List[str]] = YOLO_VARIANTS + MEGADETECTOR_VARIANTS + RTDETR_VARIANTS

# Model file extensions
MODEL_EXTENSIONS: Final[Set[str]] = {'.pt', '.onnx', '.engine'}
PRIMARY_MODEL_EXTENSION: Final[str] = '.pt'

# =============================================================================
# PROCESSING METHOD CONSTANTS
# =============================================================================

# Motion detection methods
MOTION_METHODS: Final[List[str]] = ['MOG2', 'KNN']
DEFAULT_MOTION_METHOD: Final[str] = 'MOG2'

# Tracking methods
TRACKING_METHODS: Final[List[str]] = ['auto', 'deepsort', 'simple']
DEFAULT_TRACKING_METHOD: Final[str] = 'auto'
TRACKING_METHOD_ALIASES: Final[Dict[str, str]] = {
    'bbox': 'simple',
    'bbox_simple': 'simple'
}

# =============================================================================
# DIRECTORY AND FILE CONSTANTS
# =============================================================================

# Default directories
DEFAULT_VIDEO_DIR: Final[str] = './videos'
DEFAULT_CACHE_DIR: Final[str] = './models_cache'
DEFAULT_ANALYSIS_DIR: Final[str] = './analysis'

# Subdirectory names
TRACKING_SUBDIR: Final[str] = '.tracking'
CACHE_SUBDIR: Final[str] = '.cache'
DEBUG_SUBDIR: Final[str] = 'debug'

# File markers and extensions
PROCESSED_MARKER: Final[str] = '.processed'
ANALYSIS_EXTENSION: Final[str] = '.json'
CLUSTER_FILENAME: Final[str] = 'clusters.json'
FEATURES_FILENAME: Final[str] = 'features.pkl'

# Video file extensions
VIDEO_EXTENSIONS: Final[Set[str]] = {
    '.mp4', '.MP4',
    '.mov', '.MOV', 
    '.avi', '.AVI',
    '.mkv', '.MKV',
    '.wmv', '.WMV'
}

# =============================================================================
# DETECTION AND CLASSIFICATION CONSTANTS
# =============================================================================

# MegaDetector class names
MEGADETECTOR_CLASSES: Final[List[str]] = ['animal', 'person', 'vehicle', 'empty']
MEGADETECTOR_ANIMAL_CLASS: Final[str] = 'animal'
MEGADETECTOR_PERSON_CLASS: Final[str] = 'person'
MEGADETECTOR_VEHICLE_CLASS: Final[str] = 'vehicle'
MEGADETECTOR_EMPTY_CLASS: Final[str] = 'empty'

# Processing result types
PROCESSING_RESULTS: Final[List[str]] = [
    'animals_detected',
    'no_animals_detected', 
    'camera_handling_detected',
    'processing_error'
]

# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

# Logger names
MAIN_LOGGER_NAME: Final[str] = 'wildcams'
ANALYSIS_LOGGER_NAME: Final[str] = 'analysis'
DEBUG_LOGGER_NAME: Final[str] = 'debug'

# Log message prefixes
LOG_PREFIXES: Final[Dict[str, str]] = {
    'video_start': '🎬',
    'video_success': '✅',
    'video_error': '❌', 
    'video_skip': '⚪',
    'frame_result': '📹',
    'detection': '🦌',
    'tracking': '🎯',
    'motion': '🔍',
    'camera_handling': '📷',
    'pipeline': '⚙️',
    'timing': '⏱️',
    'summary': '📊'
}

# =============================================================================
# THRESHOLD AND PARAMETER CONSTANTS
# =============================================================================

# Default confidence thresholds
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.8
DEFAULT_MEGADETECTOR_HIGH_CONF: Final[float] = 0.3
DEFAULT_YOLO_HIGH_CONF: Final[float] = 0.4
DEFAULT_WEAK_EVIDENCE_THRESHOLD: Final[float] = 0.25
DEFAULT_WILDLIFE_MODEL_CONFIDENCE: Final[float] = 0.2

# Default motion detection parameters
DEFAULT_MOTION_VAR_THRESHOLD: Final[int] = 32
DEFAULT_MIN_MOTION_AREA: Final[int] = 300
DEFAULT_MAX_MOTION_AREA: Final[int] = 80000
DEFAULT_MOTION_HISTORY: Final[int] = 100

# Default tracking parameters
DEFAULT_TRACKING_MAX_AGE: Final[int] = 50
DEFAULT_TRACKING_N_INIT: Final[int] = 3
DEFAULT_MIN_TRACK_DETECTIONS: Final[int] = 3

# Default temporal parameters
DEFAULT_MIN_TRACK_DURATION: Final[float] = 0.1
DEFAULT_MOTION_GAP_SECONDS: Final[float] = 1.0
DEFAULT_MIN_CONSECUTIVE_DETECTION_SECONDS: Final[float] = 0.2

# Frame processing constants
STANDARD_FRAME_WIDTH: Final[int] = 1600
STANDARD_FRAME_HEIGHT: Final[int] = 900
INITIAL_BEST_SCORE: Final[float] = 0.0
INITIAL_CONFIDENCE_THRESHOLD: Final[float] = 0.1

# Dictionary keys for consistency (legacy support)
DETECTION_KEYS: Final[Dict[str, str]] = {
    'bbox': 'bbox',
    'confidence': 'confidence', 
    'source': 'source',
    'class': 'class',
    'unknown': 'unknown'
}

# Common analysis status messages
ANALYSIS_STATUS_MESSAGES: Final[Dict[str, str]] = {
    'video_success': 'Animal detected with temporal consistency',
    'video_skipped': 'No consistent animal movement detected',
    'video_error': 'Processing failed',
    'pipeline_complete': 'Pipeline completed',
    'features_extracted': 'Features extracted for clustering'
}

# =============================================================================
# PROCESSOR VERSION CONSTANTS
# =============================================================================

# Processor versions for backwards compatibility
PROCESSOR_VERSIONS: Final[Dict[str, str]] = {
    'legacy': 'legacy_processor',
    'phase3': 'modular_pipeline_phase3',
    'phase4': 'modular_pipeline_phase4', 
    'phase5': 'core_architecture_phase5'
}

CURRENT_PROCESSOR_VERSION: Final[str] = PROCESSOR_VERSIONS['phase5']

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Processing step names for pipeline validation
PIPELINE_STEP_NAMES: Final[List[str]] = [
    'motion_detection',
    'camera_handling_filter', 
    'fullframe_validation'
]

# Early exit reasons
EARLY_EXIT_REASONS: Final[List[str]] = [
    'no_motion_detected',
    'camera_handling_detected',
    'insufficient_motion_data',
    'processing_error'
]

# Status constants for processed files
PROCESSING_STATUS: Final[Dict[str, str]] = {
    'success': 'success',
    'no_animals': 'no_animals',
    'camera_handling': 'camera_handling',
    'error': 'error'
}

# =============================================================================
# FEATURE EXTRACTION CONSTANTS
# =============================================================================

# Feature extraction models
FEATURE_EXTRACTION_MODELS: Final[List[str]] = [
    'resnet18',
    'resnet50',
    'efficientnet_b0'
]

DEFAULT_FEATURE_MODEL: Final[str] = 'resnet18'
EXPECTED_FEATURE_DIMENSIONS: Final[Dict[str, int]] = {
    'resnet18': 512,
    'resnet50': 2048,
    'efficientnet_b0': 1280
}

# =============================================================================
# ERROR MESSAGE CONSTANTS
# =============================================================================

# Common error messages
ERROR_MESSAGES: Final[Dict[str, str]] = {
    'video_not_found': 'Video file not found: {path}',
    'model_not_found': 'Model file not found: {model_name}',
    'invalid_config': 'Invalid configuration parameter: {param}',
    'processing_failed': 'Video processing failed: {error}',
    'feature_extraction_failed': 'Feature extraction failed: {error}',
    'clustering_failed': 'Clustering analysis failed: {error}'
}

# =============================================================================
# UTILITY FUNCTIONS FOR CONSTANTS
# =============================================================================

def is_valid_model_name(model_name: str) -> bool:
    """Check if model name is in the list of available models."""
    return model_name in ALL_AVAILABLE_MODELS

def is_valid_motion_method(method: str) -> bool:
    """Check if motion detection method is valid."""
    return method in MOTION_METHODS

def is_valid_tracking_method(method: str) -> bool:
    """Check if tracking method is valid."""
    return method in TRACKING_METHODS or method in TRACKING_METHOD_ALIASES

def normalize_tracking_method(method: str) -> str:
    """Normalize tracking method name using aliases."""
    return TRACKING_METHOD_ALIASES.get(method, method)

def is_video_file(filename: str) -> bool:
    """Check if filename has a valid video extension."""
    return any(filename.endswith(ext) for ext in VIDEO_EXTENSIONS)

def get_log_prefix(category: str) -> str:
    """Get emoji prefix for log category."""
    return LOG_PREFIXES.get(category, '📝')