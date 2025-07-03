# Wildlife Camera Trap Video Processing System

## Project Overview
Automated wildlife video processing system for Costa Rican jungle camera footage. Detects animals using ensemble ML models, filters false positives, and clusters videos based on visual similarity of detected animals.

## TODO list
* ✅ Camera handling detection fixed (inverted logic implemented)
* ✅ Full-frame analysis prioritized (crop analysis eliminated)
* Tune motion sensitivity for small animals (test MIN_MOTION_AREA 200-500, MOTION_VAR_THRESHOLD 15-35)
* Test full video set (20 videos) with current pipeline to validate performance

## Current System Architecture (Next-Generation 4-Step Pipeline)

### Core Components
1. **SD Card Watcher** (`sd_watcher.py`) - Automatically detects and downloads videos from camera SD cards
2. **Processing System**:
   - **`process.py`** - Next-generation 4-step pipeline processor (primary)
   - **`video_processor_base.py`** - Shared base class with common functionality and CLI management
   - **`ml_detection.py`** - Shared 3-model ML ensemble
3. **Nix Development Environment** (`flake.nix`) - Reproducible development setup with uv Python package manager

### 3-Step Pipeline Architecture

The system uses an optimized 3-step pipeline that eliminates crop analysis issues by connecting motion tracking directly to full-frame validation:

#### **Step 1: Motion Detection + DeepSORT Temporal Tracking**
- **Motion Detection**: MOG2/KNN background subtraction to identify movement regions
- **DeepSORT Integration**: Robust temporal consistency tracking of detections across frames (Note: DeepSORT not currently implemented - using simple bbox linking)
- **Temporal Consistency**: Builds motion tracks across multiple frames with configurable skip frames
- **Motion Region Collection**: Captures spatial regions of movement for later spatial validation
- **Fallback Mode**: Simple bbox linking if DeepSORT unavailable

#### **Step 2: Camera Handling Detection (Early Filtering)**
- **Spatial Dispersion Analysis**: Ratio of spatial clusters to tracks (dispersed movement = camera handling)
- **Motion Sparsity Analysis**: Inverted motion density (sparse erratic movement = camera handling)
- **Composite Scoring**: spatial_dispersion × motion_sparsity with configurable threshold (default: 8.0)
- **Early Exit**: Skips expensive ML processing on obvious false positives

#### **Step 3: Full-Frame Analysis with Spatial Overlap Validation (NEW APPROACH)**
- **Direct Motion Track Input**: Takes motion tracks directly from Step 2 (no crop analysis)
- **Full Ensemble Processing**: Complete ensemble on full frames using configured models (YOLO + RT-DETR + MegaDetector variants)
- **Temporal Frame Sampling**: Selects representative frames from each motion track (configurable via --max-validation-frames)
- **Spatial Overlap Validation**: Requires 30% overlap between full-frame detections and motion regions
- **Model Contribution Tracking**: Comprehensive tracking of all ensemble models in full-frame context
- **Validation Logic**: Simple confidence thresholding with spatial overlap requirement
- **False Positive Reduction**: Spatial validation eliminates detections that don't correlate with motion
- **Full-Frame Optimization**: All models process complete frames for optimal context and spatial understanding

### Available ML Models

**Current Default Ensemble**: yolo12x, yolo12m, MDV6-yolov10-e, rtdetr-l

**Additional Available Models**:
- **YOLOv10**: n/s/m/b/l/x variants (end-to-end optimized, no NMS)
- **YOLOe11**: n/s/m/l/x variants (newer efficient architecture)
- **RT-DETR**: l/x variants (Transformer-based, full-frame only)
- **MegaDetector v6**: YOLOv9-c/e, YOLOv10-c/e variants

### RT-DETR Technical Details

**Architecture**: Vision Transformer-based real-time object detector with hybrid encoder
**Key Features**:
- **Global Context Processing**: Transformer architecture for comprehensive spatial understanding
- **Multi-scale Feature Fusion**: Efficient hybrid encoder with cross-scale fusion capabilities  
- **IoU-aware Query Selection**: Focuses on most relevant objects for improved accuracy
- **End-to-End Detection**: No NMS required, reducing post-processing overhead
- **Full-Frame Requirement**: Requires complete image context for optimal performance

**Integration**: 
- **Full-Frame Only**: Runs in Step 3 alongside all other models on complete frames
- **Spatial Context**: Leverages transformer architecture for comprehensive scene understanding
- **Motion Correlation**: Detections analyzed for spatial overlap with motion regions from Step 1

### MegaDetector v6 Technical Details

**Training Data**: 3+ million camera trap images from diverse global ecosystems
**Classes**: 3-class taxonomy (Animal, Person, Vehicle) optimized for conservation monitoring
**Performance**: 94.6% accuracy on motion-triggered camera trap images

**Domain-Specific Optimizations**:
- **Camera Trap Training**: Specialized on motion-triggered images from static camera perspectives
- **Environmental Robustness**: Multi-location training across varying lighting, weather, seasonal conditions
- **Confidence Calibration**: Thresholds optimized specifically for wildlife detection scenarios
- **Model Efficiency**: MDV6-compact has 2% of parameters vs. previous versions while maintaining accuracy

**Repeat Detection Elimination (RDE)**:
- **Static Object Filtering**: Leverages fixed camera perspectives to identify false positives
- **Temporal Analysis**: Detects objects (branches, snow, litter) appearing in same locations across frames
- **Camera Trap Context**: Designed for cameras taking thousands of images from identical perspectives
- **Implementation Status**: Available in PyTorch-Wildlife framework, not currently integrated in wildcams system

**Limitations**:
- **Time-Lapse Performance**: Poor accuracy (≤61.6%) on time-lapse vs. motion-triggered images
- **Class Restriction**: Limited to 3 classes vs. 80+ in COCO-trained models
- **Extended Classes**: Some models output non-standard class IDs outside the standard mapping:
  - **Standard Classes**: 0=animal, 1=person, 2=vehicle
  - **Unknown Classes**: 102, 147, 166, 178, 197, 250, 252, 278 (UNKNOWN classes not in standard MegaDetector mapping)

### Frame-First Full-Frame Analysis Algorithm

The Step 3 full-frame analysis follows a specific algorithm to optimize ML model usage and ensure comprehensive track evaluation:

```python
# PHASE 1: Frame-First Processing (avoid duplicate ML runs)
for frame in {sampled frames with sufficient density}:
    # Run each model against the frame
    for model in models:
        detect = run_model_detection(model, frame)
        
        # For each detection, determine overlap with ALL tracks
        for track in tracks:
            overlap = calculate_overlap(detection.bbox, track.bbox_for_frame)
            if overlap >= spatial_overlap_threshold:
                log_detection(✅, model, detection, overlap, track, "spatial_valid")
            elif overlap > 0:
                log_detection(⚠️, model, detection, overlap, track, "threshold_failed") 
            # Note: No logging for zero overlap (spatial_invalid) unless debug enabled
        
        # If no detections overlapped with any track
        if no_detections_overlapped:
            log_summary(❌, model, f"{num_detections} detections with below threshold overlap")
    
    # Calculate ensemble score for this frame across all tracks
    ensemble_score = calculate_weighted_ensemble_score(all_model_results)
    log_ensemble(frame, ensemble_score, valid_models, valid_detections, reason)

# PHASE 2: Track-Level Evaluation (collect statistics and validate)
for track in tracks:
    # Collect all frame results for this track
    track_detections = get_validated_detections_for_track(track)
    track_stats = calculate_track_statistics(track_detections)
    
    # Evaluate track validation criteria
    confidence_passed = track_stats.summed_confidence >= confidence_threshold
    frames_passed = len(track_detections) >= min_track_frames
    temporal_continuity_passed = check_temporal_gaps(track_detections, max_gap_seconds)
    
    # Final validation decision
    validation_passed = confidence_passed AND frames_passed AND temporal_continuity_passed
    log_track_evaluation(✅/❌, track, validation_passed, all_metrics)
```

**Key Principles:**
1. **Frame-First**: Process each frame once across all models to avoid duplicate ML runs
2. **Model-Track Matrix**: Each model's detections checked against ALL tracks for comprehensive overlap analysis  
3. **Hierarchical Logging**: EVAL (individual) → ENSEMBLE (frame-level) → TRACK (final validation)
4. **Spatial Validation**: All detections must spatially correlate with motion regions (explicit or implicit via backfill/forward-fill)
5. **Temporal Continuity**: Final track validation requires detections within temporal gap thresholds

### False Positive Filtering (Multi-Level)
**Hardware-Level Detection (Step 2)**:
- **Camera Handling**: >20 temporal tracks with high density/duration
- **Camera Shake**: >10 tracks lasting >300 seconds

**Algorithm-Level Validation (Step 3)**:
- **Spatial Correlation**: Requires 30% overlap between full-frame detections and motion regions
- **Confidence Thresholding**: Simple confidence-based validation with spatial requirement
- **Temporal Sampling**: Representative frame selection from motion tracks

## Hardware & Performance
- **Target Platform**: AMD Ryzen 7 5700G (16 threads)
- **Processing Speed**: Next-Gen Pipeline: ~5-10 seconds per video (estimated)
- **Memory Usage**: 2-4GB RAM during inference
- **Storage**: Debug frames and logs organized in subdirectories

## Configuration (.env)
```bash
# Video processing
MAX_FRAMES_PER_VIDEO=20          # Sample frames for anchor detection
CONFIDENCE_THRESHOLD=0.1         # Base ML detection threshold

# Validation thresholds (tuned for Costa Rica jungle footage)
MEGADETECTOR_HIGH_CONFIDENCE=0.3  # MegaDetector validation threshold
YOLO_HIGH_CONFIDENCE=0.4          # YOLO validation threshold
MIN_YOLO_DETECTIONS=3             # Minimum YOLO detections for validation
WEAK_EVIDENCE_THRESHOLD=0.25      # Weak evidence validation threshold

# Camera handling detection (Step 2)
DETECTION_DENSITY_THRESHOLD=15.0
LOW_CONFIDENCE_RATIO_THRESHOLD=0.7
LOW_CONFIDENCE_CUTOFF=0.2

# Motion detection settings (Step 1 - camera trap optimized)
MOTION_METHOD=MOG2
MOTION_VAR_THRESHOLD=32           # Less sensitive to small movements
MIN_MOTION_AREA=300               # Focus on significant motion (reduced from 2000)
MAX_MOTION_AREA=80000             # Avoid full-frame motion detection
MOTION_HISTORY=100                # Background model adaptation speed
MAX_REGIONS_PER_FRAME=10          # Limit processed regions to reduce noise
MIN_REGION_WIDTH=30               # Filter out thin/small regions
MIN_REGION_HEIGHT=30
MAX_ASPECT_RATIO=5.0              # Reject overly elongated regions
MOTION_MARGIN=30                  # Pixels to expand regions for ML context

# Temporal consistency parameters (Step 1 - DeepSORT integration)
MIN_TRACK_DURATION=1.0            # Minimum track duration seconds (reduced from 2.0)
MAX_SKIP_FRAMES=3                 # Max frames to skip in tracking
TRACKING_DISTANCE_THRESHOLD=100.0 # Max distance for tracking association pixels
ANCHOR_CONFIDENCE_THRESHOLD=0.5   # Minimum confidence for anchor point detection
MIN_TRACK_FRAMES=1                # Minimum frames required for valid track

# Step 3 validation parameters (updated for direct full-frame approach)
MAX_VALIDATION_FRAMES=5           # Maximum frames to sample per motion track
SPATIAL_OVERLAP_THRESHOLD=0.3     # Required overlap between detections and motion regions (30%)
TEMPORAL_SPREAD_SECONDS=2.0       # Temporal spread for frame sampling from motion tracks
```

## Commands
```bash
# SD card monitoring
start      # Start daemon
stop       # Stop daemon  
logs       # View logs
check      # Check daemon status

# Next-Generation 3-Step Pipeline (Primary)
process -v 7 8 9                 # Process specific videos
process                          # Process all videos
process -v IMG_0011.MP4          # Process by filename

# Next-Generation tuning examples
process --min-motion-area 100 --min-track-duration 0.5       # More sensitive motion
process --max-validation-frames 3 --conf 0.35               # Stricter validation
process --tracking-distance-threshold 50 --max-skip-frames 5 # Stricter tracking

# Available MegaDetector variants (-m):
# MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e

# Available ensemble models (-e):
# yolo12x, yolo12m, yolov8x, yolov8m, yolov8n, yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x, yoloe11n, yoloe11s, yoloe11m, yoloe11l, yoloe11x, rtdetr-l, rtdetr-x, MDV6-yolov10-e (comma-separated)

# Next-Generation Parameters:
# --conf                          Detection confidence threshold (0.0-1.0, default: 0.25)
# --max-validation-frames         Max frames to sample per motion track (1-10, default: 5)
# --spatial-overlap-threshold     Spatial overlap threshold between detections and motion regions (0.0-1.0, default: 0.5)
# --detection-density-threshold  Camera handling detection threshold (1.0-50.0, default: 15.0)
# --composite-motion-threshold   Composite motion threshold for camera handling (default: 1000000)
# --min-motion-threshold         Minimum motion threshold to avoid processing static videos (default: 100)
# --motion-frames-weight         Weight exponent for motion frames in composite score (default: 1.2)
# --motion-regions-weight        Weight exponent for motion regions in composite score (default: 1.1)
# --motion-tracks-weight         Weight exponent for motion tracks in composite score (default: 1.0)
# --large-region-multiplier      Multiplier for large region percentage in composite score (default: 15.0)

# Motion Detection Parameters (Step 1):
# --motion-method                Motion detection method (MOG2/KNN, default: MOG2)
# --motion-var-threshold         Variance threshold - higher = less sensitive (default: 32)
# --min-motion-area              Min motion area pixels (default: 300)
# --max-motion-area              Max motion area pixels (default: 80000)
# --motion-history               Background history frames (default: 100)
# --max-regions-per-frame        Max regions processed per frame (default: 10)
# --min-region-width             Min region width pixels (default: 30)
# --min-region-height            Min region height pixels (default: 30)
# --max-aspect-ratio             Max width/height ratio (default: 5.0)
# --motion-margin                Margin to expand regions pixels (default: 30)

# Temporal Consistency Parameters (Step 1 - DeepSORT):
# --min-track-duration           Min track duration seconds (default: 1.0)
# --max-skip-frames              Max frames to skip in tracking (default: 3)
# --tracking-distance-threshold  Max distance for tracking association pixels (default: 100.0)
# --anchor-confidence-threshold  Minimum confidence for anchor point detection (default: 0.5)
# --min-track-frames             Minimum frames required for valid track (default: 1)

# Step 3 Full-Frame Analysis Parameters:
# --max-validation-frames        Maximum frames to sample per motion track (default: 5)
# --spatial-overlap-threshold    Spatial overlap threshold between detections and motion regions (default: 0.5)
# --temporal-spread-seconds      Temporal spread for frame sampling (default: 2.0)
```

## Scientific Logging Format

The system produces structured scientific logs with three types of evaluation rows:

### EVAL Rows (Model Evaluation)
```
EVAL | {video} | {timestamp} | {frame} | {track_id}
{✅/❌} | {model} | {bbox} | {conf} | {overlap%} | {motion_bbox} | {score}
```
- **Header**: Video name, timestamp, frame number, track ID
- **Rows**: One per model showing detection outcome
- **✅**: Model detected animal with spatial overlap ≥ threshold
- **❌**: Model found no detection or insufficient spatial overlap

### ENSEMBLE Rows (Frame-Level Combination)
```
ENSEMBLE | {video} | {timestamp} | {frame} | {track_id}
{✅/❌} | combined | valid_models={count} | total_score={sum} | valid_detections={count} | frame_score={avg}
```
- **Header**: Same context as EVAL
- **Row**: Combined results from all models for this frame
- **✅**: At least one model found spatially valid detection
- **❌**: No models found spatially valid detections

### TRACK Rows (Final Track Validation)
```
TRACK | {video} | {track_id}
{✅/❌} | duration={s} | frames={count} | detections={count} | models_active={count} | summed_conf={total} | avg_conf={avg} | max_conf={max} | duration_norm={norm} | validated={bool}
```
- **Header**: Video name and track ID
- **Row**: Final track validation decision
- **✅**: Track passed validation (summed_conf ≥ confidence_threshold)
- **❌**: Track failed validation

### Key Metrics
- **ensemble_score**: Confidence-weighted combination of all model outputs (0.0-1.0 scale)
- **spatial_valid**: Detection bbox overlaps ≥ accepted_rtdetr_overlap with motion regions
- **temporal_continuity**: Validated detections must be within max_skip_frames of each other
- **models_active**: Number of models contributing spatially valid detections

### Current Validation Approach

The system uses **spatial overlap validation** with simple confidence thresholding:

#### Validation Algorithm
1. **Frame Sampling**: Select representative frames from each motion track (--max-validation-frames)
2. **Model Evaluation**: Run each ensemble model on selected full frames
3. **Spatial Validation**: Require minimum overlap (--spatial-overlap-threshold) between detections and motion regions
4. **Temporal Consistency**: Validate detection continuity across consecutive frames (--min-consecutive-detection-seconds)
5. **Track Validation**: Accept tracks with sufficient validated detections meeting confidence threshold

#### Key Properties
- **Spatial Correlation**: All detections must spatially correlate with motion regions
- **Temporal Continuity**: Validated detections must form temporally consistent sequences
- **Simple Scoring**: Direct confidence-based validation without complex ensemble weighting
- **Model Transparency**: Each model's contribution explicitly logged and tracked

## File Structure
```
wildcams/
├── Core Processing
│   ├── process.py                  # Next-generation 3-step pipeline (primary)
│   ├── video_processor_base.py     # Shared base class and CLI management
│   └── ml_detection.py             # Shared 4-model ensemble with RT-DETR handling
├── Infrastructure
│   ├── sd_watcher.py               # SD card monitoring
│   ├── flake.nix                   # Nix environment
│   └── pyproject.toml              # Python dependencies
├── Documentation
│   ├── CLAUDE.md                   # This file
│   ├── README.md                   # Project overview
│   └── motion_detection.md         # Motion detection strategy
├── Output
│   ├── logs/                       # Centralized logging
│   └── videos/analysis/            # Analysis results
└── Utilities
    └── check_gpu.py               # GPU compatibility check
```



---

## Research Applications

### Biodiversity Monitoring
- **Species Richness**: Automated counts of unique species detected
- **Activity Patterns**: When different species are most active
- **Habitat Usage**: Which areas have highest species diversity
- **Population Trends**: Changes in detection frequency over time

### Conservation Impact
- **Threat Assessment**: Monitor presence of endangered species (*Panthera onca*, *Tapirus bairdii*)
- **Ecosystem Health**: Species diversity as environmental indicator
- **Anti-Poaching**: Automated alerts for rare/protected species
- **Research Data**: High-quality taxonomic data for ecological studies

## Technical Considerations

### Model Size & Dependencies
- **BioCLIP Model**: ~600MB download
- **PyTorch Dependency**: Already included in current system
- **Memory Usage**: +1-2GB RAM during species classification
- **Storage**: Species embeddings cached for clustering efficiency

### Integration with 4-Step Pipeline
- **Efficiency Gain**: Step 3 provides focused crops, Step 4 validates before BioCLIP
- **Quality Improvement**: Validated detections should improve species identification accuracy
- **Resource Optimization**: Only validated animals classified = minimal processing overhead

---

### CRITICAL RULE: NO HARDCODED CONSTANTS
**NEVER hardcode values in the code.** Always use CLI parameters with defaults.

**Wrong:**
```python
results = detector(frame, conf=0.05, verbose=False)  # WRONG - hardcoded
```

**Right:**
```python
results = detector(frame, conf=config.detection_threshold, verbose=False)  # CLI param
```

## Experimental Analysis Guide

### How to Generate Strategy Comparison Tables

For future Claude instances analyzing experimental results, follow this methodology:

#### 1. Log File Analysis
```bash
# Identify experimental log files
ls logs/wildcams_*.log | grep [date_pattern]

# Extract key information from each log:
# - Start/end timestamps for processing time
# - Video processing results (SUCCESS/SKIPPED/FAILED)
# - Animal detection results per video
# - 4-step pipeline metrics (Step 1-4 outcomes)
# - DeepSORT tracking statistics
# - Weighted validation scores
```

#### 2. Ground Truth Reference
Always reference the validated sample set:
- **Videos 7, 8, 9, 11, 12**: True positives (should detect animals)
- **Videos 10**: Marginal (ideally should detect animals but not required)
- **Videos 1-6**: False positives (should NOT detect animals)  
- **Videos 13-19**: Camera handling (should be filtered in Step 2)

#### 3. Results Table Structure
Create tables with these columns:
- **Strategy/Model**: Processing approach (Next-Gen 4-step vs Legacy)
- **Animals Detected**: List video numbers where animals were found
- **No Animals**: List video numbers where no animals were found  
- **Step 2 Filtered**: Videos filtered by camera handling detection
- **True Positives**: Count and percentage of correct animal detections
- **False Positives**: Count and percentage of incorrect animal detections
- **Precision**: TP / (TP + FP) - accuracy of positive predictions
- **Processing Time**: Average time per video

#### 4. Analysis Steps
1. **Extract Pipeline Metrics**: Parse Step 1-4 outcomes and timing
2. **Calculate Validation Scores**: Analyze crop vs full-frame scoring weights
3. **Track DeepSORT Performance**: Temporal consistency improvements
4. **Measure RT-DETR Impact**: Full-frame validation accuracy
5. **Processing Time Analysis**: 4-step pipeline vs legacy performance

#### 5. Critical Validation Points
- **Step Separation**: Verify clean boundaries between processing steps
- **Model Usage**: Confirm RT-DETR exclusion from Step 3, inclusion in Step 4
- **Temporal Tracking**: Validate DeepSORT improvements vs simple linking
- **Weighted Scoring**: Analyze crop+full-frame combination effectiveness

#### 6. Recommendations Format
Based on results, provide:
- **Optimal Parameters**: Best performing weight combinations and thresholds
- **Architecture Validation**: Confirm 4-step design benefits  
- **Performance Trade-offs**: Processing time vs accuracy analysis
- **Next Optimizations**: Parameter tuning recommendations for Costa Rica footage

This methodology ensures consistent analysis of the next-generation pipeline performance and optimization opportunities.

### Critical Code Quality Rules

#### No Hardcoded Constants
**NEVER hardcode values in the code. Always use CLI parameters with defaults.**

- ❌ `detector(frame, conf=0.001)` 
- ✅ `detector(frame, conf=MODEL_DETECTION_THRESHOLD)`
- ❌ `threshold = 0.05`
- ✅ `threshold = config.confidence_threshold`

All thresholds, parameters, and configuration values must be:
1. Defined as constants with descriptive names
2. Configurable via CLI parameters with reasonable defaults
3. Never embedded directly in function calls or logic

