# Wildlife Video Processing Theory

## System Architecture Theory

### 4-Step Pipeline Architecture

The system uses an optimized 4-step pipeline that eliminates crop analysis issues by connecting motion tracking directly to full-frame validation, then filtering for animals specifically:

#### **Step 1: Motion Detection + Temporal Tracking**
- **Motion Detection**: MOG2/KNN background subtraction to identify movement regions
- **Temporal Tracking**: Simple bbox linking across frames (Note: DeepSORT not currently implemented)
- **Temporal Consistency**: Builds motion tracks across multiple frames with configurable skip frames
- **Motion Region Collection**: Captures spatial regions of movement for later spatial validation

#### **Step 2: Camera Handling Detection (Early Filtering)**
- **Spatial Dispersion Analysis**: Ratio of spatial clusters to tracks (dispersed movement = camera handling)
- **Motion Sparsity Analysis**: Inverted motion density (sparse erratic movement = camera handling)
- **Composite Scoring**: spatial_dispersion × motion_sparsity with configurable threshold (--composite-motion-threshold, default: 0.5)
- **Early Exit**: Skips expensive ML processing on obvious false positives

#### **Step 3: Full-Frame Analysis with Spatial Overlap Validation**
- **Direct Motion Track Input**: Takes motion tracks directly from Step 2 (no crop analysis)
- **Full Ensemble Processing**: Complete ensemble on full frames using configured models (YOLO + RT-DETR + MegaDetector variants)
- **Temporal Frame Sampling**: Selects representative frames from each motion track (configurable via --max-validation-frames)
- **Spatial Overlap Validation**: Requires minimum overlap between full-frame detections and motion regions (--spatial-overlap-threshold, default: 0.1)
- **Model Contribution Tracking**: Comprehensive tracking of all ensemble models in full-frame context
- **Validation Logic**: Simple confidence thresholding with spatial overlap requirement
- **False Positive Reduction**: Spatial validation eliminates detections that don't correlate with motion
- **Full-Frame Optimization**: All models process complete frames for optimal context and spatial understanding

#### **Step 4: Animal Classification (enabled by default)**
- **Purpose**: Filter validated sequences down to animals specifically, removing non-animal movement that survived Step 3.
- **Classifier ensemble**: BioCLIP and DeepFaune (--classification-models, default: bioclip,deepfaune).
- **"Either model passes" logic**: A sequence is confirmed as an animal if any classifier's animal confidence clears its threshold (--animal-confidence-threshold; per-model --bioclip-threshold, --deepfaune-threshold).
- **Species labels**: Species-capable models (BioCLIP top-k, DeepFaune's class head) supply a species guess when the ensemble confirms an animal.
- **Toggle**: --skip-animal-classification runs the pipeline as the older 3-step flow.
- **Known geographic limitation**: DeepFaune's label space is European/Eurasian and BioCLIP is a general foundation model; neither is tuned for Costa Rican species. See models.md for the current model-currency assessment and roadmap.

## Model Theory

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

## Validation Theory

### Current Validation Approach

The system uses **spatial overlap validation** with simple confidence thresholding:

#### Validation Algorithm
1. **Frame Sampling**: Select representative frames from each motion track (--max-validation-frames)
2. **Model Evaluation**: Run each ensemble model on selected full frames
3. **Spatial Validation**: Require minimum overlap (--spatial-overlap-threshold) between detections and motion regions
4. **Temporal Consistency**: Validate detection continuity across consecutive frames (--min-consecutive-detection-seconds)
   - **Confidence Bridge Enhancement**: Medium-confidence frames (≥ --confidence-bridge-threshold, default: 0.6) between high-confidence frames are automatically validated
   - **Biological Rationale**: Animals don't disappear/reappear; brief confidence dips between strong detections represent the same animal in challenging pose/lighting
   - **Bridge Algorithm**: For frames that fail ensemble threshold but have medium confidence, if both adjacent frames pass, the middle frame is promoted to "passed"
5. **Track Validation**: Accept tracks with sufficient validated detections meeting confidence threshold

#### Key Properties
- **Spatial Correlation**: All detections must spatially correlate with motion regions
- **Temporal Continuity**: Validated detections must form temporally consistent sequences
- **Simple Scoring**: Direct confidence-based validation without complex ensemble weighting
- **Model Transparency**: Each model's contribution explicitly logged and tracked

#### Composite Score (track ranking)
The composite score multiplies a base ensemble confidence by four multipliers. All
coefficients are CLI-tunable (defaults preserve the historical hardcoded values):

- **base_score**: sum of boosted per-model confidences for the track.
- **temporal multiplier** = `min(--composite-temporal-multiplier-cap, 1 + detection_density)`.
- **consensus multiplier** = `1 + --composite-consensus-boost-per-model × (unique_models − 1)`.
- **motion multiplier** = `--composite-motion-multiplier-base + --composite-motion-multiplier-span × avg_motion_overlap`.
- **duration bonus** = `min(--composite-duration-bonus-cap, --composite-duration-bonus-base + duration_s / --composite-duration-bonus-divisor)`.

Per-frame, within a single model, overlapping detections get a small consensus boost of
`1 + --consensus-boost-per-detection × (n − 1)` before scoring. FPS falls back to
`--default-fps` when a video reports none.

### Scientific Logging Format

The system produces structured scientific logs with three types of evaluation rows:

#### EVAL Rows (Model Evaluation)
```
EVAL | {video} | {timestamp} | {frame} | {track_id}
{✅/❌} | {model} | {bbox} | {conf} | {overlap%} | {motion_bbox} | {score}
```
- **Header**: Video name, timestamp, frame number, track ID
- **Rows**: One per model showing detection outcome
- **✅**: Model detected animal with spatial overlap ≥ threshold
- **❌**: Model found no detection or insufficient spatial overlap

#### ENSEMBLE Rows (Frame-Level Combination)
```
ENSEMBLE | {video} | {timestamp} | {frame} | {track_id}
{✅/❌} | combined | valid_models={count} | total_score={sum} | valid_detections={count} | frame_score={avg}
```
- **Header**: Same context as EVAL
- **Row**: Combined results from all models for this frame
- **✅**: At least one model found spatially valid detection
- **❌**: No models found spatially valid detections

#### TRACK Rows (Final Track Validation)
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
- **spatial_valid**: Detection bbox overlaps ≥ --spatial-overlap-threshold with motion regions
- **temporal_continuity**: Optional track-level check (--enable-temporal-continuity-check, default off). When on, the largest gap between a track's passed frames must be ≤ --temporal-continuity-max-gap-seconds. Default off preserves the prior always-pass behavior.
- **models_active**: Number of models contributing spatially valid detections