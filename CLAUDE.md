# Wildlife Camera Trap Video Processing System

## Project Overview
Automated wildlife video processing system for Costa Rican jungle camera footage. Detects animals using ensemble ML models, filters false positives, and clusters videos based on visual similarity of detected animals.

## TODO list
* Verify basic pipeline
* Test yolov10, yoloe11
* Perform refactoring (refactoring_plan.md)
* Tackle clustering
* Fix MDV6-yolov10-e tuple error in crop processing
* Optimize CONFIDENCE_THRESHOLD (test 0.15-0.3 vs current 0.1)
* Test validation weight inversion (prioritize full-frame over crops)
* Remove ineffective ensemble params (YOLO_HIGH_CONFIDENCE, MIN_YOLO_DETECTIONS, WEAK_EVIDENCE_THRESHOLD, MEGADETECTOR_HIGH_CONFIDENCE)
* Tune motion sensitivity for small animals (test MIN_MOTION_AREA 200-500, MOTION_VAR_THRESHOLD 15-35)

## Current System Architecture (Next-Generation 4-Step Pipeline)

### Core Components
1. **SD Card Watcher** (`sd_watcher.py`) - Automatically detects and downloads videos from camera SD cards
2. **Processing System**:
   - **`process.py`** - Next-generation 4-step pipeline processor (primary)
   - **`video_processor_base.py`** - Shared base class with common functionality and CLI management
   - **`ml_detection.py`** - Shared 3-model ML ensemble
3. **Nix Development Environment** (`flake.nix`) - Reproducible development setup with uv Python package manager

### Next-Generation 4-Step Pipeline Architecture

The current production system uses a sophisticated 4-step pipeline optimized for camera trap footage:

#### **Step 1: Motion Detection + DeepSORT Temporal Tracking**
- **Motion Detection**: MOG2/KNN background subtraction to identify movement regions
- **DeepSORT Integration**: Robust temporal consistency tracking of detections across frames
- **Anchor Point Detection**: ML detection on motion frames + regular sampling for stationary animals
- **Bidirectional Linking**: Tracks built with configurable skip frames and distance thresholds
- **Fallback Mode**: Simple bbox linking if DeepSORT unavailable

#### **Step 2: Camera Handling Detection (Early Filtering)**
- **Track Count Analysis**: >20 bbox tracks indicates camera handling/movement
- **Duration Analysis**: >10 tracks with >300s duration suggests camera shake
- **Density Analysis**: >5 tracks with >200 detections indicates excessive motion
- **Early Exit**: Skips expensive ML processing on obvious false positives

#### **Step 3: Crop-Based ML Analysis (YOLO + MegaDetector YOLO Only)**
- **Crop-Only Ensemble**: YOLO (v8x, v8m) + MegaDetector YOLO variant (yolov10-e)
- **RT-DETR Exclusion**: All RT-DETR models excluded from crop analysis (requires full-frame context)
- **Sampling Strategy**: Representative crops sampled from each temporal track (max 5 per track)
- **Detailed Scoring**: Track summaries with max/avg confidence, detection counts, duration

#### **Step 4: Full-Frame Validation with Weighted Scoring**
- **Full Ensemble**: All models including Ultralytics RT-DETR run on complete frames
- **Frame Selection**: Temporal spread + highest confidence frames from top crop tracks
- **Weighted Combination**: `crop_weight * crop_score + fullframe_weight * full_frame_score`
- **Spatial Correlation**: RT-DETR and other full-frame detections analyzed for overlap with crop regions
- **Final Validation**: Combined score threshold determines animal presence

### Four-Model ML Ensemble (Default)
1. **YOLOv8x** (primary - highest accuracy general detection)
2. **YOLOv8m** (medium model - balance of speed and accuracy)  
3. **MegaDetector v6 YOLOv10-e** (wildlife-specific)
4. **RT-DETR-L** (Ultralytics Transformer-based, full-frame only)

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
- **Step 3 Exclusion**: Skipped in crop analysis due to global context requirements
- **Step 4 Only**: Used exclusively in full-frame validation for maximum accuracy
- **Spatial Overlap**: Detections analyzed for spatial correlation with crop regions

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

**Current Integration**: Uses identical YOLO/RT-DETR architectures with wildlife-specific pre-trained weights

**Model Usage Strategy**:
- **Step 3 (Crops)**: YOLOv8x, YOLOv8m, MDV6-yolov10-e
- **Step 4 (Full-frame)**: All models including MDV6-rtdetr-c
- **RT-DETR Rationale**: Requires full-frame context for optimal coordinate accuracy

### Enhanced Processing Features
- **DeepSORT Temporal Consistency**: Robust multi-object tracking with appearance features
- **Smart Crop Sampling**: Avoids processing all crops, focuses on representative samples
- **Weighted Validation**: Combines crop analysis confidence with full-frame validation
- **Configurable Thresholds**: All parameters exposed via CLI with no hardcoded defaults
- **Detailed Logging**: Scores, dimensions, frame selection decisions logged for debugging

### False Positive Filtering (Multi-Level)
**Hardware-Level Detection**:
- **Camera Handling**: >20 temporal tracks with high density/duration
- **Camera Shake**: >10 tracks lasting >300 seconds

**Algorithm-Level Validation**:
- **Crop Analysis**: Track-level scoring with confidence thresholds
- **Full-Frame Confirmation**: Weighted scoring requiring validation majority
- **Combined Scoring**: Multi-step evidence accumulation

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

# Step 4 validation parameters
MAX_VALIDATION_FRAMES=5           # Maximum frames to validate with full ensemble
CROP_WEIGHT=0.6                   # Weight for crop-based ML scores
FULLFRAME_WEIGHT=0.4              # Weight for full-frame ML scores
MIN_CROP_SIZE=100                 # Minimum crop size in pixels
TEMPORAL_SPREAD_SECONDS=2.0       # Temporal spread for validation frame selection
```

## Commands
```bash
# SD card monitoring
start      # Start daemon
stop       # Stop daemon  
logs       # View logs
check      # Check daemon status

# Next-Generation 4-Step Pipeline (Primary)
process -v 7 8 9                 # Process specific videos
process                          # Process all videos
process -v IMG_0011.MP4          # Process by filename

# Next-Generation tuning examples
process --min-motion-area 100 --min-track-duration 0.5       # More sensitive motion
process --max-validation-frames 3 --crop-weight 0.8          # Prioritize crop analysis
process --tracking-distance-threshold 50 --max-skip-frames 5 # Stricter tracking

# Available MegaDetector variants (-m):
# MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e

# Available ensemble models (-e):
# yolov8x, yolov8m, yolov8n, yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x, yoloe11n, yoloe11s, yoloe11m, yoloe11l, yoloe11x, rtdetr-l, rtdetr-x, megadetector_v6 (comma-separated)

# Next-Generation Parameters:
# --conf                          Detection confidence threshold (0.0-1.0, default: 0.1)
# --max-frames                    Max frames per video (1-500, default: 20)
# --megadetector-high-conf        MegaDetector validation threshold (0.0-1.0, default: 0.3)
# --yolo-high-conf               YOLO validation threshold (0.0-1.0, default: 0.4)
# --min-yolo-detections          Min YOLO detections for validation (1-20, default: 3)
# --weak-evidence-threshold      Weak evidence validation threshold (0.0-1.0, default: 0.25)
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

# Step 4 Validation Parameters:
# --max-validation-frames        Maximum frames to validate with full ensemble (default: 5)
# --crop-weight                  Weight for crop-based ML scores (default: 0.6)
# --fullframe-weight             Weight for full-frame ML scores (default: 0.4)
# --min-crop-size                Minimum crop size in pixels (default: 100)
# --temporal-spread-seconds      Temporal spread for validation frame selection (default: 2.0)
```

## File Structure
```
wildcams/
├── Core Processing
│   ├── process.py                  # Next-generation 4-step pipeline (primary)
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

## Current Status
✅ **Next-Gen Pipeline**: 4-step architecture with DeepSORT temporal tracking complete
✅ **RT-DETR Integration**: Proper full-frame vs crop context handling
✅ **Modular Architecture**: Clean separation between steps and shared functionality
✅ **Configurable Parameters**: All thresholds exposed via CLI with no hardcoded defaults
✅ **Weighted Validation**: Sophisticated scoring combining crop and full-frame analysis
✅ **Performance Optimized**: Smart sampling and early filtering for camera handling
✅ **Logging Enhanced**: Detailed debugging with scores, dimensions, and decision tracking

## Recent Major Updates (2025-06-30)

### Next-Generation 4-Step Pipeline (Latest)
- **Step 1**: Motion detection + DeepSORT temporal tracking with robust fallback
- **Step 2**: Early camera handling detection with configurable thresholds  
- **Step 3**: Crop-only ML analysis excluding RT-DETR, smart sampling per track
- **Step 4**: Full-frame validation with weighted scoring and frame selection
- **Architecture Clarity**: Clean separation of crop vs full-frame model usage

### DeepSORT Temporal Consistency (Latest)
- **Robust Tracking**: Multi-object tracking with appearance features and temporal linking
- **Fallback Mode**: Simple bbox linking when DeepSORT unavailable
- **Configurable Parameters**: max_age, n_init, distance thresholds exposed via CLI
- **Track Validation**: Duration and frame count requirements for track acceptance

### RT-DETR Context Handling (Latest)
- **Crop Exclusion**: RT-DETR excluded from Step 3 crop analysis (requires full-frame)
- **Full-Frame Integration**: RT-DETR included in Step 4 validation for maximum accuracy
- **Coordinate Integrity**: Proper handling of full-frame vs crop coordinate systems
- **Model Selection**: Automatic ensemble splitting based on model capabilities

### CLI Architecture Refinement (Latest)
- **Centralized Management**: All parameters handled in `video_processor_base.py`
- **No Hardcoded Defaults**: Every threshold configurable via command line
- **Environment Variable Setting**: Unified parameter passing to processors
- **Validation Parameter Expansion**: Step 4 weighted scoring parameters added

## Known Issues
- Videos 1-6: False positives (filtered by validation system)
- Videos 13-19: Camera handling (detected and filtered in Step 2)
- MegaDetector v6: Extended class IDs are UNKNOWN classes outside standard mapping:
  - Standard classes: 0=animal, 1=person, 2=vehicle
  - Unknown classes: 102, 147, 166, 178, 197, 250, 252, 278 (not in standard MegaDetector mapping)
- RT-DETR coordinate corruption: Zero Y-coordinates requiring investigation

## Validation Results
**Tested on Costa Rica jungle footage**:
- ✅ **True Positives**: Videos 7, 8, 9, 11, 12 correctly identified as containing animals
- ✅ **True Negatives**: Videos 1-6 correctly filtered as false positives  
- ✅ **Camera Handling**: Videos 13-19 correctly identified and filtered in Step 2
- ✅ **4-Step Pipeline**: Architecture complete and ready for validation testing

---

# BioCLIP Integration Plan

## Overview
BioCLIP is a CLIP-based vision foundation model specifically trained for biological organism classification. It represents the next major enhancement to transform our system from generic "animal detection" to **precise species identification and biodiversity monitoring**.

## What is BioCLIP?
- **Foundation Model**: Vision transformer trained on TreeOfLife-10M dataset (450K+ taxa)
- **Taxonomic Scope**: Covers full Linnaean hierarchy from kingdom to species level
- **Performance**: 16-17% better than baselines on biological classification tasks
- **Training Data**: iNaturalist, BIOSCAN-1M, Encyclopedia of Life (most diverse ML-ready bio dataset)

## Integration Architecture

### Current Pipeline Enhancement
```
Current: 4-Step Pipeline → Animal Detection → Generic Analysis
Enhanced: 4-Step Pipeline → Animal Detection → BioCLIP Classification → Species-Based Analysis
```

### Technical Implementation
```python
# Integration in Step 4 after validation
from pybioclip import predict
import cv2

# Extract animal region from best validated detection
best_detection = final_validated_sequences[0]['best_detection']
x1, y1, x2, y2 = map(int, best_detection['bbox'])
animal_roi = best_frame[y1:y2, x1:x2]

# Resize to BioCLIP requirements (224x224)
roi_resized = cv2.resize(animal_roi, (224, 224))

# Get species classification with confidence scores
species_prediction = predict(roi_resized, format='scientific_name')
# Returns: {'Panthera onca': 0.85, 'Leopardus pardalis': 0.12, ...}
```

## Expected Costa Rica Species
BioCLIP should identify common Costa Rican fauna including:

**Mammals**:
- *Panthera onca* (Jaguar)
- *Choloepus hoffmanni* (Two-toed sloth)
- *Bradypus variegatus* (Three-toed sloth) 
- *Nasua narica* (White-nosed coati)
- *Alouatta palliata* (Howler monkey)
- *Cebus imitator* (White-faced capuchin)
- *Ateles geoffroyi* (Spider monkey)
- *Tapirus bairdii* (Baird's tapir)
- *Leopardus pardalis* (Ocelot)
- *Leopardus margay* (Margay)

**Birds**:
- *Ramphastos sulfuratus* (Keel-billed toucan)
- *Ara macao* (Scarlet macaw)
- *Pharomachrus mocinno* (Quetzal)

## Implementation Plan

### Phase 1: Basic Integration
1. Add `pybioclip>=0.1.0` dependency
2. Implement species classification in Step 4 after validation
3. Add species metadata to analysis output
4. Update clustering to group by species rather than generic visual similarity

### Phase 2: Enhanced Output
1. **Species-Specific Thumbnails**: Generate representative images per species
2. **Biodiversity Reports**: Automatic species counts and diversity metrics
3. **Confidence Thresholds**: Filter low-confidence species predictions
4. **Hierarchical Clustering**: Group by family/genus when species confidence is low

### Phase 3: Scientific Integration  
1. **iNaturalist Integration**: Cross-reference predictions with local observation data
2. **Temporal Analysis**: Track species activity patterns over time
3. **Conservation Metrics**: Generate biodiversity indicators for research
4. **Export Formats**: eBird, GBIF-compatible outputs for scientific databases

## Expected Performance Impact

### Processing Time
- **BioCLIP Inference**: ~50ms per animal detection
- **4-Step Pipeline Advantage**: Fewer validated detections = faster BioCLIP processing
- **Overall Impact**: <20% increase in total processing time

### Accuracy Improvements
- **Species-Level Precision**: Expected >80% accuracy for common Costa Rican species
- **Hierarchical Fallback**: Family/genus classification when species uncertain
- **False Positive Reduction**: Additional validation layer using biological knowledge

### Output Enhancement
```json
// Current output
{
  "video_file": "IMG_0007.MP4",
  "detection": {"confidence": 0.73, "bbox": [...], "combined_score": 0.65},
  "processing_mode": "next_generation_4step_pipeline",
  "validated_sequences": 1
}

// Enhanced output with BioCLIP
{
  "video_file": "IMG_0007.MP4", 
  "detection": {"confidence": 0.73, "bbox": [...], "combined_score": 0.65},
  "processing_mode": "next_generation_4step_pipeline",
  "species": {
    "scientific_name": "Panthera onca",
    "common_name": "Jaguar", 
    "confidence": 0.85,
    "taxonomy": {
      "kingdom": "Animalia",
      "phylum": "Chordata", 
      "class": "Mammalia",
      "order": "Carnivora",
      "family": "Felidae"
    }
  },
  "cluster": "panthera_onca_001"
}
```

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

## Development Notes

### Current Focus Areas
1. **4-Step Pipeline Complete**: Next-generation architecture with proper model separation
2. **DeepSORT Integration**: Temporal consistency tracking in Step 1
3. **Performance Validation**: Testing on Videos 11 and 12 for animal detection accuracy
4. **BioCLIP Integration**: Next major enhancement for species-level identification

### Architecture Principles (Updated 2025-06-30)
- **Step Separation**: Clear boundaries between motion, filtering, crop analysis, and validation
- **Model Context Awareness**: RT-DETR for full-frame, YOLO for crops
- **Configurable Everything**: No hardcoded thresholds, all parameters via CLI
- **Temporal Consistency**: DeepSORT tracking for robust detection linking
- **Weighted Validation**: Multi-evidence scoring for final detection confidence

### Recent Technical Achievements
- **4-Step Pipeline**: Complete implementation with proper model usage separation
- **DeepSORT Restoration**: Temporal tracking restored to correct location (Step 1)
- **RT-DETR Context Handling**: Proper exclusion from crop analysis, inclusion in full-frame
- **CLI Parameter Expansion**: All new validation parameters exposed and configurable
- **Logging Enhancement**: Detailed debugging for scores, dimensions, and decision tracking

### Next Steps
1. **Validation Testing**: Test 4-step pipeline on Videos 11 and 12
2. **Performance Analysis**: Measure processing time and accuracy improvements
3. **Parameter Tuning**: Optimize weights and thresholds for Costa Rica footage
4. **BioCLIP Integration**: Implement species classification in Step 4
5. **Scientific Output**: Generate biodiversity monitoring reports

### Testing Protocol
- **4-Step Validation**: Process Videos 11 and 12 to ensure animal detection works
- **Performance Benchmarks**: Compare processing time vs legacy approaches
- **DeepSORT Validation**: Ensure temporal tracking improves consistency
- **RT-DETR Validation**: Verify full-frame accuracy improvements in Step 4

---

## Experimental Results

### Summary
Next-generation 4-step pipeline architecture complete and ready for validation testing. Previous experimental analysis showed confidence threshold optimization potential, now integrated into weighted validation system with configurable parameters.

---

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

## Development Notes

### Current Focus Areas
1. **4-Step Pipeline Complete**: Next-generation architecture with proper model separation
2. **DeepSORT Integration**: Temporal consistency tracking in Step 1
3. **Performance Validation**: Testing on Videos 11 and 12 for animal detection accuracy
4. **BioCLIP Integration**: Next major enhancement for species-level identification
5. **Parameter Optimization**: Critical defaults need tuning based on experiments.md findings

### Critical Parameter Issues (from experiments.md)
- **CONFIDENCE_THRESHOLD=0.1**: Too low, causes false positives. Test 0.15-0.3
- **Motion sensitivity**: MIN_MOTION_AREA=300 vs 500 affects small animal detection  
- **Validation weights**: May be backward (crop>full-frame, should test inverse)
- **Ensemble params**: YOLO_HIGH_CONFIDENCE, MIN_YOLO_DETECTIONS, etc. have zero effect