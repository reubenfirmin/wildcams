# Wildlife Camera Trap Video Processing System

## Project Overview
Automated wildlife video processing system for Costa Rican jungle camera footage. Detects animals using ensemble ML models, filters false positives, and clusters videos based on visual similarity of detected animals.

## Current System Architecture (Modular Design)

### Core Components
1. **SD Card Watcher** (`sd_watcher.py`) - Automatically detects and downloads videos from camera SD cards
2. **Unified Processing System**:
   - **`process.py`** - Unified processor with strategy selection (`-s ff|md`)
   - **`video_processor_base.py`** - Shared base class with common functionality
   - **`process_fullframe.py`** - Full-frame ML ensemble approach
   - **`process_motiondetection.py`** - Motion detection + crop-based streaming processing
   - **`ml_detection.py`** - Shared 5-model ML ensemble
3. **Nix Development Environment** (`flake.nix`) - Reproducible development setup with uv Python package manager

### Processing Approaches

#### Full-Frame Processing (`process -s ff` or `process-ff`)
- Runs complete 5-model ensemble on entire frames
- Enhanced preprocessing (histogram equalization)
- Multi-scale analysis (0.8x, 1.2x) 
- Maximum detection accuracy but slower processing
- Samples frames evenly throughout video (default: 20 frames)

#### Motion Detection Processing (`process -s md` or `process-md`)
- **Real-time streaming motion detection** using MOG2/KNN background subtraction
- **Frame-by-frame analysis** with temporal background modeling
- ML ensemble only on detected motion region crops
- **Camera trap optimized**: Focuses on sustained animal movement vs brief noise
- **Configurable filtering**: Area thresholds, aspect ratios, region limits
- Expected 6-7x performance improvement while maintaining accuracy

### Five-Model ML Ensemble
1. **YOLOv8x** (primary - highest accuracy)
2. **YOLOv8m** (backup - medium model)  
3. **YOLOv8n** (MegaDetector fallback - fastest)
4. **MegaDetector v6** (wildlife-specific via PyTorch-Wildlife) - **MDV6-rtdetr-c for maximum accuracy**
5. **DeepFaune** (classification model, not used in detection ensemble)

**Model Selection Priority**: **Accuracy over speed** - using MDV6-rtdetr-c variant for maximum detection accuracy. RT-DETR architectures typically achieve highest accuracy in object detection, chosen specifically for research-quality wildlife analysis where detection precision is prioritized over processing speed.

### Enhanced Processing Pipeline
1. **Frame Processing**: 
   - **Full-frame**: Sample up to 20 frames evenly throughout video
   - **Motion detection**: Stream through video frame-by-frame, analyze motion regions only
2. **Preprocessing**: 
   - Motion detection (MD approach) or full-frame (FF approach)
   - Histogram equalization for jungle lighting conditions
   - Multi-scale analysis to catch animals at different distances
3. **Ensemble Validation**: Multi-level evidence system to filter false positives
4. **Feature Extraction**: ResNet18 for visual similarity clustering
5. **DBSCAN Clustering**: Group videos by animal visual similarity

### False Positive Filtering
**Three-tier validation system**:
- **Strong Evidence**: Multiple models agree OR high confidence detections (MegaDetector ≥0.6, YOLO ≥0.7)
- **Medium Evidence**: Sufficient YOLO detections (≥8 detections)
- **Weak Evidence**: Wildlife-specific model detection with avg confidence ≥0.4

**Camera Handling Detection**:
- Detection density >15 per frame + >70% low confidence detections = likely equipment handling
- Successfully filters videos 13-19 (camera handling) from videos 7,8,9,11,12 (real animals)

## Hardware & Performance
- **Target Platform**: AMD Ryzen 7 5700G (16 threads)
- **Processing Speed**: 
  - Full-frame: ~15-20 seconds per video
  - Motion detection: ~3-5 seconds per video (estimated 6-7x improvement)
- **Memory Usage**: 2-4GB RAM during inference
- **Storage**: Debug frames and logs organized in subdirectories

## Configuration (.env)
```bash
# Video processing
MAX_FRAMES_PER_VIDEO=20          # Reduced from 200 for faster processing
CONFIDENCE_THRESHOLD=0.1

# Validation thresholds (tuned for Costa Rica jungle footage)
MEGADETECTOR_HIGH_CONFIDENCE=0.6
YOLO_HIGH_CONFIDENCE=0.7
MIN_YOLO_DETECTIONS=8
WEAK_EVIDENCE_THRESHOLD=0.4

# Camera handling detection
DETECTION_DENSITY_THRESHOLD=15.0
LOW_CONFIDENCE_RATIO_THRESHOLD=0.7
LOW_CONFIDENCE_CUTOFF=0.2

# Motion detection settings (camera trap optimized)
MOTION_METHOD=MOG2
MOTION_VAR_THRESHOLD=32           # Less sensitive to small movements
MIN_MOTION_AREA=2000             # Focus on significant motion only
MAX_MOTION_AREA=80000            # Avoid full-frame motion detection
MOTION_HISTORY=100               # Background model adaptation speed
MAX_REGIONS_PER_FRAME=10         # Limit processed regions to reduce noise
MIN_REGION_WIDTH=30              # Filter out thin/small regions
MIN_REGION_HEIGHT=30
MAX_ASPECT_RATIO=5.0             # Reject overly elongated regions
MOTION_MARGIN=30                 # Pixels to expand regions for ML context
```

## Commands
```bash
# SD card monitoring
start      # Start daemon
stop       # Stop daemon  
logs       # View logs
check      # Check daemon status

# Unified video processing 
process -s ff -v 7 8 9           # Full-frame strategy
process -s md -v 7 8 9           # Motion detection strategy
process -s ff -m MDV6-yolov9-e   # Custom MegaDetector model
process -s md -e yolov8x,yolov8m # Custom ensemble

# Direct processing commands (legacy)
process-ff                       # Full-frame processing (all videos)
process-md                       # Motion detection processing (all videos)
process-ff -v 7 8 9             # Process specific videos (full-frame)
process-md -v IMG_0015.MP4       # Process specific video (motion detection)

# Model configuration options
process -s ff -m MDV6-yolov9-e                     # Use balanced YOLOv9 variant
process -s ff -m MDV6-rtdetr-c                     # Use RT-DETR (highest accuracy)
process -s ff -e megadetector_v6                   # Use only MegaDetector v6
process -s md -e yolov8x,megadetector_v6          # Custom ensemble combination

# Motion detection tuning examples
process -s md --min-motion-area 1000 --motion-var-threshold 20    # More sensitive
process -s md --max-regions-per-frame 5 --min-region-width 50     # Stricter filtering
process -s md --motion-margin 50 --max-aspect-ratio 3.0           # Better crop context

# Full-frame tuning examples  
process -s ff --conf 0.05 --max-frames 50         # Lower threshold, fewer frames
process -s ff --yolo-high-conf 0.5 --weak-evidence-threshold 0.3  # Stricter validation

# Available MegaDetector variants (-m):
# MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e, MDV6-rtdetr-c

# Available ensemble models (-e):
# yolov8x, yolov8m, yolov8n, megadetector_v6 (comma-separated)

# Common Parameters (both strategies):
# --conf                          Detection confidence threshold (0.0-1.0, default: 0.1)
# --max-frames                    Max frames per video (1-500, default: 20)
# --megadetector-high-conf        MegaDetector validation threshold (0.0-1.0, default: 0.3)
# --yolo-high-conf               YOLO validation threshold (0.0-1.0, default: 0.4)
# --min-yolo-detections          Min YOLO detections for validation (1-20, default: 3)
# --weak-evidence-threshold      Weak evidence validation threshold (0.0-1.0, default: 0.25)
# --detection-density-threshold  Camera handling detection threshold (1.0-50.0, default: 15.0)
# --clustering-eps               DBSCAN clustering epsilon (0.1-1.0, default: 0.3)

# Motion Detection Specific Parameters:
# --motion-method                Motion detection method (MOG2/KNN, default: MOG2)
# --motion-var-threshold         Variance threshold - higher = less sensitive (default: 32)
# --min-motion-area              Min motion area pixels (default: 2000)
# --max-motion-area              Max motion area pixels (default: 80000)
# --motion-history               Background history frames (default: 100)
# --max-regions-per-frame        Max regions processed per frame (default: 10)
# --min-region-width             Min region width pixels (default: 30)
# --min-region-height            Min region height pixels (default: 30)
# --max-aspect-ratio             Max width/height ratio (default: 5.0)
# --motion-margin                Margin to expand regions pixels (default: 30)
```

## File Structure
```
wildcams/
├── Core Processing
│   ├── video_processor_base.py     # Shared base class
│   ├── process_fullframe.py        # Full-frame approach  
│   ├── process_motiondetection.py  # Motion detection approach
│   └── ml_detection.py             # Shared 5-model ensemble
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
✅ **Working**: Animal detection, false positive filtering, clustering
✅ **Validated**: Successfully distinguishes real animals (videos 7,8,9,11,12) from false positives
✅ **Fixed**: PyTorch-Wildlife KeyError issues - MegaDetector v6 now handles extended class IDs properly
✅ **Modular**: Clean separation between preprocessing approaches and shared functionality
✅ **Unified**: Single `process.py` script with strategy selection plus legacy direct commands
✅ **Configurable**: All parameters exposed via CLI with sensible defaults
✅ **Logging**: Simplified single logger with emoji highlights for significant steps

## Recent Major Updates

### MegaDetector v6 Class ID Fix (Latest)
- **Issue Resolved**: Fixed KeyError when MegaDetector v6 detects classes outside standard mapping (0=animal, 1=person, 2=vehicle)
- **Raw Class IDs**: Now captures and logs extended class IDs (102, 147, 166, 197, 278) for analysis
- **Coordinate Fix**: Proper handling of bounding box coordinates from raw Ultralytics results
- **No Suppression**: All model output visible, raw class IDs logged for future analysis

### Motion Detection Optimization (Latest)
- **Camera Trap Focused**: Tuned for sustained animal movement vs brief noise/wind
- **Streaming Implementation**: True frame-by-frame processing with temporal background modeling
- **Noise Reduction**: Aggressive morphological filtering, aspect ratio limits, area thresholds
- **Configurable Parameters**: All motion detection settings exposed via CLI
- **Performance Optimized**: Max 10 regions per frame, focus on significant motion only

### Unified Architecture (Latest)
- **Single Entry Point**: `process.py` with `-s ff|md` strategy selection
- **Legacy Support**: Direct `process-ff` and `process-md` commands still available
- **Simplified Logging**: Single log file per session with emoji highlights
- **CLI Consistency**: All parameters available across both strategies
- **No Hardcoding**: Every threshold and setting configurable via command line

### PyTorch-Wildlife Integration (Fixed)
- **MegaDetector v6**: Handles unknown class IDs gracefully, captures raw detections
- **Model Caching**: Proper TORCH_HOME and PYTORCH_WILDLIFE_CACHE setup
- **Extended Classes**: Logs class IDs beyond standard MegaDetector mapping
- **Model Loading**: All ensemble models working correctly with visible initialization

### Performance Optimization
- **Frame Reduction**: Default max frames reduced from 200 → 20 for faster processing
- **Motion Detection Strategy**: Real streaming analysis with intelligent noise filtering
- **Expected Speedup**: 6-7x performance improvement via motion region crops
- **Filter Override**: When video filter provided (`-v`), ignores .processed files for forced reprocessing

## Known Issues
- Videos 1-6 were false positives (now filtered by ensemble validation)
- Videos 13-19 are camera handling (now detected and filtered)
- MegaDetector v6 detects extended class IDs (102, 147, 166, 197, 278) beyond standard mapping - logged for analysis

## Validation Results
**Tested on Costa Rica jungle footage**:
- ✅ **True Positives**: Videos 7, 8, 9, 11, 12 correctly identified as containing animals
- ✅ **True Negatives**: Videos 1-6 correctly filtered as false positives  
- ✅ **Camera Handling**: Videos 13-19 correctly identified and filtered
- ✅ **Ensemble Working**: All 5 models contributing to detection pipeline

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
Current: Motion/Full-frame Detection → Validation → Generic Clustering
Enhanced: Motion/Full-frame Detection → Validation → BioCLIP Classification → Species-Based Clustering
```

### Technical Implementation
```python
# After YOLO detects animal and crops ROI
from pybioclip import predict
import cv2

# Extract animal region from best frame
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
2. Implement species classification function in `video_processor_base.py`
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
- **Motion Detection Advantage**: Fewer crops = faster BioCLIP processing
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
  "detection": {"confidence": 0.73, "bbox": [...]},
  "processing_mode": "motion_detection",
  "cluster": "cluster_001"
}

// Enhanced output with BioCLIP
{
  "video_file": "IMG_0007.MP4", 
  "detection": {"confidence": 0.73, "bbox": [...]},
  "processing_mode": "motion_detection",
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

### Integration with Motion Detection
- **Efficiency Gain**: Motion detection provides focused crops for BioCLIP
- **Quality Improvement**: Better animal crops should improve species identification accuracy
- **Resource Optimization**: Fewer regions to classify = faster processing

---

## Development Notes

### Current Focus Areas
1. **Architecture Complete**: Modular system with clean separation of concerns
2. **Performance Comparison**: Ready to compare full-frame vs motion detection approaches
3. **BioCLIP Integration**: Next major enhancement for species-level identification

### Recent Technical Progress
- **Modular Refactoring**: Extracted common functionality, eliminated code duplication
- **PyTorch-Wildlife Fixed**: All 5 models now working correctly in ensemble
- **Logging Enhanced**: Clear preprocessing mode identification for analysis
- **Performance Optimized**: Motion detection implementation complete

### Next Steps
1. **Performance Analysis**: Compare full-frame vs motion detection processing times and accuracy
2. **Motion Detection Tuning**: Optimize parameters for Costa Rica jungle conditions
3. **BioCLIP Integration**: Implement species classification once motion detection is validated
4. **Scientific Output**: Create biodiversity monitoring reports
5. **Field Testing**: Deploy system for continuous monitoring

### Testing Protocol
- **Approach Comparison**: Process same videos with both `process-ff` and `process-md`
- **Performance Benchmarks**: Processing time and detection accuracy comparison
- **Motion Detection Validation**: Ensure motion regions capture all animals
- **Species Classification**: Validate BioCLIP predictions against expert identification

---

## Experimental Results

### Summary
Comprehensive experimental analysis documented in `experiments.md` shows systematic testing of wildlife video processing strategies. Key findings: confidence threshold of 0.15-0.2 successfully filters some false positives, achieving 50% precision improvement, but challenges remain with stubborn false positive videos requiring new approaches.

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
# - Error patterns and technical issues
```

#### 2. Ground Truth Reference
Always reference the validated sample set:
- **Videos 7, 8, 9, 11, 12**: True positives (should detect animals)
- **Videos 1-6**: False positives (should NOT detect animals)  
- **Videos 13-19**: Camera handling (should NOT detect animals)

#### 3. Results Table Structure
Create tables with these columns:
- **Strategy/Model**: Processing approach and MegaDetector variant
- **Animals Detected**: List video numbers where animals were found
- **No Animals**: List video numbers where no animals were found  
- **Not Processed**: List videos not processed (with reason)
- **True Positives**: Count and percentage of correct animal detections
- **False Positives**: Count and percentage of incorrect animal detections
- **Precision**: TP / (TP + FP) - accuracy of positive predictions
- **Recall**: TP / (TP + True Negatives attempted) - coverage of actual positives

#### 4. Analysis Steps
1. **Extract Processing Results**: Parse logs for video-by-video outcomes
2. **Calculate Metrics**: Compute precision, recall, F1-score for each strategy
3. **Identify Patterns**: Look for systematic failures or advantages
4. **Processing Time Analysis**: Calculate total duration and per-video averages
5. **Technical Issues**: Document errors, incomplete runs, or reliability problems

#### 5. Critical Validation Points
- **Verify ground truth assumptions** - ensure test videos match expected categories
- **Distinguish technical failures from detection results** - processing errors vs detection outcomes
- **Account for incomplete experiments** - logs may be truncated or interrupted
- **Consider conservative behavior beneficial** - skipping videos may indicate good filtering

#### 6. Recommendations Format
Based on results, provide:
- **Best performing strategy** with specific parameters
- **Identified issues** requiring fixes or improvements  
- **Next experimental parameters** to test variations
- **Processing time vs accuracy trade-offs** for production decisions

This methodology ensures consistent, accurate analysis of experimental results for strategy optimization.