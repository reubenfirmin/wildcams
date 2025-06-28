# Wildlife Camera Trap Video Processing System

## Project Overview
Automated wildlife video processing system for Costa Rican jungle camera footage. Detects animals using ensemble ML models, filters false positives, and clusters videos based on visual similarity of detected animals.

## Current System Architecture (Modular Design)

### Core Components
1. **SD Card Watcher** (`sd_watcher.py`) - Automatically detects and downloads videos from camera SD cards
2. **Modular Processing System**:
   - **`video_processor_base.py`** - Shared base class with common functionality
   - **`process_fullframe.py`** - Full-frame ML ensemble approach
   - **`process_motiondetection.py`** - Motion detection + crop-based processing
   - **`ml_detection.py`** - Shared 5-model ML ensemble
3. **Nix Development Environment** (`flake.nix`) - Reproducible development setup with uv Python package manager

### Processing Approaches

#### Full-Frame Processing (`process-ff`)
- Runs complete 5-model ensemble on entire frames
- Enhanced preprocessing (histogram equalization)
- Multi-scale analysis (0.8x, 1.2x) 
- Maximum detection accuracy but slower processing

#### Motion Detection Processing (`process-md`)
- Motion detection pre-filtering using MOG2/KNN background subtraction
- ML ensemble only on motion region crops
- 6-7x performance improvement expected
- Intelligent filtering for vegetation, aspect ratios, temporal consistency
- Better focus on actual moving animals

### Five-Model ML Ensemble
1. **YOLOv8x** (primary - highest accuracy)
2. **YOLOv8m** (backup - medium model)  
3. **YOLOv8n** (MegaDetector fallback - fastest)
4. **MegaDetector v6** (wildlife-specific via PyTorch-Wildlife) ✅ Fixed tensor format issues
5. **DeepFaune** (European wildlife model via PyTorch-Wildlife) ✅ Fixed supervision.Detections parsing

### Enhanced Processing Pipeline
1. **Frame Extraction**: Evenly sample up to 200 frames throughout each video
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
MAX_FRAMES_PER_VIDEO=200
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

# Motion detection settings
MOTION_METHOD=MOG2
MOTION_VAR_THRESHOLD=16
MIN_MOTION_AREA=500
MAX_MOTION_AREA=100000
```

## Commands
```bash
# SD card monitoring
start      # Start daemon
stop       # Stop daemon  
logs       # View logs
check      # Check daemon status

# Video processing (new modular commands)
process-ff                    # Full-frame processing (all videos)
process-md                    # Motion detection processing (all videos)
process-ff -v 7 8 9          # Process specific videos (full-frame)
process-md -v IMG_0015.MP4    # Process specific video (motion detection)
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
✅ **Fixed**: PyTorch-Wildlife tensor format issues - MegaDetector v6 and DeepFaune now working
✅ **Modular**: Clean separation between preprocessing approaches and shared functionality
✅ **Logging**: Centralized logging with preprocessing mode identification for analysis comparison
✅ **Architecture**: Removed 8 obsolete files, clean codebase with focused responsibilities

## Recent Major Updates

### Modular Architecture (Latest)
- **Extracted Common Functionality**: `video_processor_base.py` contains shared logic
- **Clean Separation**: Each processor focuses only on its preprocessing approach
- **DRY Principle**: No duplicate code between full-frame and motion detection
- **Maintainable**: Changes to common logic update both approaches automatically

### PyTorch-Wildlife Integration (Fixed)
- **MegaDetector v6**: Fixed `'tuple' object has no attribute 'get'` error via proper supervision.Detections handling
- **DeepFaune**: Fixed `only length-1 arrays can be converted to Python scalars` via tensor conversion
- **Transform Pipeline**: Proper preprocessing with tensor-to-numpy conversion for API compatibility
- **Model Loading**: Successfully loading and running all 5 models in ensemble

### Logging Enhancement
- **Preprocessing Mode Identification**: Logs clearly show FULL_FRAME vs MOTION_DETECTION mode
- **Ensemble Step Tracking**: Each model execution logged with ENSEMBLE_STEP_1-5
- **Motion Analysis**: Detailed logging of motion regions, crops, and scaling
- **Centralized Storage**: All logs go to `logs/` directory with timestamps

### Performance Optimization
- **Motion Detection Strategy**: Comprehensive implementation with intelligent filtering
- **Expected Speedup**: 6-7x performance improvement via motion region crops
- **Filter Override**: When video filter provided (`-v`), ignores .processed files for forced reprocessing

## Known Issues
- Videos 1-6 were false positives (now filtered by ensemble validation)
- Videos 13-19 are camera handling (now detected and filtered)
- Small artifact detections occasionally occur (improved with ensemble validation)

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