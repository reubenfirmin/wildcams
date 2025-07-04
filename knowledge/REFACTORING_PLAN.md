# Wildlife Camera Processing System - Refactoring Plan

## Overview
Transform the current partially-refactored codebase into a clean, modular, object-oriented architecture. Build upon recent improvements while breaking up remaining large classes into focused components with clear responsibilities and well-defined interfaces.

## Current State (Updated July 2025)

### Progress Made
✅ **Configuration Management**: Global config object replaced with ProcessingConfig dataclass  
✅ **Module Separation**: ML detection extracted to separate module (ml_detection.py)  
✅ **Base Class Extraction**: Common functionality moved to VideoProcessorBase  
✅ **Specialized Components**: DeepSORT tracking isolated to separate module  
✅ **Algorithm Improvements**: 3-step pipeline with camera handling detection, spatial overlap validation, frame sampling optimizations

### Current Problems

#### Code Quality Issues (Updated)
- **God Classes**: `NextGenVideoProcessor` (~2,504 lines), `VideoProcessorBase` (~846 lines)
- **Mixed Concerns**: Motion detection, ML inference, file I/O, and configuration still mixed in base class
- **Tight Coupling**: While config is improved, components still tightly coupled through inheritance
- **Poor Testability**: Large classes with multiple responsibilities are hard to unit test
- **Difficult Maintenance**: Changes require understanding the entire system

#### Specific Issues by File (Updated)
- **process.py**: Massive NextGenVideoProcessor class (2,504 lines) handling 3-step pipeline, motion detection, tracking, ML coordination
- **video_processor_base.py**: God class (846 lines) managing CLI, environment, video I/O, validation, clustering
- **ml_detection.py**: Model loading, preprocessing, inference, and validation all in one class (973 lines)
- **sd_watcher.py**: Well-structured, minor improvements needed
- **deepsort_tracker.py**: Well-structured, but could benefit from better integration

## Refactoring Strategy

### Phase 1: Configuration & Infrastructure (Week 1)

#### 1.1 Enhance Configuration Management ✅ PARTIALLY DONE
**Status**: ProcessingConfig dataclass implemented, but CLI parsing still in base class

**New Files:**
```
config/
├── __init__.py
├── configuration_manager.py   # ConfigurationManager class
├── cli_parser.py             # CLIArgumentParser class (extract from VideoProcessorBase)
├── environment_config.py     # EnvironmentConfigLoader class
└── processing_config.py      # ProcessingConfig (already exists, move here)
```

**Extract From:**
- CLI parsing methods from `video_processor_base.py` (lines 1-200)
- Environment variable loading from `video_processor_base.py`
- Move existing `ProcessingConfig` from `process.py` to dedicated module

**New Architecture:**
```python
# Enhanced version of existing ProcessingConfig approach
config_manager = ConfigurationManager()
config_manager.load_from_environment()
config_manager.load_from_cli_args(sys.argv)
processing_config = config_manager.get_processing_config()
```

#### 1.2 Extract Video I/O Operations
**New Files:**
```
io/
├── __init__.py
├── video_reader.py           # VideoReader class
├── frame_extractor.py        # FrameExtractor class (includes recent frame sampling improvements)
├── analysis_writer.py        # AnalysisWriter class
└── processed_tracker.py      # ProcessedVideoTracker class
```

**Extract From:**
- Video opening/reading methods from `video_processor_base.py` (lines 300-500)
- Frame extraction and sampling logic (incorporate recent temporal clustering improvements)
- Analysis output writing and `.processed` file management
- Clustering and similarity analysis methods

**New Architecture:**
```python
video_reader = VideoReader(video_path)
frame_extractor = FrameExtractor(max_frames=20, sampling_strategy='temporal_clustering')
frames = frame_extractor.extract_frames(video_reader)
```

### Phase 2: ML Model Management (Week 2)

#### 2.1 Break Up MLDetectionEnsemble ⚠️ CRITICAL PRIORITY
**Status**: MLDetectionEnsemble extracted to separate module but still monolithic at 973 lines

**New Files:**
```
ml/
├── __init__.py
├── model_manager.py          # ModelManager class
├── model_loaders/
│   ├── __init__.py
│   ├── yolo_loader.py        # YOLOModelLoader class (YOLOv8x/m, YOLOv10, YOLOv11)
│   ├── megadetector_loader.py # MegaDetectorLoader class (all MDV6-* variants)
│   └── rtdetr_loader.py      # RTDETRLoader class (standalone RT-DETR)
├── inference/
│   ├── __init__.py
│   ├── inference_engine.py   # InferenceEngine class
│   ├── yolo_inference.py     # YOLOInferenceEngine class
│   ├── megadetector_inference.py # MegaDetectorInferenceEngine class
│   ├── rtdetr_inference.py   # RTDETRInferenceEngine class
│   └── ensemble_coordinator.py # EnsembleCoordinator class
├── preprocessing.py          # PreprocessingPipeline class
└── postprocessing.py         # PostprocessingPipeline class
```

**Extract From:**
- Model initialization logic from `MLDetectionEnsemble` (973 lines)
- Individual model inference methods
- Preprocessing transformations (TTA, multi-scale)
- NMS and detection filtering
- Current ensemble models: yolo12x, yolo12m, MDV6-yolov10-e, rtdetr-l

**Updated Model Boundaries:**
- **YOLOLoader**: Handles YOLO variants (YOLOv8x/m, YOLOv10 n/s/m/b/l/x, YOLOv11 n/s/m/l/x)
- **MegaDetectorLoader**: Handles all MegaDetector v6 variants:
  - `MDV6-yolov9-e` (MegaDetector wrapper around YOLOv9)
  - `MDV6-yolov10-e` (MegaDetector wrapper around YOLOv10)  
  - `MDV6-rtdetr-c` (MegaDetector wrapper around RT-DETR)
- **RTDETRLoader**: Handles standalone RT-DETR models (rtdetr-l, rtdetr-x)
- Current default ensemble: yolo12x, yolo12m, MDV6-yolov10-e, rtdetr-l

**New Architecture:**
```python
# Replace single ensemble class with coordinated components
model_manager = ModelManager(cache_dir, ensemble_models)
inference_engine = InferenceEngine(model_manager)
ensemble_coordinator = EnsembleCoordinator(inference_engine)

# Clean separation of concerns for 3-step pipeline
detections = ensemble_coordinator.detect(image, full_frame_only=True)  # No crop mode in current system
```

#### 2.2 Extract Preprocessing & Postprocessing
**Responsibilities:**
- **PreprocessingPipeline**: TTA, multi-scale, histogram equalization
- **PostprocessingPipeline**: NMS, confidence filtering, coordinate transformation

### Phase 3: Processing Pipeline (Week 3)

#### 3.1 Create Pipeline Architecture ⚠️ CRITICAL PRIORITY
**Status**: 3-step pipeline implemented within monolithic NextGenVideoProcessor class (2,504 lines)

**New Files:**
```
pipeline/
├── __init__.py
├── pipeline_orchestrator.py  # PipelineOrchestrator class
├── step_interface.py         # PipelineStep abstract base class
└── steps/
    ├── __init__.py
    ├── motion_detection_step.py    # MotionDetectionStep class
    ├── camera_handling_step.py     # CameraHandlingFilterStep class
    └── fullframe_validation_step.py # FullFrameValidationStep class
```

**Extract From:**
- 3-step pipeline logic from `NextGenVideoProcessor` (2,504 lines)
- Motion detection + temporal tracking (Step 1)
- Camera handling detection with spatial dispersion + motion sparsity (Step 2)
- Full-frame analysis with spatial overlap validation (Step 3)
- Each step becomes a separate class with clear input/output contracts

**Updated Architecture (3-Step Pipeline):**
```python
class PipelineStep(ABC):
    @abstractmethod
    def process(self, input_data: StepInput) -> StepOutput:
        pass

# Clean pipeline execution (updated for current 3-step approach)
orchestrator = PipelineOrchestrator([
    MotionDetectionStep(motion_config),           # Step 1: Motion + Tracking
    CameraHandlingFilterStep(filter_config),      # Step 2: Camera Handling Detection
    FullFrameValidationStep(ml_coordinator)       # Step 3: Full-Frame Analysis
])

result = orchestrator.process(video_path)
```

#### 3.2 Extract Motion Detection
**Status**: Motion detection implemented within NextGenVideoProcessor, needs extraction

**New Files:**
```
motion/
├── __init__.py
├── motion_detector.py        # MotionDetector class
├── background_subtractor.py  # BackgroundSubtractorFactory class
├── motion_config.py          # MotionDetectionConfig class
└── region_analyzer.py        # MotionRegionAnalyzer class
```

**Extract From:**
- Motion detection methods from `NextGenVideoProcessor`
- OpenCV background subtraction setup
- Motion region analysis and filtering

**New Architecture:**
```python
motion_detector = MotionDetector(motion_config)
motion_regions = motion_detector.detect_motion_regions(video_reader)

# Clear input/output contracts
@dataclass
class MotionRegion:
    frame_number: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    area: int
```

### Phase 4: Temporal Tracking (Week 4)

#### 4.1 Extract Tracking Systems
**New Files:**
```
tracking/
├── __init__.py
├── tracking_interface.py     # TemporalTracker abstract base class
├── deepsort_tracker.py       # DeepSORTTracker class
├── simple_tracker.py         # SimpleBboxTracker class  
├── tracking_factory.py       # TrackerFactory class
└── track_data.py            # TrackingInfo, Track dataclasses
```

**Extract From:**
- DeepSORT integration from `NextGenVideoProcessor`
- Simple bbox tracking fallback logic
- Track validation and filtering

**New Architecture:**
```python
class TemporalTracker(ABC):
    @abstractmethod
    def update_tracks(self, detections: List[Detection]) -> List[Track]:
        pass

# Pluggable tracking systems
tracker = TrackerFactory.create_tracker('deepsort', tracking_config)
tracks = tracker.update_tracks(motion_regions)
```

### Phase 5: Clean Main Classes (Week 5)

#### 5.1 New Main Application Structure
**New Files:**
```
core/
├── __init__.py
├── wildlife_processor.py     # WildlifeVideoProcessor (new main class)
├── session_manager.py        # ProcessingSessionManager class
└── batch_processor.py        # BatchVideoProcessor class
```

**Replace:**
- `NextGenVideoProcessor` with clean orchestration class
- Complex inheritance hierarchy with composition

**New Architecture:**
```python
class WildlifeVideoProcessor:
    def __init__(self, config: ProcessingConfig):
        self.pipeline = PipelineOrchestrator.from_config(config)
        self.session_manager = ProcessingSessionManager(config)
        
    def process_video(self, video_path: Path) -> ProcessingResult:
        return self.pipeline.process(video_path)
```

#### 5.2 Remove VideoProcessorBase
**Strategy:**
- Break up responsibilities into focused components
- Use composition instead of inheritance
- Extract utilities into separate modules

### Phase 6: Final Structure

#### 6.1 New Directory Structure
```
src/
├── core/                     # Main application classes
│   ├── wildlife_processor.py
│   ├── session_manager.py
│   └── batch_processor.py
├── pipeline/                 # Processing pipeline
│   ├── pipeline_orchestrator.py
│   ├── step_interface.py
│   └── steps/
├── ml/                       # ML model management
│   ├── model_manager.py
│   ├── model_loaders/
│   ├── inference/
│   ├── preprocessing.py
│   └── postprocessing.py
├── tracking/                 # Temporal tracking
│   ├── tracking_interface.py
│   ├── deepsort_tracker.py
│   └── simple_tracker.py
├── motion/                   # Motion detection
│   ├── motion_detector.py
│   └── background_subtractor.py
├── io/                       # Video I/O operations
│   ├── video_reader.py
│   ├── frame_extractor.py
│   └── analysis_writer.py
├── config/                   # Configuration management
│   ├── configuration_manager.py
│   ├── cli_parser.py
│   └── environment_config.py
├── validation/               # Analysis validation
│   └── camera_handling_detector.py
└── utils/                    # Shared utilities
    └── logging_setup.py
```

#### 6.2 Updated Scripts
- **process.py** → Thin wrapper that creates `WildlifeVideoProcessor` and calls it
- **sd_watcher.py** → Minor cleanup, mostly stays the same
- **New entry points** for testing individual components

## Current State Assessment (July 2025)

### What Has Been Accomplished ✅
1. **Configuration Improvements**: ProcessingConfig dataclass replaces global config
2. **3-Step Pipeline**: Motion detection → Camera handling → Full-frame analysis
3. **Algorithm Refinements**: 
   - Camera handling detection with spatial dispersion + motion sparsity
   - Spatial overlap validation with motion regions
   - Frame sampling improvements for temporal consistency
4. **Module Extraction**: ML detection separated into dedicated module
5. **Component Isolation**: DeepSORT tracker extracted to separate module
6. **Knowledge Base**: Comprehensive documentation and procedures

### What Still Needs Refactoring ⚠️
1. **Monolithic Classes**: NextGenVideoProcessor (2,504 lines), VideoProcessorBase (846 lines)
2. **Package Structure**: No src/ directory or proper package organization
3. **ML Ensemble**: 973-line MLDetectionEnsemble needs breakdown
4. **Mixed Concerns**: I/O, CLI, processing still mixed in base class
5. **Testing**: No unit tests for individual components

### Priority Ranking
1. **CRITICAL**: Break up NextGenVideoProcessor (2,504 lines) - Phase 3
2. **HIGH**: Extract MLDetectionEnsemble (973 lines) - Phase 2  
3. **HIGH**: Create package structure - Phase 1
4. **MEDIUM**: Extract I/O operations from VideoProcessorBase - Phase 1
5. **LOW**: Final cleanup and optimization - Phase 5

## Implementation Benefits

### Code Quality Improvements
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed Principle**: Easy to add new pipeline steps, tracking algorithms, ML models
- **Dependency Injection**: Components receive dependencies, not create them
- **Clear Interfaces**: Well-defined contracts between components

### Maintainability Benefits
- **Smaller Classes**: 100-200 lines instead of 2,500+ line monsters
- **Focused Testing**: Each component can be unit tested in isolation
- **Easier Debugging**: Problems isolated to specific components
- **Clear Dependencies**: Explicit rather than implicit coupling
- **Algorithm Isolation**: Motion detection, camera handling, validation can be tested independently

### Extensibility Benefits
- **BioCLIP Integration**: Simply add new pipeline step
- **New Tracking Algorithms**: Implement `TemporalTracker` interface
- **New ML Models**: Add new model loader and inference engine (current: YOLO, RT-DETR, MegaDetector)
- **Different Validation Strategies**: Implement new validation steps
- **Algorithm Experimentation**: Easier to test motion detection, camera handling, validation parameters

## Migration Strategy

### Phase-by-Phase Approach
1. **Phase 1**: Extract config and I/O (low risk, immediate benefits)
2. **Phase 2**: Break up ML ensemble (isolated, testable)
3. **Phase 3**: Create pipeline architecture (major structural change)
4. **Phase 4**: Extract tracking (complex but well-bounded)
5. **Phase 5**: Clean up main classes (final cleanup)

### Backward Compatibility
- Keep existing scripts working during transition
- Add deprecation warnings for old interfaces
- Provide migration guide for any breaking changes

### Testing Strategy
- **Unit Tests**: Each new class gets comprehensive unit tests
- **Integration Tests**: Pipeline components tested together
- **Regression Tests**: Ensure output identical to current system
- **Performance Tests**: Ensure no performance degradation

## Success Metrics

### Code Quality Metrics (Updated Targets)
- **Average Class Size**: Reduce from 1,400+ lines to 100-200 lines
- **Current State**: NextGenVideoProcessor (2,504), VideoProcessorBase (846), MLDetectionEnsemble (973)
- **Target State**: All classes under 300 lines
- **Cyclomatic Complexity**: Reduce from 15+ to 5-8 per method
- **Coupling**: Minimize dependencies between components
- **Test Coverage**: Achieve 80%+ coverage on new components

### Maintainability Metrics
- **Time to Add New Feature**: Should be significantly reduced
- **Time to Debug Issues**: Should be faster with isolated components
- **Developer Onboarding**: New developers should understand system faster
- **Algorithm Experimentation**: Easier to test single components

### Current Performance Baselines
- **3-Step Pipeline**: Motion detection → Camera handling → Full-frame analysis
- **Ground Truth**: Videos 7,8,9,11,12 (animals), 1-6 (false positives), 13-19 (camera handling)
- **Current Models**: yolo12x, yolo12m, MDV6-yolov10-e, rtdetr-l
- **Validation**: Spatial overlap validation with motion regions

This refactoring will transform the codebase from a collection of monolithic classes into a clean, modular architecture that follows SOLID principles and is easy to maintain, test, and extend, while preserving the sophisticated 3-step pipeline and algorithmic improvements already developed.