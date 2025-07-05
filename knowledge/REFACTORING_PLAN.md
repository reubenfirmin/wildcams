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

### Phase 1: Configuration & Infrastructure ✅ COMPLETED

#### 1.1 Configuration Management ✅ COMPLETED
**Status**: Complete configuration system implemented with clean architecture

**Completed Files:**
```
config/
├── __init__.py                 ✅ Done
├── configuration_manager.py    ✅ Done - ConfigurationManager class with CLI parsing
└── processing_config.py        ✅ Done - ProcessingConfig dataclass with all parameters
```

**Accomplishments:**
- ✅ Complete CLI argument parsing extracted from VideoProcessorBase  
- ✅ ProcessingConfig dataclass with 80+ parameters including new confidence bridge
- ✅ Direct CLI-to-config mapping without environment variable intermediate layer
- ✅ All hardcoded constants eliminated per project rules
- ✅ Configuration passed as parameters throughout system

#### 1.2 Video I/O Operations ✅ COMPLETED  
**Status**: Complete video I/O extraction with clean separation

**Completed Files:**
```
video_io/
├── __init__.py                 ✅ Done
├── video_reader.py            ✅ Done - VideoReader class with fallback backends
├── frame_extractor.py         ✅ Done - FrameExtractor with temporal sampling
├── analysis_writer.py         ✅ Done - AnalysisWriter with JSON/pickle output
└── processed_tracker.py       ✅ Done - ProcessedVideoTracker with .processed files
```

**Accomplishments:**
- ✅ Video I/O completely extracted from VideoProcessorBase (reduced from 846 to 480 lines)
- ✅ Frame extraction with sophisticated temporal clustering
- ✅ Analysis output with proper numpy serialization
- ✅ Processed video tracking with IMG_xxxx.MP4 format support
- ✅ Clean interfaces with dependency injection

#### 1.3 Bug Fixes and Enhancements ✅ COMPLETED
**Accomplishments:**
- ✅ Fixed conf=0.000 issue by adding ensemble_score tracking to validated sequences
- ✅ Fixed motion threshold regression (int vs float type error)
- ✅ Added confidence bridge feature with --confidence-bridge-threshold parameter
- ✅ Fixed model contribution analysis to show all ensemble models including zero-detection models
- ✅ Enhanced temporal continuity validation with biological reasoning

### Phase 2: ML Model Management ✅ COMPLETED

#### 2.1 Break Up MLDetectionEnsemble ✅ COMPLETED
**Status**: Successfully refactored 973-line monolithic class into modular architecture

**Completed Files:**
```
ml/
├── __init__.py                              ✅ Done
├── model_manager.py                         ✅ Done - Model loading and caching
├── preprocessing.py                         ✅ Done - TTA and multi-scale processing
├── postprocessing.py                        ✅ Done - NMS and coordinate transformations
├── feature_extractor.py                     ✅ Done - ResNet18 feature extraction
├── inference/
│   ├── __init__.py                          ✅ Done
│   ├── yolo_inference.py                    ✅ Done - YOLO inference engine
│   ├── megadetector_inference.py            ✅ Done - MegaDetector inference engine
│   ├── rtdetr_inference.py                  ✅ Done - RT-DETR inference engine
│   └── ensemble_coordinator.py              ✅ Done - Orchestrates all inference engines
└── ensemble_wrapper.py                      ✅ Done - Backward compatibility wrapper
```

**Accomplishments:**
- ✅ Broke down 973-line monolithic class into 6+ focused components
- ✅ Created modular architecture with clear separation of concerns:
  - **ModelManager**: Model loading, caching, and availability checking
  - **PreprocessingPipeline**: TTA transformations and multi-scale detection
  - **PostprocessingPipeline**: NMS, filtering, and coordinate transformations
  - **FeatureExtractor**: ResNet18 feature extraction for clustering
  - **Inference Engines**: Specialized engines for YOLO, RT-DETR, and MegaDetector
  - **EnsembleCoordinator**: Orchestrates inference across all model types
- ✅ Maintained backward compatibility through wrapper pattern
- ✅ Updated imports in video_processor_base.py to use new ml package
- ✅ Verified package structure and syntax correctness
- ✅ Created pluggable architecture supporting different model types

**Model Support:**
- **YOLO Models**: YOLOv8x/m, YOLOv10 (all variants), YOLOv11 (all variants), YOLOv12x/m
- **MegaDetector**: MDV6-yolov9-e, MDV6-yolov10-e, MDV6-rtdetr-c
- **RT-DETR**: rtdetr-l, rtdetr-x (standalone models)
- **Current Ensemble**: yolo12x, yolo12m, MDV6-yolov10-e, rtdetr-l

**New Architecture:**
```python
# Clean modular architecture with dependency injection
model_manager = ModelManager(ensemble_models, cache_dir)
ensemble_coordinator = EnsembleCoordinator(model_manager)

# Backward compatibility maintained through wrapper
ml_ensemble = MLDetectionEnsemble(confidence_threshold, ensemble_models, cache_dir)
detections = ml_ensemble.run_ensemble_detection(frame, timestamp, frame_idx, full_frame)
```

#### 2.2 CLI Argument Centralization ✅ COMPLETED
**Status**: All argument parsing consolidated in ConfigurationManager

**Accomplishments:**
- ✅ Moved stray `--videos` argument from process.py to ConfigurationManager
- ✅ Added centralized `parse_video_filter()` method
- ✅ Cleaned up scattered argument parsing logic
- ✅ Updated default parameters to match optimized workflow:
  - Ensemble: `yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l`
  - Confidence threshold: `0.8`
  - Spatial overlap threshold: `0.1` 
  - Track infilling: enabled by default
  - Infill gap: `0.7s`, distance: `350px`

**Before/After:**
```python
# Before (scattered in process.py)
parser.add_argument('--videos', '-v', nargs='+', help='...')
video_filter = None
if args.videos:
    # manual parsing logic...

# After (centralized in ConfigurationManager)
config_manager.setup_common_arguments(parser)  # includes --videos
video_filter = config_manager.parse_video_filter(args)
```

### Phase 3: Processing Pipeline ✅ COMPLETED

#### 3.1 Create Pipeline Architecture ✅ COMPLETED
**Status**: Successfully extracted 3-step pipeline from monolithic NextGenVideoProcessor into modular architecture

**Completed Files:**
```
pipeline/
├── __init__.py                              ✅ Done
├── pipeline_orchestrator.py                ✅ Done - Orchestrates execution of pipeline steps
├── step_interface.py                       ✅ Done - Abstract base class and data structures
├── camera_handling_filter.py               ✅ Done - Core camera handling detection logic
├── fullframe_validator.py                  ✅ Done - Full-frame validation with spatial overlap
└── steps/
    ├── __init__.py                          ✅ Done
    ├── motion_detection_step.py             ✅ Done - Step 1: Motion detection & tracking
    ├── camera_handling_step.py              ✅ Done - Step 2: Camera handling filter
    └── fullframe_validation_step.py         ✅ Done - Step 3: Full-frame ML validation

motion/
├── __init__.py                              ✅ Done
├── background_subtractor.py                ✅ Done - OpenCV background subtractor factory
├── motion_detector.py                      ✅ Done - Motion detection with region filtering
└── motion_tracker.py                       ✅ Done - Temporal tracking with infilling
```

**Accomplishments:**
- ✅ Extracted 3-step pipeline from 2,413-line NextGenVideoProcessor into focused components
- ✅ Created modular pipeline architecture with clear step interfaces
- ✅ Implemented sophisticated motion detection and tracking with infilling
- ✅ Built camera handling filter with spatial clustering and frame coverage analysis
- ✅ Developed full-frame validation with ensemble models and spatial overlap verification
- ✅ Maintained all algorithmic sophistication while improving maintainability
- ✅ Created clean separation of concerns with dependency injection
- ✅ Implemented comprehensive logging and debugging throughout pipeline

**New Architecture:**
```python
# Modular pipeline execution with clear step boundaries
steps = [
    MotionDetectionStep(motion_config),           # Step 1: Motion + Tracking
    CameraHandlingFilterStep(filter_config),      # Step 2: Camera Handling Detection  
    FullFrameValidationStep(ml_ensemble, config)  # Step 3: Full-Frame Analysis
]

orchestrator = PipelineOrchestrator(steps)
result = orchestrator.process(video_path)

# Each step has clear input/output contracts
class PipelineStep(ABC):
    @abstractmethod
    def process(self, step_input: StepInput) -> StepOutput:
        pass
```

**Key Features Preserved:**
- **Motion Detection**: OpenCV background subtraction with intelligent region expansion
- **Track Infilling**: Bridges gaps between motion tracks with spatial/temporal criteria  
- **Camera Handling**: Frame coverage analysis with spatial clustering and consistency penalties
- **Spatial Overlap**: Validates ML detections against motion regions with configurable thresholds
- **Ensemble Scoring**: Multi-model consensus with confidence boosting
- **Early Exit Logic**: Prevents expensive ML processing when camera handling detected

#### 3.2 Extract Motion Detection ✅ COMPLETED
**Status**: Motion detection successfully extracted from NextGenVideoProcessor with clean architecture

**Completed Files:**
```
motion/
├── __init__.py                              ✅ Done
├── motion_detector.py                       ✅ Done - MotionDetector class with region filtering
├── background_subtractor.py                ✅ Done - BackgroundSubtractorFactory class
└── motion_tracker.py                       ✅ Done - MotionTracker class with infilling
```

**Accomplishments:**
- ✅ Motion detection completely extracted from NextGenVideoProcessor
- ✅ Clean architecture with ProcessingConfig object passing (no magic strings)
- ✅ Sophisticated region filtering and expansion for ML context
- ✅ OpenCV background subtraction with MOG2/KNN support
- ✅ Temporal tracking with gap infilling based on spatial/temporal criteria
- ✅ Comprehensive logging and parameter validation

**New Architecture:**
```python
# Clean config object passing instead of parameter dictionaries
motion_detector = MotionDetector(config)
motion_regions = motion_detector.detect_motion_regions(frame, config)

# All parameter access via ProcessingConfig attributes
if area < config.min_motion_area or area > config.max_motion_area:
    continue
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

## Phase 5.0 - cleanup
* Objects for everything. No dicts for "on the fly" objects. 
* Consistent typing throughout. All parameters need a type. All functions need a return type. No kwargs. No loose dicts. No magic strings.
* Final sweep to ensure that config is passed through to all functions that need it.
* Get rid of as many self references as possible. Object oriented with a strong functional approach.
* 'yolov8n' is repeated in code 3 times (for one example). DRY. yolo models should be handled in the yolo inference and cli, but nowhere else.


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

## Current State Assessment (Updated July 2025)

### ✅ Phase 1 COMPLETED - Configuration & Infrastructure
1. **Configuration System**: Complete ProcessingConfig dataclass with ConfigurationManager
2. **Video I/O Extraction**: All I/O operations extracted to video_io/ package 
3. **Code Reduction**: VideoProcessorBase reduced from 846 to 480 lines (-43%)
4. **Bug Fixes**: Fixed confidence tracking, motion thresholds, model analysis
5. **New Features**: Confidence bridge, enhanced temporal validation

### ✅ Phase 2 COMPLETED - ML Model Management & CLI Cleanup
1. **ML Architecture Refactoring**: 973-line MLDetectionEnsemble broken into 6+ focused components
2. **Modular ML Package**: Created ml/ package with clear separation of concerns
3. **Backward Compatibility**: Maintained through ensemble wrapper pattern
4. **CLI Centralization**: Moved all argument parsing to ConfigurationManager
5. **Updated Defaults**: Set optimized parameter defaults for current workflow
6. **Import Updates**: Updated all imports to use new modular architecture

### ✅ Phase 3 COMPLETED - Processing Pipeline Architecture
1. **Pipeline Extraction**: Successfully extracted 2,413-line NextGenVideoProcessor into modular pipeline
2. **Motion Detection Package**: Created motion/ package with sophisticated detection and tracking
3. **Pipeline Architecture**: Built flexible pipeline with step interface and orchestrator
4. **Camera Handling Filter**: Extracted complex frame coverage analysis with spatial clustering
5. **Full-Frame Validation**: Modularized ensemble ML validation with spatial overlap verification
6. **Algorithm Preservation**: Maintained all sophisticated algorithms while improving maintainability
7. **Parameter Dictionary Elimination**: Eliminated all magic string parameter access, replaced with ProcessingConfig object passing
8. **Clean Architecture**: Config objects passed to functions instead of stored in self, following SOLID principles

### ⚠️ Remaining Refactoring Priorities (Updated)
1. **MEDIUM**: Extract tracking systems (DeepSORT integration) - Phase 4
2. **MEDIUM**: Create final application wrapper classes - Phase 5
3. **LOW**: Final cleanup and package structure - Phase 5

### Updated Code Metrics (After Phase 3)
- **process.py**: 2,413 lines → **102 lines** (-95.8% reduction) ✅ Massive breakthrough
- **wildlife_processor.py**: 229 lines (clean pipeline orchestration) ✅ New modular implementation  
- **pipeline/ package**: 10+ focused components (motion detection, camera handling, validation) ✅ Major improvement
- **motion/ package**: 4 focused components (detection, tracking, background subtraction) ✅ Major improvement
- **video_processor_base.py**: 480 lines (was 846, -43% reduction) ✅ Major improvement
- **ml/ package**: 6+ focused components (was 973-line monolith) ✅ Major improvement
- **CLI Management**: Centralized in ConfigurationManager ✅ Major improvement
- **Parameter Management**: Zero magic strings, all config object based ✅ Clean architecture achieved
- **Total complexity**: Dramatically reduced with modular architecture - **core refactoring objectives achieved**

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