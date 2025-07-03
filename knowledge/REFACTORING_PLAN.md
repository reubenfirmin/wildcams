# Wildlife Camera Processing System - Refactoring Plan

## Overview
Transform the current monolithic codebase into a clean, modular, object-oriented architecture. Break up large classes into focused components with clear responsibilities and well-defined interfaces.

## Current Problems

### Code Quality Issues
- **God Classes**: `NextGenVideoProcessor` (~1,400 lines), `VideoProcessorBase` (~747 lines)
- **Mixed Concerns**: Motion detection, ML inference, file I/O, and configuration all mixed together
- **Tight Coupling**: Global config object creates dependencies across components
- **Poor Testability**: Large classes with multiple responsibilities are hard to unit test
- **Difficult Maintenance**: Changes require understanding the entire system

### Specific Issues by File
- **process.py**: Single massive class handling 4-step pipeline, motion detection, tracking, ML coordination
- **video_processor_base.py**: God class managing CLI, environment, video I/O, validation, clustering
- **ml_detection.py**: Model loading, preprocessing, inference, and validation all in one class
- **sd_watcher.py**: Actually well-structured, minor improvements needed

## Refactoring Strategy

### Phase 1: Configuration & Infrastructure (Week 1)

#### 1.1 Extract Configuration Management
**New Files:**
```
config/
├── __init__.py
├── configuration_manager.py   # ConfigurationManager class
├── cli_parser.py             # CLIArgumentParser class
├── environment_config.py     # EnvironmentConfigLoader class
└── processing_config.py      # ProcessingConfig (improved dataclass)
```

**Extract From:**
- Global `config` object from `process.py`
- Static CLI methods from `video_processor_base.py`
- Environment variable loading scattered across files

**New Architecture:**
```python
# Instead of global config object
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
├── frame_extractor.py        # FrameExtractor class
├── analysis_writer.py        # AnalysisWriter class
└── processed_tracker.py      # ProcessedVideoTracker class
```

**Extract From:**
- Video opening/reading methods from `video_processor_base.py`
- Frame extraction and sampling logic
- Analysis output writing and `.processed` file management

**New Architecture:**
```python
video_reader = VideoReader(video_path)
frame_extractor = FrameExtractor(max_frames=20, sampling_strategy='uniform')
frames = frame_extractor.extract_frames(video_reader)
```

### Phase 2: ML Model Management (Week 2)

#### 2.1 Break Up MLDetectionEnsemble
**New Files:**
```
ml/
├── __init__.py
├── model_manager.py          # ModelManager class
├── model_loaders/
│   ├── __init__.py
│   ├── yolo_loader.py        # YOLOModelLoader class (standalone YOLOv8x/m)
│   └── megadetector_loader.py # MegaDetectorLoader class (all MDV6-* variants)
├── inference/
│   ├── __init__.py
│   ├── inference_engine.py   # InferenceEngine class
│   ├── yolo_inference.py     # YOLOInferenceEngine class
│   ├── megadetector_inference.py # MegaDetectorInferenceEngine class
│   └── ensemble_coordinator.py # EnsembleCoordinator class
├── preprocessing.py          # PreprocessingPipeline class
└── postprocessing.py         # PostprocessingPipeline class
```

**Extract From:**
- Model initialization logic from `MLDetectionEnsemble`
- Individual model inference methods
- Preprocessing transformations (TTA, multi-scale)
- NMS and detection filtering

**Model Boundaries:**
- **YOLOLoader**: Handles standalone YOLOv8x and YOLOv8m models
- **MegaDetectorLoader**: Handles all MegaDetector v6 variants:
  - `MDV6-yolov9-e` (MegaDetector wrapper around YOLOv9)
  - `MDV6-yolov10-e` (MegaDetector wrapper around YOLOv10)  
  - `MDV6-rtdetr-c` (MegaDetector wrapper around RT-DETR)
- RT-DETR is only accessed through MegaDetector interface, never directly

**New Architecture:**
```python
# Replace single ensemble class with coordinated components
model_manager = ModelManager(cache_dir, ensemble_models)
inference_engine = InferenceEngine(model_manager)
ensemble_coordinator = EnsembleCoordinator(inference_engine)

# Clean separation of concerns
detections = ensemble_coordinator.detect(image, crop_mode=True)
```

#### 2.2 Extract Preprocessing & Postprocessing
**Responsibilities:**
- **PreprocessingPipeline**: TTA, multi-scale, histogram equalization
- **PostprocessingPipeline**: NMS, confidence filtering, coordinate transformation

### Phase 3: Processing Pipeline (Week 3)

#### 3.1 Create Pipeline Architecture
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
    ├── crop_analysis_step.py       # CropAnalysisStep class
    └── fullframe_validation_step.py # FullFrameValidationStep class
```

**Extract From:**
- 4-step pipeline logic from `NextGenVideoProcessor`
- Each step becomes a separate class with clear input/output contracts

**New Architecture:**
```python
class PipelineStep(ABC):
    @abstractmethod
    def process(self, input_data: StepInput) -> StepOutput:
        pass

# Clean pipeline execution
orchestrator = PipelineOrchestrator([
    MotionDetectionStep(motion_config),
    CameraHandlingFilterStep(filter_config),
    CropAnalysisStep(ml_coordinator),
    FullFrameValidationStep(validation_config)
])

result = orchestrator.process(video_path)
```

#### 3.2 Extract Motion Detection
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

## Implementation Benefits

### Code Quality Improvements
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed Principle**: Easy to add new pipeline steps, tracking algorithms, ML models
- **Dependency Injection**: Components receive dependencies, not create them
- **Clear Interfaces**: Well-defined contracts between components

### Maintainability Benefits
- **Smaller Classes**: 100-200 lines instead of 1,400+ line monsters
- **Focused Testing**: Each component can be unit tested in isolation
- **Easier Debugging**: Problems isolated to specific components
- **Clear Dependencies**: Explicit rather than implicit coupling

### Extensibility Benefits
- **BioCLIP Integration**: Simply add new pipeline step
- **New Tracking Algorithms**: Implement `TemporalTracker` interface
- **New ML Models**: Add new model loader and inference engine
- **Different Validation Strategies**: Implement new validation steps

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

### Code Quality Metrics
- **Average Class Size**: Reduce from 500+ lines to 100-200 lines
- **Cyclomatic Complexity**: Reduce from 15+ to 5-8 per method
- **Coupling**: Minimize dependencies between components
- **Test Coverage**: Achieve 80%+ coverage on new components

### Maintainability Metrics
- **Time to Add New Feature**: Should be significantly reduced
- **Time to Debug Issues**: Should be faster with isolated components
- **Developer Onboarding**: New developers should understand system faster

This refactoring will transform the codebase from a collection of monolithic classes into a clean, modular architecture that follows SOLID principles and is easy to maintain, test, and extend.