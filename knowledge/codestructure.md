# Code Structure

```
wildcams/
├── process.py                          # Main CLI entry point (uv PEP 723 inline script)
├── process-videos.sh                   # Convenience wrapper around process.py
├── watch.sh                            # Start/restart the SD card watcher daemon
├── stop-watcher.sh                     # Stop the SD card watcher daemon
├── CLAUDE.md                           # Project instructions
├── README.md                           # Project documentation
├── pyproject.toml                      # Python project configuration (deps mirror process.py header)
├── sd_watcher.py                       # SD card watcher utility
├── .env                                # Environment variables (used by sd_watcher only)
├── .envrc                              # Directory environment setup (uv + direnv; Nix removed)
├── .gitignore                          # Git ignore patterns
├── .github/workflows/ci.yml            # CI: uv sync + pytest on push/PR
├── tests/                              # Automated tests
│   ├── conftest.py                     # Path setup so packages import under pytest
│   ├── test_imports.py                 # Import smoke test (module-graph guardrail)
│   ├── test_functional_utils.py        # Video-selection pure-function tests
│   └── test_config.py                  # CLI-argument parsing / default-value tests
├── docs/superpowers/specs/             # Design specs (recovery & updates)
├── core/                               # Core architecture
│   ├── __init__.py
│   ├── batch_processor.py              # Batch processing orchestration
│   ├── constants.py                    # All system constants
│   ├── data_types.py                   # All typed data structures
│   ├── functional_utils.py             # Pure functional utilities
│   ├── session_manager.py              # Session tracking and logging
│   └── wildlife_processor.py           # Individual video processing
├── config/                             # Configuration system
│   ├── __init__.py
│   ├── configuration_manager.py        # CLI argument parsing
│   └── processing_config.py            # Typed configuration dataclass
├── pipeline/                           # Processing pipeline
│   ├── __init__.py
│   ├── camera_handling_filter.py       # Camera handling detection logic
│   ├── fullframe_validator.py          # Full-frame validation logic
│   ├── step_interface_v2.py            # Pipeline interfaces and orchestrator
│   └── steps_v2/                       # Current step implementations
│       ├── __init__.py
│       ├── animal_classification_step.py   # Step 4: Animal classification
│       ├── camera_handling_step.py         # Step 2: Camera handling filter
│       ├── fullframe_validation_step.py    # Step 3: Full-frame validation
│       └── motion_detection_step.py        # Step 1: Motion detection
├── ml/                                 # Machine learning
│   ├── __init__.py
│   ├── constants.py                    # ML-specific constants
│   ├── ensemble_wrapper.py             # ML model ensemble coordination
│   ├── model_manager.py               # Model loading and caching
│   ├── postprocessing.py              # Detection post-processing
│   ├── preprocessing.py               # Input preprocessing
│   └── inference/                      # Individual model inference engines
│       ├── __init__.py
│       ├── bioclip_inference.py            # BioCLIP species classification
│       ├── classification_coordinator.py   # Classification ensemble coordination
│       ├── deepfaune_inference.py          # DeepFaune animal classification
│       ├── ensemble_coordinator.py         # Detection ensemble coordination
│       ├── megadetector_inference.py       # MegaDetector inference
│       ├── rtdetr_inference.py             # RT-DETR inference
│       └── yolo_inference.py               # YOLO inference
├── motion/                             # Motion detection
│   ├── __init__.py
│   ├── background_subtractor.py        # Background subtraction algorithms
│   ├── motion_detector.py             # Motion detection coordination
│   └── motion_tracker.py              # Motion tracking and linking
├── video_io/                           # Video I/O
│   ├── __init__.py
│   ├── analysis_writer.py             # Analysis result serialization
│   ├── frame_extractor.py            # Frame extraction utilities
│   ├── processed_tracker.py          # Processed video tracking
│   └── video_reader.py               # Video reading and frame access
└── knowledge/                          # Knowledge base
    ├── CLAUDE.md                       # Project overview and instructions
    ├── codestructure.md               # This file - code organization
    ├── experiments.md                 # Experimental results and findings
    ├── procedures.md                  # Common task procedures
    └── theory.md                      # Technical architecture and algorithms
```