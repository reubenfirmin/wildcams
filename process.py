#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "python-dotenv>=1.0.0",
#   "opencv-python>=4.8.0",
#   "ultralytics>=8.0.0",
#   "scikit-learn>=1.3.0",
#   "numpy>=1.24.0",
#   "pillow>=10.0.0",
#   "tqdm>=4.66.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
#   "transformers>=4.35.0",
#   "open-clip-torch>=2.20.0",
#   "pybioclip>=0.1.0",
#   "huggingface-hub>=0.19.0",
#   "timm>=0.9.0"
# ]
# ///
"""
Next Generation Wildlife Video Processor.
Combines motion detection with temporal consistency tracking and full-frame validation.
Uses modular pipeline architecture for maintainability and extensibility.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Import new configuration modules
from config import ProcessingConfig, ConfigurationManager
from core.constants import DEFAULT_ENSEMBLE_MODELS, MAIN_LOGGER_NAME
from core.data_types import ProcessingStatus

# Import new core classes
from core import BatchVideoProcessor

# Global config instance
config: ProcessingConfig = None

# Get loggers
logger = logging.getLogger('wildcams')


def initialize_config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    """Initialize global config from CLI arguments using ConfigurationManager."""
    global config
    
    # Create configuration manager and pass the already parsed args object
    config_manager = ConfigurationManager()
    config_manager.load_from_cli_args(args, include_motion=True)
    config = config_manager.get_processing_config()
    return config


def setup_logging_early() -> None:
    """Setup logging BEFORE any other components are initialized."""
    from datetime import datetime
    
    # Create timestamp for this session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup the root wildcams logger with propagation
    logger = logging.getLogger('wildcams')
    logger.propagate = False  # Don't propagate to root logger
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create logs directory in CWD
    logs_dir = Path('./logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file handler
    log_file = logs_dir / f'wildcams_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Set logger level
    logger.setLevel(logging.DEBUG)
    
    # Configure all Python logging to be quiet except our logger
    logging.getLogger().setLevel(logging.WARNING)
    
    logger.info(f"📋 Logging initialized - session {timestamp}")
    logger.info(f"📋 Log file: logs/{log_file.name}")
    
    # Test that this logger configuration works
    test_logger = logging.getLogger('wildcams')
    test_logger.info(f"📋 Test log message - logger configuration verified")


def main() -> None:
    """Main entry point for next-generation processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Next Generation Wildlife Video Processor with Temporal Consistency')
    
    # Add all arguments from config module
    config_manager = ConfigurationManager()
    config_manager.setup_common_arguments(parser)
    config_manager.setup_motion_detection_arguments(parser)
    
    args = parser.parse_args()
    
    # Parse video filter using config manager
    video_filter = config_manager.parse_video_filter(args)
    
    # Initialize global config from CLI arguments
    config = initialize_config_from_args(args)
    
    # Setup logging EARLY before any other components
    setup_logging_early()
    
    try:
        # Create batch processor with new core architecture
        batch_processor = BatchVideoProcessor(config)
        
        # Log ALL parameters explicitly AFTER logger is initialized
        logger.info("================================================================================")
        logger.info("🎯 COMMAND PARAMETERS")
        logger.info("================================================================================")
        for attr_name in sorted(vars(args)):
            attr_value = getattr(args, attr_name)
            logger.info(f"{attr_name}: {attr_value}")
        logger.info("================================================================================")
        
        print(f"🎬 Starting Next Generation wildlife video processing...")
        print(f"📊 Mode: Motion detection + temporal consistency + full-frame validation")
        print(f"🕒 Temporal parameters: {args.min_track_duration}s duration, motion gap {args.motion_tracking_gap_seconds}s, min consecutive detection {args.min_consecutive_detection_seconds}s")
        
        logger.info(f"🎯 Processing strategy: Next Generation Temporal Consistency")
        logger.info(f"🕒 Min track duration: {args.min_track_duration}s")
        logger.info(f"✅ Full-frame validation frames: {args.full_frame_validation_frames}")
        
        # Process all videos using new batch processor
        result = batch_processor.process_all_videos(video_filter=video_filter)
        
    except KeyboardInterrupt:
        print("🛑 Processing interrupted by user")
    except Exception as e:
        import traceback
        print(f"❌ Processing failed: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()