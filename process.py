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
#   "pybioclip>=0.1.0",
#   "pytorchwildlife>=1.0.0"
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
from constants import DEFAULT_ENSEMBLE_MODELS, MAIN_LOGGER_NAME
from data_types import ProcessingStatus

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
    config_manager.load_from_cli_args(args, include_motion=True, include_tracking=True)
    config = config_manager.get_processing_config()
    return config


def main() -> None:
    """Main entry point for next-generation processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Next Generation Wildlife Video Processor with Temporal Consistency')
    
    # Add all arguments from config module
    config_manager = ConfigurationManager()
    config_manager.setup_common_arguments(parser)
    config_manager.setup_motion_detection_arguments(parser)
    config_manager.setup_tracking_arguments(parser)
    
    args = parser.parse_args()
    
    # Parse video filter using config manager
    video_filter = config_manager.parse_video_filter(args)
    
    # Initialize global config from CLI arguments
    config = initialize_config_from_args(args)
    
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
        print(f"🎯 Tracking method: {args.tracking_method}")
        
        logger.info(f"🎯 Processing strategy: Next Generation Temporal Consistency")
        logger.info(f"🕒 Min track duration: {args.min_track_duration}s")
        logger.info(f"✅ Full-frame validation frames: {args.full_frame_validation_frames}")
        logger.info(f"🎯 Tracking method: {args.tracking_method}")
        
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