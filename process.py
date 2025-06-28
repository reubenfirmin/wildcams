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
Unified wildlife video processor with configurable strategies.
Supports both full-frame and motion detection approaches.
"""

import sys
import logging

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Wildlife video processor with configurable strategies')
    parser.add_argument('--videos', '-v', nargs='+', help='Optional list of video indices (e.g. 7 8 9) or names (e.g. IMG_0007.MP4) to process')
    parser.add_argument('--strategy', '-s', choices=['ff', 'fullframe', 'md', 'motiondetection'], default='ff',
                       help='Processing strategy: ff/fullframe (full-frame) or md/motiondetection (motion detection) [default: ff]')
    
    # Import here to avoid circular imports
    from video_processor_base import VideoProcessorBase
    
    # Add common arguments from base class
    VideoProcessorBase.setup_common_arguments(parser)
    
    # Add motion detection arguments conditionally based on strategy
    args, unknown = parser.parse_known_args()
    if args.strategy in ['md', 'motiondetection']:
        VideoProcessorBase.setup_motion_detection_arguments(parser)
    
    # Re-parse with all arguments
    args = parser.parse_args()
    
    # Convert video arguments to appropriate format
    video_filter = None
    if args.videos:
        video_filter = []
        for video in args.videos:
            try:
                # Try to parse as integer first
                video_filter.append(int(video))
            except ValueError:
                # If not an integer, treat as string
                video_filter.append(video)
    
    # Set environment variables from parsed arguments
    include_motion = args.strategy in ['md', 'motiondetection']
    VideoProcessorBase.set_environment_from_args(args, include_motion=include_motion)
    
    try:
        # Import and instantiate the appropriate processor
        if args.strategy in ['ff', 'fullframe']:
            from process_fullframe import FullFrameVideoProcessor
            processor = FullFrameVideoProcessor()
            strategy_name = "Full Frame"
            print(f"🎬 Starting wildlife video processing ({strategy_name})...")
            print(f"📊 Mode: Full frame ML ensemble processing")
        else:  # md, motiondetection
            from process_motiondetection import MotionDetectionVideoProcessor  
            processor = MotionDetectionVideoProcessor()
            strategy_name = "Motion Detection"
            print(f"🎬 Starting wildlife video processing ({strategy_name})...")
            print(f"📊 Mode: Motion detection + crop-based ML processing")
        
        logger = logging.getLogger('wildcams')
        logger.info(f"🎯 Processing strategy: {strategy_name}")
        logger.info(f"🎯 MegaDetector version: {args.megadetector_version}")
        logger.info(f"🤖 Ensemble models: {args.ensemble}")
        
        processor.process_all_videos(video_filter=video_filter)
        
    except KeyboardInterrupt:
        print("🛑 Processing interrupted by user")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()