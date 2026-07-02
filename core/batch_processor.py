"""
Batch video processor for wildlife video processing.

Orchestrates batch processing of multiple videos with clustering,
feature extraction, and comprehensive session management.
"""

import logging
import time
from pathlib import Path

from config.processing_config import ProcessingConfig
from core.data_types import BatchProcessingResult, VideoAnalysis

from .session_manager import ProcessingSessionManager
from .wildlife_processor import WildlifeVideoProcessor

# BatchProcessingResult now imported from data.py above

logger = logging.getLogger("wildcams")


class BatchVideoProcessor:
    """
    Batch processor for multiple wildlife videos.

    Orchestrates processing of multiple videos through the WildlifeVideoProcessor,
    manages sessions, handles clustering, and provides comprehensive reporting.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize the batch processor.

        Args:
            config: ProcessingConfig with processing parameters
        """
        self.config = config
        self.video_processor = WildlifeVideoProcessor(config)
        self.session_manager = ProcessingSessionManager(config)

        logger.info("🎯 Batch processor initialized")

    def process_all_videos(
        self, video_filter: list | None = None, force_reprocess: bool = False
    ) -> BatchProcessingResult:
        """
        Process all videos in batch with comprehensive tracking.

        Args:
            video_filter: Optional list of video indices or names to filter
            force_reprocess: If True, reprocess already processed videos

        Returns:
            Dictionary with batch processing results
        """
        # Get videos to process
        if force_reprocess or video_filter:
            videos_to_process = self.video_processor.get_video_files(video_filter)
        else:
            videos_to_process = self.video_processor.get_unprocessed_videos(video_filter)

        if not videos_to_process:
            if video_filter:
                logger.info(f"BATCH RESULT: No videos found matching filter: {video_filter}")
                logger.info(f"⚠️ No videos found matching filter: {video_filter}")
            else:
                logger.info("BATCH RESULT: No unprocessed videos found")
                logger.info("✅ No unprocessed videos found")
            return BatchProcessingResult(success=False, reason="no_videos_to_process")

        # Start processing session
        self.session_manager.start_session(videos_to_process)

        # Process each video
        for video_idx, video_path in enumerate(videos_to_process, 1):
            logger.info("=" * 80)
            logger.info(f"⏱️  VIDEO {video_idx}/{len(videos_to_process)} START: {video_path.name}")
            logger.info("=" * 80)
            video_start_time = self.session_manager.record_video_start(video_path, video_idx, len(videos_to_process))

            # Process single video - let errors bubble up
            force_reprocess_flag = force_reprocess or video_filter is not None
            analysis = self.video_processor.process_video(video_path, force_reprocess_flag)
            processing_time = time.time() - video_start_time
            logger.info("=" * 80)
            logger.info(
                f"⏱️  VIDEO {video_idx}/{len(videos_to_process)} COMPLETE: {video_path.name} - {processing_time:.2f}s"
            )
            logger.info("=" * 80)

            if analysis and not getattr(analysis, "_is_failure", False):
                # Record success
                self.session_manager.record_video_success(video_path, analysis, processing_time)
            else:
                # Record failure with step data if available
                step3_data = getattr(analysis, "_step3_data", None) if analysis else None
                step4_data = getattr(analysis, "_step4_data", None) if analysis else None
                reason = analysis.validation_result.reason if analysis else "processing_error"
                self.session_manager.record_video_failure(
                    video_path, processing_time, reason=reason, step3_data=step3_data, step4_data=step4_data
                )

        # Generate final summary
        summary = self.session_manager.generate_final_summary()

        logger.info("###############################################")
        logger.info("WILDLIFE VIDEO PROCESSING SESSION END")
        logger.info("###############################################")

        return BatchProcessingResult(success=True, summary=summary, videos_processed=len(videos_to_process))

    def process_single_video(self, video_path: Path, force_reprocess: bool = False) -> VideoAnalysis | None:
        """
        Process a single video.

        Args:
            video_path: Path to video file
            force_reprocess: If True, reprocess even if already processed

        Returns:
            Analysis results if successful, None if failed
        """
        # Check if already processed (unless forcing reprocess)
        if not force_reprocess and self.video_processor.processed_tracker.is_processed(video_path):
            logger.info(f"⏭️ Skipping already processed: {video_path.name}")
            return None

        logger.info(f"🎬 Processing single video: {video_path.name}")

        return self.video_processor.process_video(video_path)

    def get_processing_status(self) -> dict:
        """
        Get current processing status.

        Returns:
            Dictionary with current status information
        """
        session_summary = self.session_manager.get_session_summary()

        # Add additional status information
        video_files = self.video_processor.get_video_files()
        unprocessed_files = self.video_processor.get_unprocessed_videos()

        return {
            "session": session_summary,
            "total_video_files": len(video_files),
            "unprocessed_videos": len(unprocessed_files),
            "processing_complete": len(unprocessed_files) == 0,
        }

    def force_reprocess_all(self, video_filter: list | None = None) -> BatchProcessingResult:
        """
        Force reprocessing of all videos, ignoring processed status.

        Args:
            video_filter: Optional list of video indices or names to filter

        Returns:
            Dictionary with batch processing results
        """
        logger.info("🔄 Forcing reprocessing of all videos (ignoring .processed files)")
        return self.process_all_videos(video_filter=video_filter, force_reprocess=True)
