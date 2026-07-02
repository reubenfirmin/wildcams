"""
Processing session manager for wildlife video processing.

Records per-video outcomes and session state during batch processing. The
end-of-session summary logging lives in core/session_reporter.SessionReporter.
"""

import logging
import time
from pathlib import Path

from config.processing_config import ProcessingConfig
from core.data_types import ModelContribution, ProcessingSessionData, VideoAnalysis
from core.session_reporter import SessionReporter

logger = logging.getLogger("wildcams")


class ProcessingSessionManager:
    """
    Manages video processing sessions with comprehensive tracking.

    Handles session initialization, progress tracking, and per-video outcome
    recording for batch processing operations. Final summary logging is
    delegated to SessionReporter.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize the session manager.

        Args:
            config: ProcessingConfig with processing parameters
        """
        self.config = config
        self.session_start_time: float | None = None
        self.session_data = ProcessingSessionData()
        self.all_analyses: list[VideoAnalysis] = []
        self.failed_videos: list[Path] = []

        # Store detailed step data for failures
        self.failed_video_details: dict[str, dict] = {}  # video_name -> {'step3_data': ..., 'step4_data': ...}

    def start_session(self, videos_to_process: list[Path]) -> None:
        """
        Start a new processing session.

        Args:
            videos_to_process: List of video files to process
        """
        self.session_start_time = time.time()
        self.session_data = ProcessingSessionData(
            videos_to_process=[v.name for v in videos_to_process], total_videos=len(videos_to_process)
        )
        self.all_analyses = []
        self.failed_videos = []

        logger.info("###############################################")
        logger.info("WILDLIFE VIDEO PROCESSING SESSION START")
        logger.info("###############################################")
        logger.info(f"🎬 Found {len(videos_to_process)} videos to process")
        logger.info(f"Videos: {[v.name for v in videos_to_process]}")

    def record_video_start(self, video_path: Path, video_idx: int, total_videos: int) -> float:
        """
        Record the start of video processing.

        Args:
            video_path: Path to video being processed
            video_idx: Current video index (1-based)
            total_videos: Total number of videos

        Returns:
            Start time for timing tracking
        """
        video_start_time = time.time()
        logger.info(f"\n🎬 Processing video {video_idx}/{total_videos}: {video_path.name}")
        return video_start_time

    def record_video_success(self, video_path: Path, analysis: VideoAnalysis, processing_time: float) -> None:
        """
        Record successful video processing.

        Args:
            video_path: Path to processed video
            analysis: Analysis results
            processing_time: Time taken to process video
        """
        self.all_analyses.append(analysis)
        self.session_data.processing_times.append((video_path.name, processing_time))

        # Extract model contributions from fullframe validation metadata.
        # Copy the list: extending it below would otherwise mutate the list stored
        # on the analysis metadata, injecting Step-4 contributions into the Step-3 record.
        model_contributions = list(analysis.fullframe_result.metadata.model_contributions or [])

        # Add Step 4 classification model contributions if available
        if analysis.classification_result and analysis.classification_result.classification_enabled:
            step4_contributions = self._create_step4_model_contributions(analysis.classification_result)
            model_contributions.extend(step4_contributions)

        if model_contributions:
            # Store as tuple (video_name, list_of_contributions) as expected by SessionData
            self.session_data.model_contributions.append((video_path.name, model_contributions))

        logger.info(f"VIDEO SUCCESS: {video_path.name} - Animal detected with temporal consistency")

    def record_video_failure(
        self,
        video_path: Path,
        processing_time: float,
        reason: str | None = "no_consistent_animal_movement",
        step3_data: dict | None = None,
        step4_data: dict | None = None,
    ) -> None:
        """
        Record failed video processing with detailed step scores.

        Args:
            video_path: Path to failed video
            processing_time: Time taken to process video
            reason: Reason for failure
            step3_data: Optional Step 3 validation scores and sequences
            step4_data: Optional Step 4 classification scores and filtering details
        """
        self.failed_videos.append(video_path)
        self.session_data.processing_times.append((video_path.name, processing_time))
        self.session_data.rejection_reasons.append((video_path.name, reason))

        # Store detailed step data for summary
        self.failed_video_details[video_path.name] = {
            "step3_data": step3_data,
            "step4_data": step4_data,
            "reason": reason,
        }

        logger.info(f"VIDEO SKIPPED: {video_path.name} - {reason}")
        logger.info(f"⚪ {video_path.name}: {reason}")

        # Log detailed Step 3 scores for failure videos
        if step3_data and "validated_sequences" in step3_data:
            sequences = step3_data["validated_sequences"]
            if sequences:
                logger.info(f"📊 Step 3 Details: {len(sequences)} validated sequences")
                for i, seq in enumerate(sequences):
                    if hasattr(seq, "composite_score"):
                        logger.info(
                            f"  Sequence {i + 1}: composite_score={seq.composite_score:.3f}, detections={len(seq.detections)}"
                        )
                    if hasattr(seq, "ensemble_score"):
                        logger.info(f"    ensemble_score={seq.ensemble_score:.3f}")

                    # Log individual detection scores
                    for j, det in enumerate(seq.detections[:3]):  # Show first 3 detections
                        logger.info(f"    Detection {j + 1}: conf={det.confidence:.3f}, frame={det.frame_idx}")
            else:
                logger.info("📊 Step 3 Details: No validated sequences found")

        # Log detailed Step 4 scores for failure videos
        if step4_data:
            if "filtered_sequences" in step4_data:
                logger.info(f"📊 Step 4 Details: {step4_data['filtered_sequences']} sequences filtered")

            if "classification_results" in step4_data:
                results = step4_data["classification_results"]
                for i, result in enumerate(results):
                    crop_info = f" | {result.crop_info}" if result.crop_info else ""
                    crop_path = f" | crop: {result.crop_path}" if result.crop_path else ""
                    logger.info(
                        f"  Classification {i + 1}: animal={result.is_animal}, conf={result.animal_confidence:.3f}{crop_info}{crop_path}"
                    )
                    for individual_result in result.individual_results:
                        logger.info(
                            f"    {individual_result.model_name}: animal={individual_result.is_animal} (conf={individual_result.animal_confidence:.3f})"
                        )

            if "animal_sequences" in step4_data:
                confirmed = len(step4_data["animal_sequences"])
                logger.info(f"📊 Step 4 Summary: {confirmed} sequences confirmed as animals")

    def record_video_error(self, video_path: Path, processing_time: float, error: str) -> None:
        """
        Record video processing error.

        Args:
            video_path: Path to video that errored
            processing_time: Time taken before error
            error: Error message
        """
        self.failed_videos.append(video_path)
        # For now, just log the error - detailed tracking can be re-implemented later
        logger.info(f"⏰ Processing time: {processing_time:.2f}s")
        logger.info(f"❌ Error reason: {error}")

        logger.error(f"VIDEO ERROR: {video_path.name} - {error}")
        logger.error(f"❌ {video_path.name}: Processing failed - {error}")

    def generate_final_summary(self) -> dict:
        """
        Generate comprehensive final processing summary.

        Delegates the (large) summary logging to SessionReporter.

        Returns:
            Summary statistics dictionary
        """
        if self.session_start_time is None:
            raise ValueError("Session not started - call start_session() first")

        reporter = SessionReporter(
            config=self.config,
            session_start_time=self.session_start_time,
            session_data=self.session_data,
            all_analyses=self.all_analyses,
            failed_videos=self.failed_videos,
            failed_video_details=self.failed_video_details,
        )
        return reporter.generate_final_summary()

    def get_session_summary(self) -> dict[str, float | int | str | None]:
        """
        Get current session summary without logging.

        Returns:
            Dictionary with session statistics
        """
        if self.session_start_time is None:
            return {}

        current_time = time.time()
        elapsed_time = current_time - self.session_start_time

        return {
            "session_start_time": self.session_start_time,
            "elapsed_time": elapsed_time,
            "total_videos": self.session_data.total_videos,
            "processed_videos": len(self.session_data.processing_times),
            "successful_videos": len(self.all_analyses),
            "failed_videos": len(self.failed_videos),
            "average_processing_time": (
                sum(time for name, time in self.session_data.processing_times)
                / max(1, len(self.session_data.processing_times))
                if self.session_data.processing_times
                else 0.0
            ),
        }

    def _create_step4_model_contributions(self, classification_result) -> list:
        """
        Create model contributions for Step 4 classification models.

        Args:
            classification_result: AnimalClassificationResult from Step 4

        Returns:
            List of ModelContribution objects for BioCLIP and DeepFaune
        """
        contributions = []

        # Count total sequences and confirmed animals
        confirmed_animals = len(classification_result.animal_sequences)

        # Model contributions from individual results
        all_individual_results = [
            result for cs in classification_result.animal_sequences for result in cs.classification.individual_results
        ]

        # Group by model name and create contributions
        model_groups: dict[str, list] = {}
        for result in all_individual_results:
            model_name = result.model_name.lower()
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(result)

        for model_name, results in model_groups.items():
            confidences = [result.animal_confidence for result in results]
            if confidences:
                contrib = ModelContribution(
                    model_name=model_name,
                    total_detections=len(confidences),
                    max_confidence=max(confidences),
                    contributing_tracks=confirmed_animals,
                )
                contributions.append(contrib)

        return contributions
