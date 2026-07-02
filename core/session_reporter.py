"""
Session reporting for wildlife video processing.

Renders the end-of-session summary logging (successful/failed videos, timing,
and model-contribution analysis) from the data collected by
ProcessingSessionManager. Split out so the manager holds recording/state and
the reporter holds the (large) logging surface.
"""

import logging
import time
from pathlib import Path

from config.processing_config import ProcessingConfig
from core.data_types import ProcessingSessionData, VideoAnalysis

logger = logging.getLogger("wildcams")


class SessionReporter:
    """Renders the final processing summary from collected session data."""

    def __init__(
        self,
        config: ProcessingConfig,
        session_start_time: float,
        session_data: ProcessingSessionData,
        all_analyses: list[VideoAnalysis],
        failed_videos: list[Path],
        failed_video_details: dict[str, dict],
    ):
        self.config = config
        self.session_start_time = session_start_time
        self.session_data = session_data
        self.all_analyses = all_analyses
        self.failed_videos = failed_videos
        self.failed_video_details = failed_video_details

    def generate_final_summary(self) -> dict:
        """
        Generate comprehensive final processing summary.

        Returns:
            Summary statistics dictionary
        """
        total_videos = self.session_data.total_videos
        animal_videos = len(self.all_analyses)
        no_animal_videos = total_videos - animal_videos
        batch_total_time = time.time() - self.session_start_time

        logger.info("=" * 80)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"🎬 Total videos processed: {total_videos}")
        logger.info(f"🦌 Videos with animals detected: {animal_videos}")
        logger.info(f"⚪ Videos with no animals detected: {no_animal_videos}")
        logger.info("")

        # Log successful videos
        self._log_successful_videos()

        # Log failed videos
        self._log_failed_videos()

        # Log timing summary
        timing_stats = self._log_timing_summary(batch_total_time)

        # Generate model contribution analysis
        self._generate_model_contribution_analysis()

        logger.info(f"📁 Analysis files saved to: {Path(self.config.video_dir) / 'analysis'}")

        if self.all_analyses:
            logger.info(f"✅ Successfully processed {len(self.all_analyses)} videos with temporal consistency")

        logger.info("=" * 80)

        # Return summary statistics
        return {
            "total_videos": total_videos,
            "successful_videos": animal_videos,
            "failed_videos": no_animal_videos,
            "batch_time": batch_total_time,
            "timing_stats": timing_stats,
            "analyses": self.all_analyses,
            "failed_video_paths": self.failed_videos,
        }

    def _log_successful_videos(self) -> None:
        """Log details of successfully processed videos."""
        if not self.all_analyses:
            return

        logger.info("🦌 VIDEOS WITH ANIMALS DETECTED:")
        processing_times = dict(self.session_data.processing_times)
        for analysis in self.all_analyses:
            video_name = analysis.video_path.name
            confidence = analysis.validation_result.confidence
            ensemble_score = analysis.validation_result.ensemble_score
            composite_score = analysis.validation_result.composite_score
            validated_sequences = analysis.validation_result.validated_sequences_count
            processing_time = processing_times.get(video_name, 0.0)

            # Extract timing info from detection
            best_detection = analysis.validation_result.best_detection
            timestamp = best_detection.timestamp

            # Add species information if available
            species_info = ""
            if hasattr(analysis, "species_summary") and analysis.species_summary:
                species_info = f", species=[{analysis.species_summary}]"

            # Add model approval information if available
            approving_model_info = ""
            if (
                hasattr(analysis, "classification_result")
                and analysis.classification_result
                and analysis.classification_result.classification_enabled
            ):
                # Find the model that approved this video
                animal_sequences = analysis.classification_result.animal_sequences
                if animal_sequences:
                    # Get the approving model from the first confirmed sequence
                    approving_model = animal_sequences[0].classification.approving_model
                    if approving_model:
                        approving_model_info = f", approved_by={approving_model}"

            logger.info(
                f"  ✅ {video_name}: time_range={timestamp:.1f}s, "
                f"conf={confidence:.3f}, ensemble={ensemble_score:.3f}, "
                f"composite={composite_score:.3f}, validated={validated_sequences}"
                f"{species_info}{approving_model_info}, runtime={processing_time:.1f}s"
            )
        logger.info("")

    def _log_failed_videos(self) -> None:
        """Log details of failed videos."""
        if not self.failed_videos:
            return

        logger.info("⚪ VIDEOS WITH NO ANIMALS DETECTED:")
        rejection_reasons = dict(self.session_data.rejection_reasons)
        processing_times = dict(self.session_data.processing_times)
        for video_path in self.failed_videos:
            video_name = video_path.name
            rejection_reason = rejection_reasons.get(video_name, "unknown_reason")
            processing_time = processing_times.get(video_name, 0.0)

            logger.info(f"  ⚪ {video_name}: reason={rejection_reason}, runtime={processing_time:.1f}s")

            # Add detailed step statistics
            if video_name in self.failed_video_details:
                details = self.failed_video_details[video_name]

                # Step 3 statistics - show actual sequence data
                if details["step3_data"]:
                    sequences = details["step3_data"]["validated_sequences"]
                    sequence_count = len(sequences)

                    if sequences:
                        # Show actual sequence statistics
                        total_detections = sum(len(getattr(seq, "detections", [])) for seq in sequences)
                        avg_composite = sum(getattr(seq, "composite_score", 0) for seq in sequences) / len(sequences)
                        max_composite = max(getattr(seq, "composite_score", 0) for seq in sequences)
                        avg_ensemble = sum(getattr(seq, "ensemble_score", 0) for seq in sequences) / len(sequences)

                        logger.info(
                            f"     📊 Step 3: ✅ PASSED - {sequence_count} sequences, {total_detections} total detections"
                        )
                        logger.info(
                            f"     📈 Scores: avg_composite={avg_composite:.3f}, max_composite={max_composite:.3f}, avg_ensemble={avg_ensemble:.3f}"
                        )

                        # Show top sequence details
                        top_seq = max(sequences, key=lambda s: getattr(s, "composite_score", 0))
                        det_count = len(getattr(top_seq, "detections", []))
                        logger.info(
                            f"     🔍 Top sequence: {det_count} detections, composite={getattr(top_seq, 'composite_score', 0):.3f}"
                        )
                    else:
                        logger.info("     📊 Step 3: ❌ FAILED - 0 sequences passed validation")

                # Step 4 statistics
                if details["step4_data"]:
                    step4 = details["step4_data"]
                    input_count = step4.get("input_sequences_count", 0)
                    confirmed_count = step4.get("confirmed_animals", 0)
                    filtered_count = step4.get("filtered_sequences_count", 0)
                    all_results = step4.get("all_classification_results", [])

                    if confirmed_count > 0:
                        logger.info(
                            f"     🔬 Step 4: ✅ PASSED - {input_count} input → {confirmed_count} confirmed animals"
                        )
                        # Show successful classifications
                        if step4.get("animal_sequences"):
                            for i, seq in enumerate(step4["animal_sequences"][:3]):
                                classification = seq.classification
                                species = classification.species or "unidentified"
                                approving_model = classification.approving_model or "unknown"
                                logger.info(
                                    f"       ✅ {species} (conf={classification.animal_confidence:.3f}, approved_by={approving_model})"
                                )
                    else:
                        logger.info(
                            f"     🔬 Step 4: ❌ FAILED - {input_count} input → 0 confirmed, {filtered_count} filtered"
                        )

                        # Show actual classification scores from all attempts
                        if all_results:
                            logger.info(
                                f"       🔬 Classification scores (all below threshold {self.config.animal_confidence_threshold}):"
                            )
                            for i, classified_seq in enumerate(all_results[:3]):  # Show first 3
                                classification = classified_seq.classification
                                conf = classification.animal_confidence

                                # Add crop information to sequence logging
                                crop_info = f" | {classification.crop_info}" if classification.crop_info else ""
                                crop_path = f" | crop: {classification.crop_path}" if classification.crop_path else ""
                                logger.info(
                                    f"         Sequence {i + 1}: ensemble_conf={conf:.3f}{crop_info}{crop_path}"
                                )

                                # Show individual model scores from the inference results
                                for result in classification.individual_results:
                                    logger.info(
                                        f"           {result.model_name}: animal={result.is_animal} (conf={result.animal_confidence:.3f})"
                                    )
                                    if result.species:
                                        logger.info(
                                            f"             Species: {result.species} (conf={result.species_confidence:.3f})"
                                        )
                        else:
                            logger.info(
                                f"       Reason: All classifications below confidence threshold ({self.config.animal_confidence_threshold})"
                            )
                else:
                    logger.info("     🔬 Step 4: ⏭️ SKIPPED - Classification disabled")
        logger.info("")

    def _log_timing_summary(self, batch_total_time: float) -> dict:
        """
        Log timing summary and return timing statistics.

        Args:
            batch_total_time: Total time for batch processing

        Returns:
            Dictionary with timing statistics
        """
        # Extract just the timing values from the list of tuples
        processing_times = [time for name, time in self.session_data.processing_times]
        timing_stats = {}

        if processing_times:
            total_video_time = sum(t for t in processing_times if isinstance(t, (int, float)))
            avg_time = total_video_time / len(processing_times)

            timing_stats = {
                "total_batch_time": batch_total_time,
                "total_video_time": total_video_time,
                "average_per_video": avg_time,
                "videos_processed": len(processing_times),
            }

            logger.info("⏱️  TIMING SUMMARY:")
            logger.info(f"  📊 Total batch time: {batch_total_time:.1f}s")
            logger.info(f"  📊 Total video processing time: {total_video_time:.1f}s")
            logger.info(f"  📊 Average per video: {avg_time:.1f}s")
            logger.info(f"  📊 Videos processed: {len(processing_times)}")
            logger.info("")

        return timing_stats

    def _generate_model_contribution_analysis(self) -> None:
        """Generate comprehensive model contribution analysis."""
        logger.info("🤖 MODEL CONTRIBUTION ANALYSIS:")
        logger.info("   (YOLO vs MegaDetector performance comparison)")
        logger.info("")

        # Use the model contributions collected during processing
        model_contributions_data = self.session_data.model_contributions

        if not model_contributions_data:
            logger.info("  ❌ No model contribution data collected")
            logger.info("     (No videos reached Step 3 ML analysis)")
            logger.info("")
            return

        # Aggregate model statistics across ALL videos with ML data
        all_models_stats = {}
        videos_with_animals = [analysis.video_path.name for analysis in self.all_analyses]

        for video_name, contributions_list in model_contributions_data:
            for contrib in contributions_list:
                model_name = contrib.model_name
                if model_name not in all_models_stats:
                    all_models_stats[model_name] = {
                        "total_detections": 0,
                        "videos_contributed": 0,
                        "max_confidence": 0.0,
                        "total_tracks": 0,
                    }
                all_models_stats[model_name]["total_detections"] += contrib.total_detections
                all_models_stats[model_name]["videos_contributed"] += 1
                all_models_stats[model_name]["max_confidence"] = max(
                    all_models_stats[model_name]["max_confidence"], contrib.max_confidence
                )
                all_models_stats[model_name]["total_tracks"] += contrib.contributing_tracks

        videos_with_ml_data = [video_name for video_name, _ in model_contributions_data]
        logger.info(
            f"  📊 ANALYSIS COVERS: {len(videos_with_ml_data)} videos with ML data "
            f"({len(videos_with_animals)} successful, "
            f"{len(videos_with_ml_data) - len(videos_with_animals)} failed validation)"
        )
        logger.info("")

        logger.info("  🤖 ALL MODELS:")
        for model_name, stats in sorted(all_models_stats.items(), key=lambda x: x[1]["total_detections"], reverse=True):
            logger.info(
                f"    {model_name}: {stats['total_detections']} detections, "
                f"{stats['videos_contributed']}/{len(videos_with_ml_data)} videos, "
                f"max_conf={stats['max_confidence']:.3f}, tracks={stats['total_tracks']}"
            )
        logger.info("")

        # Per-video breakdown for detailed analysis
        logger.info("  📹 PER-VIDEO MODEL BREAKDOWN:")
        contributions_by_video = dict(model_contributions_data)
        for video_name in sorted(videos_with_ml_data):
            video_contributions = contributions_by_video.get(video_name)

            if video_contributions:
                # Indicate if video passed or failed validation
                status = "PASS" if video_name in videos_with_animals else "FAIL"

                # Find the deciding model for successful videos
                deciding_model = None
                if status == "PASS":
                    # Find the analysis for this video to get the approving model
                    for analysis in self.all_analyses:
                        if analysis.video_path.name == video_name:
                            if (
                                hasattr(analysis, "classification_result")
                                and analysis.classification_result
                                and analysis.classification_result.classification_enabled
                                and analysis.classification_result.animal_sequences
                            ):
                                deciding_model = analysis.classification_result.animal_sequences[
                                    0
                                ].classification.approving_model
                            break

                status_info = f"[{status}]"
                if deciding_model:
                    status_info = f"[{status} - decided by {deciding_model}]"

                logger.info(f"    {video_name} {status_info}:")
                # Sort by detection count for easier comparison
                sorted_contribs = sorted(video_contributions, key=lambda x: x.total_detections, reverse=True)
                for contrib in sorted_contribs:
                    # Highlight the deciding model
                    model_marker = "👑 " if contrib.model_name.lower() == (deciding_model or "").lower() else ""
                    logger.info(
                        f"      {model_marker}{contrib.model_name}: {contrib.total_detections} detections, "
                        f"max_conf={contrib.max_confidence:.3f}"
                    )
            else:
                logger.info(f"    {video_name}: No model contribution data")
        logger.info("")
