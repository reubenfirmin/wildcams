"""
Processing session manager for wildlife video processing.

Manages processing sessions with comprehensive tracking, logging,
and analysis of batch video processing operations.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from config import ProcessingConfig
from core.constants import MAIN_LOGGER_NAME, LOG_PREFIXES
from core.data_types import (
    VideoAnalysis, TimingStatistics, SessionSummary, 
    ProcessingSessionData, ModelContribution
)

logger = logging.getLogger('wildcams')


class ProcessingSessionManager:
    """
    Manages video processing sessions with comprehensive tracking.
    
    Handles session initialization, progress tracking, timing analysis,
    and comprehensive summary generation for batch processing operations.
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the session manager.
        
        Args:
            config: ProcessingConfig with processing parameters
        """
        self.config = config
        self.session_start_time = None
        self.session_data = ProcessingSessionData()
        self.all_analyses = []
        self.failed_videos = []
        
        # Store detailed step data for failures
        self.failed_video_details = {}  # video_name -> {'step3_data': ..., 'step4_data': ...}
        
    def start_session(self, videos_to_process: List[Path]) -> None:
        """
        Start a new processing session.
        
        Args:
            videos_to_process: List of video files to process
        """
        self.session_start_time = time.time()
        self.session_data = ProcessingSessionData(
            videos_to_process=[v.name for v in videos_to_process],
            total_videos=len(videos_to_process)
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
        
    def record_video_failure(self, video_path: Path, processing_time: float, reason: str = 'no_consistent_animal_movement', 
                           step3_data: Optional[Dict] = None, step4_data: Optional[Dict] = None) -> None:
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
            'step3_data': step3_data,
            'step4_data': step4_data,
            'reason': reason
        }
        
        logger.info(f"VIDEO SKIPPED: {video_path.name} - {reason}")
        logger.info(f"⚪ {video_path.name}: {reason}")
        
        # Log detailed Step 3 scores for failure videos
        if step3_data and 'validated_sequences' in step3_data:
            sequences = step3_data['validated_sequences']
            if sequences:
                logger.info(f"📊 Step 3 Details: {len(sequences)} validated sequences")
                for i, seq in enumerate(sequences):
                    if hasattr(seq, 'composite_score'):
                        logger.info(f"  Sequence {i+1}: composite_score={seq.composite_score:.3f}, detections={len(seq.detections)}")
                    if hasattr(seq, 'ensemble_score'):
                        logger.info(f"    ensemble_score={seq.ensemble_score:.3f}")
                    
                    # Log individual detection scores
                    for j, det in enumerate(seq.detections[:3]):  # Show first 3 detections
                        logger.info(f"    Detection {j+1}: conf={det.confidence:.3f}, frame={det.frame_idx}")
            else:
                logger.info(f"📊 Step 3 Details: No validated sequences found")
        
        # Log detailed Step 4 scores for failure videos  
        if step4_data:
            if 'filtered_sequences' in step4_data:
                logger.info(f"📊 Step 4 Details: {step4_data['filtered_sequences']} sequences filtered")
            
            if 'classification_results' in step4_data:
                results = step4_data['classification_results']
                for i, result in enumerate(results):
                    crop_info = f" | {result.crop_info}" if result.crop_info else ""
                    crop_path = f" | crop: {result.crop_path}" if result.crop_path else ""
                    logger.info(f"  Classification {i+1}: animal={result.is_animal}, conf={result.animal_confidence:.3f}{crop_info}{crop_path}")
                    for individual_result in result.individual_results:
                        logger.info(f"    {individual_result.model_name}: animal={individual_result.is_animal} (conf={individual_result.animal_confidence:.3f})")
                        
            if 'animal_sequences' in step4_data:
                confirmed = len(step4_data['animal_sequences'])
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
        
    def generate_final_summary(self) -> SessionSummary:
        """
        Generate comprehensive final processing summary.
        
        Returns:
            Summary statistics dictionary
        """
        if self.session_start_time is None:
            raise ValueError("Session not started - call start_session() first")
            
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
            'total_videos': total_videos,
            'successful_videos': animal_videos,
            'failed_videos': no_animal_videos,
            'batch_time': batch_total_time,
            'timing_stats': timing_stats,
            'analyses': self.all_analyses,
            'failed_video_paths': self.failed_videos
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
            if hasattr(analysis, 'species_summary') and analysis.species_summary:
                species_info = f", species=[{analysis.species_summary}]"
            
            # Add model approval information if available
            approving_model_info = ""
            if (hasattr(analysis, 'classification_result') and 
                analysis.classification_result and 
                analysis.classification_result.classification_enabled):
                # Find the model that approved this video
                animal_sequences = analysis.classification_result.animal_sequences
                if animal_sequences:
                    # Get the approving model from the first confirmed sequence
                    approving_model = animal_sequences[0].classification.approving_model
                    if approving_model:
                        approving_model_info = f", approved_by={approving_model}"
            
            logger.info(f"  ✅ {video_name}: time_range={timestamp:.1f}s, "
                       f"conf={confidence:.3f}, ensemble={ensemble_score:.3f}, "
                       f"composite={composite_score:.3f}, validated={validated_sequences}"
                       f"{species_info}{approving_model_info}, runtime={processing_time:.1f}s")
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
            rejection_reason = rejection_reasons.get(video_name, 'unknown_reason')
            processing_time = processing_times.get(video_name, 0.0)
            
            logger.info(f"  ⚪ {video_name}: reason={rejection_reason}, runtime={processing_time:.1f}s")
            
            # Add detailed step statistics
            if video_name in self.failed_video_details:
                details = self.failed_video_details[video_name]
                
                # Step 3 statistics - show actual sequence data
                if details['step3_data']:
                    sequences = details['step3_data']['validated_sequences']
                    sequence_count = len(sequences)
                    
                    if sequences:
                        # Show actual sequence statistics
                        total_detections = sum(len(getattr(seq, 'detections', [])) for seq in sequences)
                        avg_composite = sum(getattr(seq, 'composite_score', 0) for seq in sequences) / len(sequences)
                        max_composite = max(getattr(seq, 'composite_score', 0) for seq in sequences)
                        avg_ensemble = sum(getattr(seq, 'ensemble_score', 0) for seq in sequences) / len(sequences)
                        
                        logger.info(f"     📊 Step 3: ✅ PASSED - {sequence_count} sequences, {total_detections} total detections")
                        logger.info(f"     📈 Scores: avg_composite={avg_composite:.3f}, max_composite={max_composite:.3f}, avg_ensemble={avg_ensemble:.3f}")
                        
                        # Show top sequence details
                        top_seq = max(sequences, key=lambda s: getattr(s, 'composite_score', 0))
                        det_count = len(getattr(top_seq, 'detections', []))
                        logger.info(f"     🔍 Top sequence: {det_count} detections, composite={getattr(top_seq, 'composite_score', 0):.3f}")
                    else:
                        logger.info(f"     📊 Step 3: ❌ FAILED - 0 sequences passed validation")
                
                # Step 4 statistics
                if details['step4_data']:
                    step4 = details['step4_data']
                    input_count = step4.get('input_sequences_count', 0)
                    confirmed_count = step4.get('confirmed_animals', 0)
                    filtered_count = step4.get('filtered_sequences_count', 0)
                    all_results = step4.get('all_classification_results', [])
                    
                    if confirmed_count > 0:
                        logger.info(f"     🔬 Step 4: ✅ PASSED - {input_count} input → {confirmed_count} confirmed animals")
                        # Show successful classifications
                        if step4.get('animal_sequences'):
                            for i, seq in enumerate(step4['animal_sequences'][:3]):
                                classification = seq.classification
                                species = classification.species or "unidentified"
                                approving_model = classification.approving_model or "unknown"
                                logger.info(f"       ✅ {species} (conf={classification.animal_confidence:.3f}, approved_by={approving_model})")
                    else:
                        logger.info(f"     🔬 Step 4: ❌ FAILED - {input_count} input → 0 confirmed, {filtered_count} filtered")
                        
                        # Show actual classification scores from all attempts
                        if all_results:
                            logger.info(f"       🔬 Classification scores (all below threshold {self.config.animal_confidence_threshold}):")
                            for i, classified_seq in enumerate(all_results[:3]):  # Show first 3
                                classification = classified_seq.classification
                                conf = classification.animal_confidence
                                
                                # Add crop information to sequence logging
                                crop_info = f" | {classification.crop_info}" if classification.crop_info else ""
                                crop_path = f" | crop: {classification.crop_path}" if classification.crop_path else ""
                                logger.info(f"         Sequence {i+1}: ensemble_conf={conf:.3f}{crop_info}{crop_path}")
                                
                                # Show individual model scores from the inference results
                                for result in classification.individual_results:
                                    logger.info(f"           {result.model_name}: animal={result.is_animal} (conf={result.animal_confidence:.3f})")
                                    if result.species:
                                        logger.info(f"             Species: {result.species} (conf={result.species_confidence:.3f})")
                        else:
                            logger.info(f"       Reason: All classifications below confidence threshold ({self.config.animal_confidence_threshold})")
                else:
                    logger.info(f"     🔬 Step 4: ⏭️ SKIPPED - Classification disabled")
        logger.info("")
        
    def _log_timing_summary(self, batch_total_time: float) -> Dict:
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
                'total_batch_time': batch_total_time,
                'total_video_time': total_video_time,
                'average_per_video': avg_time,
                'videos_processed': len(processing_times)
            }
            
            logger.info(f"⏱️  TIMING SUMMARY:")
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
                        'total_detections': 0,
                        'videos_contributed': 0,
                        'max_confidence': 0.0,
                        'total_tracks': 0
                    }
                all_models_stats[model_name]['total_detections'] += contrib.total_detections
                all_models_stats[model_name]['videos_contributed'] += 1
                all_models_stats[model_name]['max_confidence'] = max(
                    all_models_stats[model_name]['max_confidence'], 
                    contrib.max_confidence
                )
                all_models_stats[model_name]['total_tracks'] += contrib.contributing_tracks
        
        videos_with_ml_data = [video_name for video_name, _ in model_contributions_data]
        logger.info(f"  📊 ANALYSIS COVERS: {len(videos_with_ml_data)} videos with ML data "
                   f"({len(videos_with_animals)} successful, "
                   f"{len(videos_with_ml_data) - len(videos_with_animals)} failed validation)")
        logger.info("")
        
        logger.info("  🤖 ALL MODELS:")
        for model_name, stats in sorted(all_models_stats.items(), 
                                      key=lambda x: x[1]['total_detections'], reverse=True):
            logger.info(f"    {model_name}: {stats['total_detections']} detections, "
                       f"{stats['videos_contributed']}/{len(videos_with_ml_data)} videos, "
                       f"max_conf={stats['max_confidence']:.3f}, tracks={stats['total_tracks']}")
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
                            if (hasattr(analysis, 'classification_result') and 
                                analysis.classification_result and 
                                analysis.classification_result.classification_enabled and
                                analysis.classification_result.animal_sequences):
                                deciding_model = analysis.classification_result.animal_sequences[0].classification.approving_model
                            break
                
                status_info = f"[{status}]"
                if deciding_model:
                    status_info = f"[{status} - decided by {deciding_model}]"
                
                logger.info(f"    {video_name} {status_info}:")
                # Sort by detection count for easier comparison
                sorted_contribs = sorted(video_contributions, 
                                       key=lambda x: x.total_detections, reverse=True)
                for contrib in sorted_contribs:
                    # Highlight the deciding model
                    model_marker = "👑 " if contrib.model_name.lower() == (deciding_model or "").lower() else ""
                    logger.info(f"      {model_marker}{contrib.model_name}: {contrib.total_detections} detections, "
                               f"max_conf={contrib.max_confidence:.3f}")
            else:
                logger.info(f"    {video_name}: No model contribution data")
        logger.info("")
        
    def get_session_summary(self) -> Dict[str, Union[float, int, Optional[str]]]:
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
            'session_start_time': self.session_start_time,
            'elapsed_time': elapsed_time,
            'total_videos': self.session_data.total_videos,
            'processed_videos': len(self.session_data.processing_times),
            'successful_videos': len(self.all_analyses),
            'failed_videos': len(self.failed_videos),
            'average_processing_time': (sum(time for name, time in self.session_data.processing_times) / 
                                     max(1, len(self.session_data.processing_times)) if self.session_data.processing_times else 0.0)
        }
    
    def _create_step4_model_contributions(self, classification_result) -> List:
        """
        Create model contributions for Step 4 classification models.
        
        Args:
            classification_result: AnimalClassificationResult from Step 4
            
        Returns:
            List of ModelContribution objects for BioCLIP and DeepFaune
        """
        from core.data_types import ModelContribution
        contributions = []
        
        # Count total sequences and confirmed animals
        total_sequences = classification_result.input_sequences_count
        confirmed_animals = len(classification_result.animal_sequences)
        
        
        # Model contributions from individual results
        all_individual_results = [
            result for cs in classification_result.animal_sequences 
            for result in cs.classification.individual_results
        ]
        
        # Group by model name and create contributions
        model_groups = {}
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
                    contributing_tracks=confirmed_animals
                )
                contributions.append(contrib)
        
        return contributions