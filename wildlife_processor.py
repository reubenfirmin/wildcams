"""
Wildlife Video Processor using modular pipeline architecture.
Replaces the monolithic NextGenVideoProcessor with a clean, maintainable implementation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from video_processor_base import VideoProcessorBase
from config import ProcessingConfig
from pipeline import PipelineOrchestrator
from pipeline.steps import MotionDetectionStep, CameraHandlingFilterStep, FullFrameValidationStep

logger = logging.getLogger('wildcams')


class WildlifeVideoProcessor(VideoProcessorBase):
    """Wildlife video processor using modular pipeline architecture."""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        
        # Create tracking subdirectory for .processed files
        self.tracking_dir = self.video_dir / '.tracking'
        self.tracking_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline steps
        motion_step = MotionDetectionStep(config)
        camera_step = CameraHandlingFilterStep(config)
        validation_step = FullFrameValidationStep(config, self.ml_ensemble)
        
        # Create pipeline orchestrator
        self.pipeline = PipelineOrchestrator([
            motion_step,
            camera_step, 
            validation_step
        ])
        
        logger.info(f"🎯 Wildlife video processor initialized with modular pipeline")
        logger.info(f"🕒 Temporal consistency: min {config.min_track_duration}s, motion gap {config.motion_tracking_gap_seconds}s")
        logger.info(f"🔍 Motion method: {config.motion_method}")
        logger.info(f"✅ Pipeline steps: {len(self.pipeline.get_step_names())}")
    
    
    def process_all_videos(self, config, video_filter=None):
        """Process all videos using the modular pipeline."""
        import time
        batch_start_time = time.time()
        
        logger.info("###############################################")
        logger.info("NEXT-GENERATION BATCH PROCESSING SESSION START")
        logger.info("###############################################")
        
        # Get videos to process
        videos_to_process = self.get_filtered_videos(video_filter)
        
        if not videos_to_process:
            if video_filter:
                logger.info(f"BATCH RESULT: No videos found matching filter: {video_filter}")
                logger.info(f"⚠️ No videos found matching filter: {video_filter}")
            else:
                logger.info("BATCH RESULT: No unprocessed videos found")
                logger.info("✅ No unprocessed videos found")
            return
        
        logger.info(f"Videos to process: {[v.name for v in videos_to_process]}")
        logger.info(f"🎬 Found {len(videos_to_process)} videos to process")
        
        # Clear previous session data
        self.all_features = []
        self.video_metadata = []
        
        all_analyses = []
        failed_videos = []
        session_data = {
            'processing_times': {},
            'composite_scores': {},
            'rejection_reasons': {},
            'model_contributions': {},
            'failed_video_data': {}
        }
        
        for video_idx, video_path in enumerate(videos_to_process, 1):
            video_start_time = time.time()
            try:
                logger.info(f"\n🎬 Processing video {video_idx}/{len(videos_to_process)}: {video_path.name}")
                
                # Process video using pipeline
                result = self.process_video_with_pipeline(video_path, config)
                
                # Record processing time
                processing_time = time.time() - video_start_time
                session_data['processing_times'][video_path.name] = processing_time
                
                if result:
                    all_analyses.append(result)
                    self.mark_as_processed(video_path, result, success=True)
                    
                    # Log individual video success
                    logger.info(f"VIDEO SUCCESS: {video_path.name} - Animal detected with temporal consistency")
                    
                    # Extract model contributions from pipeline metadata
                    if 'model_contributions' in result.get('pipeline_metadata', {}):
                        session_data['model_contributions'][video_path.name] = result['pipeline_metadata']['model_contributions']
                    
                else:
                    failed_videos.append(video_path)
                    self.mark_as_processed(video_path, None, success=False)
                    
                    # Log individual video failure
                    logger.info(f"VIDEO SKIPPED: {video_path.name} - No consistent animal movement detected")
                    logger.info(f"⚪ {video_path.name}: No consistent animal movement")
                    
                    session_data['rejection_reasons'][video_path.name] = 'no_consistent_animal_movement'
                
            except KeyboardInterrupt:
                logger.info("🛑 Processing interrupted by user")
                break
            except Exception as e:
                processing_time = time.time() - video_start_time
                session_data['processing_times'][video_path.name] = processing_time
                
                logger.error(f"VIDEO ERROR: {video_path.name} - {str(e)}")
                logger.error(f"❌ {video_path.name}: Processing failed - {str(e)}")
                failed_videos.append(video_path)
                session_data['rejection_reasons'][video_path.name] = f'error: {str(e)}'
                continue
        
        # Generate comprehensive final summary (like original implementation)
        self._generate_comprehensive_summary(videos_to_process, all_analyses, failed_videos, session_data, batch_start_time)
        
        # Generate clustering
        if self.all_features:
            # Pass config to clustering function
            clusters = self.cluster_animal_videos(self.video_metadata, config)
            self.save_clustering_results(clusters)
        
        logger.info("###############################################")
        logger.info("NEXT-GENERATION BATCH PROCESSING SESSION END")
        logger.info("###############################################")
    
    def process_video_with_pipeline(self, video_path: Path, config) -> Optional[Dict]:
        """Process a single video using the modular pipeline."""
        video_start_time = datetime.now()
        
        logger.info(f"🎬 PIPELINE PROCESSING: {video_path.name}")
        
        try:
            # Run the pipeline
            result = self.pipeline.process(video_path, config)
            
            if not result.success:
                logger.error(f"❌ Pipeline failed: {result.metadata.get('error', 'Unknown error')}")
                return None
            
            if result.early_exit:
                logger.info(f"⏹️ Early exit: {result.early_exit_reason}")
                return None
            
            # Extract validated sequences from result
            validated_sequences = result.data.get('validated_sequences', [])
            
            if not validated_sequences:
                logger.info(f"❌ No validated sequences found")
                return None
            
            # Process results similar to original implementation
            logger.info(f"✅ Pipeline completed: {len(validated_sequences)} validated sequences")
            
            # Extract features from best sequence
            best_sequence = max(validated_sequences, key=lambda s: s['ensemble_score'])
            features = self.extract_features_from_best_sequence(video_path, best_sequence)
            
            if features is not None:
                self.all_features.append(features)
                
                # Create video metadata
                analysis = {
                    'video_path': str(video_path),
                    'video_name': video_path.name,
                    'detection': best_sequence['best_detection'],
                    'ensemble_score': best_sequence['ensemble_score'],
                    'composite_score': best_sequence['composite_score'],  # Extract composite score
                    'validation_result': 'animals_detected',
                    'processing_time': (datetime.now() - video_start_time).total_seconds(),
                    'validated_sequences_count': len(validated_sequences),
                    'pipeline_metadata': result.metadata
                }
                
                self.video_metadata.append(analysis)
                self.save_analysis(analysis, video_path)
                
                return analysis
        
        except Exception as e:
            logger.error(f"💥 Pipeline processing failed for {video_path.name}: {e}")
            return None
        
        return None
    
    def extract_features_from_best_sequence(self, video_path: Path, best_sequence: Dict):
        """Extract ResNet features from best detection in sequence."""
        import cv2
        import numpy as np
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"💥 Failed to open video: {video_path}")
                return None
            
            best_detection = best_sequence['best_detection']
            frame_idx = best_detection.get('frame_idx', 0)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"💥 Failed to read frame {frame_idx} from {video_path}")
                return None
            
            # Extract features using ML ensemble's ResNet feature extractor
            bbox = best_detection['bbox']
            features = self.ml_ensemble.extract_features(frame, bbox)
            
            if features is not None:
                logger.info(f"🎯 Features extracted from detection: conf={best_detection['confidence']:.3f}, dims={len(features)}")
            else:
                logger.error(f"💥 Feature extraction returned None")
            
            return features
            
        except Exception as e:
            logger.error(f"💥 Feature extraction failed: {e}")
            return None
    
    def _generate_comprehensive_summary(self, videos_to_process, all_analyses, failed_videos, session_data, batch_start_time):
        """Generate detailed processing summary matching original implementation."""
        import time
        
        total_videos = len(videos_to_process)
        animal_videos = len(all_analyses)
        no_animal_videos = total_videos - animal_videos
        batch_total_time = time.time() - batch_start_time
        
        logger.info("=" * 80)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"🎬 Total videos processed: {total_videos}")
        logger.info(f"🦌 Videos with animals detected: {animal_videos}")
        logger.info(f"⚪ Videos with no animals detected: {no_animal_videos}")
        logger.info("")
        
        # Videos with animals detected
        if all_analyses:
            logger.info("🦌 VIDEOS WITH ANIMALS DETECTED:")
            for analysis in all_analyses:
                video_name = Path(analysis['video_path']).name
                confidence = analysis['detection']['confidence']
                ensemble_score = analysis.get('ensemble_score', 0.0)
                composite_score = analysis.get('composite_score', 0.0)
                validated_sequences = analysis.get('validated_sequences_count', 1)
                processing_time = session_data['processing_times'].get(video_name, 0.0)
                
                # Extract timing info from detection
                best_detection = analysis['detection']
                timestamp = best_detection.get('timestamp', 0.0)
                
                logger.info(f"  ✅ {video_name}: time_range={timestamp:.1f}s, conf={confidence:.3f}, ensemble={ensemble_score:.3f}, composite={composite_score:.3f}, validated={validated_sequences}, runtime={processing_time:.1f}s")
            logger.info("")
        
        # Videos with no animals
        if failed_videos:
            logger.info("⚪ VIDEOS WITH NO ANIMALS DETECTED:")
            for video_path in failed_videos:
                video_name = video_path.name
                rejection_reason = session_data['rejection_reasons'].get(video_name, 'unknown_reason')
                processing_time = session_data['processing_times'].get(video_name, 0.0)
                
                logger.info(f"  ⚪ {video_name}: reason={rejection_reason}, runtime={processing_time:.1f}s")
            logger.info("")
        
        # Timing summary
        processing_times = session_data['processing_times']
        if processing_times:
            total_video_time = sum(t for t in processing_times.values() if isinstance(t, (int, float)))
            avg_time = total_video_time / len(processing_times)
            logger.info(f"⏱️  TIMING SUMMARY:")
            logger.info(f"  📊 Total batch time: {batch_total_time:.1f}s")
            logger.info(f"  📊 Total video processing time: {total_video_time:.1f}s")
            logger.info(f"  📊 Average per video: {avg_time:.1f}s")
            logger.info(f"  📊 Videos processed: {len(processing_times)}")
            logger.info("")
        
        # Model contribution analysis
        self._generate_model_contribution_analysis(session_data['model_contributions'], all_analyses)
        
        logger.info("📁 Analysis files saved to: /home/rfirmin/Videos/wildcams/analysis/")
        
        if all_analyses:
            logger.info(f"✅ Successfully processed {len(all_analyses)} videos with temporal consistency")
        
        logger.info("=" * 80)
    
    def _generate_model_contribution_analysis(self, model_contributions, all_analyses):
        """Generate model contribution analysis matching original implementation."""
        logger.info("🤖 MODEL CONTRIBUTION ANALYSIS:")
        logger.info("   (YOLO vs MegaDetector performance comparison)")
        logger.info("")
        
        if not model_contributions:
            logger.info("  ❌ No model contribution data collected")
            logger.info("     (No videos reached Step 3 ML analysis)")
            logger.info("")
            return
        
        # Aggregate model statistics across ALL videos with ML data
        all_models_stats = {}
        videos_with_ml_data = list(model_contributions.keys())
        videos_with_animals = [Path(analysis['video_path']).name for analysis in all_analyses]
        
        for video_name in videos_with_ml_data:
            video_stats = model_contributions.get(video_name, {})
            for model_name, stats in video_stats.items():
                if model_name not in all_models_stats:
                    all_models_stats[model_name] = {
                        'total_detections': 0,
                        'videos_contributed': 0,
                        'max_confidence': 0.0,
                        'total_tracks': 0
                    }
                all_models_stats[model_name]['total_detections'] += stats.get('total_detections', 0)
                all_models_stats[model_name]['videos_contributed'] += 1
                all_models_stats[model_name]['max_confidence'] = max(
                    all_models_stats[model_name]['max_confidence'], 
                    stats.get('max_confidence', 0.0)
                )
                all_models_stats[model_name]['total_tracks'] += stats.get('contributing_tracks', 0)
        
        logger.info(f"  📊 ANALYSIS COVERS: {len(videos_with_ml_data)} videos with ML data ({len(videos_with_animals)} successful, {len(videos_with_ml_data) - len(videos_with_animals)} failed validation)")
        logger.info("")
        
        logger.info("  🤖 ALL MODELS:")
        for model_name, stats in sorted(all_models_stats.items(), key=lambda x: x[1]['total_detections'], reverse=True):
            logger.info(f"    {model_name}: {stats['total_detections']} detections, {stats['videos_contributed']}/{len(videos_with_ml_data)} videos, max_conf={stats['max_confidence']:.3f}, tracks={stats['total_tracks']}")
        logger.info("")
        
        # Per-video breakdown for detailed analysis
        logger.info("  📹 PER-VIDEO MODEL BREAKDOWN:")
        for video_name in sorted(videos_with_ml_data):
            video_stats = model_contributions.get(video_name, {})
            if video_stats:
                # Indicate if video passed or failed validation
                status = "PASS" if video_name in videos_with_animals else "FAIL"
                logger.info(f"    {video_name} [{status}]:")
                # Sort by detection count for easier comparison
                sorted_models = sorted(video_stats.items(), key=lambda x: x[1].get('total_detections', 0), reverse=True)
                for model_name, stats in sorted_models:
                    logger.info(f"      {model_name}: {stats.get('total_detections', 0)} detections, max_conf={stats.get('max_confidence', 0.0):.3f}")
            else:
                logger.info(f"    {video_name}: No model contribution data")
        logger.info("")
    
    def mark_as_processed(self, video_path: Path, analysis: Optional[Dict] = None, success: bool = True):
        """Mark video as processed with detailed information."""
        processed_file = self.tracking_dir / f"{video_path.stem}.processed"
        
        # Create processed data
        processed_data = {
            'video_file': video_path.name,
            'processed_timestamp': datetime.now().isoformat(),
            'processing_status': 'success' if success else 'no_animals',
            'processor_version': 'modular_pipeline_phase3',
        }
        
        if analysis and success:
            # Add detailed information about strongest detections
            processed_data.update({
                'animals_detected': True,
                'confidence': analysis['detection']['confidence'],
                'ensemble_score': analysis.get('ensemble_score', 0.0),
                'validated_sequences': analysis.get('validated_sequences_count', 1),
                'processing_time': analysis.get('processing_time', 0.0),
                'best_detection': {
                    'timestamp': analysis['detection'].get('timestamp', 0.0),
                    'bbox': analysis['detection']['bbox'],
                    'confidence': analysis['detection']['confidence'],
                    'source': analysis['detection'].get('source', 'unknown')
                }
            })
        else:
            processed_data.update({
                'animals_detected': False,
                'reason': 'no_consistent_animal_movement'
            })
        
        # Write processed file with JSON data
        import json
        with open(processed_file, 'w') as f:
            # Convert numpy types to JSON-serializable types
            processed_data_clean = self._convert_for_json(processed_data)
            json.dump(processed_data_clean, f, indent=2)
        
        logger.info(f"📝 Marked {video_path.name} as processed: {processed_file}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj