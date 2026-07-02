"""
Clean wildlife video processor using composition-based architecture.

This replaces the inheritance-heavy VideoProcessorBase with a focused
class that orchestrates video processing through component composition.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from config import ProcessingConfig
from pipeline.step_interface_v2 import PipelineOrchestrator
from pipeline.steps_v2 import (
    MotionDetectionStepImpl,
    CameraHandlingStepImpl,
    FullFrameValidationStepImpl
)
from pipeline.steps_v2.animal_classification_step import AnimalClassificationStepImpl
from ml.ensemble_wrapper import MLDetectionEnsemble
from video_io import VideoReader, AnalysisWriter, ProcessedVideoTracker
from core.constants import (
    DEFAULT_VIDEO_DIR, CACHE_SUBDIR, TRACKING_SUBDIR, 
    VIDEO_EXTENSIONS, MAIN_LOGGER_NAME
)
from core.data_types import (
    VideoAnalysis, ValidationResult, Detection, BoundingBox,
    ProcessedVideoRecord, create_processing_record
)
# ELIMINATED: create_video_analysis_from_pipeline_result violates Phase 5.0
from core.functional_utils import (
    collect_video_files, validate_video_filter
)

logger = logging.getLogger('wildcams')


class WildlifeVideoProcessor:
    """
    Main wildlife video processor using clean composition architecture.
    
    This class orchestrates video processing through component composition
    rather than inheritance, making it easier to test and maintain.
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the wildlife video processor.
        
        Args:
            config: ProcessingConfig with all processing parameters
        """
        self.config = config
        self.video_dir = Path(config.video_dir)
        
        # Initialize components through composition
        self.analysis_writer = AnalysisWriter(self.video_dir / 'analysis')
        self.processed_tracker = ProcessedVideoTracker(self.video_dir / '.tracking')
        
        # Initialize ML ensemble
        self.ml_ensemble = MLDetectionEnsemble(
            confidence_threshold=config.confidence_threshold,
            ensemble_models=config.ensemble_models,
            cache_dir=self.video_dir / '.cache'
        )
        
        # Initialize pipeline steps with dependency injection
        motion_step = MotionDetectionStepImpl(config)
        camera_step = CameraHandlingStepImpl(config)
        validation_step = FullFrameValidationStepImpl(config, self.ml_ensemble)
        
        # Initialize Step 4: Animal Classification (optional)
        classification_step = None
        if config.enable_animal_classification:
            classification_step = AnimalClassificationStepImpl(config)
            logger.info(f"🔬 Step 4: Animal Classification enabled")
        else:
            logger.info(f"⏭️ Step 4: Animal Classification disabled")
        
        # Create pipeline orchestrator
        self.pipeline = PipelineOrchestrator(
            motion_step=motion_step,
            camera_step=camera_step,
            validation_step=validation_step,
            classification_step=classification_step
        )
        
        logger.info(f"🎯 Wildlife video processor initialized with modular pipeline")
        logger.info(f"📁 Video directory: {self.video_dir}")
        logger.info(f"🤖 ML ensemble: {len(config.ensemble_models)} models")
    
    def process_video(self, video_path: Path, force_reprocess: bool = False) -> Optional[VideoAnalysis]:
        """
        Process a single video through the pipeline.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoAnalysis object if successful, None if failed/rejected
        """
        video_start_time = datetime.now()
        
        logger.info(f"🎬 Processing: {video_path.name}")
        
        # Check if already processed (unless force reprocessing)
        if not force_reprocess and self.processed_tracker.is_processed(video_path):
            logger.info(f"⏭️ Skipping already processed: {video_path.name}")
            return None
        
        # Run the TYPED pipeline
        analysis = self.pipeline.process(video_path, self.config)
        
        # analysis is now a VideoAnalysis object, not a dictionary result
        
        # Always save analysis and mark as processed regardless of outcome
        # Convert to simple dict for JSON serialization
        analysis_dict = {
            'video_path': str(analysis.video_path),
            'video_name': analysis.video_name,
            'processing_time': analysis.processing_time,
            'animals_detected': analysis.validation_result.animals_detected,
            'confidence': analysis.validation_result.confidence,
            'ensemble_score': analysis.validation_result.ensemble_score,
            'composite_score': analysis.validation_result.composite_score,
            'validated_sequences_count': analysis.validation_result.validated_sequences_count,
            'reason': analysis.validation_result.reason or 'no_animals'
        }
        self.analysis_writer.save_analysis(video_path, analysis_dict)
        self.processed_tracker.mark_as_processed(video_path)
        
        # Extract step data for ALL videos (success and failure)
        # Get validated sequences from fullframe_result
        validated_sequences = []
        if hasattr(analysis, 'fullframe_result') and analysis.fullframe_result and hasattr(analysis.fullframe_result, 'data'):
            validated_sequences = getattr(analysis.fullframe_result.data, 'validated_sequences', [])
        
        step3_data = {
            'validated_sequences': validated_sequences,
            'confidence': analysis.validation_result.confidence,
            'ensemble_score': analysis.validation_result.ensemble_score,
            'composite_score': analysis.validation_result.composite_score,
            'sequence_count': len(validated_sequences)
        }
        
        step4_data = None
        # Always capture Step 4 data when classification is enabled, even if no animals confirmed
        if self.config.enable_animal_classification:
            if hasattr(analysis, 'classification_result') and analysis.classification_result:
                # Extract ALL classification results including filtered ones
                all_results = []
                if hasattr(analysis.classification_result, 'all_classified_sequences'):
                    all_results = analysis.classification_result.all_classified_sequences
                
                step4_data = {
                    'input_sequences_count': analysis.classification_result.input_sequences_count,
                    'animal_sequences': analysis.classification_result.animal_sequences,
                    'confirmed_animals': len(analysis.classification_result.animal_sequences),
                    'filtered_sequences_count': analysis.classification_result.filtered_sequences_count,
                    'classification_enabled': analysis.classification_result.classification_enabled,
                    'all_classification_results': all_results  # Include ALL attempts with actual scores
                }
            else:
                # Step 4 was enabled but no classification result - this shouldn't happen
                step4_data = {
                    'input_sequences_count': 0,
                    'animal_sequences': [],
                    'confirmed_animals': 0,
                    'filtered_sequences_count': 0,
                    'classification_enabled': True,
                    'all_classification_results': []
                }
        
        # Store detailed step data in analysis for ALL videos
        analysis._step3_data = step3_data
        analysis._step4_data = step4_data
        
        # Determine success/failure ONLY at the end
        if not analysis.validation_result.animals_detected:
            logger.info(f"⏹️ No animals detected: {analysis.validation_result.reason or 'no_animals'}")
            analysis._is_failure = True
            return analysis
        else:
            logger.info(f"✅ Pipeline completed: {analysis.validation_result.validated_sequences_count} validated sequences")
            analysis._is_failure = False
            return analysis
    
    def get_video_files(self, video_filter: Optional[List[Union[int, str]]] = None) -> List[Path]:
        """
        Get list of video files to process.
        
        Args:
            video_filter: Optional list of video indices or names to filter
            
        Returns:
            List of Path objects for video files
        """
        # Use pure function to collect video files
        all_videos = collect_video_files(self.video_dir, VIDEO_EXTENSIONS)
        
        # Apply filter if provided using pure function
        if video_filter:
            filtered_videos, warnings = validate_video_filter(video_filter, all_videos)
            
            # Log any warnings
            for warning in warnings:
                logger.warning(f"⚠️ {warning}")
            
            return filtered_videos
        
        return all_videos
    
    def get_unprocessed_videos(self, video_filter: Optional[List[Union[int, str]]] = None) -> List[Path]:
        """
        Get list of unprocessed video files.
        
        Args:
            video_filter: Optional list of video indices or names to filter
            
        Returns:
            List of Path objects for unprocessed video files
        """
        all_videos = self.get_video_files(video_filter)
        unprocessed_videos = []
        
        for video_path in all_videos:
            if not self.processed_tracker.is_processed(video_path):
                unprocessed_videos.append(video_path)
        return unprocessed_videos
    
