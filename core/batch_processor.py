"""
Batch video processor for wildlife video processing.

Orchestrates batch processing of multiple videos with clustering,
feature extraction, and comprehensive session management.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from core.data_types import VideoAnalysis, BatchProcessingResult
import time

from config import ProcessingConfig
from .wildlife_processor import WildlifeVideoProcessor
from .session_manager import ProcessingSessionManager
# BatchProcessingResult now imported from data.py above

logger = logging.getLogger('wildcams')


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
        
        # Storage for clustering and analysis
        self.all_features = []
        self.video_metadata = []
        
        logger.info(f"🎯 Batch processor initialized")
        
    def process_all_videos(self, video_filter: Optional[List] = None, force_reprocess: bool = False) -> BatchProcessingResult:
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
            return BatchProcessingResult(success=False, reason='no_videos_to_process')
        
        # Start processing session
        self.session_manager.start_session(videos_to_process)
        
        # Clear previous session data for clustering
        self.all_features = []
        self.video_metadata = []
        
        # Process each video
        for video_idx, video_path in enumerate(videos_to_process, 1):
            video_start_time = self.session_manager.record_video_start(
                video_path, video_idx, len(videos_to_process)
            )
            
            # Process single video - let errors bubble up
            force_reprocess_flag = force_reprocess or video_filter is not None
            analysis = self.video_processor.process_video(video_path, force_reprocess_flag)
            processing_time = time.time() - video_start_time
            
            if analysis and not getattr(analysis, '_is_failure', False):
                # Record success
                self.session_manager.record_video_success(video_path, analysis, processing_time)
                
                # Extract features for clustering
                self._extract_features_for_clustering(video_path, analysis)
                
            else:
                # Record failure with step data if available
                step3_data = getattr(analysis, '_step3_data', None) if analysis else None
                step4_data = getattr(analysis, '_step4_data', None) if analysis else None
                reason = analysis.validation_result.reason if analysis else 'processing_error'
                self.session_manager.record_video_failure(video_path, processing_time, reason=reason,
                                                        step3_data=step3_data, step4_data=step4_data)
        
        # Generate final summary
        summary = self.session_manager.generate_final_summary()
        
        # Generate clustering if we have features AND clustering is enabled
        clustering_results = None
        if self.all_features and self.config.enable_clustering:
            clustering_results = self._perform_clustering()
        
        logger.info("###############################################")
        logger.info("WILDLIFE VIDEO PROCESSING SESSION END")
        logger.info("###############################################")
        
        return BatchProcessingResult(
            success=True,
            summary=summary,
            features_extracted=len(self.all_features),
            videos_processed=len(videos_to_process)
        )
    
    def process_single_video(self, video_path: Path, force_reprocess: bool = False) -> Optional[VideoAnalysis]:
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
    
    def _extract_features_for_clustering(self, video_path: Path, analysis: VideoAnalysis) -> None:
        """
        Extract features from analysis for clustering.
        
        Args:
            video_path: Path to video file
            analysis: Analysis results with detection information
        """
        # Extract features from best detection using efficient video reader
        from video_io import VideoReader
        
        best_detection = analysis.validation_result.best_detection
        
        # Open video reader once for efficiency
        video_reader = VideoReader(video_path)
        video_reader.open()
        
        try:
            features = self.video_processor.extract_features_from_detection(
                video_path, best_detection, video_reader
            )
            
            if features is not None:
                self.all_features.append(features)
                self.video_metadata.append(analysis)
                logger.info(f"🎯 Features extracted for clustering: {video_path.name}")
            else:
                logger.warning(f"⚠️ Failed to extract features for clustering: {video_path.name}")
        finally:
            video_reader.close()
    
    def _perform_clustering(self) -> Optional[Dict]:
        """
        Perform clustering analysis on extracted features.
        
        Returns:
            Clustering results if successful, None if failed
        """
        logger.info(f"🔍 Performing clustering analysis on {len(self.all_features)} feature vectors")
        
        # Import clustering dependencies
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        if len(self.all_features) < 2:
            logger.info("CLUSTERING: Not enough features for clustering")
            return {}
        
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(self.all_features)
            
            # Apply PCA for dimensionality reduction if needed
            if features_scaled.shape[1] > 50:
                pca = PCA(n_components=50)
                features_scaled = pca.fit_transform(features_scaled)
                logger.info(f"CLUSTERING: Applied PCA, reduced to {features_scaled.shape[1]} dimensions")
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=self.config.clustering_eps, min_samples=self.config.min_samples)
            cluster_labels = clustering.fit_predict(features_scaled)
            
            # Process clustering results
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label == -1:  # Noise point
                    continue
                    
                if label not in clusters:
                    clusters[label] = []
                
                video_info = self.video_metadata[idx]
                clusters[label].append({
                    'video_name': video_info['video_name'],
                    'confidence': video_info['detection']['confidence'],
                    'ensemble_score': video_info.get('ensemble_score', 0.0)
                })
            
            # Calculate similarity scores for clusters
            for cluster_id, videos in clusters.items():
                if len(videos) > 1:
                    # Calculate average similarity within cluster
                    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                    cluster_features = features_scaled[cluster_indices]
                    similarity_matrix = cosine_similarity(cluster_features)
                    avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                    
                    for video in videos:
                        video['similarity_score'] = avg_similarity
                else:
                    videos[0]['similarity_score'] = 1.0
            
            # Save clustering results
            self._save_clustering_results(clusters)
            
            logger.info(f"✅ Clustering completed: {len(clusters)} clusters identified")
            
            return {
                'num_clusters': len(clusters),
                'clusters': clusters,
                'total_videos': len(self.video_metadata)
            }
            
        except Exception as e:
            logger.error(f"❌ Clustering failed: {e}")
            return None
    
    def _save_clustering_results(self, clusters: Dict) -> None:
        """Save clustering results and feature data."""
        from video_io import AnalysisWriter
        
        # Convert clusters dict to list format expected by AnalysisWriter
        clusters_list = [clusters] if clusters else []
        
        # Create analysis writer
        analysis_writer = AnalysisWriter(self.config)
        analysis_writer.save_clustering_results(clusters_list, self.all_features, self.video_metadata)
    
    def get_processing_status(self) -> Dict:
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
            'session': session_summary,
            'total_video_files': len(video_files),
            'unprocessed_videos': len(unprocessed_files),
            'features_extracted': len(self.all_features),
            'processing_complete': len(unprocessed_files) == 0
        }
    
    def force_reprocess_all(self, video_filter: Optional[List] = None) -> BatchProcessingResult:
        """
        Force reprocessing of all videos, ignoring processed status.
        
        Args:
            video_filter: Optional list of video indices or names to filter
            
        Returns:
            Dictionary with batch processing results
        """
        logger.info("🔄 Forcing reprocessing of all videos (ignoring .processed files)")
        return self.process_all_videos(video_filter=video_filter, force_reprocess=True)
    
    def clear_processed_status(self, video_filter: Optional[List] = None) -> int:
        """
        Clear processed status for videos to allow reprocessing.
        
        Args:
            video_filter: Optional list of video indices or names to filter
            
        Returns:
            Number of videos cleared
        """
        videos = self.video_processor.get_video_files(video_filter)
        cleared_count = 0
        
        for video_path in videos:
            if self.video_processor.processed_tracker.clear_processed_status(video_path):
                cleared_count += 1
                logger.info(f"🧹 Cleared processed status: {video_path.name}")
        
        logger.info(f"✅ Cleared processed status for {cleared_count} videos")
        return cleared_count