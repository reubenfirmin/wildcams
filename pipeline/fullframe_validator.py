"""Full-frame validation for wildlife video processing."""

import cv2
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from data import CompositeScore
from data import MotionTrack, ValidationSequence, Detection, BoundingBox, Track

logger = logging.getLogger('wildcams')


class FullFrameValidator:
    """Validates motion tracks using full-frame ML ensemble analysis with spatial overlap."""
    
    def __init__(self, ml_ensemble):
        """
        Initialize full-frame validator.
        
        Args:
            ml_ensemble: ML ensemble for running detections
        """
        self.ml_ensemble = ml_ensemble
        
        # Statistics tracking
        self._failed_tracks_data = []
        self._model_contributions = {}
        
        logger.info(f"🎯 Full-Frame Validator initialized")
    
    def validate_motion_tracks(self, video_path: Path, motion_tracks: List[MotionTrack], config) -> List[ValidationSequence]:
        """
        Run full-frame analysis on motion tracks with spatial overlap validation.
        
        Args:
            video_path: Path to video file
            motion_tracks: List of motion tracks from previous steps
            
        Returns:
            List of validated sequences with spatial overlap confirmation
        """
        if not motion_tracks:
            return []
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"[STEP3] Running full-frame analysis on {len(motion_tracks)} motion tracks")
        
        # Initialize model contributions tracking for this video
        self._model_contributions = {}
        for model_name in config.ensemble_models:
            self._model_contributions[model_name] = {
                'total_detections': 0,
                'max_confidence': 0.0,
                'contributing_tracks': 0,
                'spatial_valid_count': 0,
                'total_score': 0.0
            }
        
        # Convert motion tracks to extended tracks for validation
        extended_tracks = self._build_extended_tracks(motion_tracks, fps, total_frames)
        
        # Sample frames for analysis
        all_sample_frames, track_sample_frames = self._sample_track_frames(extended_tracks, config)
        
        logger.info(f"[STEP3] Processing {len(all_sample_frames)} unique frames across {len(extended_tracks)} tracks")
        
        # Initialize tracking
        track_detections = {track['track_id']: [] for track in extended_tracks}
        frame_results = []
        
        # Process each sampled frame
        for frame_idx in sorted(all_sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            timestamp = frame_idx / fps
            frame_result = self._process_frame(frame, frame_idx, timestamp, extended_tracks, track_detections, config, video_path)
            frame_results.append(frame_result)
        
        cap.release()
        
        # Evaluate tracks
        validated_results = self._evaluate_tracks(extended_tracks, track_detections, frame_results, track_sample_frames, fps, config, video_path)
        
        return validated_results
    
    def _build_extended_tracks(self, motion_tracks: List[MotionTrack], fps: float, total_frames: int) -> List[Dict]:
        """Convert basic motion tracks to extended bbox tracks for spatial validation."""
        extended_tracks = []
        
        for motion_track in motion_tracks:
            track_id = motion_track.track_id
            motion_frames = [region.frame_idx for region in motion_track.regions]
            motion_regions = motion_track.regions
            
            if not motion_frames:
                continue
            
            # Get representative bbox from motion regions
            if motion_regions and len(motion_regions) > 0:
                first_region = motion_regions[0]
                last_region = motion_regions[-1]
                first_known_position = [first_region.bbox.x1, first_region.bbox.y1, first_region.bbox.x2, first_region.bbox.y2]
                last_known_position = [last_region.bbox.x1, last_region.bbox.y1, last_region.bbox.x2, last_region.bbox.y2]
            else:
                first_known_position = [100, 100, 200, 200]
                last_known_position = first_known_position[:]
            
            # Build full video coverage detections
            full_detections = []
            motion_start_frame = min(motion_frames)
            motion_end_frame = max(motion_frames)
            
            # Backfill: frame 0 to motion_start-1
            for frame_idx in range(0, motion_start_frame):
                full_detections.append({
                    'frame': frame_idx,
                    'bbox': first_known_position.copy(),
                    'timestamp': frame_idx / fps,
                    'motion_detected': False,
                    'fill_type': 'backfill'
                })
            
            # Motion frames: use actual motion regions
            for i, frame_idx in enumerate(motion_frames):
                # Find the motion region that corresponds to this frame
                matching_region = None
                for region in motion_regions:
                    if region.frame_idx == frame_idx:
                        matching_region = region
                        break
                
                if matching_region:
                    bbox = [matching_region.bbox.x1, matching_region.bbox.y1, 
                           matching_region.bbox.x2, matching_region.bbox.y2]
                else:
                    bbox = first_known_position.copy()
                    
                full_detections.append({
                    'frame': frame_idx,
                    'bbox': bbox,
                    'timestamp': frame_idx / fps,
                    'motion_detected': True,
                    'fill_type': 'motion'
                })
            
            # Forward-fill: motion_end+1 to video end
            for frame_idx in range(motion_end_frame + 1, total_frames):
                full_detections.append({
                    'frame': frame_idx,
                    'bbox': last_known_position.copy(),
                    'timestamp': frame_idx / fps,
                    'motion_detected': False,
                    'fill_type': 'forward_fill'
                })
            
            # Sort detections by frame
            full_detections.sort(key=lambda x: x['frame'])
            
            # Create extended track
            extended_track = {
                'track_id': track_id,
                'bbox_track': {
                    'detections': full_detections,
                    'start_frame': 0,
                    'end_frame': total_frames - 1,
                    'motion_start_frame': motion_start_frame,
                    'motion_end_frame': motion_end_frame
                },
                'duration_seconds': motion_track.duration_seconds,
                'motion_frames': len(motion_frames),
                'original_motion_track': motion_track
            }
            
            extended_tracks.append(extended_track)
        
        return extended_tracks
    
    def _sample_track_frames(self, extended_tracks: List[Dict], config) -> Tuple[set, Dict]:
        """Sample frames for analysis from all tracks."""
        all_sample_frames = set()
        track_sample_frames = {}
        
        for track in extended_tracks:
            track_id = track['track_id']
            motion_track = track['original_motion_track']
            motion_frames = [region.frame_idx for region in motion_track.regions]
            
            # Sample frames from motion track
            if len(motion_frames) <= config.max_validation_frames:
                sample_frames = motion_frames
            else:
                # Sample evenly distributed frames
                step = len(motion_frames) / config.max_validation_frames
                sample_frames = [motion_frames[int(i * step)] for i in range(config.max_validation_frames)]
                # Always include the last frame
                if motion_frames[-1] not in sample_frames:
                    sample_frames[-1] = motion_frames[-1]
            
            track_sample_frames[track_id] = sample_frames
            all_sample_frames.update(sample_frames)
        
        return all_sample_frames, track_sample_frames
    
    def _process_frame(self, frame, frame_idx: int, timestamp: float, extended_tracks: List[Dict], track_detections: Dict, config, video_path: Path) -> Dict:
        """Process a single frame with ensemble models and validate against tracks."""
        frame_valid = False
        track_ensemble_results = []
        
        logger.info(f"EVAL | {video_path.stem} | {timestamp:.2f}s | {frame_idx}")
        
        # Run each model against the frame (original nested loop structure)
        for model_name in config.ensemble_models:
            model_detections = self.ml_ensemble.run_single_model_detection(
                model_name, frame, config,
                timestamp_seconds=timestamp, frame_idx=frame_idx,
                full_frame=frame,
                accepted_rtdetr_overlap=config.spatial_overlap_threshold
            )
            
            # Track model contributions
            if model_detections:
                self._model_contributions[model_name]['total_detections'] += len(model_detections)
                max_conf = max(det['confidence'] for det in model_detections)
                self._model_contributions[model_name]['max_confidence'] = max(
                    self._model_contributions[model_name]['max_confidence'], max_conf
                )
            
            # For each track, determine if there was an overlap above the threshold
            for track in extended_tracks:
                track_id = track['track_id']
                track_bbox = self._get_track_bbox_for_frame(track, frame_idx)
                
                if track_bbox is not None:
                    fill_type = self._get_track_fill_type(track, frame_idx)
                    overlap_type = 'explicit' if fill_type == 'motion' else f'implicit_{fill_type}'
                    motion_bbox_str = f"motn:{track_bbox[0]:.0f},{track_bbox[1]:.0f},{track_bbox[2]:.0f},{track_bbox[3]:.0f}"
                    
                    overlapping_count = 0
                    valid_detections = []
                    
                    if model_detections:
                        for det in model_detections:
                            det_bbox = det['bbox']
                            overlap = self._calculate_bbox_overlap(track_bbox, det_bbox)
                            
                            if overlap >= config.spatial_overlap_threshold:
                                valid_detections.append({
                                    'detection': det,
                                    'overlap': overlap,
                                    'bbox_str': f"bbox:{det_bbox[0]:.0f},{det_bbox[1]:.0f},{det_bbox[2]:.0f},{det_bbox[3]:.0f}"
                                })
                                overlapping_count += 1
                            elif overlap > 0.0:
                                overlapping_count += 1
                        
                        # Process valid detections - create ONE synthetic detection per model per track
                        if valid_detections:
                            # Find best detection by score (confidence * overlap) after consensus boosting
                            consensus_boost = 1.0 + 0.1 * (len(valid_detections) - 1)
                            
                            best_detection = None
                            best_score = 0.0
                            
                            for valid_det in valid_detections:
                                det = valid_det['detection']
                                overlap = valid_det['overlap']
                                boosted_conf = min(1.0, det['confidence'] * consensus_boost)
                                score = boosted_conf * overlap
                                
                                if score > best_score:
                                    best_score = score
                                    best_detection = {
                                        'detection': det,
                                        'overlap': overlap,
                                        'bbox_str': valid_det['bbox_str'],
                                        'boosted_conf': boosted_conf,
                                        'score': score
                                    }
                            
                            # Log ONE synthetic detection representing the best from this model for this track
                            det = best_detection['detection']
                            overlap = best_detection['overlap']
                            bbox_str = best_detection['bbox_str']
                            boosted_conf = best_detection['boosted_conf']
                            overall_score = best_detection['score']
                            
                            consensus_note = f"consensus:{len(valid_detections)}" if len(valid_detections) > 1 else "single"
                            logger.info(f"✅ | {model_name} | {bbox_str} | conf:{det['confidence']:.3f}→{boosted_conf:.3f} | ovlp:{overlap:.3f} | {motion_bbox_str} | scor:{overall_score:.3f} | spatial_valid | {overlap_type} | track_{track_id} | {consensus_note}")
                            
                            # Track spatial valid contributions
                            self._model_contributions[model_name]['spatial_valid_count'] += 1
                            self._model_contributions[model_name]['total_score'] += overall_score
                            
                            # Store ONE synthetic detection for track evaluation (use original confidence for reporting)
                            det_copy = det.copy()
                            det_copy['source'] = model_name
                            det_copy['confidence'] = det['confidence']  # Use original confidence for final reporting
                            det_copy['boosted_confidence'] = boosted_conf
                            det_copy['original_confidence'] = det['confidence']
                            det_copy['consensus_boost'] = consensus_boost
                            det_copy['consensus_count'] = len(valid_detections)
                            det_copy['motion_overlap'] = overlap
                            det_copy['frame_idx'] = frame_idx
                            det_copy['timestamp'] = timestamp
                            det_copy['overlap_type'] = overlap_type
                            track_detections[track_id].append(det_copy)
                    
                    # If no detections overlapped with this track, log summary  
                    if overlapping_count == 0:
                        if model_detections:
                            non_overlapping = len(model_detections)
                            logger.info(f"❌ | {model_name} | {non_overlapping}_detections | conf:0.000 | ovlp:0.000 | {motion_bbox_str} | scor:0.000 | no_overlap | {overlap_type} | track_{track_id}")
                        else:
                            logger.info(f"❌ | {model_name} | none | conf:0.000 | ovlp:0.000 | {motion_bbox_str} | scor:0.000 | no_detection | {overlap_type} | track_{track_id}")
                    elif len(valid_detections) == 0 and overlapping_count > 0:
                        # All detections had some overlap but failed threshold
                        logger.info(f"⚠️ | {model_name} | {overlapping_count}_detections | conf:0.000 | ovlp:0.000 | {motion_bbox_str} | scor:0.000 | threshold_failed | {overlap_type} | track_{track_id}")
                else:
                    # Track has no bbox for this frame
                    if model_detections:
                        logger.info(f"❌ | {model_name} | {len(model_detections)}_detections | conf:0.000 | ovlp:0.000 | motn:none | scor:0.000 | no_track_bbox | track_{track_id}")
                    else:
                        logger.info(f"❌ | {model_name} | none | conf:0.000 | ovlp:0.000 | motn:none | scor:0.000 | no_detection | track_{track_id}")
        
        # Calculate ensemble score per track for this frame (after all models processed)
        for track in extended_tracks:
            track_id = track['track_id']
            track_frame_detections = [d for d in track_detections[track_id] if d['frame_idx'] == frame_idx]
            
            # Always evaluate each track, even if no detections (use boosted confidence for ensemble scoring)
            track_ensemble_score = sum(d['boosted_confidence'] for d in track_frame_detections) if track_frame_detections else 0.0
            track_passed = track_ensemble_score >= config.confidence_threshold
            
            if track_passed:
                frame_valid = True
            
            track_ensemble_results.append({
                'track_id': track_id,
                'ensemble_score': track_ensemble_score,
                'detections': len(track_frame_detections),
                'passed': track_passed
            })
        
        # Log overall frame result
        frame_icon = "✅" if frame_valid else "❌"
        passed_tracks = [r for r in track_ensemble_results if r['passed']]
        frame_reason = f"{len(passed_tracks)}_tracks_passed" if frame_valid else "no_tracks_passed"
        
        logger.info(f"================================================================================")
        logger.info(f"{frame_icon} | FRAME_RESULT | {video_path.stem} | frame_{frame_idx} | {timestamp:.2f}s | tracks_evaluated={len(track_ensemble_results)} | tracks_passed={len(passed_tracks)} | {frame_reason}")
        logger.info(f"================================================================================")
        
        return {
            'frame_idx': frame_idx,
            'track_results': track_ensemble_results,
            'frame_valid': frame_valid
        }
    
    def _get_track_bbox_for_frame(self, track: Dict, frame_idx: int) -> Optional[List[float]]:
        """Get bbox for specific track at specific frame."""
        bbox_track = track['bbox_track']
        for det in bbox_track['detections']:
            if det['frame'] == frame_idx:
                return det['bbox']
        return None
    
    def _get_track_fill_type(self, track: Dict, frame_idx: int) -> str:
        """Get fill type for specific track at specific frame."""
        bbox_track = track['bbox_track']
        for det in bbox_track['detections']:
            if det['frame'] == frame_idx:
                return det.get('fill_type', 'motion')
        return 'unknown'
    
    def _evaluate_tracks(self, extended_tracks: List[Dict], track_detections: Dict, frame_results: List[Dict], track_sample_frames: Dict, fps: float, config, video_path: Path) -> List[Dict]:
        """Evaluate tracks based on validation criteria."""
        validated_results = []
        
        for track in extended_tracks:
            track_id = track['track_id']
            detections = track_detections[track_id]
            
            if not detections:
                continue
            
            # Calculate track statistics (use original confidence for reporting)
            avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
            max_confidence = max(d['confidence'] for d in detections)
            summed_confidence = sum(d['confidence'] for d in detections)
            
            # Calculate boosted statistics for validation logic
            avg_boosted_confidence = sum(d['boosted_confidence'] for d in detections) / len(detections)
            max_boosted_confidence = max(d['boosted_confidence'] for d in detections)
            summed_boosted_confidence = sum(d['boosted_confidence'] for d in detections)
            
            # Count passed frames for this track
            track_frames = track_sample_frames[track_id]
            passed_frames = 0
            
            for frame_result in frame_results:
                if frame_result['frame_idx'] in track_frames:
                    track_result = next((tr for tr in frame_result['track_results'] if tr['track_id'] == track_id), None)
                    if track_result and track_result['passed']:
                        passed_frames += 1
            
            # Validation criteria (use boosted confidence for thresholding)
            confidence_passed = summed_boosted_confidence >= config.confidence_threshold
            frames_passed = len(detections) >= config.min_track_frames
            temporal_continuity_passed = True  # Simplified for now
            
            validation_passed = confidence_passed and frames_passed and temporal_continuity_passed
            
            # Log track evaluation in original format
            logger.info(f"TRACK | {video_path.stem} | track_{track_id}")
            track_icon = "✅" if validation_passed else "❌"
            # Calculate composite score for logging (even for failed tracks)
            composite_score = self._calculate_composite_score(track, detections, track_frames, fps) if detections else CompositeScore.empty().to_dict()
            
            logger.info(f"{track_icon} | duration={track['duration_seconds']:.2f}s | frames_evaluated={len(track_frames)} | frames_passed={passed_frames} | detections={len(detections)} | summed_conf={summed_confidence:.3f} | avg_conf={avg_confidence:.3f} | max_conf={max_confidence:.3f} | ensemble={summed_boosted_confidence:.3f} | composite={composite_score['final_score']:.3f} | models={composite_score['consensus_models']} | motion_align={composite_score['motion_alignment']:.3f} | temp_density={composite_score['temporal_density']:.3f} | conf_pass={confidence_passed} | frames_pass={frames_passed} | temporal_pass={temporal_continuity_passed} | validated={validation_passed}")
            
            if validation_passed:
                # Find best detection
                best_detection_dict = max(detections, key=lambda d: d['confidence'])
                
                # Calculate multi-dimensional confidence score
                composite_score = self._calculate_composite_score(track, detections, track_frames, fps)
                
                # Create typed best detection
                best_detection_bbox = BoundingBox(
                    best_detection_dict['bbox'][0],
                    best_detection_dict['bbox'][1],
                    best_detection_dict['bbox'][2],
                    best_detection_dict['bbox'][3]
                )
                
                best_detection = Detection(
                    confidence=best_detection_dict['confidence'],
                    bbox=best_detection_bbox,
                    source=f"fullframe_{best_detection_dict.get('source', 'unknown')}",
                    class_name='animal',
                    timestamp=best_detection_dict.get('timestamp', track_frames[0] / fps),
                    frame_idx=best_detection_dict.get('frame_idx', track_frames[0])
                )
                
                # Convert all detections to typed objects
                typed_detections = []
                for det in detections:
                    det_bbox = BoundingBox(det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3])
                    typed_det = Detection(
                        confidence=det['confidence'],
                        bbox=det_bbox,
                        source=det.get('source', 'unknown'),
                        class_name='animal',
                        timestamp=det.get('timestamp', 0.0),
                        frame_idx=det.get('frame_idx', 0)
                    )
                    typed_detections.append(typed_det)
                
                # Create track object
                track_obj = Track(
                    track_id=track_id,
                    detections=typed_detections,
                    start_frame=min(track_frames),
                    end_frame=max(track_frames),
                    duration_seconds=track['duration_seconds'],
                    confidence_scores=[d.confidence for d in typed_detections],
                    bbox_sequence=[d.bbox for d in typed_detections]
                )
                
                # Create validation sequence
                validated_sequence = ValidationSequence(
                    sequence_id=track_id,
                    track=track_obj,
                    detections=typed_detections,
                    ensemble_score=summed_boosted_confidence,
                    composite_score=composite_score['final_score'],
                    best_detection=best_detection,
                    frame_range=(min(track_frames), max(track_frames)),
                    duration_seconds=track['duration_seconds']
                )
                
                validated_results.append(validated_sequence)
        
        return validated_results
    
    def _calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU (Intersection over Union) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def get_model_contributions(self) -> Dict:
        """Get model contribution statistics for this video."""
        return self._model_contributions.copy()
    
    def _calculate_composite_score(self, track: Dict, detections: List[Dict], track_frames: List[int], fps: float) -> Dict:
        """
        Calculate multi-dimensional confidence score that combines multiple strong signals.
        
        Args:
            track: Track information with duration
            detections: List of detections for this track
            track_frames: List of sampled frame indices for this track
            fps: Video frame rate
            
        Returns:
            Dict with final score and breakdown of components
        """
        if not detections:
            return {
                'final_score': 0.0,
                'base_score': 0.0,
                'temporal_density': 0.0,
                'consensus_models': 0,
                'motion_alignment': 0.0,
                'duration_seconds': track['duration_seconds'],
                'temporal_multiplier': 0.0,
                'consensus_multiplier': 0.0,
                'motion_multiplier': 0.0,
                'duration_bonus': 0.0
            }
        
        # Base ensemble score (sum of boosted model confidences)
        # Base score is sum of boosted confidences (distinct from ensemble/conf scores)
        base_score = sum(d.get('boosted_confidence', 0.0) for d in detections)
        
        # 1. Temporal consistency (how dense are detections across track)
        track_duration_frames = len(track_frames)
        temporal_density = len(detections) / max(1, track_duration_frames)
        temporal_multiplier = min(2.0, 1.0 + temporal_density)  # 1.0-2.0x based on detection density
        
        # 2. Multi-model consensus (how many different models contributed)
        unique_models = len(set(d['source'] for d in detections))
        consensus_multiplier = 1.0 + (0.2 * (unique_models - 1))  # +20% per additional model beyond first
        
        # 3. Motion correlation (how well do ML detections align with motion regions)
        avg_motion_overlap = sum(d.get('motion_overlap', 0.0) for d in detections) / len(detections)
        motion_multiplier = 0.5 + (1.5 * avg_motion_overlap)  # 0.5-2.0x based on motion alignment
        
        # 4. Track duration bonus (longer tracks are more reliable indicators)
        duration_seconds = track['duration_seconds']
        duration_bonus = min(1.5, 0.8 + (duration_seconds / 6.0))  # Up to 1.5x for 4+ second tracks
        
        # Calculate final score with all multipliers
        final_score = base_score * temporal_multiplier * consensus_multiplier * motion_multiplier * duration_bonus
        
        
        return {
            'final_score': final_score,
            'base_score': base_score,
            'temporal_density': temporal_density,
            'consensus_models': unique_models,
            'motion_alignment': avg_motion_overlap,
            'duration_seconds': duration_seconds,
            'temporal_multiplier': temporal_multiplier,
            'consensus_multiplier': consensus_multiplier,
            'motion_multiplier': motion_multiplier,
            'duration_bonus': duration_bonus
        }