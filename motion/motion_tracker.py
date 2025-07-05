"""Motion tracker for wildlife video processing."""

import logging
from pathlib import Path
from typing import Dict, List, Any
import cv2

from .motion_detector import MotionDetector

logger = logging.getLogger('wildcams')


class MotionTracker:
    """Tracks motion sequences and applies infilling to create temporal tracks."""
    
    def __init__(self, motion_detector: MotionDetector, config):
        """
        Initialize motion tracker.
        
        Args:
            motion_detector: MotionDetector instance
            config: ProcessingConfig object
        """
        self.motion_detector = motion_detector
        self.config = config
        
        # Statistics tracking
        self._large_region_count = 0
        self._total_region_count = 0
        self._initial_track_count = 0
    
    def find_consistent_motion_sequences_and_tracks(self, video_path: Path, fps: float, total_frames: int, config) -> List[Dict]:
        """STEP 1: Find sequences with consistent motion/tracking."""
        cap = self.motion_detector.open_video_stream(video_path)
        if not cap:
            return []
        
        # Reset motion detector for this video
        self.motion_detector._init_motion_detector(self.config)
        
        motion_sequences = []
        current_sequence = None
        
        logger.info(f"Analyzing motion across {total_frames} frames")
        logger.info(f"📊 Motion detection parameters:")
        logger.info(f"  🎯 Min motion area: {config.min_motion_area} pixels")
        logger.info(f"  📈 Variance threshold: {config.motion_var_threshold}")
        logger.info(f"  ⏱️ Min track duration: {config.min_track_duration}s")
        logger.info(f"  📦 Max regions per frame: {config.max_regions_per_frame}")
        
        frame_motion_count = 0
        total_motion_regions = 0
        motion_region_sizes = []
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            motion_regions = self.motion_detector.detect_motion_regions(frame, config)
            
            if motion_regions:
                frame_motion_count += 1
                total_motion_regions += len(motion_regions)
                
                # Track motion region sizes
                for region in motion_regions:
                    x1, y1, x2, y2 = region
                    area = (x2 - x1) * (y2 - y1)
                    motion_region_sizes.append(area)
                
                if frame_idx % 50 == 0:  # Log every 50 frames
                    logger.info(f"📊 Frame {frame_idx}: {len(motion_regions)} motion regions detected")
                    
                if current_sequence is None:
                    # Start new sequence
                    current_sequence = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'start_timestamp': timestamp,
                        'end_timestamp': timestamp,
                        'motion_regions': [motion_regions],
                        'frames': [frame_idx]
                    }
                else:
                    # Continue current sequence
                    current_sequence['end_frame'] = frame_idx
                    current_sequence['end_timestamp'] = timestamp
                    current_sequence['motion_regions'].append(motion_regions)
                    current_sequence['frames'].append(frame_idx)
            else:
                # No motion - end current sequence (duration filtering happens after infilling)
                if current_sequence is not None:
                    duration = current_sequence['end_timestamp'] - current_sequence['start_timestamp']
                    motion_sequences.append(current_sequence)
                    logger.info(f"Motion sequence found: frames {current_sequence['start_frame']}-{current_sequence['end_frame']} ({duration:.2f}s)")
                    current_sequence = None
        
        # Handle sequence that extends to end of video (duration filtering happens after infilling)
        if current_sequence is not None:
            duration = current_sequence['end_timestamp'] - current_sequence['start_timestamp']
            motion_sequences.append(current_sequence)
            logger.info(f"Motion sequence found: frames {current_sequence['start_frame']}-{current_sequence['end_frame']} ({duration:.2f}s)")
        
        cap.release()
        
        # Add detailed motion detection summary
        logger.info(f"📊 MOTION DETECTION SUMMARY:")
        logger.info(f"  🎯 Total frames: {total_frames}")
        logger.info(f"  📈 Frames with motion: {frame_motion_count} ({100*frame_motion_count/total_frames:.1f}%)")
        logger.info(f"  📦 Total motion regions: {total_motion_regions}")
        
        if motion_region_sizes:
            avg_region_size = sum(motion_region_sizes) / len(motion_region_sizes)
            max_region_size = max(motion_region_sizes)
            min_region_size = min(motion_region_sizes)
            # Camera handling typically has many large regions vs animal movement with smaller focused regions
            # For 1600x900 videos, large regions would be >100k pixels (e.g. 400x250px or ~17% of frame)
            large_regions = len([size for size in motion_region_sizes if size > 100000])  # >400x250px
            
            # Store for use in composite score calculation
            self._large_region_count = large_regions
            self._total_region_count = len(motion_region_sizes)
            
            logger.info(f"  📏 Region sizes - Avg: {avg_region_size:.0f}px, Min: {min_region_size:.0f}px, Max: {max_region_size:.0f}px")
            logger.info(f"  📦 Large regions (>100k px): {large_regions}/{len(motion_region_sizes)} ({100*large_regions/len(motion_region_sizes):.1f}%)")
        else:
            # Store defaults if no motion regions found
            self._large_region_count = 0
            self._total_region_count = 0
            
        logger.info(f"  ⏱️ Motion sequences found: {len(motion_sequences)}")
        
        if frame_motion_count == 0:
            logger.warning(f"⚠️ NO MOTION DETECTED - Check parameters:")
            logger.warning(f"  🎯 Min area: {config.min_motion_area} (try lower like 100-200)")
            logger.warning(f"  📈 Var threshold: {config.motion_var_threshold} (try lower like 16-25)")
        elif len(motion_sequences) == 0:
            logger.warning(f"⚠️ MOTION DETECTED BUT NO SEQUENCES - Duration too strict:")
            logger.warning(f"  ⏱️ Min duration: {config.min_track_duration}s (try lower like 1.0s)")
        
        # Convert motion sequences to motion tracks for Step 2 filtering
        motion_tracks = []
        for i, sequence in enumerate(motion_sequences):
            duration = sequence['end_timestamp'] - sequence['start_timestamp']
            
            # Create motion track from sequence
            motion_track = {
                'track_id': i,
                'start_frame': sequence['start_frame'],
                'end_frame': sequence['end_frame'],
                'duration_seconds': duration,
                'frames': sequence['frames'],
                'motion_regions': sequence['motion_regions'],
                'detection_count': len(sequence['frames']),  # Number of frames with motion
                'avg_regions_per_frame': sum(len(regions) for regions in sequence['motion_regions']) / len(sequence['motion_regions']) if sequence['motion_regions'] else 0
            }
            motion_tracks.append(motion_track)
        
        # Store initial track count for camera handling filter
        self._initial_track_count = len(motion_tracks)
        
        # Apply track infilling if enabled
        if config.enable_track_infilling:
            motion_tracks = self._infill_motion_tracks(motion_tracks, fps, config)
        else:
            # Print track summary even without infilling
            self._print_track_summary(motion_tracks, fps)
        
        return motion_tracks
    
    def _infill_motion_tracks(self, motion_tracks: List[Dict], fps: float, config) -> List[Dict]:
        """Infill gaps between nearby motion tracks to create continuous tracks."""
        if not motion_tracks:
            return motion_tracks
        
        logger.info(f"[INFILL] Starting with {len(motion_tracks)} tracks, checking for infilling opportunities")
        
        # Sort tracks by start frame
        sorted_tracks = sorted(motion_tracks, key=lambda t: t['frames'][0])
        infilled_tracks = []
        used_track_ids = set()
        
        for i, track_a in enumerate(sorted_tracks):
            if track_a['track_id'] in used_track_ids:
                continue
                
            # Start with this track
            merged_track = track_a.copy()
            used_track_ids.add(track_a['track_id'])
            
            # Look for compatible tracks to merge
            changed = True
            while changed:
                changed = False
                
                for j, track_b in enumerate(sorted_tracks):
                    if track_b['track_id'] in used_track_ids:
                        continue
                    
                    # Check if tracks can be infilled
                    if self._can_infill_tracks(merged_track, track_b, fps, config):
                        logger.info(f"[INFILL] Merging track_{track_b['track_id']} into track_{merged_track['track_id']}")
                        merged_track = self._merge_tracks_with_infill(merged_track, track_b, fps)
                        used_track_ids.add(track_b['track_id'])
                        changed = True
                        break
            
            infilled_tracks.append(merged_track)
        
        logger.info(f"[INFILL] Result: {len(infilled_tracks)} tracks after infilling ({len(motion_tracks) - len(infilled_tracks)} tracks merged)")
        
        # Filter out tracks shorter than max(min_track_duration, min_consecutive_detection_seconds)
        min_required_duration = max(config.min_track_duration, config.min_consecutive_detection_seconds)
        filtered_tracks = [track for track in infilled_tracks if track['duration_seconds'] >= min_required_duration]
        
        removed_count = len(infilled_tracks) - len(filtered_tracks)
        if removed_count > 0:
            logger.info(f"[INFILL] Removed {removed_count} tracks shorter than {min_required_duration:.2f}s (max of min_track_duration={config.min_track_duration:.2f}s, min_consecutive_detection={config.min_consecutive_detection_seconds:.2f}s)")
        
        # Print detailed track summary
        self._print_track_summary(filtered_tracks, fps)
        
        return filtered_tracks
    
    def _can_infill_tracks(self, track_a: Dict, track_b: Dict, fps: float, config) -> bool:
        """Check if two tracks can be infilled based on spatial and temporal criteria."""
        # Get track boundaries
        frames_a = track_a['frames']
        frames_b = track_b['frames']
        
        end_frame_a = frames_a[-1]
        start_frame_b = frames_b[0]
        
        # Check temporal gap
        gap_frames = start_frame_b - end_frame_a
        gap_seconds = gap_frames / fps
        
        if gap_frames <= 0:  # Overlapping or adjacent tracks
            return False
        
        if gap_seconds > config.infill_max_gap_seconds:
            return False
        
        # Check spatial proximity using end bbox of track_a vs start bbox of track_b
        motion_regions_a = track_a.get('motion_regions', [])
        motion_regions_b = track_b.get('motion_regions', [])
        
        if not motion_regions_a or not motion_regions_b:
            return False
        
        # Get end bbox from track_a and start bbox from track_b
        end_region_a = motion_regions_a[-1][-1] if motion_regions_a[-1] else None
        start_region_b = motion_regions_b[0][0] if motion_regions_b[0] else None
        
        if not end_region_a or not start_region_b:
            return False
        
        # Calculate distance between track end and track start centers
        center_a = ((end_region_a[0] + end_region_a[2]) / 2, (end_region_a[1] + end_region_a[3]) / 2)
        center_b = ((start_region_b[0] + start_region_b[2]) / 2, (start_region_b[1] + start_region_b[3]) / 2)
        
        distance = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
        
        if distance > config.infill_max_distance_pixels:
            return False
        
        # Check bbox overlap ratio between end and start positions
        overlap = self._calculate_bbox_overlap(end_region_a, start_region_b)
        if overlap < config.infill_min_overlap_ratio:
            return False
        
        return True
    
    def _merge_tracks_with_infill(self, track_a: Dict, track_b: Dict, fps: float) -> Dict:
        """Merge two tracks with interpolated frames in the gap."""
        frames_a = track_a['frames']
        frames_b = track_b['frames']
        regions_a = track_a['motion_regions']
        regions_b = track_b['motion_regions']
        
        end_frame_a = frames_a[-1]
        start_frame_b = frames_b[0]
        
        # Create interpolated frames for the gap
        gap_frames = list(range(end_frame_a + 1, start_frame_b))
        
        # Interpolate regions for gap frames (simple: use last region from track_a)
        last_region_a = regions_a[-1][-1] if regions_a and regions_a[-1] else track_a.get('representative_region')
        first_region_b = regions_b[0][0] if regions_b and regions_b[0] else track_b.get('representative_region')
        
        # Simple interpolation: use last region from track_a for all gap frames
        gap_regions = [[last_region_a] for _ in gap_frames] if last_region_a else []
        
        # Merge the tracks
        merged_track = {
            'track_id': track_a['track_id'],  # Keep first track's ID
            'start_frame': frames_a[0],
            'end_frame': frames_b[-1],
            'frames': frames_a + gap_frames + frames_b,
            'motion_regions': regions_a + gap_regions + regions_b,
            'duration_seconds': (frames_b[-1] - frames_a[0]) / fps,
            'representative_region': track_a.get('representative_region'),
            'detection_count': len(frames_a) + len(gap_frames) + len(frames_b),
            'avg_regions_per_frame': track_a.get('avg_regions_per_frame', 0),
            'infilled_from': [track_a['track_id'], track_b['track_id']],
            'infill_gap_frames': len(gap_frames)
        }
        
        return merged_track
    
    def _print_track_summary(self, motion_tracks: List[Dict], fps: float) -> None:
        """Print a detailed summary of motion tracks."""
        if not motion_tracks:
            logger.info("📋 TRACK SUMMARY: No motion tracks found")
            return
        
        logger.info("📋 MOTION TRACK SUMMARY:")
        logger.info("================================================================================")
        
        for track in motion_tracks:
            track_id = track['track_id']
            frames = track['frames']
            start_frame, end_frame = frames[0], frames[-1]
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = track['duration_seconds']
            
            # Get start and end bbox from motion regions
            motion_regions = track.get('motion_regions', [])
            if motion_regions:
                # Get first and last motion regions
                start_region = motion_regions[0][0] if motion_regions[0] else None
                end_region = motion_regions[-1][-1] if motion_regions[-1] else None
                
                if start_region:
                    start_bbox_str = f"start_bbox:{start_region[0]:.0f},{start_region[1]:.0f},{start_region[2]:.0f},{start_region[3]:.0f}"
                    start_width = start_region[2] - start_region[0]
                    start_height = start_region[3] - start_region[1]
                    start_size_str = f"start_size:{start_width:.0f}x{start_height:.0f}"
                else:
                    start_bbox_str = "start_bbox:unknown"
                    start_size_str = "start_size:unknown"
                
                if end_region:
                    end_bbox_str = f"end_bbox:{end_region[0]:.0f},{end_region[1]:.0f},{end_region[2]:.0f},{end_region[3]:.0f}"
                    end_width = end_region[2] - end_region[0]
                    end_height = end_region[3] - end_region[1]
                    end_size_str = f"end_size:{end_width:.0f}x{end_height:.0f}"
                else:
                    end_bbox_str = "end_bbox:unknown"
                    end_size_str = "end_size:unknown"
                
                bbox_str = f"{start_bbox_str} | {end_bbox_str}"
                size_str = f"{start_size_str} | {end_size_str}"
            else:
                bbox_str = "start_bbox:unknown | end_bbox:unknown"
                size_str = "start_size:unknown | end_size:unknown"
            
            # Infill information if available
            infill_info = ""
            if 'infilled_from' in track:
                original_tracks = track['infilled_from']
                gap_frames = track.get('infill_gap_frames', 0)
                infill_info = f" | infilled:{len(original_tracks)}tracks,{gap_frames}gap_frames"
            
            logger.info(f"  🎯 track_{track_id}: frames:{start_frame}-{end_frame} ({len(frames)}frames) | "
                       f"time:{start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s) | {bbox_str} | {size_str}{infill_info}")
        
        logger.info("================================================================================")
        
        # Summary statistics
        total_frames = sum(len(track['frames']) for track in motion_tracks)
        avg_duration = sum(track['duration_seconds'] for track in motion_tracks) / len(motion_tracks)
        longest_track = max(motion_tracks, key=lambda t: t['duration_seconds'])
        
        logger.info(f"📊 SUMMARY: {len(motion_tracks)} tracks | {total_frames} total frames | "
                   f"avg_duration:{avg_duration:.2f}s | longest:{longest_track['duration_seconds']:.2f}s (track_{longest_track['track_id']})")
    
    def _calculate_bbox_overlap(self, motion_bbox: List[float], detection_bbox: List[float]) -> float:
        """Calculate IoU (Intersection over Union) between motion and detection bboxes."""
        mx1, my1, mx2, my2 = motion_bbox
        dx1, dy1, dx2, dy2 = detection_bbox
        
        # Calculate intersection
        ix1 = max(mx1, dx1)
        iy1 = max(my1, dy1)
        ix2 = min(mx2, dx2)
        iy2 = min(my2, dy2)
        
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0  # No overlap
        
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        motion_area = (mx2 - mx1) * (my2 - my1)
        detection_area = (dx2 - dx1) * (dy2 - dy1)
        union_area = motion_area + detection_area - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        # IoU = intersection / union
        return intersection_area / union_area
    
    # Getter methods for statistics
    def get_large_region_count(self) -> int:
        """Get count of large motion regions (for camera handling detection)."""
        return self._large_region_count
    
    def get_total_region_count(self) -> int:
        """Get total count of motion regions."""
        return self._total_region_count
    
    def get_initial_track_count(self) -> int:
        """Get initial track count before infilling."""
        return self._initial_track_count