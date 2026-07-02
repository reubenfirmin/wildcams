"""Camera handling filter for wildlife video processing."""

import logging
from pathlib import Path

import cv2
import numpy as np

from core.data_types import MotionTrack

logger = logging.getLogger("wildcams")


class CameraHandlingFilter:
    """Filters motion tracks to detect camera handling using frame coverage analysis."""

    def __init__(self):
        """
        Initialize camera handling filter.
        """
        # Statistics tracking
        self._composite_scores = {}
        self._rejection_reasons = {}
        self._initial_track_count = 0

        logger.info("📹 Camera Handling Filter initialized")

    def filter_motion_tracks_for_camera_handling(
        self, video_path: Path, motion_tracks: list[MotionTrack], config, initial_track_count: int | None = None
    ) -> list[MotionTrack]:
        """STEP 2: Filter motion tracks for camera handling detection using frame coverage analysis."""
        logger.info(f"[STEP2] {video_path.name}: Filtering {len(motion_tracks)} motion tracks for camera handling")

        # Store initial track count for penalty calculation
        if initial_track_count is not None:
            self._initial_track_count = initial_track_count
        else:
            self._initial_track_count = len(motion_tracks)

        # Get video dimensions for frame coverage calculation
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_area = frame_width * frame_height
        cap.release()

        # Calculate frame coverage: "everything moves most of the time" = camera handling
        frames_with_motion = set()
        total_motion_area_per_frame: dict[int, float] = {}

        for track in motion_tracks:
            # Work with typed MotionTrack objects
            for region in track.regions:
                frame_idx = region.frame_idx
                frames_with_motion.add(frame_idx)

                # Add this region's area to the frame total
                if frame_idx not in total_motion_area_per_frame:
                    total_motion_area_per_frame[frame_idx] = 0
                total_motion_area_per_frame[frame_idx] += region.area

        # Calculate frame coverage metrics
        temporal_coverage = len(frames_with_motion) / max(1, total_frames)  # 0.0-1.0

        avg_spatial_coverage = 0.0
        if total_motion_area_per_frame:
            avg_spatial_coverage = sum(
                min(1.0, area / total_frame_area) for area in total_motion_area_per_frame.values()
            ) / len(total_motion_area_per_frame)

        # Combined coverage score: high = camera handling
        frame_coverage_score = temporal_coverage * avg_spatial_coverage

        num_tracks = len(motion_tracks)

        # Get track count before filtering for penalty calculation
        filtering_penalty = 1.0 + ((self._initial_track_count - num_tracks) / max(1, self._initial_track_count)) * 2.0

        # Spatial clustering: group tracks by bbox overlap
        spatial_clusters = self._cluster_tracks_by_spatial_overlap(motion_tracks)
        effective_regions = len(spatial_clusters)

        # Log spatial clustering details
        logger.info("  📊 SPATIAL CLUSTERING DEBUG:")
        for i, cluster in enumerate(spatial_clusters):
            cluster_frames = sum(len(track.regions) for track in cluster)
            track_ids = [track.track_id for track in cluster]
            logger.info(f"    Cluster {i}: tracks={track_ids}, total_frames={cluster_frames}")

            # Show bbox positions for debugging
            for track in cluster:
                if track.regions:
                    start_bbox = track.regions[0].bbox
                    logger.info(
                        f"      track_{track.track_id}: start_bbox=({start_bbox.x1}, {start_bbox.y1}, {start_bbox.x2}, {start_bbox.y2})"
                    )

        # Calculate consistency penalty based on bbox variance within clusters
        consistency_penalty = self._calculate_bbox_consistency_penalty(spatial_clusters)

        # No large region calculation needed with spatial clustering

        # Camera handling detection: HIGH scores = camera handling
        # frame_coverage_score: high = "everything moves most of the time"
        # spatial_dispersion: high = dispersed movement across frame

        # Calculate spatial dispersion: ratio of clusters to tracks
        spatial_dispersion = effective_regions / max(1, num_tracks)

        # Calculate composite score for camera handling detection
        # Higher values indicate MORE camera handling characteristics
        base_score = (
            frame_coverage_score**config.motion_frames_weight
            * spatial_dispersion**config.motion_regions_weight
            * num_tracks**config.motion_tracks_weight
        )

        composite_score = base_score * consistency_penalty * filtering_penalty

        # Use CLI parameter for camera handling detection
        threshold = config.composite_motion_threshold

        logger.info(f"[STEP2] {video_path.name}: Camera handling score = {composite_score:.6f}")
        logger.info(
            f"  📊 Frame coverage: temporal={temporal_coverage:.3f} × spatial={avg_spatial_coverage:.3f} = {frame_coverage_score:.6f}"
        )
        logger.info(f"  📊 Spatial dispersion: {effective_regions}/{num_tracks} = {spatial_dispersion:.3f}")
        logger.info(
            f"  📊 Base: coverage^{config.motion_frames_weight:.1f}={frame_coverage_score:.6f}^{config.motion_frames_weight:.1f} * dispersion^{config.motion_regions_weight:.1f}={spatial_dispersion:.3f}^{config.motion_regions_weight:.1f} * tracks^{config.motion_tracks_weight:.1f}={num_tracks}^{config.motion_tracks_weight:.1f} = {base_score:.6f}"
        )
        logger.info(
            f"  📊 Penalties: consistency={consistency_penalty:.2f}x, filtering={filtering_penalty:.2f}x (initial_tracks={self._initial_track_count}→{num_tracks})"
        )
        logger.info(f"  📊 Spatial clusters: {len(spatial_clusters)} effective regions from {num_tracks} tracks")

        # Store composite score for summary reporting
        self._composite_scores[video_path.name] = composite_score

        # Check for excessive motion (camera handling)
        if composite_score > threshold:
            logger.warning(f"⚠️  CAMERA HANDLING: score={composite_score} > {threshold}")
            # Store rejection reason for summary
            self._rejection_reasons[video_path.name] = f"camera_handling (score={composite_score:.0f})"
            return []  # Early exit - skip expensive ML processing

        return motion_tracks

    def _cluster_tracks_by_spatial_overlap(self, motion_tracks: list[MotionTrack]) -> list[list[MotionTrack]]:
        """Group tracks into spatial clusters based on bbox overlap."""
        if not motion_tracks:
            return []

        clusters: list = []
        for track in motion_tracks:
            # Get representative bbox for this track (use first region)
            if not track.regions:
                continue

            track_bbox = track.regions[0].bbox
            if not track_bbox:
                continue

            # Find if this track overlaps with any existing cluster
            assigned = False
            for cluster in clusters:
                for cluster_track in cluster:
                    if not cluster_track.regions:
                        continue

                    cluster_bbox = cluster_track.regions[0].bbox
                    # Convert BoundingBox objects to coordinate lists for overlap calculation
                    track_coords = [track_bbox.x1, track_bbox.y1, track_bbox.x2, track_bbox.y2]
                    cluster_coords = [cluster_bbox.x1, cluster_bbox.y1, cluster_bbox.x2, cluster_bbox.y2]

                    if self._calculate_bbox_overlap(track_coords, cluster_coords) > 0.3:  # 30% overlap threshold
                        cluster.append(track)
                        assigned = True
                        break
                if assigned:
                    break

            # If no overlap found, create new cluster
            if not assigned:
                clusters.append([track])

        return clusters

    def _calculate_ensemble_score(self, model_contributions: dict[str, float]) -> float:
        """Simple sum: each model contributes at most once per evaluation."""
        return sum(model_contributions.values())

    def _calculate_bbox_consistency_penalty(self, spatial_clusters: list[list[MotionTrack]]) -> float:
        """Calculate consistency reward using logarithmic decay for repeated spatial regions."""
        if not spatial_clusters:
            return 1.0

        total_weight = 0.0

        logger.info("  📊 CONSISTENCY PENALTY DEBUG:")
        for i, cluster in enumerate(spatial_clusters):
            # Each spatial cluster starts with weight 1.0
            cluster_frames = sum(len(track.regions) for track in cluster)

            if cluster_frames <= 1:
                # Single frame = full weight
                cluster_weight = 1.0
                total_weight += cluster_weight
                logger.info(f"    Cluster {i}: {cluster_frames} frames → weight={cluster_weight:.3f} (single frame)")
            else:
                # Logarithmic decay: consistent movement gets lower weight
                # log(frames) rewards staying in same spatial area longer
                cluster_weight = 1.0 / max(1.0, np.log(cluster_frames))
                total_weight += cluster_weight
                logger.info(
                    f"    Cluster {i}: {cluster_frames} frames → log({cluster_frames})={np.log(cluster_frames):.3f} → weight={cluster_weight:.3f}"
                )

        # Normalize by number of clusters to get average consistency
        avg_weight = total_weight / len(spatial_clusters) if spatial_clusters else 1.0

        # Convert to penalty: consistent movement (low avg_weight) = low penalty
        # Scale: 0.1 avg_weight → 1.0x penalty, 1.0 avg_weight → 5.0x penalty
        consistency_penalty = 1.0 + (avg_weight * 4.0)

        logger.info(
            f"    Total weight: {total_weight:.3f}, avg_weight: {avg_weight:.3f} → penalty: {consistency_penalty:.3f}"
        )

        return min(consistency_penalty, 5.0)  # Cap at 5x penalty

    def _calculate_bbox_overlap(self, motion_bbox: list[float], detection_bbox: list[float]) -> float:
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
    def get_composite_scores(self) -> dict[str, float]:
        """Get composite scores for all processed videos."""
        return self._composite_scores.copy()

    def get_rejection_reasons(self) -> dict[str, str]:
        """Get rejection reasons for all rejected videos."""
        return self._rejection_reasons.copy()
