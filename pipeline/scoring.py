"""Pure scoring functions for Step-3 full-frame validation.

Extracted from FullFrameValidator so the composite-score math, IoU, and
temporal-continuity checks can be unit-tested in isolation. No side effects.
"""

from core.data_types import CompositeScore, ExtendedTrack, ScoredDetection


def calculate_bbox_overlap(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate IoU (Intersection over Union) between two [x1,y1,x2,y2] boxes."""
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


def check_temporal_continuity(passed_frame_indices: list[int], fps: float, config) -> bool:
    """
    Check that a track's passed frames form a temporally continuous sequence.

    Opt-in via --enable-temporal-continuity-check. When disabled (default), this
    returns True to preserve the pipeline's prior always-pass behavior. When enabled,
    the largest gap between consecutive passed frames must not exceed
    --temporal-continuity-max-gap-seconds (animals do not teleport; a large gap between
    strong detections suggests two unrelated events rather than one continuous track).
    """
    if not config.enable_temporal_continuity_check:
        return True

    if len(passed_frame_indices) <= 1:
        # A single passed frame is trivially continuous.
        return True

    ordered = sorted(passed_frame_indices)
    max_gap_seconds = max((ordered[i + 1] - ordered[i]) / fps for i in range(len(ordered) - 1))
    return max_gap_seconds <= config.temporal_continuity_max_gap_seconds


def calculate_composite_score(
    track: ExtendedTrack, detections: list[ScoredDetection], track_frames: list[int], fps: float, config
) -> CompositeScore:
    """
    Calculate the multi-dimensional composite score combining several strong signals.

    Args:
        track: The extended track (provides duration).
        detections: List of ScoredDetection for this track.
        track_frames: List of sampled frame indices for this track.
        fps: Video frame rate.

    Returns:
        CompositeScore with the final score and its component breakdown.
    """
    if not detections:
        return CompositeScore.empty()

    # Base ensemble score = sum of boosted model confidences.
    base_score = sum(d.boosted_confidence for d in detections)

    # 1. Temporal consistency (how dense are detections across the track).
    track_duration_frames = len(track_frames)
    temporal_density = len(detections) / max(1, track_duration_frames)
    temporal_multiplier = min(config.composite_temporal_multiplier_cap, 1.0 + temporal_density)

    # 2. Multi-model consensus (how many different models contributed).
    unique_models = len({d.detection.source for d in detections})
    consensus_multiplier = 1.0 + (config.composite_consensus_boost_per_model * (unique_models - 1))

    # 3. Motion correlation (how well ML detections align with motion regions).
    avg_motion_overlap = sum(d.motion_overlap for d in detections) / len(detections)
    motion_multiplier = config.composite_motion_multiplier_base + (
        config.composite_motion_multiplier_span * avg_motion_overlap
    )

    # 4. Track duration bonus (longer tracks are more reliable indicators).
    duration_seconds = track.duration_seconds
    duration_bonus = min(
        config.composite_duration_bonus_cap,
        config.composite_duration_bonus_base + (duration_seconds / config.composite_duration_bonus_divisor),
    )

    final_score = base_score * temporal_multiplier * consensus_multiplier * motion_multiplier * duration_bonus

    return CompositeScore(
        final_score=final_score,
        base_score=base_score,
        temporal_density=temporal_density,
        consensus_models=unique_models,
        motion_alignment=avg_motion_overlap,
        duration_seconds=duration_seconds,
        temporal_multiplier=temporal_multiplier,
        consensus_multiplier=consensus_multiplier,
        motion_multiplier=motion_multiplier,
        duration_bonus=duration_bonus,
    )
