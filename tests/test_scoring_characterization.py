"""Characterization tests that pin the scoring core's behavior.

They assert the exact outputs of the composite-score and IoU math so refactors
(the dict->dataclass migration, the split into pipeline/scoring.py) can prove
they changed nothing. A failure here means the math moved.
"""

from types import SimpleNamespace

import pytest

from config import ConfigurationManager
from core.data_types import BoundingBox, Detection, ScoredDetection
from pipeline.scoring import calculate_bbox_overlap, calculate_composite_score


def _scored(boosted_confidence, source, motion_overlap):
    """Build a ScoredDetection carrying only the fields composite scoring reads."""
    return ScoredDetection(
        detection=Detection(
            confidence=boosted_confidence,
            bbox=BoundingBox(0, 0, 1, 1),
            source=source,
            class_name="animal",
        ),
        boosted_confidence=boosted_confidence,
        motion_overlap=motion_overlap,
        overlap_type="explicit",
    )


@pytest.fixture
def config():
    m = ConfigurationManager()
    m.load_from_cli_args([], include_motion=True)
    return m.get_processing_config()


def test_bbox_overlap_iou():
    # Identical boxes -> IoU 1.0; half-overlapping -> 1/3; disjoint -> 0.0
    assert calculate_bbox_overlap([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert calculate_bbox_overlap([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3, abs=1e-6)
    assert calculate_bbox_overlap([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_composite_score_known_input(config):
    track = SimpleNamespace(duration_seconds=4.0)  # scoring only reads duration_seconds
    detections = [
        _scored(boosted_confidence=0.4, source="yolo12x", motion_overlap=0.2),
        _scored(boosted_confidence=0.3, source="rtdetr-l", motion_overlap=0.4),
    ]
    track_frames = [10, 20, 30, 40, 50]
    result = calculate_composite_score(track, detections, track_frames, fps=30.0, config=config)

    assert result.base_score == pytest.approx(0.7)
    assert result.temporal_density == pytest.approx(0.4)
    assert result.temporal_multiplier == pytest.approx(1.4)
    assert result.consensus_models == 2
    assert result.consensus_multiplier == pytest.approx(1.2)
    assert result.motion_alignment == pytest.approx(0.3)
    assert result.motion_multiplier == pytest.approx(0.95)
    assert result.duration_bonus == pytest.approx(0.8 + 4 / 6)
    expected = 0.7 * 1.4 * 1.2 * 0.95 * (0.8 + 4 / 6)
    assert result.final_score == pytest.approx(expected)


def test_composite_score_empty_detections(config):
    result = calculate_composite_score(SimpleNamespace(duration_seconds=1.0), [], [1, 2], fps=30.0, config=config)
    assert result.final_score == 0.0
