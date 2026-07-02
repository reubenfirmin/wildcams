"""Characterization tests for the postprocessing pure functions.

These pin the current numeric behavior of NMS, IoU, area/confidence filtering,
and coordinate conversion so the dict->Detection migration can prove it changed
nothing. During the migration the INPUTS are rewritten to Detection objects; the
asserted OUTPUTS (which detections survive, the IoU value) must stay identical.
"""

import pytest

from core.data_types import Detection, BoundingBox
from ml.postprocessing import PostprocessingPipeline


@pytest.fixture
def pp():
    return PostprocessingPipeline()


def _det(bbox, confidence, source="yolo12x"):
    return Detection(
        confidence=confidence, bbox=BoundingBox(*bbox), source=source, class_name="animal",
    )


def test_calculate_iou(pp):
    # calculate_iou operates on raw [x1,y1,x2,y2] coordinate lists.
    assert pp.calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert pp.calculate_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3, abs=1e-6)
    assert pp.calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_filter_by_area(pp):
    dets = [
        _det([0, 0, 5, 5], 0.9),      # area 25
        _det([0, 0, 100, 100], 0.9),  # area 10000
    ]
    kept = pp.filter_by_area(dets, min_area=100, max_area=None)
    assert len(kept) == 1
    assert kept[0].bbox == BoundingBox(0, 0, 100, 100)


def test_filter_by_confidence(pp):
    dets = [_det([0, 0, 10, 10], 0.9), _det([0, 0, 10, 10], 0.1)]
    kept = pp.filter_by_confidence(dets, threshold=0.5)
    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_advanced_nms_keeps_higher_confidence(pp):
    # Two heavily-overlapping boxes -> NMS keeps the higher-confidence one.
    dets = [_det([0, 0, 10, 10], 0.9), _det([1, 1, 11, 11], 0.4)]
    kept = pp.apply_advanced_nms(dets, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_advanced_nms_keeps_disjoint(pp):
    dets = [_det([0, 0, 10, 10], 0.9), _det([100, 100, 110, 110], 0.4)]
    kept = pp.apply_advanced_nms(dets, iou_threshold=0.5)
    assert len(kept) == 2


def test_convert_coordinates_scales(pp):
    dets = [_det([10, 20, 30, 40], 0.9)]
    out = pp.convert_coordinates(dets, source_size=(100, 100), target_size=(200, 200))
    assert out[0].bbox == BoundingBox(20, 40, 60, 80)
