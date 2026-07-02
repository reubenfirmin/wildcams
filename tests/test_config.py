"""Tests for CLI-argument parsing into ProcessingConfig.

Guards against regressions in default values (several docs have drifted from the
code defaults) and confirms the config plumbing wires each flag to the right field.
"""

from argparse import Namespace

import pytest

from config import ConfigurationManager, ProcessingConfig


@pytest.fixture
def default_config():
    manager = ConfigurationManager()
    manager.load_from_cli_args([], include_motion=True)
    return manager.get_processing_config()


def test_defaults_load(default_config):
    assert isinstance(default_config, ProcessingConfig)


@pytest.mark.parametrize(
    "field,expected",
    [
        ("confidence_threshold", 0.8),
        ("max_frames_per_video", 20),
        ("composite_motion_threshold", 0.5),
        ("motion_method", "MOG2"),
        ("spatial_overlap_threshold", 0.1),
        ("enable_animal_classification", True),
        ("bioclip_threshold", 0.30),
        ("deepfaune_threshold", 0.62),
        # Composite-score tuning defaults must preserve the prior hardcoded values.
        ("default_fps", 30.0),
        ("consensus_boost_per_detection", 0.1),
        ("composite_temporal_multiplier_cap", 2.0),
        ("composite_consensus_boost_per_model", 0.2),
        ("composite_motion_multiplier_base", 0.5),
        ("composite_motion_multiplier_span", 1.5),
        ("composite_duration_bonus_base", 0.8),
        ("composite_duration_bonus_cap", 1.5),
        ("composite_duration_bonus_divisor", 6.0),
        ("enable_temporal_continuity_check", False),
        ("temporal_continuity_max_gap_seconds", 1.0),
    ],
)
def test_default_values(default_config, field, expected):
    assert getattr(default_config, field) == expected


def test_composite_motion_threshold_is_float_not_truncated():
    """Regression: this flag was once typed int, truncating 0.5 -> 0."""
    manager = ConfigurationManager()
    manager.load_from_cli_args(["--composite-motion-threshold", "0.5"], include_motion=True)
    assert manager.get_processing_config().composite_motion_threshold == 0.5


def test_ensemble_parsed_into_list():
    manager = ConfigurationManager()
    manager.load_from_cli_args(["--ensemble", "yolo12x, yolo12m ,rtdetr-l"], include_motion=True)
    assert manager.get_processing_config().ensemble_models == ["yolo12x", "yolo12m", "rtdetr-l"]


def test_parse_video_filter_splits_indices_and_names():
    manager = ConfigurationManager()
    args = Namespace(videos=["7", "8", "IMG_1234"])
    assert manager.parse_video_filter(args) == [7, 8, "IMG_1234"]


def test_parse_video_filter_none_when_absent():
    manager = ConfigurationManager()
    assert manager.parse_video_filter(Namespace(videos=None)) is None


def test_clustering_flags_removed():
    manager = ConfigurationManager()
    manager.load_from_cli_args([], include_motion=True)
    cfg = manager.get_processing_config()
    assert not hasattr(cfg, "enable_clustering")


def test_frame_pass_threshold_defaults_to_confidence_threshold():
    """Unset, the per-frame gate must mirror --confidence-threshold (prior behavior)."""
    manager = ConfigurationManager()
    manager.load_from_cli_args(["--confidence-threshold", "0.55"], include_motion=True)
    cfg = manager.get_processing_config()
    assert cfg.frame_pass_confidence_threshold == 0.55


def test_frame_pass_threshold_can_be_set_independently():
    manager = ConfigurationManager()
    manager.load_from_cli_args(
        ["--confidence-threshold", "0.8", "--frame-pass-confidence-threshold", "0.3"],
        include_motion=True,
    )
    cfg = manager.get_processing_config()
    assert cfg.confidence_threshold == 0.8
    assert cfg.frame_pass_confidence_threshold == 0.3


def test_get_config_before_load_raises():
    with pytest.raises(ValueError):
        ConfigurationManager().get_processing_config()
