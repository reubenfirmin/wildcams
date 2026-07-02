"""Unit tests for the temporal-continuity check in pipeline.scoring."""

import pytest

from config import ConfigurationManager
from pipeline.scoring import check_temporal_continuity


@pytest.fixture
def config():
    manager = ConfigurationManager()
    manager.load_from_cli_args([], include_motion=True)
    return manager.get_processing_config()


def test_temporal_continuity_disabled_by_default_always_passes(config):
    # Even a wildly discontinuous set passes when the check is off (default behavior).
    assert check_temporal_continuity([0, 300, 600], fps=30.0, config=config) is True


def test_temporal_continuity_single_frame_is_continuous(config):
    config.enable_temporal_continuity_check = True
    assert check_temporal_continuity([42], fps=30.0, config=config) is True


def test_temporal_continuity_small_gap_passes(config):
    config.enable_temporal_continuity_check = True
    config.temporal_continuity_max_gap_seconds = 1.0
    # Frames 0,15,30 at 30fps -> gaps of 0.5s, under the 1.0s limit.
    assert check_temporal_continuity([0, 15, 30], fps=30.0, config=config) is True


def test_temporal_continuity_large_gap_fails(config):
    config.enable_temporal_continuity_check = True
    config.temporal_continuity_max_gap_seconds = 1.0
    # Frames 0 and 60 at 30fps -> a 2.0s gap, over the 1.0s limit.
    assert check_temporal_continuity([0, 60], fps=30.0, config=config) is False


def test_temporal_continuity_unordered_input(config):
    config.enable_temporal_continuity_check = True
    config.temporal_continuity_max_gap_seconds = 1.0
    # Same frames as the passing case but shuffled; must be order-independent.
    assert check_temporal_continuity([30, 0, 15], fps=30.0, config=config) is True
