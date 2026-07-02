"""Unit tests for Step-3 scoring helpers that are pure enough to test in isolation."""

import pytest

from config import ConfigurationManager
from pipeline.fullframe_validator import FullFrameValidator


@pytest.fixture
def config():
    manager = ConfigurationManager()
    manager.load_from_cli_args([], include_motion=True)
    return manager.get_processing_config()


@pytest.fixture
def validator():
    # The temporal-continuity helper does not touch the ML ensemble.
    return FullFrameValidator(ml_ensemble=None)


def test_temporal_continuity_disabled_by_default_always_passes(validator, config):
    # Even a wildly discontinuous set passes when the check is off (default behavior).
    assert validator._check_temporal_continuity([0, 300, 600], fps=30.0, config=config) is True


def test_temporal_continuity_single_frame_is_continuous(validator, config):
    config.enable_temporal_continuity_check = True
    assert validator._check_temporal_continuity([42], fps=30.0, config=config) is True


def test_temporal_continuity_small_gap_passes(validator, config):
    config.enable_temporal_continuity_check = True
    config.temporal_continuity_max_gap_seconds = 1.0
    # Frames 0,15,30 at 30fps -> gaps of 0.5s, under the 1.0s limit.
    assert validator._check_temporal_continuity([0, 15, 30], fps=30.0, config=config) is True


def test_temporal_continuity_large_gap_fails(validator, config):
    config.enable_temporal_continuity_check = True
    config.temporal_continuity_max_gap_seconds = 1.0
    # Frames 0 and 60 at 30fps -> a 2.0s gap, over the 1.0s limit.
    assert validator._check_temporal_continuity([0, 60], fps=30.0, config=config) is False


def test_temporal_continuity_unordered_input(validator, config):
    config.enable_temporal_continuity_check = True
    config.temporal_continuity_max_gap_seconds = 1.0
    # Same frames as the passing case but shuffled; must be order-independent.
    assert validator._check_temporal_continuity([30, 0, 15], fps=30.0, config=config) is True
