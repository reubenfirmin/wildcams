"""Unit tests for the pure video-selection utilities in core.functional_utils."""

from pathlib import Path

import pytest

from core.functional_utils import (
    collect_video_files,
    filter_videos_by_criteria,
    validate_video_filter,
)


def _make_videos(tmp_path, names):
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return [tmp_path / name for name in names]


def test_collect_video_files_matches_extensions_and_sorts(tmp_path):
    _make_videos(tmp_path, ["b.mp4", "a.mp4", "c.mov"])
    (tmp_path / "notes.txt").write_text("ignore me")

    result = collect_video_files(tmp_path, extensions={".mp4", ".mov"})

    assert [p.name for p in result] == ["a.mp4", "b.mp4", "c.mov"]


def test_collect_video_files_empty_dir(tmp_path):
    assert collect_video_files(tmp_path, extensions={".mp4"}) == []


def test_filter_videos_by_index_is_one_based():
    videos = [Path("v1.mp4"), Path("v2.mp4"), Path("v3.mp4")]

    result = filter_videos_by_criteria(videos, [1, 3])

    assert [p.name for p in result] == ["v1.mp4", "v3.mp4"]


def test_filter_videos_out_of_range_index_is_dropped():
    videos = [Path("v1.mp4"), Path("v2.mp4")]

    result = filter_videos_by_criteria(videos, [5])

    assert result == []


def test_filter_videos_by_name_substring():
    videos = [Path("IMG_0007.mp4"), Path("IMG_0008.mp4"), Path("other.mp4")]

    result = filter_videos_by_criteria(videos, ["0007", "other"])

    assert [p.name for p in result] == ["IMG_0007.mp4", "other.mp4"]


def test_validate_video_filter_reports_out_of_range_and_missing_name():
    videos = [Path("v1.mp4"), Path("v2.mp4")]

    filtered, warnings = validate_video_filter([1, 9, "nope"], videos)

    assert [p.name for p in filtered] == ["v1.mp4"]
    assert any("9" in w for w in warnings)
    assert any("nope" in w for w in warnings)


def test_validate_video_filter_clean_input_has_no_warnings():
    videos = [Path("v1.mp4"), Path("v2.mp4")]

    filtered, warnings = validate_video_filter([2], videos)

    assert [p.name for p in filtered] == ["v2.mp4"]
    assert warnings == []
