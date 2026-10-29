"""Tests for the helper functions added in 1215116/62cff5f."""

import numpy as np
import pytest

from preprocessing import deg_to_px, euclidean_distance


def test_deg_to_px_known_value():
    # mirror main.py's density: mean of 1920/552 and 1080/307 px/mm
    px_per_mm = np.mean([1920 / 552, 1080 / 307])
    expected = np.tan(np.radians(1.0)) * 729 * px_per_mm  # ~44.5 px
    assert deg_to_px(1.0, 729, px_per_mm) == pytest.approx(expected, rel=1e-6)


def test_deg_to_px_zero_and_monotonic():
    assert deg_to_px(0.0, 729, 3.5) == 0.0
    assert deg_to_px(2.0, 729, 3.5) > deg_to_px(1.0, 729, 3.5)


def test_euclidean_distance_3_4_5(make_events):
    a = make_events([{"fix_avg_x": 500.0, "fix_avg_y": 400.0}]).iloc[0]
    b = make_events([{"fix_avg_x": 503.0, "fix_avg_y": 404.0}]).iloc[0]
    assert euclidean_distance(a, b) == pytest.approx(5.0)


def test_euclidean_distance_same_point(make_events):
    a = make_events([{"fix_avg_x": 500.0, "fix_avg_y": 400.0}]).iloc[0]
    assert euclidean_distance(a, a) == 0.0


def test_euclidean_distance_with_nan_coords(make_events):
    # NaN coordinates -> NaN distance -> the merge comparison is False,
    # i.e. fixations with missing positions are never merged
    a = make_events([{"fix_avg_x": np.nan}]).iloc[0]
    b = make_events([{"fix_avg_x": 500.0}]).iloc[0]
    assert np.isnan(euclidean_distance(a, b))
