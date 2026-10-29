"""Tests for preprocessing.calculate_saccade_angle_in_df.

Screen coordinates: origin top-left, y grows downward. The function flips
the sign of dy so angles are counter-clockwise from rightwards (math coords).
"""

import numpy as np
import pytest

from preprocessing import calculate_saccade_angle_in_df


def _sacc(start, end):
    return {
        "trial_type": "saccade",
        "sacc_start_x": start[0],
        "sacc_start_y": start[1],
        "sacc_end_x": end[0],
        "sacc_end_y": end[1],
    }


def test_cardinal_angles(make_events):
    df = make_events(
        [
            _sacc((100, 100), (200, 100)),  # rightwards -> 0
            _sacc((100, 100), (100, 50)),  # upwards on screen -> pi/2
            _sacc((100, 100), (0, 100)),  # leftwards -> pi
            _sacc((100, 100), (100, 200)),  # downwards on screen -> 3pi/2 (y flip)
        ]
    )
    out = calculate_saccade_angle_in_df(df)
    a = out["sacc_angle"]
    assert a.iloc[0] == pytest.approx(0.0)
    assert a.iloc[1] == pytest.approx(np.pi / 2)
    assert a.iloc[2] == pytest.approx(np.pi)
    assert a.iloc[3] == pytest.approx(3 * np.pi / 2)


def test_zero_length_saccade_is_nan(make_events):
    df = make_events([_sacc((100, 100), (100, 100))])
    out = calculate_saccade_angle_in_df(df)
    assert np.isnan(out["sacc_angle"].iloc[0])


def test_missing_coords_are_nan(make_events):
    df = make_events(
        [{"trial_type": "saccade", "sacc_end_x": np.nan}]  # start/end defaults are NaN
    )
    out = calculate_saccade_angle_in_df(df)
    assert np.isnan(out["sacc_angle"].iloc[0])


def test_non_saccade_rows_are_nan(make_events):
    df = make_events([_sacc((100, 100), (200, 100))])
    df.loc[0, "trial_type"] = "fixation"
    out = calculate_saccade_angle_in_df(df)
    assert np.isnan(out["sacc_angle"].iloc[0])


def test_input_not_mutated(make_events):
    df = make_events([_sacc((100, 100), (200, 100))])
    cols_before = list(df.columns)
    calculate_saccade_angle_in_df(df)
    assert list(df.columns) == cols_before
