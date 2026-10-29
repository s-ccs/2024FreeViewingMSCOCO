"""Tests for preprocessing.annotate_blink_saccades_in_df."""

import pytest

from preprocessing import annotate_blink_saccades_in_df

W_MS = 50.0  # window in ms


def test_flags_saccade_overlapping_blink(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.02, "trial_type": "saccade", "eye": "L"},
            {"onset": 0.98, "duration": 0.05, "trial_type": "blink", "eye": "L"},
        ]
    )
    out = annotate_blink_saccades_in_df(df, window_ms=W_MS)
    sac = out[out["trial_type"] == "saccade"]
    assert sac["blink_saccade"].all()


def test_flags_saccade_near_blink_within_window(make_events):
    # blink after the saccade, inside the +window expansion (no direct overlap)
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.02, "trial_type": "saccade", "eye": "L"},
            {"onset": 1.05, "duration": 0.01, "trial_type": "blink", "eye": "L"},
        ]
    )
    out = annotate_blink_saccades_in_df(df, window_ms=W_MS)
    assert out[out["trial_type"] == "saccade"]["blink_saccade"].all()


def test_not_flagged_when_blink_outside_window(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.02, "trial_type": "saccade", "eye": "L"},
            {"onset": 1.2, "duration": 0.01, "trial_type": "blink", "eye": "L"},
        ]
    )
    out = annotate_blink_saccades_in_df(df, window_ms=W_MS)
    assert not out[out["trial_type"] == "saccade"]["blink_saccade"].any()


def test_flags_saccade_enclosed_by_blink(make_events):
    # blink encloses the whole expanded saccade window
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.02, "trial_type": "saccade", "eye": "L"},
            {"onset": 0.5, "duration": 1.5, "trial_type": "blink", "eye": "L"},
        ]
    )
    out = annotate_blink_saccades_in_df(df, window_ms=W_MS)
    assert out[out["trial_type"] == "saccade"]["blink_saccade"].all()


def test_match_eye_true_ignores_other_eye(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.02, "trial_type": "saccade", "eye": "L"},
            {"onset": 0.98, "duration": 0.05, "trial_type": "blink", "eye": "R"},
        ]
    )
    out = annotate_blink_saccades_in_df(df, window_ms=W_MS, match_eye=True)
    assert not out[out["trial_type"] == "saccade"]["blink_saccade"].any()


def test_match_eye_false_uses_any_eye(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.02, "trial_type": "saccade", "eye": "L"},
            {"onset": 0.98, "duration": 0.05, "trial_type": "blink", "eye": "R"},
        ]
    )
    out = annotate_blink_saccades_in_df(df, window_ms=W_MS, match_eye=False)
    assert out[out["trial_type"] == "saccade"]["blink_saccade"].all()


@pytest.mark.xfail(
    reason=(
        "audit #5: crashes when the DataFrame contains no saccades — "
        "events.loc[sac_mask, 'blink_saccade'] = [] (empty list) fails "
        "column validation on pandas 3.x"
    ),
    strict=True,
)
def test_non_saccade_rows_get_false_column(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.1, "trial_type": "fixation", "eye": "L"},
            {"onset": 1.0, "duration": 0.05, "trial_type": "blink", "eye": "L"},
        ]
    )
    out = annotate_blink_saccades_in_df(df, window_ms=W_MS)
    assert "blink_saccade" in out.columns
    assert not out[out["trial_type"] == "fixation"]["blink_saccade"].any()
    assert not out[out["trial_type"] == "blink"]["blink_saccade"].any()


def test_no_blinks_no_crash(make_events):
    df = make_events(
        [{"onset": 1.0, "duration": 0.02, "trial_type": "saccade", "eye": "L"}]
    )
    out = annotate_blink_saccades_in_df(df, window_ms=W_MS)
    assert not out["blink_saccade"].any()


def test_input_not_mutated(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.02, "trial_type": "saccade", "eye": "L"},
            {"onset": 0.98, "duration": 0.05, "trial_type": "blink", "eye": "L"},
        ]
    )
    cols_before = list(df.columns)
    annotate_blink_saccades_in_df(df, window_ms=W_MS)
    assert list(df.columns) == cols_before
