"""Tests for preprocessing.annotate_matching_binocular_events.

Default tolerance: 20 ms = 0.02 s. Times in seconds.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing import annotate_matching_binocular_events

TOL_MS = 20


def test_reciprocal_pair_within_tolerance(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.1, "eye": "L"},
            {"onset": 1.005, "duration": 0.1, "eye": "R"},
        ]
    )
    out = annotate_matching_binocular_events(df, tolerance_ms=TOL_MS)
    assert out.loc[0, "has_match"] and out.loc[1, "has_match"]
    # reciprocal: each points at the other's index
    assert out.loc[0, "matching_event"] == 1
    assert out.loc[1, "matching_event"] == 0
    # score = |onset diff| + |end diff| = 0.005 + 0.005
    assert out.loc[0, "matching_score"] == pytest.approx(0.01)
    assert out.loc[1, "matching_score"] == pytest.approx(0.01)


def test_no_match_outside_tolerance(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.1, "eye": "L"},
            {"onset": 1.03, "duration": 0.1, "eye": "R"},  # onset diff 30 ms > 20
        ]
    )
    out = annotate_matching_binocular_events(df, tolerance_ms=TOL_MS)
    assert not out["has_match"].any()
    assert out["matching_event"].isna().all()
    assert out["matching_score"].isna().all()


def test_no_match_when_offset_outside_tolerance(make_events):
    # onsets 5 ms apart (inside) but ends 100 ms apart (outside)
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.1, "eye": "L"},
            {"onset": 1.005, "duration": 0.2, "eye": "R"},
        ]
    )
    out = annotate_matching_binocular_events(df, tolerance_ms=TOL_MS)
    assert not out["has_match"].any()


def test_best_candidate_selected_and_non_reciprocal_reset(make_events):
    # L is matched by two R candidates; the best (lowest score) wins,
    # the loser's match is non-reciprocal and gets reset to has_match=False.
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.1, "eye": "L"},
            {"onset": 1.005, "duration": 0.1, "eye": "R"},  # score 0.010 -> winner
            {"onset": 1.008, "duration": 0.1, "eye": "R"},  # score 0.016 -> reset
        ]
    )
    out = annotate_matching_binocular_events(df, tolerance_ms=TOL_MS)
    assert out.loc[0, "matching_event"] == 1
    assert out.loc[0, "matching_score"] == pytest.approx(0.01)
    assert out.loc[1, "has_match"]  # reciprocal with L
    assert not out.loc[2, "has_match"]
    assert pd.isna(out.loc[2, "matching_event"])


def test_select_candidate_false_keeps_all_candidates(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.1, "eye": "L"},
            {"onset": 1.005, "duration": 0.1, "eye": "R"},
            {"onset": 1.008, "duration": 0.1, "eye": "R"},
        ]
    )
    out = annotate_matching_binocular_events(df, tolerance_ms=TOL_MS, select_candidate=False)
    assert out.loc[0, "matching_event"] == [1, 2]
    assert out.loc[0, "matching_score"] == pytest.approx([0.01, 0.016])
    assert out["has_match"].all()  # no reciprocity reset in this mode


def test_no_match_across_trial_types(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.1, "trial_type": "saccade", "eye": "L"},
            {"onset": 1.0, "duration": 0.1, "trial_type": "fixation", "eye": "R"},
        ]
    )
    out = annotate_matching_binocular_events(df, tolerance_ms=TOL_MS)
    assert not out["has_match"].any()


def test_no_match_within_same_eye(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.1, "eye": "L"},
            {"onset": 1.005, "duration": 0.1, "eye": "L"},
        ]
    )
    out = annotate_matching_binocular_events(df, tolerance_ms=TOL_MS)
    assert not out["has_match"].any()


def test_output_sorted_by_onset(make_events):
    df = make_events(
        [
            {"onset": 2.0, "duration": 0.1, "eye": "L"},
            {"onset": 1.0, "duration": 0.1, "eye": "R"},
        ]
    )
    out = annotate_matching_binocular_events(df, tolerance_ms=TOL_MS)
    assert out["onset"].is_monotonic_increasing


def test_negative_tolerance_raises(make_events):
    df = make_events([{"onset": 1.0, "eye": "L"}])
    with pytest.raises(ValueError):
        annotate_matching_binocular_events(df, tolerance_ms=-1)


@pytest.mark.xfail(
    reason="audit #3: type() check rejects numpy floats (np.float64 raises ValueError)",
    strict=True,
)
def test_numpy_float_tolerance_accepted(make_events):
    df = make_events(
        [
            {"onset": 1.0, "duration": 0.1, "eye": "L"},
            {"onset": 1.005, "duration": 0.1, "eye": "R"},
        ]
    )
    out = annotate_matching_binocular_events(df, tolerance_ms=np.float64(TOL_MS))
    assert out["has_match"].all()
