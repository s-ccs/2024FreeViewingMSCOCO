"""Tests for preprocessing.merge_fixation_candidates."""

import numpy as np
import pytest

from preprocessing import merge_fixation_candidates


def _sacc(onset, duration, eye="L", angle=5.0):
    return {
        "onset": onset,
        "duration": duration,
        "trial_type": "saccade",
        "eye": eye,
        "sacc_visual_angle": angle,
    }


def _fix(onset, duration, eye="L", x=500.0, y=400.0, pupil=1000.0):
    return {
        "onset": onset,
        "duration": duration,
        "trial_type": "fixation",
        "eye": eye,
        "fix_avg_x": x,
        "fix_avg_y": y,
        "fix_avg_pupil_size": pupil,
    }


def test_drops_small_saccade_and_merges_across_it(make_events):
    df = make_events(
        [
            _fix(1.0, 0.1, x=100.0, y=100.0, pupil=800.0),
            _sacc(1.1, 0.02, angle=0.1),  # < a_min -> dropped
            _fix(1.12, 0.1, x=200.0, y=300.0, pupil=1200.0),
        ]
    )
    out = merge_fixation_candidates(
        df, a_min=0.5, binocularity_criterion=False, temporal_merge_threshold=50
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["trial_type"] == "fixation"
    assert row["onset"] == pytest.approx(1.0)
    assert row["end_time"] == pytest.approx(1.22)
    assert row["duration"] == pytest.approx(0.22)  # spans the removed saccade
    # duration-weighted averages: (100*0.1 + 200*0.1) / 0.2
    assert row["fix_avg_x"] == pytest.approx(150.0)
    assert row["fix_avg_y"] == pytest.approx(200.0)
    assert row["fix_avg_pupil_size"] == pytest.approx(1000.0)


def test_three_way_merge_weighted_average(make_events):
    df = make_events(
        [
            _fix(1.0, 0.1, x=100.0),
            _sacc(1.1, 0.02, angle=0.1),
            _fix(1.12, 0.1, x=200.0),
            _sacc(1.22, 0.02, angle=0.1),
            _fix(1.24, 0.1, x=400.0),
        ]
    )
    out = merge_fixation_candidates(
        df, a_min=0.5, binocularity_criterion=False, temporal_merge_threshold=50
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["onset"] == pytest.approx(1.0)
    assert row["end_time"] == pytest.approx(1.34)
    assert row["fix_avg_x"] == pytest.approx((100.0 + 200.0 + 400.0) / 3)


def test_no_merge_across_kept_saccade(make_events):
    # a kept saccade (>= a_min) between two fixations keeps them
    # non-adjacent, so no merge happens across it
    df = make_events(
        [
            _fix(1.0, 0.1),
            _sacc(1.1, 0.02, angle=5.0),  # >= a_min -> kept
            _fix(1.12, 0.1),
        ]
    )
    out = merge_fixation_candidates(
        df, a_min=0.5, binocularity_criterion=False, temporal_merge_threshold=50
    )
    assert (out["trial_type"] == "fixation").sum() == 2
    assert (out["trial_type"] == "saccade").sum() == 1


def test_no_merge_across_eyes(make_events):
    df = make_events([_fix(1.0, 0.1, eye="L"), _fix(1.11, 0.1, eye="R")])
    out = merge_fixation_candidates(
        df, a_min=0.5, binocularity_criterion=False, temporal_merge_threshold=50
    )
    assert len(out) == 2


def test_gap_under_threshold_merges(make_events):
    df = make_events([_fix(1.0, 0.1), _fix(1.13, 0.1)])  # gap 30 ms < 50 ms
    out = merge_fixation_candidates(
        df, a_min=0.5, binocularity_criterion=False, temporal_merge_threshold=50
    )
    assert len(out) == 1


def test_gap_over_threshold_does_not_merge(make_events):
    df = make_events([_fix(1.0, 0.1), _fix(1.16, 0.1)])  # gap 60 ms > 50 ms
    out = merge_fixation_candidates(
        df, a_min=0.5, binocularity_criterion=False, temporal_merge_threshold=50
    )
    assert len(out) == 2


def test_binocularity_drops_unmatched_saccades(make_events):
    # a fixation survives, so the (known) empty-result crash is not hit
    df = make_events(
        [
            _fix(1.0, 0.1),
            _sacc(1.1, 0.02, angle=5.0),  # no R partner -> dropped
            _fix(1.2, 0.1),  # gap 80 ms -> no merge either way
        ]
    )
    out = merge_fixation_candidates(df, a_min=0.5, binocularity_criterion=True)
    assert (out["trial_type"] == "saccade").sum() == 0
    assert (out["trial_type"] == "fixation").sum() == 2


def test_binocularity_keeps_matched_saccades(make_events):
    df = make_events(
        [
            _sacc(1.1, 0.02, eye="L", angle=5.0),
            _sacc(1.105, 0.02, eye="R", angle=5.0),  # within 20 ms -> match
        ]
    )
    out = merge_fixation_candidates(df, a_min=0.5, binocularity_criterion=True)
    assert (out["trial_type"] == "saccade").sum() == 2
    assert out["has_match"].all()


@pytest.mark.xfail(
    reason="audit #2: empty result (all rows dropped) crashes on sort_values",
    strict=True,
)
def test_all_events_dropped(make_events):
    df = make_events([_sacc(1.1, 0.02, angle=5.0)])  # no R partner -> dropped
    out = merge_fixation_candidates(df, a_min=0.5, binocularity_criterion=True)
    assert len(out) == 0


def test_output_sorted_by_onset(make_events):
    df = make_events(
        [
            _fix(2.0, 0.1, eye="R"),
            _fix(1.0, 0.1, eye="L"),
            _fix(1.3, 0.1, eye="R"),
        ]
    )
    out = merge_fixation_candidates(
        df, a_min=0.5, binocularity_criterion=False, temporal_merge_threshold=50
    )
    assert out["onset"].is_monotonic_increasing


def test_saccade_below_a_min_merged_span_includes_gap(make_events):
    # fixations separated by a dropped saccade AND a >0 gap still merge
    # when the total gap stays under the threshold; the merged fixation
    # spans from first onset to last end_time
    df = make_events(
        [
            _fix(1.0, 0.1),
            _sacc(1.1, 0.02, angle=0.1),
            _fix(1.14, 0.1),  # gap 20 ms < 50 ms
        ]
    )
    out = merge_fixation_candidates(
        df, a_min=0.5, binocularity_criterion=False, temporal_merge_threshold=50
    )
    assert len(out) == 1
    assert out.iloc[0]["duration"] == pytest.approx(0.24)


def test_real_data_smoke(real_events):
    """Run the full chain on one real subject and check pipeline invariants."""
    from preprocessing import (
        annotate_blink_saccades_in_df,
        calculate_saccade_angle_in_df,
    )

    events = annotate_blink_saccades_in_df(real_events, window_ms=50.0)
    events = calculate_saccade_angle_in_df(events)
    merged = merge_fixation_candidates(
        events,
        a_min=0.5,
        binocularity_criterion=True,
        binocular_matching_tolerance=20,
        temporal_merge_threshold=50,
    )

    assert len(merged) < len(events)
    assert merged["onset"].is_monotonic_increasing

    fs = merged[merged["trial_type"].isin(["fixation", "saccade"])]
    assert (fs["duration"] >= 0).all()
    # merged fixations carry duration == end_time - onset (5 ms slack for
    # rounding in the raw file)
    assert (fs["duration"] - (fs["end_time"] - fs["onset"])).abs().max() < 5e-3

    # every kept saccade is above a_min and has a binocular partner
    sacc = merged[merged["trial_type"] == "saccade"]
    assert len(sacc) > 0
    assert (sacc["sacc_visual_angle"] >= 0.5).all()
    assert sacc["has_match"].all()
