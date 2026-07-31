"""
preprocessing.py: Eye-tracking data loading and preprocessing functions.

Overview:

Load Input file(s):
    load_subject_tsv
Preprocessing:
    merge_events() calls:
        merge_fixation_candidates()
Helpers:
    annotate_blink_saccades_in_df()
"""

import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    A_MIN,
    BLINK_WINDOW_MS,
    SESSION,
    TASK,
)

logger = logging.getLogger(__name__)


# Load Input file(s)
# =============================================================================
def load_subject_tsv(
    filepath: Path, subject_id: str,
) -> pd.DataFrame:
    """
    Args:
        folder_path (Path): Directory containing the subject's TSV file
        subject_id (str): subject number, e.g. "005"
        window_ms (float, optional): Window in ms around each blink for blink saccade annotation. Defaults to BLINK_WINDOW_MS.

    Raises:
        FileNotFoundError: _description_

    Returns:
        pd.DataFrame: Events DataFrame
    """
    #filename = f"sub-{subject_id}_{SESSION}_task-{TASK}_et_events.tsv"
    #filepath = os.path.join(folder_path, filename)

    logger.info(f"Loading events TSV: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t")
    logger.debug(
        f"Loaded {len(df)} events."
    )

    return df


# Preprocessing
# =============================================================================
def merge_fixation_candidates(events, a_min=A_MIN, merge_threshold=None):
    """
    Event-selection procedure described by Hooge et al. (2022).
    Saccades are dropped when they are *both* smaller than `a_min` (deg) *and* shorter than the minimum saccade duration T_min, computed as:
        T_min (ms) = 2.2 * a_min + 27
    Consecutive fixations from the same eye are then merged.

    Parameters:
    events : pandas.DataFrame
        DataFrame with eye-tracking events
    a_min : float
        Minimum saccade amplitude threshold in degrees (default: 1.0).
    merge_threshold : float
        Fixations will only be merged if the time between them is below `merge_threshold` (in ms)

    Returns:
    pandas.DataFrame
        Events DataFrame after merging stage.

    Notes on the merged fixation's `duration` and centroid
    ------------------------------------------------------
    These two quantities are deliberately computed over DIFFERENT time bases.
    They are not inconsistent — each answers a different question.

    `duration` is the SPAN of the merged fixation:

        duration = end_time(last fixation) - onset(first fixation)

    i.e. it INCLUDES the gaps left by the removed micro-saccades. This follows
    Hooge et al. (2007), as reported in Hooge et al. (2022, p. 2765): when a
    saccade is removed, the durations of the removed saccade and of the
    preceding and following fixations are summed. The merged object is treated
    as one continuous fixation, so the time spent in the removed saccade counts
    toward it.

    `fix_avg_x`, `fix_avg_y` and `fix_avg_pupil_size` are averaged with weights
    equal to the COMPONENT FIXATION DURATIONS ONLY, excluding those gaps. The
    samples recorded during a removed saccade belong to neither fixation's
    position estimate, so they must not influence the centroid.

    Worked example
    --------------
        Fixation A : onset 1.000  end 1.200  duration 0.200 s   x = 100 px
          (removed saccade spanning 1.200 -> 1.220, a 20 ms gap)
        Fixation B : onset 1.220  end 1.500  duration 0.280 s   x = 130 px

        merged x        = (100*0.200 + 130*0.280) / (0.200 + 0.280)
                        = 117.5 px            <- weights total 0.480 s
        merged duration = 1.500 - 1.000
                        = 0.500 s             <- span, 20 ms longer

    So `duration` (0.500 s) exceeds the sum of the component durations
    (0.480 s) by exactly the removed saccade. That is intended.
    """
    # Compute minimum saccade duration from the paper's formula
    t_min_sac= (2.2 * a_min + 27) / 1000.0

    if merge_threshold is None:
        merge_threshold = t_min_sac
    else:
        merge_threshold /= 1000.0  

    # Sort by eye and onset to prevent cross-eye merging
    events = events.sort_values(["eye", "onset"]).reset_index(drop=True)

    n_before = (events["trial_type"] == "saccade").sum()
    # Drop saccades that are BOTH below amplitude AND duration threshold
    events = events[
        ~(
            (events["trial_type"] == "saccade")
            & (events["sacc_visual_angle"] < a_min)
            & (events["duration"] < t_min_sac)
        )
    ].reset_index(drop=True)
    n_dropped = n_before - (events["trial_type"] == "saccade").sum()
    logger.info(
        f"Dropped {n_dropped} saccades (amplitude < {a_min}° and duration < {t_min_sac*1000:.1f} ms)"
    )

    rows_to_keep = []
    i = 0

    # Merge loop: finds and merge consecutive fixations
    while i < len(events):
        current_row = events.iloc[i].copy()

        if (
            i < len(events) - 1
            and events.iloc[i]["trial_type"] == "fixation"
            and events.iloc[i + 1]["trial_type"] == "fixation"
            and events.iloc[i]["eye"] == events.iloc[i + 1]["eye"]
            and events.iloc[i+1]["onset"]-events.iloc[i]["end_time"] < merge_threshold # QUESTION:avoid merges of implausible long fixations (e.g. across two different valid fixations)
            # QUESTION: "only Fixations that are close to each other in time and space are combined."? # TBD: explore
        ):
            j = i + 1
            duration_sum = current_row["duration"]
            while (
                j < len(events)
                and events.iloc[j]["trial_type"] == "fixation"
                and events.iloc[j]["eye"] == current_row["eye"]
                and events.iloc[j]["onset"]-events.iloc[j-1]["end_time"] < merge_threshold
            ):
                next_row = events.iloc[j]
                duration_sum += next_row["duration"]
                for c in ["fix_avg_x", "fix_avg_y", "fix_avg_pupil_size"]:
                    current_row[c] = (
                        current_row[c] * (duration_sum - next_row["duration"])
                        + next_row[c] * next_row["duration"]
                    ) / duration_sum
                j += 1

            current_row["end_time"] = next_row["end_time"]
            current_row["duration"] = next_row["end_time"] - current_row["onset"]
            rows_to_keep.append(current_row)
            i = j
        else:
            rows_to_keep.append(current_row)
            i += 1

    merged_events = pd.DataFrame(rows_to_keep)
    merged_events = merged_events.sort_values(["onset"]).reset_index(drop=True)

    return merged_events

# Helpers
# ============================================================================
def annotate_blink_saccades_in_df(
    events_df: pd.DataFrame, window_ms: float, match_eye: bool = True
) -> pd.DataFrame:
    """
    Flag a saccade as a blink saccade when a blink OVERLAPS the saccade
    interval expanded by ±window_ms.

    Args:
        events_df: events DataFrame.
        window_ms: half-window in ms around each saccade.
        match_eye: if True, only blinks of the SAME eye can flag a saccade.
                   If False, any blink can (blinks are physiologically binocular).

    Adds to events_df:
        blink_saccade (bool): True for flagged saccades, False everywhere else.
    """
    events = events_df.copy()
    w = window_ms / 1000.0

    sac_mask = events["trial_type"] == "saccade"
    blinks = events.loc[events["trial_type"] == "blink"]

    # One (onset, end_time) array PER EYE, blinks sorted by onset within each eye.
    empty = np.empty((0, 2), dtype=float)
    blinks_by_eye = {}
    for eye_val in sorted(blinks["eye"].unique()):
        sort = blinks.loc[blinks["eye"] == eye_val, ["onset", "end_time"]].to_numpy(dtype=float)
        blinks_by_eye[eye_val] = sort[np.argsort(sort[:, 0], kind="stable")]

    # Fallback pool for match_eye=False, also sorted by onset.
    all_blinks = blinks[["onset", "end_time"]].to_numpy(dtype=float)
    all_blinks = all_blinks[np.argsort(all_blinks[:, 0], kind="stable")]

    blink_saccades = []
    for sac_eye, sac_start, sac_end in events.loc[
        sac_mask, ["eye", "onset", "end_time"]
    ].itertuples(index=False):
        b = blinks_by_eye.get(sac_eye, empty) if match_eye else all_blinks
        win_start, win_end = sac_start - w, sac_end + w

        near = False
        for blink_start, blink_end in b:
            if blink_start > win_end:      # sorted by onset -> no later blink can overlap
                break
            # Interval overlap: also catches a blink that ENCLOSES the whole window.
            if blink_start <= win_end and blink_end >= win_start:
                near = True
                break
        blink_saccades.append(near)

    events["blink_saccade"] = False
    events.loc[sac_mask, "blink_saccade"] = blink_saccades

    logger.info(
        f"Blink saccade annotation: {sum(blink_saccades)}/{len(blink_saccades)} saccades "
        f"flagged (window=±{window_ms} ms, match_eye={match_eye})"
    )
    return events.sort_values(["onset", "eye"], kind="stable").reset_index(drop=True)
