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

TBD maybe refactored for events.tsv ?!
    compute_saccade_amplitude()
    compute_saccade_amplitude_from_radians()
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
    filepath: Path, subject_id: str, window_ms: float = BLINK_WINDOW_MS
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
        f"Loaded {len(df)} events. Annotating blink saccades (window={window_ms} ms)..."
    )
    df = annotate_blink_saccades_in_df(df, window_ms)

    return df


# Preprocessing
# =============================================================================
def merge_fixation_candidates(events, a_min=A_MIN, merge_threshold=None):
    """
    # TBD: from Hooge et al. (2022)
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
            and np.hypot(events.iloc[i+1]["fix_avg_x"], events.iloc[i+1]["fix_avg_y"]) < a_min  # QUESTION: "only Fixations that are close to each other in time and space are combined."? # TBD: explore
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
    events_df: pd.DataFrame, window_ms: float
) -> pd.DataFrame:
    """
    Flag saccades as near a blink if the blink START or END falls within
    the saccade interval expanded by ±window_ms.
    Adds:
      - blink_saccade (bool) column to saccade rows; False for all other event types.
    """
    events = events_df.copy()
    w = window_ms / 1000.0  # convert to seconds

    saccades_mask = events["trial_type"] == "saccade"
    blinks = events[events["trial_type"] == "blink"]

    # sort by onset time
    b = blinks[["onset", "end_time"]].to_numpy(float)
    b = b[np.argsort(b[:, 0])]

    s = events.loc[saccades_mask, ["onset", "end_time"]].to_numpy(float)

    blink_saccades = []
    for sac_start, sac_end in s:
        win_start = sac_start - w
        win_end = sac_end + w
        near = False
        for blink_start, blink_end in b:
            if blink_start > win_end:
                break
            if (win_start <= blink_start <= win_end) or (
                win_start <= blink_end <= win_end
            ):
                near = True
                break
        blink_saccades.append(near)

    events["blink_saccade"] = False
    events.loc[saccades_mask, "blink_saccade"] = blink_saccades

    n_flagged = sum(blink_saccades)
    logger.info(
        f"Blink saccade annotation: {n_flagged}/{len(blink_saccades)} saccades flagged (window=±{window_ms} ms)"
    )

    return events
