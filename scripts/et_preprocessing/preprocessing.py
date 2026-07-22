"""
preprocessing.py: Eye-tracking data loading and preprocessing functions.

Overview:

Load Input file(s):
    load_subject_tsv
Preprocessing:
    merge_events() calls:
        merge_fixation_candidates()
Helpers:
    find_consecutive_trial_types()
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
def merge_fixation_candidates(events, a_min=A_MIN, merge_threshold=100):
    """
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

    # Convert merge threshold to seconds
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
            and events.iloc[i+1]["onset"]-events.iloc[i]["end_time"] < merge_threshold
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
# =============================================================================
def find_consecutive_trial_types(events, trial_type: str) -> pd.DataFrame:
    """
    Find consecutive events of the same trial_type from the same eye
    with no intervening event of a different type between them.

    Parameters:
    events : pd.DataFrame
        Raw events DataFrame (unsorted is fine, sorted internally).
    trial_type : str
        Event type to check: 'fixation' or 'saccade'.

    Returns:
    pd.DataFrame
        Copy of the original dataframe with an additional column 'consecutive_block'
        which indicates which series of consecutive events an event belongs to or
        is nan in the case that the event is not part of a consecutive series.
    """
    if trial_type not in ("fixation", "saccade"):
        raise ValueError("trial_type must be 'fixation' or 'saccade'")

    events_sorted = events.sort_values(["eye", "onset"]).reset_index() # keep the original index as a column

    block = 0
    i = 0
    while i < len(events_sorted) - 1:

        if (
            events_sorted.loc[i,"trial_type"] == trial_type
            and events_sorted.loc[i+1,"trial_type"] == trial_type
            and events_sorted.loc[i,"eye"] == events_sorted.loc[i+1,"eye"]
        ):
            events_sorted.loc[i,"consecutive_block"] = block

            # Keep adding events to a consecutive block as long as they are of the same event type
            j = 1
            while (
                events_sorted.loc[i+j,"trial_type"] == trial_type
                and events_sorted.loc[i+j,"eye"] == events_sorted.loc[i,"eye"]
            ):
                events_sorted.loc[i+j, "consecutive_block"] = block
                j += 1

            i = i+j
            block += 1
        else: 
            i += 1


    #result = events_sorted.loc[sorted(consecutives)].copy()
    result = events_sorted.sort_values("index").reset_index(drop=True) # sort according to the original index
    result.drop("index", axis=1, inplace=True)

    logger.debug(f"Found {block} blocks of consecutive events of type {trial_type}.")

    return result

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

    b = blinks[["onset", "end_time"]].to_numpy(float)
    b = b[np.argsort(b[:, 0])]

    S = events.loc[saccades_mask, ["onset", "end_time"]].to_numpy(float)

    blink_saccades = []
    for sac_start, sac_end in S:
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


def compute_angle(x1, y1, z1, x2, y2, z2):
    """
    Compute the angle (deg) between two 3D cartesian vectors.
    """
    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])

    denom = np.linalg.norm(p1) * np.linalg.norm(p2)
    if denom == 0:
        return np.nan
    alpha_rad = np.arccos(np.clip(np.dot(p1, p2) / denom, -1.0, 1.0))
    return math.degrees(alpha_rad)
