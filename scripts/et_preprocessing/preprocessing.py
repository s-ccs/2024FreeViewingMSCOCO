"""
preprocessing.py: Eye-tracking data loading and preprocessing functions.

Overview:

Load Input file(s):
    load_subject_tsv
Preprocessing:
    merge_events() calls:
        merge_fixation_candidates()
        merge_saccade_candidates()
Helpers:
    find_consecutive_trial_types()
    pixel_to_deg()
    annotate_saccades_near_blinks_in_df()

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
    SCREEN_DIST_M,
    SCREEN_RES,
    SCREEN_SIZE_M,
    SESSION,
    T_MIN_FIX,
    TASK,
)

logger = logging.getLogger(__name__)


# Load Input file(s)
# =============================================================================
def load_subject_tsv(
    folder_path: Path, subject_id: str, window_ms: float = BLINK_WINDOW_MS
) -> pd.DataFrame:
    """
    Args:
        folder_path (Path): Directory containing the subject's TSV file
        subject_id (str): subject number, e.g. "005"
        window_ms (float, optional): Window in ms around each blink for near-blink annotation. Defaults to BLINK_WINDOW_MS.

    Raises:
        FileNotFoundError: _description_

    Returns:
        pd.DataFrame: Events DataFrame
    """
    filename = f"sub-{subject_id}_{SESSION}_task-{TASK}_et_events.tsv"
    filepath = os.path.join(folder_path, filename)

    logger.info(f"Loading events TSV: {filename}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t")
    logger.debug(
        f"Loaded {len(df)} events. Annotating near-blink saccades (window={window_ms} ms)..."
    )
    df = annotate_saccades_near_blinks_in_df(df, window_ms)

    return df


# Preprocessing
# =============================================================================
def merge_events(
    events: pd.DataFrame, a_min: float = A_MIN, t_min_fix: float = T_MIN_FIX
) -> pd.DataFrame:
    """
    Run both merging stages (Hooge et al. 2022):
    1. merge_fixation_candidates(): drop micro-saccades, merge consecutive fixations
    2. merge_saccade_candidates(): drop short fixations, merge consecutive saccades

    Args:
        events (pd.DataFrame): _description_
        a_min (float, optional): Min. saccade amplitude (degrees). Defaults to A_MIN.
        t_min_fix (float, optional): Minimum fixation duration (seconds). Defaults to T_MIN_FIX.

    Returns:
        pd.DataFrame: merged events
    """
    logger.info("Stage 1: merging fixation candidates (dropping micro-saccades)...")
    s1 = merge_fixation_candidates(events, a_min)
    logger.info("Stage 2: merging saccade candidates (dropping short fixations)...")
    s2 = merge_saccade_candidates(s1, t_min_fix)
    logger.info(f"Merging complete. Events before: {len(events)}, after: {len(s2)}")
    return s2


def merge_fixation_candidates(events, a_min=A_MIN):
    """
    Saccades are dropped when they are *both* smaller than `a_min` (deg) *and* shorter than the minimum saccade duration T_min, computed as:
        T_min (ms) = 2.2 * a_min + 27
    Consecutive fixations from the same eye are then merged.

    Parameters:
    events : pandas.DataFrame
        DataFrame with eye-tracking events
    a_min : float
        Minimum saccade amplitude threshold in degrees (default: 1.0).

    Returns:
    pandas.DataFrame
        Events DataFrame after merging stage.
    """
    # Compute minimum saccade duration from the paper's formula
    t_min_sacc = (2.2 * a_min + 27) / 1000.0

    # Sort by eye and onset to prevent cross-eye merging
    events = events.sort_values(["eye", "onset"])

    n_before = (events["trial_type"] == "saccade").sum()
    # Drop saccades that are BOTH below amplitude AND duration threshold
    events = events[
        ~(
            (events["trial_type"] == "saccade")
            & (events["sacc_visual_angle"] < a_min)
            & (events["duration"] < t_min_sacc)
        )
    ].reset_index(drop=True)
    n_dropped = n_before - (events["trial_type"] == "saccade").sum()
    logger.info(
        f"Dropped {n_dropped} micro-saccades (amplitude < {a_min}° and duration < {t_min_sacc*1000:.1f} ms)"
    )

    # Identify naturally consecutive fixation pairs (should not be merged)
    consecutives = find_consecutive_trial_types(events, trial_type="fixation")
    events = events.drop(index=consecutives.index)

    rows_to_keep = []
    i = 0

    while i < len(events):
        current_row = events.iloc[i].copy()

        if (
            i < len(events) - 1
            and events.iloc[i]["trial_type"] == "fixation"
            and events.iloc[i + 1]["trial_type"] == "fixation"
            and events.iloc[i]["eye"] == events.iloc[i + 1]["eye"]
        ):
            j = i + 1
            duration_sum = current_row["duration"]
            while (
                j < len(events)
                and events.iloc[j]["trial_type"] == "fixation"
                and events.iloc[j]["eye"] == current_row["eye"]
            ):
                next_row = events.iloc[j]
                duration_sum += next_row["duration"]  # ← accumulate here
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
    merged_events = pd.concat([merged_events, consecutives])
    merged_events = merged_events.sort_values(["onset"]).reset_index(drop=True)

    return merged_events


def merge_saccade_candidates(events, t_min_fix: float = T_MIN_FIX):
    """
    Fixations are dropped when they are *both* smaller than `t_min_fix`. Consecutive saccades from the same eye are then merged.

    Parameters:
    events : pandas.DataFrame
        DataFrame with eye-tracking trial types
    a_min : float
        Minimum fixation time (default: 0.06).

    Returns:
    pandas.DataFrame
        Events DataFrame after merging stage.
    """
    events = events.sort_values(["eye", "onset"])

    n_before = (events["trial_type"] == "fixation").sum()
    # Drop fixations shorter than t_min_fix
    events = events[
        ~((events["trial_type"] == "fixation") & (events["duration"] < t_min_fix))
    ].reset_index(drop=True)
    n_dropped = n_before - (events["trial_type"] == "fixation").sum()
    logger.info(
        f"Dropped {n_dropped} short fixations (duration < {t_min_fix*1000:.0f} ms)"
    )

    # Identify naturally consecutive saccade pairs (should not be merged)
    consecutives = find_consecutive_trial_types(events, trial_type="saccade")
    events = events.drop(index=consecutives.index)

    rows_to_keep = []
    i = 0

    while i < len(events):
        current_row = events.iloc[i].copy()

        if (
            i < len(events) - 1
            and events.iloc[i]["trial_type"] == "saccade"
            and events.iloc[i + 1]["trial_type"] == "saccade"
            and events.iloc[i]["eye"] == events.iloc[i + 1]["eye"]
        ):

            duration_sum = current_row["duration"]
            peak_velocities = [current_row["peak_velocity"]]
            j = i + 1

            while (
                j < len(events)
                and events.iloc[j]["trial_type"] == "saccade"
                and events.iloc[j]["eye"] == current_row["eye"]
            ):
                next_row = events.iloc[j]
                duration_sum += next_row["duration"]
                peak_velocities.append(next_row["peak_velocity"])
                j += 1

            current_row["sacc_end_x"] = next_row["sacc_end_x"]
            current_row["sacc_end_y"] = next_row["sacc_end_y"]
            current_row["sacc_visual_angle"] = pixel_to_deg(
                current_row["sacc_start_x"],
                current_row["sacc_start_y"],
                next_row["sacc_end_x"],
                next_row["sacc_end_y"],
            )
            current_row["peak_velocity"] = max(peak_velocities)
            current_row["duration"] = duration_sum
            current_row["end_time"] = next_row["end_time"]
            rows_to_keep.append(current_row)
            i = j
        else:
            rows_to_keep.append(current_row)
            i += 1

    merged_events = pd.DataFrame(rows_to_keep)
    merged_events = pd.concat([merged_events, consecutives])
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
        Rows from the original DataFrame where a consecutive pair was found.
    """
    if trial_type not in ("fixation", "saccade"):
        raise ValueError("trial_type must be 'fixation' or 'saccade'")

    events_sorted = events.sort_values(["eye", "onset"]).reset_index(drop=True)
    consecutives = set()

    i = 0
    while i < len(events_sorted) - 1:
        current_row = events_sorted.iloc[i]
        next_row = events_sorted.iloc[i + 1]

        if (
            current_row["trial_type"] == trial_type
            and next_row["trial_type"] == trial_type
            and current_row["eye"] == next_row["eye"]
        ):
            consecutives.add(i)
            consecutives.add(i + 1)

        i += 1

    result = events_sorted.loc[sorted(consecutives)].copy()
    logger.debug(f"Found {len(consecutives) // 2} consecutive {trial_type} pairs.")

    return result


def pixel_to_deg(
    px,
    py,
    qx,
    qy,
    screen_res=SCREEN_RES,
    screen_size_m=SCREEN_SIZE_M,
    screen_dist_m=SCREEN_DIST_M,
):
    """
    Compute visual angle (deg) between two pixel coordinates.
    """
    px_per_m_x = screen_res[0] / screen_size_m[0]
    px_per_m_y = screen_res[1] / screen_size_m[1]

    dx_m = (qx - px) / px_per_m_x
    dy_m = (qy - py) / px_per_m_y
    dist_m = np.sqrt(dx_m**2 + dy_m**2)
    return np.degrees(np.arctan2(dist_m, screen_dist_m))


def annotate_saccades_near_blinks_in_df(
    events_df: pd.DataFrame, window_ms: float
) -> pd.DataFrame:
    """
    Flag saccades as near a blink if the blink START or END falls within
    the saccade interval expanded by ±window_ms.
    Adds:
      - near_blink (bool) column to saccade rows; False for all other event types.
    """
    events = events_df.copy()
    w = window_ms / 1000.0  # convert to seconds

    saccades_mask = events["trial_type"] == "saccade"
    blinks = events[events["trial_type"] == "blink"]

    b = blinks[["onset", "end_time"]].to_numpy(float)
    b = b[np.argsort(b[:, 0])]

    S = events.loc[saccades_mask, ["onset", "end_time"]].to_numpy(float)

    near_blinks = []
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
        near_blinks.append(near)

    events["near_blink"] = False
    events.loc[saccades_mask, "near_blink"] = near_blinks

    n_flagged = sum(near_blinks)
    logger.info(
        f"Near-blink annotation: {n_flagged}/{len(near_blinks)} saccades flagged (window=±{window_ms} ms)"
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
