"""
preprocessing.py: Eye-tracking data loading and preprocessing functions.

Overview:

Load Input file(s):
    load_subject_tsv
Preprocessing:
    annotate_matching_binocular_events
    annotate_blink_saccades_in_df
    merge_fixation_candidates
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


# Preprocessing: Merge fixation candidates
# =============================================================================
def merge_fixation_candidates(events, a_min=A_MIN, binocularity_criterion = True, binocular_matching_tolerance = 20, temporal_merge_threshold = None, spatial_merge_threshold = None):
    """
    Adapted version of stage 1 of the selection procedure of Hooge et al. (2022) (using only an amplitude criterion).
    Drops saccades that are smaller than `a_min`, then merges the fixations left adjacent (if they
    are close enough in time and space).

    Args:
        events (pd.DataFrame): events with columns onset, end_time, duration,
            trial_type, eye, fix_avg_x, fix_avg_y, fix_avg_pupil_size,
            sacc_visual_angle. Times in seconds, positions in pixels.
        a_min (float): minimal saccade amplitude in degrees.
        binocularity_criterion (bool, default: True): whether to exclude saccades that do
            not have a matching saccade in the opposite eye.
        binocular_matching_tolerance (float, default: 20): Only used if binocularity_criterion for saccade selection is used.
            Maximum permitted difference, in milliseconds, for both onset and end times of matching binocular events. Must be non-negative.
        temporal_merge_threshold (float): maximal time gap between
             two fixations for them to merge, in MILLISECONDS.
        spatial_merge_threshold (float): Maximal Euclidean distance (in pixels) between two fixations for them to merge. 

    Returns:
        pd.DataFrame: events after the drop and merge, sorted by onset.
            Saccades below both thresholds are removed; merged fixations carry
            the first onset, the last end_time and duration-weighted positions.

    References:
        Hooge, I. T. C., Niehorster, D. C., Nystrom, M., Andersson, R., &
        Hessels, R. S. (2022). Fixation classification: How to merge and select
        fixation candidates. Behavior Research Methods, 54(6), 2765-2776.
        https://doi.org/10.3758/s13428-021-01723-1
    """

    # Check whether temporal and spatial merging thresholds were provided
    if temporal_merge_threshold is None or spatial_merge_threshold is None:
        raise ValueError("Both temporal_merge_threshold (ms) and spatial_merge_threshold (px) must be provided.")
    
    # Convert from milliseconds to seconds 
    temporal_merge_threshold /= 1000.0  

    # Sort by eye and onset to prevent cross-eye merging
    events = events.sort_values(["eye", "onset"]).reset_index(drop=True)

    # Annotate binocular matching events
    if binocularity_criterion:
        if not "has_match" in events.columns:
            logger.info(f"Annotate events whether they have a matching event in the opposite eye (with an absolute onset/offset time diff tolerance of {binocular_matching_tolerance} ms).")
            events = annotate_matching_binocular_events(events, tolerance_ms = binocular_matching_tolerance, select_candidate=True, reset_non_reciprocal_matches=True)
             

    n_before = (events["trial_type"] == "saccade").sum()

    # Drop saccades that are below amplitude threshold (or not binocular)
    if binocularity_criterion:
        events = events[
            ~(
                (events["trial_type"] == "saccade")
                & ((events["sacc_visual_angle"] < a_min) | (events["has_match"] == False))
            )
        ].reset_index(drop=True)

    else:
        events = events[
            ~(
                (events["trial_type"] == "saccade")
                & (events["sacc_visual_angle"] < a_min)
            )
        ].reset_index(drop=True)

    n_dropped = n_before - (events["trial_type"] == "saccade").sum()
    logger.info(
        f"Dropped {n_dropped} saccades (amplitude < {a_min}° or not binocular (if using binocularity criterion))"
    )

    rows_to_keep = []
    i = 0

    # Merge loop: finds and merge consecutive fixations
    while i < len(events):
        current_row = events.iloc[i].copy()

        if (
            i < len(events) - 1
            and events.iloc[i]["trial_type"] == "fixation" # if the current row is a fixation
            and events.iloc[i + 1]["trial_type"] == "fixation" # if the next row is also a fixation
            and events.iloc[i]["eye"] == events.iloc[i + 1]["eye"] # if the current row and the next row are from the same eye
            and events.iloc[i+1]["onset"]-events.iloc[i]["end_time"] < temporal_merge_threshold # if the gap between the current row and the next row is less than the merge threshold (temporal & spatial)
            and euclidean_distance(events.iloc[i+1], events.iloc[i]) < spatial_merge_threshold
        ):
            j = i + 1
            duration_sum = current_row["duration"]
            while (
                j < len(events)
                and events.iloc[j]["trial_type"] == "fixation"
                and events.iloc[j]["eye"] == current_row["eye"]
                and events.iloc[j]["onset"]-events.iloc[j-1]["end_time"] < temporal_merge_threshold # if the gap between the current row and the next row is less than the merge threshold
                and euclidean_distance(events.iloc[j], events.iloc[i]) < spatial_merge_threshold # check if the distance between the current fixation and the first fixation in the cluster is below threshold
            ):
                next_row = events.iloc[j]
                # full span from the first onset to the last end_time and therefore includes the removed saccades
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
    merged_events.drop(columns=["has_match", "matching_event", "matching_score"], inplace=True) # Drop columns related to binocularity criterion
    merged_events = merged_events.sort_values(["onset", "eye"]).reset_index(drop=True)

    return merged_events


# Preprocessing: Calculate (absolute) saccade angle
# =============================================================================
def calculate_saccade_angle_in_df(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the angle of each saccade in radians, based on the start and end positions.

    Args:
        events_df (pd.DataFrame): DataFrame containing saccade events with columns 'sacc_start_x', 'sacc_start_y', 'sacc_end_x', 'sacc_end_y'.

    Returns:
        pd.DataFrame: DataFrame with an additional 'sacc_angle' column containing the calculated angles in radians [0, 2*pi), measured
            counter-clockwise from rightwards. NaN where undefined.
    """
    df = events_df.copy()
    coord_cols = ["sacc_start_x", "sacc_start_y", "sacc_end_x", "sacc_end_y"]

    dx = df['sacc_end_x'] - df['sacc_start_x']
    dy = df['sacc_end_y'] - df['sacc_start_y']
    # Flip the sign because the ET origin is top-left, so screen y goes downward.
    dy = -dy

    valid = (
        (df["trial_type"] == "saccade")
        & df[coord_cols].notna().all(axis=1)
        & ((dx != 0) | (dy != 0))
        )

    logger.info(
        f"Calculating saccade angles: {valid.sum()}/{len(df[df['trial_type'] == 'saccade'])} valid saccades. Non-valid saccades will have NaN in the 'sacc_angle' column."
    )
    df['sacc_angle'] = np.where(valid, np.arctan2(dy, dx) % (2 * np.pi), np.nan)
    return df

# Preprocessing: Annotate blink saccades
# =============================================================================

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


# Preprocessing: Annotate matching binocular events
# =============================================================================

def annotate_matching_binocular_events(events_df, tolerance_ms = 20, select_candidate=True, reset_non_reciprocal_matches=True):
    """Annotate events whether they have a matching event in the opposite eye.

    A candidate must have the same ``trial_type`` and an absolute difference in
    both onset and end time no greater than ``tolerance_ms`` milliseconds. If multiple
    candidates qualify and ``select_candidate=True``, the candidate with the
    smallest sum of absolute onset and end-time differences is selected. If
    ``select_candidate=False``, all candidates and their scores are retained
    when multiple candidates qualify.
    

    Parameters
    ----------
    events_df : pandas.DataFrame
        Events to annotate. Must contain ``onset``, ``end_time``,
        ``trial_type``, and ``eye`` columns. Eye values must be ``"L"`` or
        ``"R"``.
    tolerance_ms : float, default=20
        Maximum permitted difference, in milliseconds, for both onset and end
        times. Must be non-negative.
    select_candidate: bool, default=True
        Whether to select the lowest-score candidate when multiple candidates qualify. 
        If ``False``, all such candidates and their scores are retained.
    reset_non_reciprocal_matches: bool, default=True
         When ``select_candidate=True``, whether to find events that do not have a reciprocal match
         and reset them to `has_match = False`. This parameter has no effect when ``select_candidate=False``.

    Returns
    -------
    pandas.DataFrame
        A copy of the input, sorted by ``onset`` with a reset index, plus:
        ``has_match`` (whether a match is retained), ``matching_event``, and
        ``matching_score``. When ``select_candidate=True`` and
        ``reset_non_reciprocal_matches=True``, ``has_match=False`` means either
        that no candidate was found or that the selected match was not reciprocal;
        in the latter case, ``matching_event`` and ``matching_score`` are also
        cleared. A single selected match is represented by its positional index
        in the returned DataFrame and a scalar score. When
        ``select_candidate=False`` and multiple candidates qualify, these columns
        contain lists of indices and scores, respectively; the list scores are
        rounded to four decimal places. Unmatched events have ``None`` for
        ``matching_event`` and ``NaN`` for ``matching_score``.
    """

    if type(tolerance_ms) not in [int, float] or tolerance_ms < 0:
        raise ValueError("Tolerance must be a non-negative value (int or float) in milliseconds.")
    else:
        tolerance = tolerance_ms / 1000 # convert to seconds

    # Copy original data frame and sort by onset
    output_df = events_df.copy().sort_values("onset", kind="stable").reset_index(drop=True)

    other_eye = {"L": "R", "R": "L"}

    # Extract event onset and offsets as numpy arrays
    onset = output_df["onset"].to_numpy(dtype=float)
    end = output_df["end_time"].to_numpy(dtype=float)

    # Create lookup dictionary grouped by trial_type and eye with only onset and offset
    # to simplify finding matching event candidates and avoiding repeated df querying
    grouped_df = output_df.groupby(["trial_type", "eye"], sort=False)

    groups = {}
    for key, group in grouped_df:
        pos = group.index.to_numpy()
        groups[key] = (onset[pos], end[pos], pos)

    n = len(output_df)
    output_df["has_match"] = pd.Series([False] * n, index=output_df.index, dtype=bool)
    output_df["matching_event"] = pd.Series([None] * n, index=output_df.index, dtype=object)
    output_df["matching_score"] = output_df["matching_score"] = pd.Series([np.nan] * n, index=output_df.index, dtype=object)
    
    # Loop over all events and look for matching candidates in the other eye
    for i, row in output_df.iterrows():

        # Extract relevant onset and offsets from the groups dictionary
        eye = other_eye.get(row["eye"])
        events_subset = groups.get((row["trial_type"], eye)) if eye else None
        if events_subset is None:
            continue

        # Search candidates that have an onset within the onset tolerance region
        onsets_subset, offsets_subset, event_indices = events_subset
        lo = np.searchsorted(onsets_subset, onset[i] - tolerance, side="left")
        hi = np.searchsorted(onsets_subset, onset[i] + tolerance, side="right")
        candidates = event_indices[lo:hi]

        # Check whether the candidates also have their offsets within the offset tolerance region
        candidates = candidates[np.abs(end[candidates] - end[i]) <= tolerance]

        # Calculate the onset and offset differences per candidate and take the sum
        scores = np.abs(onset[candidates] - onset[i]) + np.abs(end[candidates] - end[i])

        # If there is no matching candidate continue with the next event
        if len(candidates) == 0:
            continue  
            
        else:
            output_df.loc[i, "has_match"] = True
            
        # If there is only one matching candidate set has_match to True and save the candidate index and matching score
        if len(candidates) == 1:
            output_df.at[i, "matching_event"] = int(candidates[0])
            output_df.at[i, "matching_score"] = float(scores[0])

        elif len(candidates) > 1:

            if select_candidate:
                # Find candidate with lowest score i.e. lowest onset/offset differences
                winner_idx = np.argmin(scores)
                winner_candidate = int(candidates[winner_idx])
                winner_score = float(scores[winner_idx])
    
                output_df.at[i, "matching_event"] = winner_candidate
                output_df.at[i, "matching_score"] = winner_score

            else:
                output_df.at[i, "matching_event"] = candidates.tolist()
                output_df.at[i, "matching_score"] = np.round(scores,4).tolist()

    # Find matching candidates that are not reciprocal and reset them to has_match = False
    if select_candidate and reset_non_reciprocal_matches:
        selected = output_df["matching_event"].tolist()
        reciprocal = np.zeros(n, dtype=bool)
    
        for i, j in enumerate(selected):
            if j is not None and selected[int(j)] == i:
                reciprocal[i] = True
    
        output_df.loc[~reciprocal, "has_match"] = False
        output_df.loc[~reciprocal, "matching_event"] = None
        output_df.loc[~reciprocal, "matching_score"] = np.nan
            
    return output_df


# Helper functions
# =============================================================================

def deg_to_px(size_deg, screen_distance_mm, px_per_mm):
    """Convert a visual angle in degrees to pixels based on screen distance and pixel density."""
    size_rad = np.radians(size_deg)
    size_mm = np.tan(size_rad) * screen_distance_mm
    size_px = size_mm * px_per_mm

    return size_px

def euclidean_distance(fix_1, fix_2):
    """Calculate the Euclidean distance between two fixation points given as dataframe rows."""
    p1 = np.array([fix_1['fix_avg_x'], fix_1['fix_avg_y']])
    p2 = np.array([fix_2['fix_avg_x'], fix_2['fix_avg_y']])
    return np.linalg.norm(p1 - p2)