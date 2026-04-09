"""
plotting.py: Eye-tracking visualisation.

Overview:

Preprocessing comparison:
    plot_eye_trace_both_eyes() -> Horizontal eye trace before vs. after merging (Hooge et al. 2022)

Main sequence:
    plot_main_sequence()
    detect_main_sequence_outliers() #TBD maybe delete
    log_main_sequence_outliers() #TBD maybe delete

TBD / not yet refactored for events.tsv:
    plot_fixation_duration()      # TBD
    saccade_amplitude()           # TBD
    saccade_duration()            # TBD
    fixation_frequency()          # TBD
    saccade_angular_histogram()   # TBD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import HuberRegressor

from config import (
    BY_EYE,
    OUT_FILE_FORMAT,
    MS_DROP_NEAR_BLINKS,
    MS_DROP_OUTLIERS,
    MS_OUTLIER_MAD_THRESH,  # TBD: evtl rausnehmen (Code dafür ist in ET_Data_Plotting.ipynb)
    MS_DETECT_MAD_THRESH,
    FIX_DUR_MIN_MS,
    FIX_DUR_MAX_MS,
    FIX_DUR_BIN_W,
    SACC_AMP_MAX_DEG,
    SACC_DUR_MAX_MS,
    ANG_HIST_REFINEMENT,
    ANG_HIST_MICROSACC_MIN_DEG,
    ANG_HIST_BINS_POLAR,
    ANG_HIST_BIN_WIDTH_CART,
)

# Main Sequence
# =============================================================================


def plot_main_sequence(
    events_df: pd.DataFrame,
    out_path: str,
    out_file_format: str = OUT_FILE_FORMAT,
    by_eye: str = BY_EYE,
    title: str = "Main Sequence",
    drop_near_blinks: bool = MS_DROP_NEAR_BLINKS,
):
    """
    Plots main sequence: saccade amplitude vs. peak velocity (log-log).
    Optionally drops near-blink saccades and/or main-sequence outliers. and logs them.

    Args:
        events_df (pd.DataFrame)
        out_path (str): Directory to save the figure. Pass None to skip saving.
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'eps'
        by_eye (str): One of: 'all', 'left', 'right', 'both'
        title (str) : Defaults to  'Main Sequence', gets extended by 'by_eye'(blinked-cleaned|' ') # TBD find better description
        drop_near_blinks (bool, optional): If True, exclude saccades flagged as near a blink. Defaults to config.MS_DROP_NEAR_BLINKS.
    """

    s = events_df[events_df["trial_type"] == "saccade"].copy()

    if drop_near_blinks:
        s = s[s["near_blink"] == False]

    if by_eye != "all":
        eye_map = {"left": "L", "right": "R", "both": "both"}
        s = s[s["eye"] == eye_map[by_eye]]

    base_name = f"{title.lower().replace(' ', '_')}-{by_eye}Eyes" + (
        "_blinkCleaned" if drop_near_blinks else ""
    )

    fig, ax = plt.subplots()

    if by_eye == "all":
        for eye, sub in s.groupby("eye"):
            ax.scatter(
                sub["sacc_visual_angle"], sub["peak_velocity"], s=10, label=str(eye)
            )
        ax.legend(title="Eye")
    else:
        ax.scatter(s["sacc_visual_angle"], s["peak_velocity"], s=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Saccade amplitude (deg)")
    ax.set_ylabel("Peak velocity (deg/s)")
    ax.set_title(
        f"{title} — {by_eye}" + (" (blink-cleaned)" if drop_near_blinks else "")
    )
    plt.show()  # TBD only show when flagged

    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "both": "Binocular only",
    }

    suffix = []
    if drop_near_blinks:
        suffix.append("blink-cleaned")
    if drop_ms_outliers:
        suffix.append(f"outliers-dropped (>{ms_outlier_mad_thresh} MAD)")
    ax.set_title(
        f"{title} — {title_map[by_eye]}" + (f" ({', '.join(suffix)})" if suffix else "")
    )

    fig.tight_layout()
    plt.show()

    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        fname = base_name + ("_msOutliersDropped" if drop_ms_outliers else "")
        fig.savefig(
            os.path.join(out_path, f"{fname}.{out_file_format}"), bbox_inches="tight"
        )


# Saccade Amplitude
# =============================================================================
def plot_fixation_duration(
    events_df: pd.DataFrame,
    out_path: str,
    out_file_format: str = OUT_FILE_FORMAT,
    by_eye: str = BY_EYE,
    min_ms: float = 60,
    max_ms: float = 1000,
    title: str = "Fixation Durations",
):
    """
    Histogram of fixation durations (ms), outliers dropped  (lower bound 60ms, upper bound 1000ms).

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving.
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'eps'
        by_eye (str): One of: 'all', 'left', 'right', 'both'
        min_ms (float, optional): lower bound to drop ultra-short blinks/micro-fixations
        max_ms (float, optional): upper bound to drop implausibly long fixations #TBD ist das überhaupt noch notwendig mit preprocessing?
        title (str, optional): Defaults to 'Fixation Durations'.
    Raises:
        ValueError: HIERHIERHIERHIEHR
        ValueError: _description_
    """
HIERHIERHEIRHEIRHEIRHEIRHEI
    if by_eye not in {"all", "left", "right", "both"}:
        raise ValueError("by_eye must be one of: 'all', 'left', 'right', 'both'")

    # 1) Fixations only, convert seconds → ms
    fix = events_df.loc[
        events_df["trial_type"] == "fixation", ["duration", "eye"]
    ].copy()
    fix = fix.dropna(subset=["duration"])
    fix["duration_ms"] = fix["duration"] * 1000.0

    # 2) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        fix = fix.query(f"eye == @chosen_eye").copy()

    # 3) Filter by plausible duration range
    dur = fix["duration_ms"]
    dur = dur[(dur >= min_ms) & (dur <= max_ms)]
    if dur.empty:
        raise ValueError(
            "No fixation durations post filtering. Check inputs or ranges."
        )

    fig, ax = plt.subplots()
    ax.hist(dur)
    ax.set_xlabel("Fixation duration (ms)")
    ax.set_ylabel("Count")

    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "both": "Binocular only",
    }
    ax.set_title(f"{title} — {title_map[by_eye]}")
    plt.show()

    fig = ax.figure
    fig.tight_layout()
    fig.savefig(
        f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}",
        bbox_inches="tight",
    )
    print(
        f"Plot saved to file '{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}'"
    )
