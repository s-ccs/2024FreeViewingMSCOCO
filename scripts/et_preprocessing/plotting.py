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
    ax.set_title(
        f"{title} — {title_map[by_eye]}" + (f" ({', '.join(suffix)})" if suffix else "")
    )

    fig.tight_layout()
    plt.show()

    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        fig.savefig(
            os.path.join(out_path, f"{base_name}.{out_file_format}"),
            bbox_inches="tight",
        )


# Fixation Duration
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
        out_path (str): Directory to save the figure. Pass None to skip saving. # TBD
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'eps'
        by_eye (str): One of: 'all', 'left', 'right', 'both'
        min_ms (float, optional): lower bound to drop ultra-short blinks/micro-fixations
        max_ms (float, optional): upper bound to drop implausibly long fixations #TBD ist das überhaupt noch notwendig mit preprocessing?
        title (str, optional): Defaults to 'Fixation Durations'.
    Raises:
        ValueError: No fixation durations within min_ms - max_ms found
    """
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
    else:  # tBD: verbatim?
        # Report counts
        dropouts = len(fix["duration_ms"]) - len(dur)
        print(f"Total fixations: {len(fix["duration_ms"])}")
        print(f"Kept fixations ({min_ms}ms <= duration <= {max_ms}ms): {len(dur)}")
        print(
            f"Dropped outliers: {dropouts}, {(dropouts/len(fix["duration_ms"]))*100:.2f}%"
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


# Saccade Amplitude
# =============================================================================
def plot_saccade_amplitude(
    events_df: pd.DataFrame,
    out_path: str,
    out_file_format: str = OUT_FILE_FORMAT,
    by_eye: str = BY_EYE,
    max_deg: float = SACC_AMP_MAX_DEG,
    title: str = "Saccade Amplitude",
):
    """
    Histogram of saccade amplitudes (degrees), outliers dropped (upper bound max_deg).

    Args:
        events_df (pd.DataFrame): Event dataframe containing a 'trial_type' column with saccade events.
        out_path (str): Directory to save the figure. Pass None to skip saving.
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'eps'
        by_eye (str): One of: 'all', 'left', 'right', 'both'
        max_deg (float, optional): Upper bound to drop implausibly large saccade amplitudes.
        title (str, optional): Defaults to 'Saccade Amplitude'.
    """

    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    # 1) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye").copy()

    # 2) Select saccade amplitudes in degrees
    # amplitudes = df["amplitude_deg"].dropna()
    all_amplitudes = s_df["sacc_visual_angle"].dropna()
    if max_deg is not None:
        amplitudes = all_amplitudes[all_amplitudes <= max_deg]

    # Identify dropped outliers
    dropout = len(all_amplitudes[all_amplitudes > max_deg])

    # Report counts
    # TBD: verbatim?
    print(f"Total saccades: {len(all_amplitudes)}")
    print(f"Kept saccades (<={max_deg}°): {len(amplitudes)}")
    print(
        f"Dropped outliers (>{max_deg}°): {dropout}, {(dropout/len(all_amplitudes))*100:.2f}%"
    )

    # 3) Create figure
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.hist(amplitudes, bins=40, edgecolor="black")

    # 4) Labels & title
    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "both": "Binocular only",
    }

    ax.set_title(f"{title} — {title_map[by_eye]}")
    ax.set_xlabel("Saccade amplitude (deg)")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0)

    fig.tight_layout()

    # 5) Save & show
    out_file = (
        f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
    )
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to file '{out_file}'")
    plt.show()
    plt.close(fig)


# Saccade Duration
# =============================================================================
def plot_saccade_duration(
    events_df: pd.DataFrame,
    out_path: str,
    out_file_format: str = OUT_FILE_FORMAT,
    by_eye: str = BY_EYE,
    title: str = "Saccade Duration",
    max_dur: int = SACC_DUR_MAX_MS,
):
    """
    Histogram of fixation frequency (fixations per second).

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving.
        out_file_format (str, optional):  File extension for saving, e.g. 'svg', 'pdf', 'eps'. Defaults to OUT_FILE_FORMAT.
        by_eye (str, optional): One of: 'all', 'left', 'right', 'both'. Defaults to BY_EYE.
        title (str, optional): Defaults to "Saccade Duration".
        max_dur (int, optional): _description_. Defaults to SACC_DUR_MAX_MS.
    """

    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    # 1) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye")

    # 2) Convert duration from seconds to milliseconds
    durations = (s_df["duration"] * 1000).dropna()
    # Report counts
    # TBD: verbatim?
    print(f"Total saccades: {len(durations)}")

    # 3) Drop saccades >120ms
    if max_dur is not None:
        durations = durations[durations <= max_dur]
        durations_copy = durations.copy()
        dropout = len(durations_copy[durations > max_dur])
        # Report counts
        # TBD: verbatim?
        print(f"Kept saccades (<={max_dur}°): {len(durations)}")
        print(
            f"Dropped {(dropout/len(durations))*100:.2f}% samples of duration > {max_dur} milliseconds."
        )

    # 4) Create figure
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.hist(durations, bins=40, edgecolor="black")

    # 5) Labels & title
    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "both": "Binocular only",
    }

    ax.set_title(f"{title} — {title_map[by_eye]}")
    ax.set_xlabel("Saccade duration (ms)")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0)

    fig.tight_layout()

    # 6) Save & show
    # TBD: Overwrite Flag/None/Show
    out_file = (
        f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
    )
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to file '{out_file}'")

    plt.show()
    plt.close(fig)


# Fixation Frequency
# =============================================================================


def plot_fixation_frequency(
    events_df: pd.DataFrame,
    out_path: str,
    out_file_format: str = OUT_FILE_FORMAT,
    by_eye: str = BY_EYE,
    title="Fixation frequency histogram",
):
    """
    Histogram of fixation frequency (fixations per second), binned by second-level onset buckets.

    Args:
        events_df (pd.DataFrame): Event dataframe containing a 'trial_type' column with fixation events.
        out_path (str): Directory to save the figure. Pass None to skip saving.
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'eps'.
        by_eye (str): One of: 'all', 'left', 'right', 'both'
        title (str, optional): Defaults to 'Fixation frequency histogram'.
    """
    f_df = events_df[events_df["trial_type"] == "fixation"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        f_df = f_df.query("eye == @chosen_eye").copy()

    f_df["sec"] = f_df["onset"].astype(float).floordiv(1).astype(int)
    fix_per_sec = f_df.groupby("sec").size()

    plt.figure()
    plt.hist(
        fix_per_sec.values,
        bins=np.arange(fix_per_sec.max() + 2) - 0.3,
        width=0.6,
    )
    plt.xlim(left=-0.3)
    plt.xlabel("Fixation Frequency (per Second)")
    plt.ylabel("Count")
    plt.title(title)

    # TBD: Overwrite Flag/None/Show
    out_file = (
        f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
    )
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to '{out_file}'")
    plt.show()


title = "Saccade Direction Histogram"
cart_title = "Cartesian Angular Histogram"
refinement = False  # refinements: weight by saccade amplitude, excluding microsaccades or artifacts


def plot_saccade_angles(
    events_df: pd.DataFrame,
    out_path: str,
    out_file_format: str = OUT_FILE_FORMAT,
    by_eye: str = BY_EYE,
    title: str = "Saccade Direction Histogram",
    style: str = None,
):
    """
    Histogram of saccade directions (degrees), shown as a polar rose plot, a Cartesian bar histogram, or both.

    Args:
        events_df (pd.DataFrame):
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'eps'
        by_eye (str): One of: 'all', 'left', 'right', 'both'
        title (str, optional): Defaults to 'Saccade Direction Histogram'.
        style (str, optional): One of: 'polar', 'cartesian', or None (produces both). Defaults to None.
    """

    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    # 1) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye").copy()

    # 2) Compute saccade angles
    dx = s_df["sacc_end_x"] - s_df["sacc_start_x"]
    dy = s_df["sacc_end_y"] - s_df["sacc_start_y"]

    # Angle in radians, then degrees
    angles_rad = np.arctan2(dy, dx)
    angles_deg = np.degrees(angles_rad)

    # Optional: map to [0, 360)
    angles_deg = (angles_deg + 360) % 360

    if style in ["polar", None]:
        # Convert degrees back to radians for polar plotting
        angles_rad = np.deg2rad(angles_deg)

        # 4) create Figure
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)

        ax.hist(angles_rad, bins=36, edgecolor="black")  # 10° bins

        # Configure axes
        ax.set_theta_zero_location("E")  # 0° to the right
        ax.set_theta_direction(1)  # counter-clockwise
        ax.set_title(title)
        plt.tight_layout()

        # 5) Save & show
        # TBD overwrite/show flag
        out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
        plt.savefig(out_file, bbox_inches="tight")
        print(f"Plot saved to file '{out_file}'")

        plt.show()

    if style in ["cartesian", None]:
        # 4) Create Figure 2: Cartesian Angular Histogramm
        plt.hist(angles_deg, bins=np.arange(0, 361, 10), edgecolor="black")
        plt.xlabel("Saccade direction (deg)")
        plt.ylabel("Count")
        plt.title(cart_title)

        # 5) Save & show
        # TBD overwrite/show flag
        out_file = f"{out_path}/{cart_title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
        plt.savefig(out_file, bbox_inches="tight")
        print(f"Plot saved to file '{out_file}'")

        plt.show()
