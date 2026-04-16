"""
plotting.py: Eye-tracking visualisation.

Overview:
---------

Preprocessing comparison:
    plot_eye_trace_both_eyes()
        -> Horizontal eye trace before vs. after merging (Hooge et al. 2022)

Main sequence:
    plot_main_sequence()

Fixations:
    plot_fixation_duration()
    plot_fixation_frequency()

Saccade Amplitudes:
    plot_saccade_amplitude()
    plot_saccade_duration()
    plot_saccade_angles()
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

logger = logging.getLogger(__name__)


# =============================================================================
# Eye Trace Comparison (pre/post merge)
# =============================================================================
def plot_eye_trace_both_eyes(
    events_before: pd.DataFrame,
    events_after: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    title: str = "Eye Trace Merge Comparison",
    time_window: tuple = None,
):
    """
    Horizontal eye position trace for both eyes with fixation boxes overlaid,
    comparing before and after the Hooge et al. (2022) merging procedure.

    Args:
        events_before (pd.DataFrame): Original (pre-merge) events dataframe.
        events_after (pd.DataFrame): Merged (post-merge) events dataframe.
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        title (str, optional): Defaults to 'Eye Trace Merge Comparison'.
        time_window (tuple, optional): (start_time, end_time) in seconds to zoom into a specific range.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    for ax_idx, eye in enumerate(["L", "R"]):
        ax = axes[ax_idx]

        before = events_before[events_before["eye"] == eye].copy()
        after = events_after[events_after["eye"] == eye].copy()

        if time_window:
            before = before[
                (before["onset"] >= time_window[0])
                & (before["onset"] <= time_window[1])
            ]
            after = after[
                (after["onset"] >= time_window[0]) & (after["onset"] <= time_window[1])
            ]

        pos_col = "fix_avg_x"
        before_fix = before[before["trial_type"] == "fixation"]
        after_fix = after[after["trial_type"] == "fixation"]

        # Build trace as isolated horizontal segments per fixation
        times = []
        positions = []
        for _, row in before_fix.iterrows():
            times.extend([row["onset"], row["end_time"]])
            positions.extend([row[pos_col], row[pos_col]])

        ax.plot(
            times,
            positions,
            "k-",
            linewidth=1.5,
            alpha=0.8,
            label="Eye trace" if ax_idx == 0 else "",
        )

        # Fixation boxes BEFORE merging
        for idx, row in before_fix.iterrows():
            y_center = row[pos_col]
            box_height = 20
            ax.add_patch(
                plt.Rectangle(
                    (row["onset"], y_center - box_height / 2),
                    row["duration"],
                    box_height,
                    facecolor="limegreen",
                    edgecolor="mediumorchid",
                    alpha=0.4,
                    linewidth=1.5,
                    label=(
                        "Before merging"
                        if (ax_idx == 0 and idx == before_fix.index[0])
                        else ""
                    ),
                )
            )

        # Fixation boxes AFTER merging
        for idx, row in after_fix.iterrows():
            y_center = row[pos_col]
            box_height = 20
            ax.add_patch(
                plt.Rectangle(
                    (row["onset"], y_center - box_height / 2),
                    row["duration"],
                    box_height,
                    facecolor="tomato",
                    edgecolor="crimson",
                    alpha=0.6,
                    linewidth=2,
                    label=(
                        "After merging"
                        if (ax_idx == 0 and idx == after_fix.index[0])
                        else ""
                    ),
                )
            )

        ax.set_ylabel(f"Eye {eye}\nHorizontal position", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Eye {eye}: n={len(before_fix)} → {len(after_fix)} fixations",
            fontsize=11,
            loc="right",
            style="italic",
        )
        ax.grid(True, alpha=0.3)

        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()

    if out_path is not None:
        out_file = (
            f"{out_path}/{title.lower().replace(' ', '_')}-bothEyes.{out_file_format}"
        )
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")

    plt.show()
    plt.close(fig)


# =============================================================================
# Main Sequence
# =============================================================================
def plot_main_sequence(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "both",
    title: str = "Main Sequence",
    drop_near_blinks: bool = False,
):
    """
    Plots main sequence: saccade amplitude vs. peak velocity (log-log).
    Optionally drops near-blink saccades and/or main-sequence outliers. and logs them.

    Args:
        events_df (pd.DataFrame)
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'svg'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'both'. Defaults to 'both'.
        title (str) : Defaults to  'Main Sequence', gets extended by 'by_eye'(blinked-cleaned|' ') # TBD find better description
        drop_near_blinks (bool, optional): If True, exclude saccades flagged as near a blink. Defaults to False.
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
        out_file = os.path.join(out_path, f"{base_name}.{out_file_format}")
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")


# =============================================================================
# Fixation Duration
# =============================================================================
def plot_fixation_duration(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "both",
    fix_dur_min_ms: float = 60,
    fix_dur_max_ms: float = 1000,
    title: str = "Fixation Durations",
):
    """
    Histogram of fixation durations (ms), outliers dropped  (lower bound 60ms, upper bound 1000ms).

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'both'. Defaults to 'both'.
        fix_dur_min_ms (float, optional): lower bound to drop ultra-short blinks/micro-fixations. Defaults to 60ms.
        fix_dur_max_ms (float, optional): upper bound to drop implausibly long fixations. Defaults to 1000ms.
        title (str, optional): Defaults to 'Fixation Durations'.
    Raises:
        ValueError: No fixation durations within fix_dur_min_ms - fix_dur_max_ms found
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
    dur = dur[(dur >= fix_dur_min_ms) & (dur <= fix_dur_max_ms)]
    if dur.empty:
        raise ValueError(
            "No fixation durations post filtering. Check inputs or ranges."
        )
    else:
        dropouts = len(fix["duration_ms"]) - len(dur)
        logger.info(f"Total fixations: {len(fix['duration_ms'])}")
        logger.info(
            f"Kept fixations ({fix_dur_min_ms}ms <= duration <= {fix_dur_max_ms}ms): {len(dur)}"
        )
        logger.info(
            f"Dropped outliers: {dropouts} ({(dropouts/len(fix['duration_ms']))*100:.2f}%)"
        )

    # 4) create the figure
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
    fig = ax.figure
    fig.tight_layout()

    # Save & show
    if out_path is not None:
        out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")

    plt.show()
    plt.close(fig)


# =============================================================================
# Saccade Amplitude
# =============================================================================
def plot_saccade_amplitude(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "both",
    title: str = "Saccade Amplitude",
    sac_amp_max_deg: float = 40,
):
    """
    Histogram of saccade amplitudes (degrees), outliers dropped (upper bound sac_amp_max_deg).

    Args:
        events_df (pd.DataFrame): Event dataframe containing a 'trial_type' column with saccade events.
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to "svg".
        by_eye (str): One of: 'all', 'left', 'right', 'both'. Defaults to 'both'.
        sac_amp_max_deg (float, optional): Upper bound (deg) to drop implausibly large saccade amplitudes. Defaults to 40°
        title (str, optional): Defaults to 'Saccade Amplitude'.
    Raises:
        ValueError: No saccade amplitudes within 0 - sac_amp_max_deg found.
    """

    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    # 1) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye").copy()

    # 2) Select saccade amplitudes in degrees
    all_amplitudes = s_df["sacc_visual_angle"].dropna()
    if sac_amp_max_deg is not None:
        amplitudes = all_amplitudes[all_amplitudes <= sac_amp_max_deg]

    if amplitudes.empty:
        raise ValueError(f"No saccade amplitudes within 0–{sac_amp_max_deg}° found.")

    # Identify dropped outliers
    dropout = len(all_amplitudes[all_amplitudes > sac_amp_max_deg])
    logger.info(f"Total saccades: {len(all_amplitudes)}")
    logger.info(f"Kept saccades (<={sac_amp_max_deg}°): {len(amplitudes)}")
    logger.info(
        f"Dropped outliers (>{sac_amp_max_deg}°): {dropout} ({(dropout/len(all_amplitudes))*100:.2f}%)"
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
    if out_path is not None:
        out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")

    plt.show()
    plt.close(fig)


# =============================================================================
# Saccade Duration
# =============================================================================
def plot_saccade_duration(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "both",
    title: str = "Saccade Duration",
    sac_dur_max_ms: int = 120,
):
    """
    Histogram of fixation frequency (fixations per second).

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str, optional):  File extension for saving, e.g. 'svg', 'pdf', 'eps'. Defaults to 'svg'.
        by_eye (str, optional): One of: 'all', 'left', 'right', 'both'. Defaults to 'both'.
        title (str, optional): Defaults to "Saccade Duration".
        sac_dur_max_ms (int, optional): Maximum duration of a saccade (ms). Pass None to disable clipping. Defaults to 120ms.
    """

    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    # 1) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye")

    # 2) Convert duration from seconds to milliseconds
    durations = (s_df["duration"] * 1000).dropna()
    logger.info(f"Total saccades: {len(durations)}")

    # 3) Drop saccades >120ms
    if sac_dur_max_ms is not None:
        durations = durations[durations <= sac_dur_max_ms]
        durations_copy = durations.copy()
        dropout = len(durations_copy[durations > sac_dur_max_ms])
        logger.info(f"Kept saccades (<={sac_dur_max_ms}ms): {len(durations)}")
        logger.info(
            f"Dropped outliers (>{sac_dur_max_ms}ms): {dropout} ({(dropout/len(durations))*100:.2f}%)"
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
    if out_path is not None:
        out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")

    plt.show()
    plt.close(fig)


# =============================================================================
# Fixation Frequency
# =============================================================================
def plot_fixation_frequency(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "both",
    title="Fixation frequency histogram",
):
    """
    Histogram of fixation frequency (fixations per second), binned by second-level onset buckets.

    Args:
        events_df (pd.DataFrame): Event dataframe containing a 'trial_type' column with fixation events.
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str, optional): One of: 'all', 'left', 'right', 'both'. Defaults to "both".
        title (str, optional): Defaults to 'Fixation frequency histogram'.
    Raises:
        ValueError: No fixation events found for the specified eye selection.
    """
    f_df = events_df[events_df["trial_type"] == "fixation"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        f_df = f_df.query("eye == @chosen_eye").copy()

    if f_df.empty:
        raise ValueError(f"No fixation events found for eye='{by_eye}'.")

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

    # 5) Save & show
    if out_path is not None:
        out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
        plt.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")

    plt.show()


# =============================================================================
# Saccade Angular Histogram
# =============================================================================
def plot_saccade_angles(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "both",
    title: str = "Saccade Direction Histogram",
    style: str = None,
):
    """
    Histogram of saccade directions (degrees), shown as a polar rose plot, a Cartesian bar histogram, or both.

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving. Defaults to None.
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'both'. Defaults to 'both'
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
        ax.set_title(f"Polar {title}")
        plt.tight_layout()

        # 5) Save & show
        if out_path is not None:
            out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
            plt.savefig(out_file, bbox_inches="tight")
            logger.info(f"Polar {title} plot saved to '{out_file}'")
        else:
            logger.warning(f"Polar {title} plot not saved — pass `out_path` to save.")

        plt.show()

    if style in ["cartesian", None]:
        # 4) Create Figure 2: Cartesian Angular Histogramm
        plt.hist(angles_deg, bins=np.arange(0, 361, 10), edgecolor="black")
        plt.xlabel("Saccade direction (deg)")
        plt.ylabel("Count")
        plt.title(f"Cartesian {title}")

        # 5) Save & show
        if out_path is not None:
            out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"  # noqa: F821
            plt.savefig(out_file, bbox_inches="tight")
            logger.info(f"Cartesian {title} plot saved to '{out_file}'")
        else:
            logger.warning(
                f"Cartesian {title} plot not saved — pass `out_path` to save."
            )

        plt.show()


# =============================================================================
# Summary Figure
# =============================================================================
def plot_summary(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "both",
    title: str = "Summary",
    ms_fix_dur_min: float = 60,
    ms_fix_dur_max: float = 1000,
    deg_sacc_amp_max: float = 40,
    ms_sacc_dur_max: float = 120,
    drop_near_blinks: bool = False,
):
    """
    summary figure combining all core plots into one panel (2×3 grid):
        [0] Main sequence         [1] Fixation duration    [2] Fixation frequency
        [3] Saccade amplitude     [4] Saccade duration     [5] Saccade angles (polar)

    Args:
        events_df (pd.DataFrame)
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'both'. Defaults to 'both'.
        title (str, optional): Pass None for no title. Defaults to 'Summary Plots'.
        ms_fix_dur_min (float, optional): Lower bound for fixation duration (ms). Defaults to 60.
        ms_fix_dur_max (float, optional): Upper bound for fixation duration (ms). Defaults to 1000.
        deg_sacc_amp_max (float, optional): Upper bound for saccade amplitude (deg). Defaults to 40.
        ms_sacc_dur_max (float, optional): Upper bound for saccade duration (ms). Defaults to 120.
        drop_near_blinks (bool, optional): If True, exclude near-blink saccades from main sequence. Defaults to False.
    """
    eye_mapping = {"left": "L", "right": "R", "both": "both"}
    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "both": "Binocular only",
    }

    # --- Shared data prep ---
    fix_df = events_df[events_df["trial_type"] == "fixation"].copy()
    sacc_df = events_df[events_df["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        chosen_eye = eye_mapping[by_eye]
        fix_df = fix_df[fix_df["eye"] == chosen_eye]
        sacc_df = sacc_df[sacc_df["eye"] == chosen_eye]

    # Figure layout
    fig = plt.figure(figsize=(16, 10))
    ax_ms = fig.add_subplot(2, 3, 1)  # main sequence
    ax_fdur = fig.add_subplot(2, 3, 2)  # fixation duration
    ax_ffreq = fig.add_subplot(2, 3, 3)  # fixation frequency
    ax_samp = fig.add_subplot(2, 3, 4)  # saccade amplitude
    ax_sdur = fig.add_subplot(2, 3, 5)  # saccade duration
    ax_angles = fig.add_subplot(2, 3, 6, polar=True)  # saccade directions

    # 1) Main sequence
    s_ms = sacc_df.copy()
    if drop_near_blinks:
        s_ms = s_ms[s_ms["near_blink"] == False]
    if by_eye == "all":
        for eye, sub in s_ms.groupby("eye"):
            ax_ms.scatter(
                sub["sacc_visual_angle"], sub["peak_velocity"], s=6, label=str(eye)
            )
        ax_ms.legend(title="Eye", fontsize=7)
    else:
        ax_ms.scatter(s_ms["sacc_visual_angle"], s_ms["peak_velocity"], s=6)
    ax_ms.set_xscale("log")
    ax_ms.set_yscale("log")
    ax_ms.set_xlabel("Amplitude (deg)")
    ax_ms.set_ylabel("Peak velocity (deg/s)")
    ax_ms.set_title("Main Sequence")

    # 2) Fixation duration
    dur_ms = fix_df["duration"].dropna() * 1000
    dur_ms = dur_ms[(dur_ms >= ms_fix_dur_min) & (dur_ms <= ms_fix_dur_max)]
    if dur_ms.empty:
        logger.warning("Summary: no fixation durations in range, skipping panel.")
    else:
        ax_fdur.hist(dur_ms, bins=40, edgecolor="black")
    ax_fdur.set_xlabel("Duration (ms)")
    ax_fdur.set_ylabel("Count")
    ax_fdur.set_title("Fixation Duration")

    # 3) Fixation frequency
    f_freq = fix_df.copy()
    f_freq["sec"] = f_freq["onset"].astype(float).floordiv(1).astype(int)
    fix_per_sec = f_freq.groupby("sec").size()
    if fix_per_sec.empty:
        logger.warning("Summary: no fixation frequency data, skipping panel.")
    else:
        ax_ffreq.hist(
            fix_per_sec.values,
            bins=np.arange(fix_per_sec.max() + 2) - 0.3,
            width=0.6,
            edgecolor="black",
        )
    ax_ffreq.set_xlim(left=-0.3)
    ax_ffreq.set_xlabel("Fixations per second")
    ax_ffreq.set_ylabel("Count")
    ax_ffreq.set_title("Fixation Frequency")

    # 4) Saccade amplitude
    all_amp = sacc_df["sacc_visual_angle"].dropna()
    amp = all_amp[all_amp <= deg_sacc_amp_max]
    if amp.empty:
        logger.warning("Summary: no saccade amplitudes in range, skipping panel.")
    else:
        ax_samp.hist(amp, bins=40, edgecolor="black")
    ax_samp.set_xlabel("Amplitude (deg)")
    ax_samp.set_ylabel("Count")
    ax_samp.set_xlim(left=0)
    ax_samp.set_title("Saccade Amplitude")

    # 5) Saccade duration
    all_sdur = (sacc_df["duration"] * 1000).dropna()
    sdur = all_sdur[all_sdur <= ms_sacc_dur_max]
    if sdur.empty:
        logger.warning("Summary: no saccade durations in range, skipping panel.")
    else:
        ax_sdur.hist(sdur, bins=40, edgecolor="black")
    ax_sdur.set_xlabel("Duration (ms)")
    ax_sdur.set_ylabel("Count")
    ax_sdur.set_xlim(left=0)
    ax_sdur.set_title("Saccade Duration")

    # Saccade angles (polar)
    dx = sacc_df["sacc_end_x"] - sacc_df["sacc_start_x"]
    dy = sacc_df["sacc_end_y"] - sacc_df["sacc_start_y"]
    angles_deg = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    angles_rad = np.deg2rad(angles_deg)
    ax_angles.hist(angles_rad, bins=36, edgecolor="black")
    ax_angles.set_theta_zero_location("E")
    ax_angles.set_theta_direction(1)
    ax_angles.set_title("Saccade Directions")

    # Title
    if title is not None:
        fig.suptitle(f"{title} — {title_map[by_eye]}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    # Save & show
    if out_path is not None:
        out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")

    plt.show()
    plt.close(fig)
