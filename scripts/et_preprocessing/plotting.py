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
def plot_eye_trace_pre_post_processing(
    events_before: pd.DataFrame,
    events_after: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    title: str = "Eye Trace Merge Comparison",
    window_size: int = 20,
    top_n: int = 3,
) -> dict:
    """
    Horizontal eye position trace for both eyes with fixation boxes overlaid,
    comparing before and after the Hooge et al. (2022) merging procedure.

    Args:
        events_before (pd.DataFrame): Original (pre-merge) events dataframe.
        events_after (pd.DataFrame): Merged (post-merge) events dataframe.
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        title (str, optional): Defaults to 'Eye Trace Merge Comparison'.
        time_windows (tuple, optional): (start_time, end_time) in seconds to zoom into a specific range.
        window_size: TBD update, in seconds
        top_n: TBD update, top n windows
    """
    figs = {}

    for eye in ["L", "R"]:
        before = events_before[events_before["eye"] == eye].copy()
        after = events_after[events_after["eye"] == eye].copy()

        before = before[before["trial_type"] == "fixation"]
        after = after[after["trial_type"] == "fixation"]

        t_start = before["onset"].min()
        t_end = before["end_time"].max()
        windows = np.arange(t_start, t_end, window_size)
        change_records = []

        for w_start in windows[:-1]:
            w_end = w_start + window_size

            # number of events before the merging process within the time window
            n_before = ((before["onset"] >= w_start) & (before["onset"] < w_end)).sum()
            # number of events after the merging process within the time window
            n_after = ((after["onset"] >= w_start) & (after["onset"] < w_end)).sum()

            change_records.append((w_start, w_end, int(n_before - n_after)))

        change_records.sort(key=lambda x: x[2], reverse=True)
        time_windows = change_records[:top_n]

        fig, axes = plt.subplots(top_n, 1, figsize=(14, 3 * top_n), squeeze=False)
        eye_label = "Left" if eye == "L" else "Right"
        fig.suptitle(f"{title} — {eye_label} Eye", fontsize=13, fontweight="bold")

        pos_col = "fix_avg_x"

        for rank, (w_start, w_end, _) in enumerate(time_windows):
            ax = axes[rank][0]

            w_before = before[(before["onset"] >= w_start) & (before["onset"] < w_end)]
            w_after = after[(after["onset"] >= w_start) & (after["onset"] < w_end)]

            # Eye trace as horizontal segments per fixation
            times, positions = [], []
            for _, row in w_before.iterrows():
                times.extend([row["onset"], row["end_time"]])
                positions.extend([row[pos_col], row[pos_col]])
            if times:
                ax.plot(
                    times, positions, "k-", linewidth=1.5, alpha=0.7, label="Eye trace"
                )

            # Before-merge fixation lines
            for idx, row in w_before.iterrows():
                ax.hlines(
                    y=row[pos_col]
                    + 3,  # 3 pixels offset to visualize overlapping lines
                    xmin=row["onset"],
                    xmax=row["onset"] + row["duration"],
                    colors="mediumseagreen",
                    linewidth=3,
                    alpha=0.85,
                    label="Before" if idx == w_before.index[0] else "",
                )

            # After-merge fixation lines
            for idx, row in w_after.iterrows():
                ax.hlines(
                    y=row[pos_col]
                    - 3,  # 3 pixels offset to visualize overlapping lines
                    xmin=row["onset"],
                    xmax=row["onset"] + row["duration"],
                    colors="tomato",
                    linewidth=3,
                    alpha=0.85,
                    label="After" if idx == w_after.index[0] else "",
                )

            ax.set_xlim(w_start, w_end)
            ax.set_ylabel("Horiz. pos.", fontsize=9)
            ax.set_title(
                f"Eye Trace Plot pre-/post processing  |  t=[{w_start:.2f}s, {w_end:.2f}s]  |  ",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
            if rank == 0:
                ax.legend(loc="upper right", fontsize=8)

            axes[-1][0].set_xlabel("Time (s)", fontsize=11)
            fig.tight_layout()
            if top_n > 1:
                figs[f"{rank}-{eye}"] = fig
            else:
                figs[eye] = fig

            if out_path is not None:
                eye_str = eye_label.lower()
                out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{eye_str}Eye_R{rank}.{out_file_format}"
                fig.savefig(out_file, bbox_inches="tight")
                logger.info(f"{title} ({eye_label}) plot saved to '{out_file}'")
            else:
                logger.warning(
                    f"{title} ({eye_label}) plot not saved — pass `out_path` to save."
                )

            plt.show()

    return figs


# =============================================================================
# Main Sequence
# =============================================================================
def plot_main_sequence(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "binocular",
    title: str = "Main Sequence",
    include_near_blink_sac: bool | str = True,
):
    """
    Plots main sequence: saccade amplitude vs. peak velocity (log-log).

    Args:
        events_df (pd.DataFrame)
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str): Defaults to 'Main Sequence'.
        include_near_blink_sac (bool | str):
            - True (default): include all saccades, near-blink ones treated like any other.
            - False: exclude saccades flagged as near a blink entirely.
            - 'highlight': include all saccades, but mark near-blink ones in a distinct color.
    """
    s = events_df[events_df["trial_type"] == "saccade"].copy()
    n_total = len(s)

    near_blink_mask = s["near_blink"] == True
    n_flagged = near_blink_mask.sum()

    if include_near_blink_sac == False:
        s = s[~near_blink_mask]
        logger.info(
            f"Excluded {n_flagged} of {n_total} saccades flagged as near-blink. "
            "Set include_near_blink_sac=True to include or 'highlight' to mark them."
        )
    elif include_near_blink_sac == "highlight":
        logger.info(
            f"Highlighting {n_flagged} of {n_total} saccades flagged as near-blink. "
            "Set include_near_blink_sac=True to include without marking, "
            "or False to exclude them."
        )

    if by_eye != "all":
        eye_map = {"left": "L", "right": "R", "binocular": "binocular"}
        s = s[s["eye"] == eye_map[by_eye]]

    base_name = (
        f"{title.lower().replace(' ', '_')}-{by_eye}Eyes"
        + ("_blinkExcluded" if include_near_blink_sac == False else "")
        + ("_blinkHighlighted" if include_near_blink_sac == "highlight" else "")
    )

    fig, ax = plt.subplots()

    if by_eye == "all":
        for eye, sub in s.groupby("eye"):
            ms_scatter(sub, include_near_blink_sac, ax, label=str(eye))
        ax.legend(title="Eye")
    else:
        ms_scatter(s, include_near_blink_sac, ax)
        if include_near_blink_sac == "highlight" and n_flagged > 0:
            ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Saccade amplitude (deg)")
    ax.set_ylabel("Peak velocity (deg/s)")

    title_map = {
        "all": "Left, Right and Binocular Gaze",
        "left": "Left Gaze only",
        "right": "Right Gaze only",
        "binocular": "Binocular Gaze only",
    }
    suffix_map = {
        False: "(near-blink excluded)",
        "highlight": "(near-blink highlighted)",
    }
    suffix = suffix_map.get(include_near_blink_sac, "")
    ax.set_title(f"{title} — {title_map[by_eye]}" + (f" {suffix}" if suffix else ""))

    fig.tight_layout()
    plt.show()

    if out_path is not None:
        out_file = os.path.join(out_path, f"{base_name}.{out_file_format}")
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")

    return fig


# =============================================================================
# Fixation Duration
# =============================================================================
def plot_fixation_duration(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "binocular",
    fix_dur_min: float = 60,
    fix_dur_max: float = 1000,
    title: str = "Fixation Durations",
):
    """
    Histogram of fixation durations (ms), outliers dropped  (lower bound 60ms, upper bound 1000ms).

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        fix_dur_min (float, optional): lower bound to drop ultra-short blinks/micro-fixations in ms. Defaults to 60ms.
        fix_dur_max (float, optional): upper bound to drop implausibly long fixations in ms. Defaults to 1000ms.
        title (str, optional): Defaults to 'Fixation Durations'.
    Raises:
        ValueError: No fixation durations within fix_dur_min - fix_dur_max found
    """
    if by_eye not in {"all", "left", "right", "binocular"}:
        raise ValueError("by_eye must be one of: 'all', 'left', 'right', 'binocular'")

    # 1) Fixations only, convert seconds → ms
    fix = events_df.loc[
        events_df["trial_type"] == "fixation", ["duration", "eye"]
    ].copy()
    fix = fix.dropna(subset=["duration"])
    fix["duration_ms"] = fix["duration"] * 1000.0

    # 2) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "binocular"}
        chosen_eye = eye_mapping[by_eye]
        fix = fix.query(f"eye == @chosen_eye").copy()

    # 3) Filter by plausible duration range
    dur = fix["duration_ms"]
    dur = dur[(dur >= fix_dur_min) & (dur <= fix_dur_max)]
    if dur.empty:
        raise ValueError(
            "No fixation durations post filtering. Check inputs or ranges."
        )
    else:
        dropouts = len(fix["duration_ms"]) - len(dur)
        logger.info(f"Total fixations: {len(fix['duration_ms'])}")
        logger.info(
            f"Kept fixations ({fix_dur_min}ms <= duration <= {fix_dur_max}ms): {len(dur)}"
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
        "binocular": "Binocular only",
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
    by_eye: str = "binocular",
    title: str = "Saccade Amplitude",
    sac_amp_max: float = 40,
):
    """
    Histogram of saccade amplitudes (degrees), outliers dropped (upper bound sac_amp_max).

    Args:
        events_df (pd.DataFrame): Event dataframe containing a 'trial_type' column with saccade events.
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to "svg".
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        sac_amp_max (float, optional): Upper bound (deg) to drop implausibly large saccade amplitudes. Defaults to 40°
        title (str, optional): Defaults to 'Saccade Amplitude'.
    Raises:
        ValueError: No saccade amplitudes within 0 - sac_amp_max degrees found.
    """

    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    # 1) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye").copy()

    # 2) Select saccade amplitudes in degrees
    all_amplitudes = s_df["sacc_visual_angle"].dropna()
    if sac_amp_max is not None:
        amplitudes = all_amplitudes[all_amplitudes <= sac_amp_max]

    if amplitudes.empty:
        raise ValueError(f"No saccade amplitudes within 0–{sac_amp_max}° found.")

    # Identify dropped outliers
    dropout = len(all_amplitudes[all_amplitudes > sac_amp_max])
    logger.info(f"Total saccades: {len(all_amplitudes)}")
    logger.info(f"Kept saccades (<={sac_amp_max}°): {len(amplitudes)}")
    logger.info(
        f"Dropped outliers (>{sac_amp_max}°): {dropout} ({(dropout/len(all_amplitudes))*100:.2f}%)"
    )

    # 3) Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(amplitudes, bins=40, edgecolor="black")

    # 4) Labels & title
    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "binocular": "Binocular only",
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
    by_eye: str = "binocular",
    title: str = "Saccade Duration",
    sac_dur_max: int = 120,
):
    """
    Histogram of fixation frequency (fixations per second).

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str, optional):  File extension for saving, e.g. 'svg', 'pdf', 'eps'. Defaults to 'svg'.
        by_eye (str, optional): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str, optional): Defaults to "Saccade Duration".
        sac_dur_max (int, optional): Maximum duration of a saccade (ms). Pass None to disable clipping. Defaults to 120ms.
    """

    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    # 1) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye")

    # 2) Convert duration from seconds to milliseconds
    durations = (s_df["duration"] * 1000).dropna()
    logger.info(f"Total saccades: {len(durations)}")

    # 3) Drop saccades >120ms
    if sac_dur_max is not None:
        durations = durations[durations <= sac_dur_max]
        durations_copy = durations.copy()
        dropout = len(durations_copy[durations > sac_dur_max])
        logger.info(f"Kept saccades (<={sac_dur_max}ms): {len(durations)}")
        logger.info(
            f"Dropped outliers (>{sac_dur_max}ms): {dropout} ({(dropout/len(durations))*100:.2f}%)"
        )

    # 4) Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(durations, bins=40, edgecolor="black")

    # 5) Labels & title
    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "binocular": "Binocular only",
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
    by_eye: str = "binocular",
    title="Fixation frequency histogram",
):
    """
    Histogram of fixation frequency (fixations per second), binned by second-level onset buckets.

    Args:
        events_df (pd.DataFrame): Event dataframe containing a 'trial_type' column with fixation events.
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str, optional): One of: 'all', 'left', 'right', 'binocular'. Defaults to "binocular".
        title (str, optional): Defaults to 'Fixation frequency histogram'.
    Raises:
        ValueError: No fixation events found for the specified eye selection.
    """
    f_df = events_df[events_df["trial_type"] == "fixation"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "both"}
        chosen_eye = eye_mapping[by_eye]
        f_df = f_df.query("eye == @chosen_eye").copy()

    if f_df.empty:
        raise ValueError(f"No fixation events found for eye='{by_eye}'.")

    f_df["sec"] = f_df["onset"].astype(float).floordiv(1).astype(int)
    fix_per_sec = f_df.groupby("sec").size()

    fig, ax = plt.subplots()
    ax.hist(
        fix_per_sec.values,
        bins=np.arange(fix_per_sec.max() + 2) - 0.3,
        width=0.6,
        edgecolor="black",
    )
    ax.set_xlim(left=-0.3)
    ax.set_xlabel("Fixation Frequency (per Second)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()

    if out_path is not None:
        out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")

    plt.show()
    return fig


# =============================================================================
# Saccade Angular Histogram
# =============================================================================
def plot_saccade_angles(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "binocular",
    title: str = "Saccade Direction Histogram",
    style: str = None,
):
    """
    Histogram of saccade directions (degrees), shown as a polar rose plot, a Cartesian bar histogram, or both.

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving. Defaults to None.
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'
        title (str, optional): Defaults to 'Saccade Direction Histogram'.
        style (str, optional): One of: 'polar', 'cartesian', or None (produces both). Defaults to None.
    """

    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    # 1) Filter by eye
    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "both"}
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
            out_file = f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
            logger.info(f"Cartesian {title} plot saved to '{out_file}'")
        else:
            logger.warning(
                f"Cartesian {title} plot not saved — pass `out_path` to save."
            )

        plt.show()


# =============================================================================
# Summary Plot
# =============================================================================
def plot_summary(
    events_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "binocular",
    title: str = "Summary",
    fix_dur_min: float = 60,
    fix_dur_max: float = 1000,
    sac_amp_max: float = 40,
    sac_dur_max: float = 120,
    include_near_blink_sac: bool | str = True,
):  # *kwargs?! TBD update with kwargs for individual plot options?
    """
    summary figure combining all core plots into one panel (2×3 grid):
        [1] Main sequence         [2] Fixation duration    [3] Fixation frequency
        [4] Saccade amplitude     [5] Saccade duration     [6] Saccade angles (polar)

    Args:
        events_df (pd.DataFrame)
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str, optional): Pass None for no title. Defaults to 'Summary Plots'.
        fix_dur_min (float, optional): Lower bound for fixation duration (ms). Defaults to 60.
        fix_dur_max (float, optional): Upper bound for fixation duration (ms). Defaults to 1000.
        sac_amp_max (float, optional): Upper bound for saccade amplitude (deg). Defaults to 40.
        sac_dur_max (float, optional): Upper bound for saccade duration (ms). Defaults to 120.
        include_near_blink_sac (bool | str):
            - True (default): include all saccades in the main sequence panel.
            - False: exclude near-blink saccades from the main sequence panel.
            - 'highlight': mark near-blink saccades in orange in the main sequence panel.
    """
    eye_mapping = {"left": "L", "right": "R", "binocular": "both"}
    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "binocular": "Binocular only",
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
    blink_saccade_mask = s_ms["near_blink"] == True
    n_flagged = blink_saccade_mask.sum()

    if include_near_blink_sac == False:
        s_ms = s_ms[~blink_saccade_mask]
        logger.info(
            f"Excluded {n_flagged} of {len(s_ms)} saccades flagged as blink saccades. "
            "Set include_near_blink_sac=True to include or 'highlight' to mark them."
        )
    elif include_near_blink_sac == "highlight":
        logger.info(
            f"Highlighting {n_flagged} of {len(s_ms)} saccades flagged as blink saccades. "
            "Set include_near_blink_sac=True to include without marking, "
            "or False to exclude them."
        )

    if by_eye == "all":
        for eye, sub in s_ms.groupby("eye"):
            ms_scatter(sub, include_near_blink_sac, ax_ms, label=str(eye))
        ax_ms.legend(title="Eye", fontsize=7)
    else:
        ms_scatter(s_ms, include_near_blink_sac, ax_ms)
        if (
            include_near_blink_sac == "highlight"
            and not s_ms[s_ms["near_blink"] == True].empty
        ):
            ax_ms.legend(fontsize=7)

    ax_ms.set_xscale("log")
    ax_ms.set_yscale("log")
    ax_ms.set_xlabel("Amplitude (deg)")
    ax_ms.set_ylabel("Peak velocity (deg/s)")
    ms_suffix = {False: " (blink excl.)", "highlight": " (blink highlight.)"}.get(
        include_near_blink_sac, ""
    )
    ax_ms.set_title(f"Main Sequence{ms_suffix}")

    # 2) Fixation duration
    dur_ms = fix_df["duration"].dropna() * 1000
    dur_ms = dur_ms[(dur_ms >= fix_dur_min) & (dur_ms <= fix_dur_max)]
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
    amp = all_amp[all_amp <= sac_amp_max]
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
    sdur = all_sdur[all_sdur <= sac_dur_max]
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

    return fig


# =============================================================================
# Summary Plot
# =============================================================================
def plot_summary_comparison(
    events_before: pd.DataFrame,
    events_after: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "binocular",
    title: str = "Summary",
    fix_dur_min: float = 60,
    fix_dur_max: float = 1000,
    sac_amp_max: float = 40,
    sac_dur_max: float = 120,
    include_near_blink_sac: bool | str = True,
):
    """
    summary figure combining all core plots into one panel (2×3 grid):
        [1] Main sequence         [2] Fixation duration    [3] Fixation frequency
        [4] Saccade amplitude     [5] Saccade duration     [6] Saccade angles (polar)

    Args:
        events_df (pd.DataFrame)
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str, optional): Pass None for no title. Defaults to 'Summary Plots'.
        fix_dur_min (float, optional): Lower bound for fixation duration (ms). Defaults to 60.
        fix_dur_max (float, optional): Upper bound for fixation duration (ms). Defaults to 1000.
        sac_amp_max (float, optional): Upper bound for saccade amplitude (deg). Defaults to 40.
        sac_dur_max (float, optional): Upper bound for saccade duration (ms). Defaults to 120.
        include_near_blink_sac (bool | str):
            - True (default): include all saccades in the main sequence panel.
            - False: exclude near-blink saccades from the main sequence panel.
            - 'highlight': mark near-blink saccades in orange in the main sequence panel.
    """
    eye_mapping = {"left": "L", "right": "R", "binocular": "both"}
    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "binocular": "Binocular only",
    }

    # Shared data prep
    events_before["processing_stage"] = "before"
    events_after["processing_stage"] = "after"
    fix_before = events_before[events_before["trial_type"] == "fixation"].copy()
    fix_after = events_after[events_after["trial_type"] == "fixation"].copy()
    fix_df = pd.concat([fix_before, fix_after], ignore_index=True)
    sacc_before = events_before[events_before["trial_type"] == "saccade"].copy()
    sacc_after = events_after[events_after["trial_type"] == "saccade"].copy()
    sacc_df = pd.concat([sacc_before, sacc_after], ignore_index=True)

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
    blink_saccade_mask = s_ms["near_blink"] == True
    n_flagged = blink_saccade_mask.sum()

    if include_near_blink_sac == False:
        s_ms = s_ms[~blink_saccade_mask]
        logger.info(
            f"Excluded {n_flagged} of {len(s_ms)} saccades flagged as blink saccades. "
            "Set include_near_blink_sac=True to include or 'highlight' to mark them."
        )
    elif include_near_blink_sac == "highlight":
        logger.info(
            f"Highlighting {n_flagged} of {len(s_ms)} saccades flagged as blink saccades. "
            "Set include_near_blink_sac=True to include without marking, "
            "or False to exclude them."
        )

    if by_eye == "all":
        for eye, sub in s_ms.groupby("eye"):
            ms_scatter(sub, include_near_blink_sac, ax_ms, label=str(eye))
        ax_ms.legend(title="Eye", fontsize=7)
    else:
        ms_scatter(s_ms, include_near_blink_sac, ax_ms)
        if (
            include_near_blink_sac == "highlight"
            and not s_ms[s_ms["near_blink"] == True].empty
        ):
            ax_ms.legend(fontsize=7)

    ax_ms.set_xscale("log")
    ax_ms.set_yscale("log")
    ax_ms.set_xlabel("Amplitude (deg)")
    ax_ms.set_ylabel("Peak velocity (deg/s)")
    ms_suffix = {False: " (blink excl.)", "highlight": " (blink highlight.)"}.get(
        include_near_blink_sac, ""
    )
    ax_ms.set_title(f"Main Sequence{ms_suffix}")

    # 2) Fixation duration
    fix_dur_before = fix_before.dropna() * 1000
    fix_dur_after = fix_after.dropna() * 1000
    total_fixations_before = len(fix_dur_before)
    total_fixations_after = len(fix_dur_after)
    fix_dur_before = fix_dur_before[
        (fix_dur_before >= fix_dur_min) & (fix_dur_before <= fix_dur_max)
    ]
    fix_dur_after = fix_dur_after[
        (fix_dur_after >= fix_dur_min) & (fix_dur_after <= fix_dur_max)
    ]
    n_dropped_before = total_fixations_before - len(fix_dur_before)
    n_dropped_after = total_fixations_after - len(fix_dur_after)

    logger.info(
        f"Total fixations: {total_fixations_before + total_fixations_after}, Fixations before preprocessing: {total_fixations_before}, Fixations after preprocessing: {total_fixations_after}"
    )
    logger.info(
        f"Kept fixations in range for plotting [{fix_dur_min}, {fix_dur_max}] ms: Total {len(fix_dur_before) + len(fix_dur_after)}, Fixations before preprocessing: {len(fix_dur_before)}, Fixations after preprocessing: {len(fix_dur_after)}"
    )
    logger.info(
        f"Dropped fixations outside range: Total {n_dropped_before + n_dropped_after}, Before: {n_dropped_before}, After: {n_dropped_after}"
    )

    ax_fdur.hist(
        [fix_dur_before, fix_dur_after],
        bins=40,
        edgecolor="black",
        stacked=True,
        color=["#0072B2", "#009E73"],
        label=["Before Preprocessing", "After Preprocessing"],
    )
    ax_fdur.set_xlabel("Duration (ms)")
    ax_fdur.set_ylabel("Count")
    ax_fdur.set_title("Fixation Duration (before vs. after)")

    # 3) Fixation frequency
    f_freq_before = fix_before.copy()
    f_freq_before["sec"] = f_freq_before["onset"].astype(float).floordiv(1).astype(int)
    fix_per_sec_before = f_freq_before.groupby("sec").size()

    f_freq_after = fix_after.copy()
    f_freq_after["sec"] = f_freq_after["onset"].astype(float).floordiv(1).astype(int)
    fix_per_sec_after = f_freq_after.groupby("sec").size()

    logger.info(
        f"Seconds with fixations: Total {len(fix_per_sec_before) + len(fix_per_sec_after)}, "
        f"Before preprocessing: {len(fix_per_sec_before)}, After preprocessing: {len(fix_per_sec_after)}"
    )
    logger.info(
        f"Fixations counted: Total {fix_per_sec_before.sum() + fix_per_sec_after.sum()}, "
        f"Before preprocessing: {fix_per_sec_before.sum()}, After preprocessing: {fix_per_sec_after.sum()}"
    )
    logger.info(
        f"Mean fixations/s: Before preprocessing: {fix_per_sec_before.mean():.2f}, "
        f"After preprocessing: {fix_per_sec_after.mean():.2f}"
    )
    logger.info(
        f"Max fixations/s: Before preprocessing: {fix_per_sec_before.max()}, "
        f"After preprocessing: {fix_per_sec_after.max()}"
    )

    max_val = max(fix_per_sec_before.max(), fix_per_sec_after.max())
    ax_ffreq.hist(
        [fix_per_sec_before.values, fix_per_sec_after.values],
        bins=np.arange(max_val + 2) - 0.3,
        width=0.6,
        edgecolor="black",
        stacked=True,
        color=["#0072B2", "#009E73"],
        label=["Before Preprocessing", "After Preprocessing"],
    )

    ax_ffreq.set_xlim(left=-0.3)
    ax_ffreq.set_xlabel("Fixations per second")
    ax_ffreq.set_ylabel("Count")
    ax_ffreq.set_title("Fixation Frequency")

    # 4) Saccade amplitude
    blink_mask_before = sacc_before["near_blink"] == True
    blink_mask_after = sacc_after["near_blink"] == True
    n_flagged_before = blink_mask_before.sum()
    n_flagged_after = blink_mask_after.sum()

    samp_before = sacc_before.copy()
    samp_after = sacc_after.copy()

    if include_near_blink_sac == False:
        samp_before = samp_before[~blink_mask_before]
        samp_after = samp_after[~blink_mask_after]
        logger.info(
            f"Saccade amplitude (blink saccades excluded): "
            f"Before preprocessing: {n_flagged_before}, After preprocessing: {n_flagged_after}"
        )
    elif include_near_blink_sac == "highlight" or include_near_blink_sac == True:
        logger.info(
            f"Saccade amplitude (blink saccades included, but no visual highlight in histogram for a cleaner overview): "
            f"Before preprocessing: {n_flagged_before}, After preprocessing: {n_flagged_after}"
        )

    all_amp_before = samp_before["sacc_visual_angle"].dropna()
    all_amp_after = samp_after["sacc_visual_angle"].dropna()
    amp_before = all_amp_before[all_amp_before <= sac_amp_max]
    amp_after = all_amp_after[all_amp_after <= sac_amp_max]
    n_dropped_before = len(all_amp_before) - len(amp_before)
    n_dropped_after = len(all_amp_after) - len(amp_after)

    logger.info(
        f"Total saccades: Total {len(all_amp_before) + len(all_amp_after)}, "
        f"Before preprocessing: {len(all_amp_before)}, After preprocessing: {len(all_amp_after)}"
    )
    logger.info(
        f"Kept saccades for plotting (<={sac_amp_max}°): Total {len(amp_before) + len(amp_after)}, "
        f"Before preprocessing: {len(amp_before)}, After preprocessing: {len(amp_after)}"
    )
    logger.info(
        f"Dropped saccades outside range (>{sac_amp_max}°): Total {n_dropped_before + n_dropped_after}, "
        f"Before preprocessing: {n_dropped_before}, After preprocessing: {n_dropped_after}"
    )

    if amp_before.empty and amp_after.empty:
        logger.warning("Summary: no saccade amplitudes in range, skipping panel.")
    else:
        ax_samp.hist(
            [amp_before.values, amp_after.values],
            bins=40,
            edgecolor="black",
            stacked=True,
            color=["#0072B2", "#009E73"],
            label=["Before Preprocessing", "After Preprocessing"],
        )
    ax_samp.set_xlabel("Amplitude (deg)")
    ax_samp.set_ylabel("Count")
    ax_samp.set_xlim(left=0)
    amp_suffix = {False: " (blink excl.)", "highlight": " (blink incl.)"}.get(
        include_near_blink_sac, ""
    )
    ax_samp.set_title(f"Saccade Amplitude{amp_suffix}")

    # 5) Saccade duration
    blink_mask_before = sacc_before["near_blink"] == True
    blink_mask_after = sacc_after["near_blink"] == True
    n_flagged_before = blink_mask_before.sum()
    n_flagged_after = blink_mask_after.sum()

    sdur_before_df = sacc_before.copy()
    sdur_after_df = sacc_after.copy()

    if include_near_blink_sac == False:
        sdur_before_df = sdur_before_df[~blink_mask_before]
        sdur_after_df = sdur_after_df[~blink_mask_after]
        logger.info(
            f"Saccade duration — excluded blink saccades: "
            f"Before preprocessing: {n_flagged_before}, After preprocessing: {n_flagged_after}"
        )
    elif include_near_blink_sac == "highlight" or include_near_blink_sac == True:
        logger.info(
            f"Saccade duration (blink saccades included, but no visual highlight in histogram for a cleaner overview): "
            f"Before preprocessing: {n_flagged_before}, After preprocessing: {n_flagged_after}"
        )

    all_sdur_before = (sdur_before_df["duration"] * 1000).dropna()
    all_sdur_after = (sdur_after_df["duration"] * 1000).dropna()
    sdur_before = all_sdur_before[all_sdur_before <= sac_dur_max]
    sdur_after = all_sdur_after[all_sdur_after <= sac_dur_max]
    n_dropped_before = len(all_sdur_before) - len(sdur_before)
    n_dropped_after = len(all_sdur_after) - len(sdur_after)

    logger.info(
        f"Total saccades: Total {len(all_sdur_before) + len(all_sdur_after)}, "
        f"Before preprocessing: {len(all_sdur_before)}, After preprocessing: {len(all_sdur_after)}"
    )
    logger.info(
        f"Kept saccades for plotting (<={sac_dur_max} ms): Total {len(sdur_before) + len(sdur_after)}, "
        f"Before preprocessing: {len(sdur_before)}, After preprocessing: {len(sdur_after)}"
    )
    logger.info(
        f"Dropped saccades outside range (>{sac_dur_max} ms): Total {n_dropped_before + n_dropped_after}, "
        f"Before preprocessing: {n_dropped_before}, After preprocessing: {n_dropped_after}"
    )

    if sdur_before.empty and sdur_after.empty:
        logger.warning("Summary: no saccade durations in range, skipping panel.")
    else:
        ax_sdur.hist(
            [sdur_before.values, sdur_after.values],
            bins=40,
            edgecolor="black",
            stacked=True,
            color=["#0072B2", "#009E73"],
            label=["Before Preprocessing", "After Preprocessing"],
        )
    ax_sdur.set_xlabel("Duration (ms)")
    ax_sdur.set_ylabel("Count")
    ax_sdur.set_xlim(left=0)
    sdur_suffix = {False: " (blink excl.)", "highlight": " (blink incl.)"}.get(
        include_near_blink_sac, ""
    )
    ax_sdur.set_title(f"Saccade Duration{sdur_suffix}")

    # Saccade angles (polar)
    blink_mask_before = sacc_before["near_blink"] == True
    blink_mask_after = sacc_after["near_blink"] == True
    n_flagged_before = blink_mask_before.sum()
    n_flagged_after = blink_mask_after.sum()

    sang_before = sacc_before.copy()
    sang_after = sacc_after.copy()

    if include_near_blink_sac == False:
        sang_before = sang_before[~blink_mask_before]
        sang_after = sang_after[~blink_mask_after]
        logger.info(
            f"Saccade directions — excluded blink saccades: "
            f"Before preprocessing: {n_flagged_before}, After preprocessing: {n_flagged_after}"
        )
    elif include_near_blink_sac == "highlight" or include_near_blink_sac == True:
        logger.info(
            f"Saccade directions (blink saccades included, but no visual highlight in histogram for a cleaner overview): "
            f"Before preprocessing: {n_flagged_before}, After preprocessing: {n_flagged_after}"
        )

    dx_before = sang_before["sacc_end_x"] - sang_before["sacc_start_x"]
    dy_before = sang_before["sacc_end_y"] - sang_before["sacc_start_y"]
    angles_rad_before = np.deg2rad(
        (np.degrees(np.arctan2(dy_before, dx_before)) + 360) % 360
    )

    dx_after = sang_after["sacc_end_x"] - sang_after["sacc_start_x"]
    dy_after = sang_after["sacc_end_y"] - sang_after["sacc_start_y"]
    angles_rad_after = np.deg2rad(
        (np.degrees(np.arctan2(dy_after, dx_after)) + 360) % 360
    )

    logger.info(
        f"Total saccades for direction plot: Total {len(angles_rad_before) + len(angles_rad_after)}, "
        f"Before preprocessing: {len(angles_rad_before)}, After preprocessing: {len(angles_rad_after)}"
    )

    ax_angles.hist(
        [angles_rad_before, angles_rad_after],
        bins=36,
        edgecolor="black",
        stacked=True,
        color=["#0072B2", "#009E73"],
        label=["Before Preprocessing", "After Preprocessing"],
    )
    ax_angles.set_theta_zero_location("E")
    ax_angles.set_theta_direction(1)
    ang_suffix = {False: " (blink excl.)", "highlight": " (blink incl.)"}.get(
        include_near_blink_sac, ""
    )
    ax_angles.set_title(f"Saccade Directions{ang_suffix}")

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

    return fig


def ms_scatter(sub, include_near_blink_sac, ax_ms, label=None):
    if "processing_stage" in sub.columns:
        before = sub[sub["processing_stage"] == "before"]
        after = sub[sub["processing_stage"] == "after"]

        if include_near_blink_sac == "highlight":
            ax_ms.scatter(
                before.loc[before["near_blink"] == False, "sacc_visual_angle"],
                before.loc[before["near_blink"] == False, "peak_velocity"],
                s=6,
                color="#0072B2",
                alpha=0.6,
                label="before preprocessing",
            )
            ax_ms.scatter(
                after.loc[after["near_blink"] == False, "sacc_visual_angle"],
                after.loc[after["near_blink"] == False, "peak_velocity"],
                s=6,
                color="#009E73",
                alpha=0.6,
                label="after preprocessing",
            )
            flagged = before[
                before["near_blink"] == True
            ]  # in beiden Stages gleich laut deiner Annahme
            if not flagged.empty:
                ax_ms.scatter(
                    flagged["sacc_visual_angle"],
                    flagged["peak_velocity"],
                    s=6,
                    color="#E69F00",
                    alpha=0.6,
                    label="blink saccade",
                )
        else:
            ax_ms.scatter(
                before["sacc_visual_angle"],
                before["peak_velocity"],
                s=6,
                color="#0072B2",
                alpha=0.6,
                label="before preprocessing",
            )
            ax_ms.scatter(
                after["sacc_visual_angle"],
                after["peak_velocity"],
                s=6,
                color="#009E73",
                alpha=0.6,
                label="after preprocessing",
            )

    elif include_near_blink_sac == "highlight":
        normal = sub[sub["near_blink"] == False]
        flagged = sub[sub["near_blink"] == True]
        ax_ms.scatter(
            normal["sacc_visual_angle"],
            normal["peak_velocity"],
            s=6,
            color="#0072B2",
            label=label,
        )
        if not flagged.empty:
            ax_ms.scatter(
                flagged["sacc_visual_angle"],
                flagged["peak_velocity"],
                s=6,
                color="#E69F00",
                alpha=0.6,
                label="blink saccade",
            )
    else:
        ax_ms.scatter(
            sub["sacc_visual_angle"],
            sub["peak_velocity"],
            s=6,
            label=label,
        )
