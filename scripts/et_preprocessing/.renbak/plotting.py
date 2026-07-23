"""
plotting.py: eye-tracking visualisation — public plotting API.

Each public plot_* function composes one figure: it selects events (by trial
type and eye), delegates the actual drawing to a `_graph_*` helper in graphs.py,
then titles and saves the figure. plot_summary / plot_summary_comparison arrange
all six core graphs in a 2×3 grid.

Preprocessing comparison:
    plot_eye_trace_pre_post_processing()  -> horizontal eye trace before vs. after merge

Main sequence:
    plot_main_sequence()

Fixations:
    plot_fixation_duration()
    plot_fixation_frequency()

Saccades:
    plot_saccade_amplitude()
    plot_saccade_duration()
    plot_saccade_angles()

Summaries:
    plot_summary(), plot_summary_comparison()
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graphs import (
    _graph_main_sequence,
    _graph_fixation_duration,
    _graph_fixation_frequency,
    _graph_saccade_amplitude,
    _graph_saccade_duration,
    _graph_saccade_angles,
)

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
        window_size (int, optional): window length in seconds. Defaults to 20.
        top_n (int, optional): number of most-changed windows to show. Defaults to 3.
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
# Main Sequence (amplitude vs. peak-velocity, log-log)
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
            - True (default): include all saccades; blink saccades treated like any other.
            - False: exclude blink saccades entirely.
            - 'highlight': include all saccades, but mark blink saccades in a distinct colour.
    """
    # filter out saccades and optionally by eye
    s = events_df[events_df["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        eye_map = {"left": "L", "right": "R", "binocular": "binocular"}
        s = s[s["eye"] == eye_map[by_eye]]

    base_name = (
        f"{title.lower().replace(' ', '_')}-{by_eye}Eyes"
        + ("_blinkExcluded" if include_near_blink_sac is False else "")
        + ("_blinkHighlighted" if include_near_blink_sac == "highlight" else "")
    )

    # plot the main sequence
    fig, ax = plt.subplots()
    _graph_main_sequence(ax, s, include_near_blink_sac, by_eye)

    # Labels & title
    gaze_map = {
        "all": "Left, Right and Binocular Gaze",
        "left": "Left Gaze only",
        "right": "Right Gaze only",
        "binocular": "Binocular Gaze only",
    }
    blink_decision_map = {
        False: "(blink saccades excluded)",
        "highlight": "(blink saccades highlighted)",
    }
    suffix = blink_decision_map.get(include_near_blink_sac, "")
    ax.set_title(f"{title} — {gaze_map[by_eye]}" + (f" {suffix}" if suffix else ""))

    fig.tight_layout()

    # Save & show
    if out_path is not None:
        out_file = os.path.join(out_path, f"{base_name}.{out_file_format}")
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")

    plt.show()
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
    Histogram of fixation durations (ms), outliers dropped for plotting only
    (default: lower bound 60 ms, upper bound 1000 ms).

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        fix_dur_min (float, optional): lower bound (ms). Defaults to 60.
        fix_dur_max (float, optional): upper bound (ms). Defaults to 1000.
        title (str, optional): Defaults to 'Fixation Durations'.
    """
    if by_eye not in {"all", "left", "right", "binocular"}:
        raise ValueError("by_eye must be one of: 'all', 'left', 'right', 'binocular'")

    # filter out fixations and optionally by eye
    fix = events_df[events_df["trial_type"] == "fixation"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "binocular"}
        fix = fix[fix["eye"] == eye_mapping[by_eye]]

    # plot fixation duration histogram
    fig, ax = plt.subplots()
    _graph_fixation_duration(ax, fix, fix_dur_min=fix_dur_min, fix_dur_max=fix_dur_max)

    # Labels & title
    gaze_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "binocular": "Binocular only",
    }
    ax.set_title(f"{title} — {gaze_map[by_eye]}")
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
    include_near_blink_sac: bool | str = False,
):
    """
    Histogram of saccade amplitudes (degrees), outliers dropped for plotting
    only (upper bound sac_amp_max).

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str, optional): Defaults to 'Saccade Amplitude'.
        sac_amp_max (float, optional): Upper bound (deg). Defaults to 40.
        include_near_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them. Defaults to False.
    """
    # filter out saccades and optionally by eye
    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "binocular"}
        s_df = s_df[s_df["eye"] == eye_mapping[by_eye]]

    # plot saccade amplitude histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    _graph_saccade_amplitude(
        ax, s_df, sac_amp_max=sac_amp_max, include_near_blink_sac=include_near_blink_sac
    )

    # Title
    gaze_map = {
        "all": "Left, Right and Binocular Gaze",
        "left": "Left Gaze only",
        "right": "Right Gaze only",
        "binocular": "Binocular Gaze only",
    }
    blink_decision_map = {
        False: "(blink saccades excluded)",
        "highlight": "(blink saccades highlighted)",
    }
    suffix = blink_decision_map.get(include_near_blink_sac, "")
    ax.set_title(f"{title} — {gaze_map[by_eye]}" + (f" {suffix}" if suffix else ""))

    fig.tight_layout()

    # Save & Show
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
    Histogram of saccade durations (ms), outliers dropped for plotting only
    (upper bound sac_dur_max).

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str, optional): File extension for saving. Defaults to 'svg'.
        by_eye (str, optional): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str, optional): Defaults to 'Saccade Duration'.
        sac_dur_max (int, optional): Maximum duration of a saccade (ms). Defaults to 120.
    """
    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "binocular"}
        s_df = s_df[s_df["eye"] == eye_mapping[by_eye]]

    # plot saccade duration histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    _graph_saccade_duration(
        ax, s_df, sac_dur_max=sac_dur_max, include_near_blink_sac=True
    )

    # suffix to the graph title: which gaze data is shown
    gaze_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "binocular": "Binocular only",
    }
    ax.set_title(f"{title} — {gaze_map[by_eye]}")
    fig.tight_layout()

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
    Histogram of fixation frequency (fixations per second), binned by
    second-level onset buckets.

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving. Defaults to 'svg'.
        by_eye (str, optional): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str, optional): Defaults to 'Fixation frequency histogram'.
    """
    # filter out fixations and optionally by eye
    f_df = events_df[events_df["trial_type"] == "fixation"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "binocular"}
        f_df = f_df[f_df["eye"] == eye_mapping[by_eye]]

    # plot fixation frequency histogram
    fig, ax = plt.subplots()
    _graph_fixation_frequency(ax, f_df)
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
    Histogram of saccade directions (degrees), shown as a polar rose plot, a
    Cartesian bar histogram, or both.

    Args:
        events_df (pd.DataFrame):
        out_path (str): Directory to save the figure. Pass None to skip saving. Defaults to None.
        out_file_format (str): File extension for saving. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str, optional): Defaults to 'Saccade Direction Histogram'.
        style (str, optional): One of: 'polar', 'cartesian', or None (produces both). Defaults to None.
    """
    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "binocular": "binocular"}
        s_df = s_df[s_df["eye"] == eye_mapping[by_eye]]

    base = f"{title.lower().replace(' ', '_')}-{by_eye}Eyes"

    if style in ["polar", None]:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)
        _graph_saccade_angles(ax, s_df, include_near_blink_sac=True, style="polar")
        ax.set_title(f"Polar {title}")
        fig.tight_layout()
        if out_path is not None:
            out_file = f"{out_path}/{base}_polar.{out_file_format}"
            fig.savefig(out_file, bbox_inches="tight")
            logger.info(f"Polar {title} plot saved to '{out_file}'")
        else:
            logger.warning(f"Polar {title} plot not saved — pass `out_path` to save.")
        plt.show()

    if style in ["cartesian", None]:
        fig2, ax2 = plt.subplots()
        _graph_saccade_angles(ax2, s_df, include_near_blink_sac=True, style="cartesian")
        ax2.set_title(f"Cartesian {title}")
        fig2.tight_layout()
        if out_path is not None:
            out_file = f"{out_path}/{base}_cartesian.{out_file_format}"
            fig2.savefig(out_file, bbox_inches="tight")
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
):
    """
    Summary figure combining all core plots into one graphic (2×3 grid):
        [1] Main sequence         [2] Fixation duration    [3] Fixation frequency
        [4] Saccade amplitude     [5] Saccade duration     [6] Saccade angles (polar)

    Args:
        events_df (pd.DataFrame)
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str, optional): Pass None for no title. Defaults to 'Summary'.
        fix_dur_min (float, optional): Lower bound for fixation duration (ms). Defaults to 60.
        fix_dur_max (float, optional): Upper bound for fixation duration (ms). Defaults to 1000.
        sac_amp_max (float, optional): Upper bound for saccade amplitude (deg). Defaults to 40.
        sac_dur_max (float, optional): Upper bound for saccade duration (ms). Defaults to 120.
        include_near_blink_sac (bool | str):
            - True (default): include all saccades in the main sequence graph.
            - False: exclude blink saccades.
            - 'highlight': mark blink saccades in the main sequence graph.
    """
    eye_mapping = {"left": "L", "right": "R", "binocular": "binocular"}
    gaze_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "binocular": "Binocular only",
    }

    # Shared data prep
    fix_df = events_df[events_df["trial_type"] == "fixation"].copy()
    sacc_df = events_df[events_df["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        chosen_eye = eye_mapping[by_eye]
        fix_df = fix_df[fix_df["eye"] == chosen_eye]
        sacc_df = sacc_df[sacc_df["eye"] == chosen_eye]

    # Figure layout
    fig = plt.figure(figsize=(16, 10))
    ax_ms = fig.add_subplot(2, 3, 1)
    ax_fdur = fig.add_subplot(2, 3, 2)
    ax_ffreq = fig.add_subplot(2, 3, 3)
    ax_samp = fig.add_subplot(2, 3, 4)
    ax_sdur = fig.add_subplot(2, 3, 5)
    ax_angles = fig.add_subplot(2, 3, 6, polar=True)

    _graph_main_sequence(ax_ms, sacc_df, include_near_blink_sac, by_eye)
    _graph_fixation_duration(
        ax_fdur, fix_df, fix_dur_min=fix_dur_min, fix_dur_max=fix_dur_max
    )
    _graph_fixation_frequency(ax_ffreq, fix_df)
    _graph_saccade_amplitude(
        ax_samp, sacc_df, sac_amp_max=sac_amp_max, include_near_blink_sac=include_near_blink_sac
    )
    _graph_saccade_duration(
        ax_sdur, sacc_df, sac_dur_max=sac_dur_max, include_near_blink_sac=include_near_blink_sac
    )
    _graph_saccade_angles(
        ax_angles, sacc_df, include_near_blink_sac=include_near_blink_sac
    )

    if title is not None:
        fig.suptitle(f"{title} — {gaze_map[by_eye]}", fontsize=14, fontweight="bold")
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
# Summary Plot — before/after comparison
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
    Before/after summary figure — same 2×3 grid as plot_summary, but each graph
    stacks the pre-merge ('before') and post-merge ('after') distributions:
        [1] Main sequence         [2] Fixation duration    [3] Fixation frequency
        [4] Saccade amplitude     [5] Saccade duration     [6] Saccade angles (polar)

    Args:
        events_before (pd.DataFrame): Original (pre-merge) events dataframe.
        events_after (pd.DataFrame): Merged (post-merge) events dataframe.
        out_path (str): Directory to save the figure. Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf', 'png'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'binocular'.
        title (str, optional): Pass None for no title. Defaults to 'Summary'.
        fix_dur_min (float, optional): Lower bound for fixation duration (ms). Defaults to 60.
        fix_dur_max (float, optional): Upper bound for fixation duration (ms). Defaults to 1000.
        sac_amp_max (float, optional): Upper bound for saccade amplitude (deg). Defaults to 40.
        sac_dur_max (float, optional): Upper bound for saccade duration (ms). Defaults to 120.
        include_near_blink_sac (bool | str):
            - True (default): include all saccades in the main sequence graph.
            - False: exclude blink saccades.
            - 'highlight': mark blink saccades in the main sequence graph.
    """
    eye_mapping = {"left": "L", "right": "R", "binocular": "binocular"}
    gaze_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "binocular": "Binocular only",
    }

    # Copy to avoid mutating the caller's dataframes (processing_stage tag below).
    events_before = events_before.copy()
    events_after = events_after.copy()
    events_before["processing_stage"] = "before"
    events_after["processing_stage"] = "after"

    fix_before = events_before[events_before["trial_type"] == "fixation"].copy()
    fix_after = events_after[events_after["trial_type"] == "fixation"].copy()
    sacc_before = events_before[events_before["trial_type"] == "saccade"].copy()
    sacc_after = events_after[events_after["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        chosen_eye = eye_mapping[by_eye]
        fix_before = fix_before[fix_before["eye"] == chosen_eye]
        fix_after = fix_after[fix_after["eye"] == chosen_eye]
        sacc_before = sacc_before[sacc_before["eye"] == chosen_eye]
        sacc_after = sacc_after[sacc_after["eye"] == chosen_eye]

    # Main sequence needs one frame carrying 'processing_stage' for ms_scatter.
    sacc_both = pd.concat([sacc_before, sacc_after], ignore_index=True)

    # Figure layout
    fig = plt.figure(figsize=(16, 10))
    ax_ms = fig.add_subplot(2, 3, 1)
    ax_fdur = fig.add_subplot(2, 3, 2)
    ax_ffreq = fig.add_subplot(2, 3, 3)
    ax_samp = fig.add_subplot(2, 3, 4)
    ax_sdur = fig.add_subplot(2, 3, 5)
    ax_angles = fig.add_subplot(2, 3, 6, polar=True)

    _graph_main_sequence(ax_ms, sacc_both, include_near_blink_sac, by_eye)
    _graph_fixation_duration(
        ax_fdur, fix_before, fix_after, fix_dur_min=fix_dur_min, fix_dur_max=fix_dur_max
    )
    _graph_fixation_frequency(ax_ffreq, fix_before, fix_after)
    _graph_saccade_amplitude(
        ax_samp,
        sacc_before,
        sacc_after,
        sac_amp_max=sac_amp_max,
        include_near_blink_sac=include_near_blink_sac,
    )
    _graph_saccade_duration(
        ax_sdur,
        sacc_before,
        sacc_after,
        sac_dur_max=sac_dur_max,
        include_near_blink_sac=include_near_blink_sac,
    )
    _graph_saccade_angles(
        ax_angles, sacc_before, sacc_after, include_near_blink_sac=include_near_blink_sac
    )

    if title is not None:
        fig.suptitle(
            f"{title} — {gaze_map[by_eye]} (before vs. after)",
            fontsize=14,
            fontweight="bold",
        )
    fig.tight_layout()

    if out_path is not None:
        out_file = (
            f"{out_path}/{title.lower().replace(' ', '_')}_comparison"
            f"-{by_eye}Eyes.{out_file_format}"
        )
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} comparison plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} comparison plot not saved — pass `out_path` to save.")

    plt.show()

    return fig
