"""
graphs.py: drawing helpers for the visualisationscript plotting.py

Each `_graph_*` function draws exactly one graph type onto a given Matplotlib
axis. The public plot_* / plot_summary functions in plotting.py compose these
onto figures and handle eye selection, titles and saving. Keeping the drawing
logic here means every graph has a single source for bins, axes, thresholds and
blink handling.

Contents
--------
    _darken, _exclude_blink        -> small shared helpers
    _graph_main_sequence           -> amplitude vs. peak velocity (log-log)
    _graph_fixation_duration       -> fixation duration histogram (ms)
    _graph_fixation_frequency      -> fixations per second histogram
    _graph_saccade_amplitude       -> saccade amplitude histogram (deg)
    _graph_saccade_duration        -> saccade duration histogram (ms)
    _graph_saccade_angles          -> saccade direction (polar / cartesian)
    ms_scatter                     -> main-sequence scatter primitive
"""

import logging

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Palette & small helpers
# =============================================================================
# Consistent, colour-blind-friendly palette (Wong 2011).
BEFORE_COLOR = "#0072B2"  # blue   — before preprocessing / saccades
AFTER_COLOR = "#009E73"   # green  — after preprocessing / blink saccades (single eye)
BLINK_COLOR = "#E69F00"   # orange — blink saccades (histograms)


def _darken(color, factor: float = 0.55):
    """
    Helperfunction: darken a colour by scaling its RGB channels.

    Args:
        color: any Matplotlib colour spec (hex, name, RGBA).
        factor (float): scaling factor in 0–1; smaller = darker. Defaults to 0.55.
    """
    r, g, b, a = to_rgba(color)
    return (r * factor, g * factor, b * factor, a)


def _exclude_blink(sacc, include_near_blink_sac):
    """
    Helperfunction: optionally drop near-blink saccades.

    Args:
        sacc (pd.DataFrame): saccade events (must contain a 'near_blink' column).
        include_near_blink_sac (bool | str): if False, near-blink saccades are
            dropped; otherwise the frame is returned unchanged.

    Returns:
        tuple: (saccades, number of near-blink saccades flagged).
    """
    mask = sacc["near_blink"] == True
    n_flagged = int(mask.sum())
    if include_near_blink_sac is False:
        return sacc[~mask], n_flagged
    return sacc, n_flagged


# =============================================================================
# Main sequence (amplitude vs. peak-velocity, log-log)
# =============================================================================
def _graph_main_sequence(ax, sacc_df, include_near_blink_sac: bool | str = False, by_eye: str = "binocular"):
    """
    Helperfunction: draw the main-sequence scatter (amplitude vs. peak velocity,
    log-log) onto `ax`.

    Works for single and comparison data: comparison data carries a
    'processing_stage' column, which ms_scatter uses to split before/after.

    Args:
        ax: Matplotlib axis to draw on.
        sacc_df (pd.DataFrame): saccade events (may carry 'processing_stage').
        include_near_blink_sac (bool | str): False excludes near-blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
        by_eye (str): 'all' plots one colour per eye; otherwise a single group.
            Defaults to 'binocular'.
    """
    comparison = "processing_stage" in sacc_df.columns
    s_ms, n_flagged = _exclude_blink(sacc_df, include_near_blink_sac)
    if include_near_blink_sac is False:
        logger.info(f"Main sequence — excluded {n_flagged} blink saccades.")
    elif include_near_blink_sac == "highlight":
        logger.info(f"Main sequence — highlighting {n_flagged} blink saccades.")
    elif include_near_blink_sac is True:
        logger.info(f"Main sequence — including {n_flagged} blink saccades.")

    if by_eye == "all":
        # one colour per eye (from the cycle); blinks in a darker shade of it
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (eye, sub) in enumerate(s_ms.groupby("eye")):
            eye_color = cycle[i % len(cycle)]
            ms_scatter(
                sub, include_near_blink_sac, ax, label=str(eye),
                sac_color=eye_color, blink_color=_darken(eye_color),
            )
        ax.legend(title="Eye", fontsize=7)
    else:
        # single eye: saccades in blue, blink saccades in green
        ms_scatter(
            s_ms, include_near_blink_sac, ax,
            sac_color=BEFORE_COLOR, blink_color=AFTER_COLOR,
        )
        if comparison or (
            include_near_blink_sac == "highlight"
            and not s_ms[s_ms["near_blink"] == True].empty
        ):
            ax.legend(fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Amplitude (deg)")
    ax.set_ylabel("Peak velocity (deg/s)")
    suffix = {False: " (blink excl.)", "highlight": " (blink highlight.)"}.get(
        include_near_blink_sac, ""
    )
    ax.set_title(f"Main Sequence{suffix}")


# =============================================================================
# Fixation duration
# =============================================================================
def _graph_fixation_duration(ax, fix_df, fix_after=None, fix_dur_min: float = 60, fix_dur_max: float = 1000):
    """
    Helperfunction: draw the fixation-duration histogram (ms) onto `ax`.
    Single distribution when fix_after is None; stacked before/after otherwise.

    Args:
        ax: Matplotlib axis to draw on.
        fix_df (pd.DataFrame): fixation events (single, or the 'before' stage).
        fix_after (pd.DataFrame, optional): fixation events for the 'after' stage. Defaults to None.
        fix_dur_min (float): lower bound (ms); shorter fixations dropped (plotting only). Defaults to 60.
        fix_dur_max (float): upper bound (ms); longer fixations dropped (plotting only). Defaults to 1000.
    """

    def prep(df):
        d = df["duration"].dropna() * 1000.0
        total = len(d)
        d = d[(d >= fix_dur_min) & (d <= fix_dur_max)]
        dropout = total - len(d)

        # Log kept / dropped (plotting range only)
        logger.info(f"Total fixations: {total}")
        logger.info(
            f"Only for plotting: Kept fixations ([{fix_dur_min}, {fix_dur_max}] ms): {len(d)}"
        )
        if total > 0:
            logger.info(
                f"Only for plotting: Dropped outliers (outside [{fix_dur_min}, {fix_dur_max}] ms): "
                f"{dropout} ({dropout / total * 100:.2f}%)"
            )
        return d

    if fix_after is None:
        d_before = prep(fix_df)
        if not d_before.empty:
            ax.hist(d_before, bins=40, edgecolor="black")
    else:
        d_before = prep(fix_df)
        d_after = prep(fix_after)
        if not (d_before.empty and d_after.empty):
            ax.hist(
                [d_before, d_after],
                bins=40,
                edgecolor="black",
                stacked=True,
                color=[BEFORE_COLOR, AFTER_COLOR],
                label=["Before Preprocessing", "After Preprocessing"],
            )
            ax.legend(fontsize=7)

    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Fixation Duration")


# =============================================================================
# Fixation frequency
# =============================================================================
def _graph_fixation_frequency(ax, fix_df, fix_after=None):
    """
    Helperfunction: draw the fixation-frequency histogram (fixations per second)
    onto `ax`. Single distribution when fix_after is None; stacked before/after
    otherwise.

    Args:
        ax: Matplotlib axis to draw on.
        fix_df (pd.DataFrame): fixation events (single, or the 'before' stage).
        fix_after (pd.DataFrame, optional): fixation events for the 'after' stage. Defaults to None.
    """

    def per_sec(df):
        f = df.copy()
        f["sec"] = f["onset"].astype(float).floordiv(1).astype(int)
        return f.groupby("sec").size()

    fps_before = per_sec(fix_df)

    if fix_after is None:
        if not fps_before.empty:
            ax.hist(
                fps_before.values,
                bins=np.arange(fps_before.max() + 2) - 0.3,
                width=0.6,
                edgecolor="black",
            )
    else:
        fps_after = per_sec(fix_after)
        max_val = max(
            fps_before.max() if not fps_before.empty else 0,
            fps_after.max() if not fps_after.empty else 0,
        )
        logger.info(
            f"Fixation frequency — mean/s: before {fps_before.mean():.2f}, "
            f"after {fps_after.mean():.2f}."
        )
        ax.hist(
            [fps_before.values, fps_after.values],
            bins=np.arange(max_val + 2) - 0.3,
            width=0.6,
            edgecolor="black",
            stacked=True,
            color=[BEFORE_COLOR, AFTER_COLOR],
            label=["Before Preprocessing", "After Preprocessing"],
        )
        ax.legend(fontsize=7)

    ax.set_xlim(left=-0.3)
    ax.set_xlabel("Fixations per second")
    ax.set_ylabel("Count")
    ax.set_title("Fixation Frequency")


# =============================================================================
# Saccade amplitude
# =============================================================================
def _graph_saccade_amplitude(ax, sacc_df, sacc_after=None, sac_amp_max: float = 40, include_near_blink_sac: bool | str = False):
    """
    Helperfunction: draw the saccade amplitude histogram (deg) onto `ax`.
    Single distribution when sacc_after is None; stacked before/after otherwise.
    With include_near_blink_sac='highlight' (single only), blink saccades are
    drawn as a separate highlighted layer.

    Args:
        ax: Matplotlib axis to draw on.
        sacc_df (pd.DataFrame): saccade events (single, or the 'before' stage).
        sacc_after (pd.DataFrame, optional): saccade events for the 'after' stage. Defaults to None.
        sac_amp_max (float): Upper bound (deg) to drop implausibly large amplitudes (plotting only). Defaults to 40.
        include_near_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
    """

    def prep_data(s_df):
        # 1) Optionally exclude blink saccades (only when include is False)
        if include_near_blink_sac is False:
            s_df, _ = _exclude_blink(s_df, include_near_blink_sac)

        # 2) Select saccades with a valid amplitude, drop outliers (plotting range only)
        s = s_df.dropna(subset=["sacc_visual_angle"])
        total = len(s)
        s = s[s["sacc_visual_angle"] <= sac_amp_max]
        dropout = total - len(s)

        # 3) Log the numbers of items kept / dropped
        logger.info(f"Total saccades: {total}")
        logger.info(f"Only for plotting: Kept saccades (<={sac_amp_max}°): {len(s)}")
        if total > 0:
            logger.info(
                f"Only for plotting: Dropped outliers (>{sac_amp_max}°): {dropout} ({dropout / total * 100:.2f}%)"
            )
        return s

    # 4) Create figure: single, highlighted, or stacked before/after
    if sacc_after is None:
        logger.info("Plotting saccade amplitude histogram (deg) — single distribution.")
        sacc_df = prep_data(sacc_df)
        if sacc_df.empty:
            raise ValueError(f"No saccade amplitudes within 0–{sac_amp_max}° found.")
        if include_near_blink_sac == "highlight":
            saccade = sacc_df.loc[sacc_df["near_blink"] == False, "sacc_visual_angle"]
            blink = sacc_df.loc[sacc_df["near_blink"] == True, "sacc_visual_angle"]
            ax.hist(
                [saccade.values, blink.values],
                bins=40,
                edgecolor="black",
                stacked=True,
                color=[BEFORE_COLOR, BLINK_COLOR],
                label=["saccades", "blink saccades"],
            )
            ax.legend(fontsize=7)
        else:
            ax.hist(sacc_df["sacc_visual_angle"], bins=40, edgecolor="black")

    # stacked figure: before vs. after preprocessing
    else:
        logger.info("Plotting saccade amplitude histogram (deg) — stacked before/after.")
        if include_near_blink_sac == "highlight":
            logger.warning(
                "include_near_blink_sac='highlight' is not supported for stacked before/after plots. Blink saccades will be included without highlighting."
            )
        s_before = prep_data(sacc_df)
        s_after = prep_data(sacc_after)
        if s_before.empty and s_after.empty:
            raise ValueError(f"No saccade amplitudes within 0–{sac_amp_max}° found.")
        ax.hist(
            [s_before["sacc_visual_angle"].values, s_after["sacc_visual_angle"].values],
            bins=40,
            edgecolor="black",
            stacked=True,
            color=[BEFORE_COLOR, AFTER_COLOR],
            label=["Before Preprocessing", "After Preprocessing"],
        )
        ax.legend(fontsize=7)

    # 5) Labels & title
    ax.set_xlabel("Saccade amplitude (deg)")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0)
    title = "Saccade Amplitude"
    if include_near_blink_sac is False:
        title += " (blink saccades excl.)"
    elif include_near_blink_sac == "highlight":
        title += " (blink saccades highlighted)"
    ax.set_title(title)


# =============================================================================
# Saccade duration
# =============================================================================
def _graph_saccade_duration(ax, sacc_df, sacc_after=None, sac_dur_max: float = 120, include_near_blink_sac: bool | str = False):
    """
    Helperfunction: draw the saccade duration histogram (ms) onto `ax`.
    Single distribution when sacc_after is None; stacked before/after otherwise.
    With include_near_blink_sac='highlight' (single only), blink saccades are
    drawn as a separate highlighted layer.

    Args:
        ax: Matplotlib axis to draw on.
        sacc_df (pd.DataFrame): saccade events (single, or the 'before' stage).
        sacc_after (pd.DataFrame, optional): saccade events for the 'after' stage. Defaults to None.
        sac_dur_max (float): Upper bound (ms) to drop implausibly long saccades (plotting only). Defaults to 120.
        include_near_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
    """

    def prep_data(s_df):
        # 1) Optionally exclude blink saccades (only when include is False)
        if include_near_blink_sac is False:
            s_df, _ = _exclude_blink(s_df, include_near_blink_sac)

        # 2) Convert duration from seconds to ms, drop outliers (plotting range only)
        s = s_df.dropna(subset=["duration"]).copy()
        s["duration_ms"] = s["duration"] * 1000
        total = len(s)
        s = s[s["duration_ms"] <= sac_dur_max]
        dropout = total - len(s)

        # 3) Log the numbers of items kept / dropped
        logger.info(f"Total saccades: {total}")
        logger.info(f"Only for plotting: Kept saccades (<={sac_dur_max}ms): {len(s)}")
        if total > 0:
            logger.info(
                f"Only for plotting: Dropped outliers (>{sac_dur_max}ms): {dropout} ({dropout / total * 100:.2f}%)"
            )
        return s

    # 4) Create figure: single, highlighted, or stacked before/after
    if sacc_after is None:
        logger.info("Plotting saccade duration histogram (ms) — single distribution.")
        sacc_df = prep_data(sacc_df)
        if sacc_df.empty:
            raise ValueError(f"No saccade durations within 0–{sac_dur_max}ms found.")
        if include_near_blink_sac == "highlight":
            saccade = sacc_df.loc[sacc_df["near_blink"] == False, "duration_ms"]
            blink = sacc_df.loc[sacc_df["near_blink"] == True, "duration_ms"]
            ax.hist(
                [saccade.values, blink.values],
                bins=40,
                edgecolor="black",
                stacked=True,
                color=[BEFORE_COLOR, BLINK_COLOR],
                label=["saccades", "blink saccades"],
            )
            ax.legend(fontsize=7)
        else:
            ax.hist(sacc_df["duration_ms"], bins=40, edgecolor="black")

    # stacked figure: before vs. after preprocessing
    else:
        logger.info("Plotting saccade duration histogram (ms) — stacked before/after.")
        if include_near_blink_sac == "highlight":
            logger.warning(
                "include_near_blink_sac='highlight' is not supported for stacked before/after plots. Blink saccades will be included without highlighting."
            )
        s_before = prep_data(sacc_df)
        s_after = prep_data(sacc_after)
        if s_before.empty and s_after.empty:
            raise ValueError(f"No saccade durations within 0–{sac_dur_max}ms found.")
        ax.hist(
            [s_before["duration_ms"].values, s_after["duration_ms"].values],
            bins=40,
            edgecolor="black",
            stacked=True,
            color=[BEFORE_COLOR, AFTER_COLOR],
            label=["Before Preprocessing", "After Preprocessing"],
        )
        ax.legend(fontsize=7)

    # 5) Labels & title
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0)
    title = "Saccade Duration"
    if include_near_blink_sac is False:
        title += " (blink saccades excl.)"
    elif include_near_blink_sac == "highlight":
        title += " (blink saccades highlighted)"
    ax.set_title(title)


# =============================================================================
# Saccade angular histogram
# =============================================================================
def _graph_saccade_angles(ax, sacc_df, sacc_after=None, include_near_blink_sac: bool | str = False, style: str = "polar"):
    """
    Helperfunction: draw the saccade direction histogram onto `ax`.
    style='polar'     -> rose plot; ax must be a polar axis (radians, 36 bins).
    style='cartesian' -> bar histogram over 0–360°; ax must be a normal axis (10° bins).
    Single distribution when sacc_after is None; stacked before/after otherwise.
    With include_near_blink_sac='highlight' (single only), blink saccades are
    drawn as a separate highlighted layer.

    Args:
        ax: Matplotlib axis to draw on (polar for style='polar').
        sacc_df (pd.DataFrame): saccade events (single, or the 'before' stage).
        sacc_after (pd.DataFrame, optional): saccade events for the 'after' stage. Defaults to None.
        include_near_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
        style (str): 'polar' or 'cartesian'. Defaults to 'polar'.
    """
    # Histogram bins depend on the plotting style
    bins = 36 if style == "polar" else np.arange(0, 361, 10)

    def prep_data(s_df):
        # 1) Optionally exclude blink saccades (only when include is False)
        if include_near_blink_sac is False:
            s_df, _ = _exclude_blink(s_df, include_near_blink_sac)

        # 2) Compute saccade direction: radians for polar, degrees [0, 360) for cartesian
        s = s_df.copy()
        dx = s["sacc_end_x"] - s["sacc_start_x"]
        dy = s["sacc_end_y"] - s["sacc_start_y"]
        if style == "polar":
            s["angle"] = np.arctan2(dy, dx) % (2 * np.pi)
        else:
            s["angle"] = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        # 3) Log the number of saccades
        logger.info(f"Total saccades: {len(s)}")
        return s

    # 4) Create figure: single, highlighted, or stacked before/after
    if sacc_after is None:
        logger.info("Plotting saccade direction histogram — single distribution.")
        sacc_df = prep_data(sacc_df)
        if sacc_df.empty:
            raise ValueError("No saccade directions found.")
        if include_near_blink_sac == "highlight":
            saccade = sacc_df.loc[sacc_df["near_blink"] == False, "angle"]
            blink = sacc_df.loc[sacc_df["near_blink"] == True, "angle"]
            ax.hist(
                [saccade.values, blink.values],
                bins=bins,
                edgecolor="black",
                stacked=True,
                color=[BEFORE_COLOR, BLINK_COLOR],
                label=["saccades", "blink saccades"],
            )
            ax.legend(fontsize=6)
        else:
            ax.hist(sacc_df["angle"], bins=bins, edgecolor="black")

    # stacked figure: before vs. after preprocessing
    else:
        logger.info("Plotting saccade direction histogram — stacked before/after.")
        if include_near_blink_sac == "highlight":
            logger.warning(
                "include_near_blink_sac='highlight' is not supported for stacked before/after plots. Blink saccades will be included without highlighting."
            )
        s_before = prep_data(sacc_df)
        s_after = prep_data(sacc_after)
        if s_before.empty and s_after.empty:
            raise ValueError("No saccade directions found.")
        ax.hist(
            [s_before["angle"].values, s_after["angle"].values],
            bins=bins,
            edgecolor="black",
            stacked=True,
            color=[BEFORE_COLOR, AFTER_COLOR],
            label=["Before Preprocessing", "After Preprocessing"],
        )
        anchor = (1.15, 1.12) if style == "polar" else (1.0, 1.0)
        ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=anchor)

    # 5) Labels & title
    if style == "polar":
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
    else:
        ax.set_xlabel("Saccade direction (deg)")
        ax.set_ylabel("Count")
    title = "Saccade Directions"
    if include_near_blink_sac is False:
        title += " (blink saccades excl.)"
    elif include_near_blink_sac == "highlight":
        title += " (blink saccades highlighted)"
    ax.set_title(title)


# =============================================================================
# Main-sequence scatter primitive
# =============================================================================
def ms_scatter(sub, include_near_blink_sac, ax_ms, label=None, sac_color=None, blink_color=None):
    """
    Helperfunction: draw main-sequence scatter points onto `ax_ms`.

    Handles three cases: before/after comparison data (carries 'processing_stage'),
    single/all-eye data with 'highlight' (saccades vs. blink saccades in separate
    colours), and plain data (one colour).

    Args:
        sub (pd.DataFrame): saccade events for one group.
        include_near_blink_sac (bool | str): 'highlight' marks blink saccades; else ignored here.
        ax_ms: Matplotlib axis to draw on.
        label (str, optional): legend label for the normal saccades. Defaults to None.
        sac_color (optional): colour for normal saccades. Defaults to None (auto).
        blink_color (optional): colour for blink saccades. Defaults to None (auto).
    """
    if "processing_stage" in sub.columns:
        before = sub[sub["processing_stage"] == "before"]
        after = sub[sub["processing_stage"] == "after"]

        if include_near_blink_sac == "highlight":
            ax_ms.scatter(
                before.loc[before["near_blink"] == False, "sacc_visual_angle"],
                before.loc[before["near_blink"] == False, "peak_velocity"],
                s=6,
                color=BEFORE_COLOR,
                alpha=0.6,
                label="before preprocessing",
            )
            ax_ms.scatter(
                after.loc[after["near_blink"] == False, "sacc_visual_angle"],
                after.loc[after["near_blink"] == False, "peak_velocity"],
                s=6,
                color=AFTER_COLOR,
                alpha=0.6,
                label="after preprocessing",
            )
            flagged = before[before["near_blink"] == True]
            if not flagged.empty:
                ax_ms.scatter(
                    flagged["sacc_visual_angle"],
                    flagged["peak_velocity"],
                    s=6,
                    color=BLINK_COLOR,
                    alpha=0.6,
                    label="blink saccade",
                )
        else:
            ax_ms.scatter(
                before["sacc_visual_angle"],
                before["peak_velocity"],
                s=6,
                color=BEFORE_COLOR,
                alpha=0.6,
                label="before preprocessing",
            )
            ax_ms.scatter(
                after["sacc_visual_angle"],
                after["peak_velocity"],
                s=6,
                color=AFTER_COLOR,
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
            color=sac_color,
            label=label if label is not None else "saccades",
        )
        if not flagged.empty:
            ax_ms.scatter(
                flagged["sacc_visual_angle"],
                flagged["peak_velocity"],
                s=6,
                color=blink_color,
                label=f"{label} blink" if label is not None else "blink saccade",
            )
    else:
        ax_ms.scatter(
            sub["sacc_visual_angle"],
            sub["peak_velocity"],
            s=6,
            label=label,
        )
      