"""
graphs.py: drawing helpers for the visualisationscript plotting.py

Each `_graph_*` function draws exactly one graph type onto a given Matplotlib
axis. The public plot_* / plot_summary functions in plotting.py compose these
onto figures and handle eye selection, titles and saving. Keeping the drawing
logic here means every graph has a single source for bins, axes, thresholds and
blink handling.

Contents
--------
    _exclude_blink        -> small shared helpers
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
# Palette
# =============================================================================
# colour-blind-friendly palette
BEFORE_COLOR = "#0072B2"  # blue   — before preprocessing / saccades
AFTER_COLOR = "#009E73"   # green  — after preprocessing / blink saccades (single eye)
BLINK_COLOR = "#E69F00"   # orange — blink saccades (histograms)
# Fixed colour per eye, when plotting multiple eyes in one graph (by_eye='all')
EYE_COLORS = {
    "L": "#0072B2",          # blue
    "R": "#E69F00",          # orange
    "binocular": "#009E73",  # green
}

# =============================================================================
# Helper's helper: Exclude blink saccades
# =============================================================================
def _exclude_blink(sac, include_blink_sac):
    """
    Helperfunction: optionally drop blink saccades.

    Args:
        sac (pd.DataFrame): saccade events (must contain a 'blink_saccade' column).
        include_blink_sac (bool | str): if False, blink saccades are
            dropped; otherwise the frame is returned unchanged.

    Returns:
        tuple: (saccades, number of blink saccades flagged).
    """
    mask = sac["blink_saccade"] == True
    n_flagged = int(mask.sum())
    if include_blink_sac is False:
        return sac[~mask], n_flagged
    return sac, n_flagged


# =============================================================================
# Main sequence (amplitude vs. peak-velocity, log-log)
# =============================================================================
def _graph_main_sequence(ax, sac_df, include_blink_sac: bool | str = False, by_eye: str = "binocular"):
    """
    Helperfunction: draw the main-sequence scatter (amplitude vs. peak velocity,
    log-log) onto `ax`.

    Works for single and comparison data: comparison data carries a
    'processing_stage' column, which ms_scatter uses to split before/after.

    Args:
        ax: Matplotlib axis to draw on.
        sac_df (pd.DataFrame): saccade events (may carry 'processing_stage').
        include_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
        by_eye (str): 'all' plots one colour per eye; otherwise a single group.
            Defaults to 'binocular'.
    """
    # determine whether this is a comparison plot (df contains 'processing_stage' column) or a single plot
    comparison = "processing_stage" in sac_df.columns
    # deal with blink saccades (excluded, only when include is False)
    s_ms, n_flagged = _exclude_blink(sac_df, include_blink_sac)
    if include_blink_sac is False:
        logger.info(f"Main sequence — excluded {n_flagged} blink saccades.")
    elif include_blink_sac == "highlight":
        logger.info(f"Main sequence — highlighting {n_flagged} blink saccades.")
    elif include_blink_sac is True:
        logger.info(f"Main sequence — including {n_flagged} blink saccades.")

    if by_eye == "all":
        # one colour per eye; blinks in a darker shade of it
        for eye, sub in s_ms.groupby("eye"):
            eye_color = EYE_COLORS.get(str(eye), "#333333")  # Fallback für unbekannte Augen
            r, g, b, a = to_rgba(eye_color)
            blink_color = (r * 0.55, g * 0.55, b * 0.55, a)
            ms_scatter(
                sub, include_blink_sac, ax, label=str(eye),
                sac_color=eye_color, blink_color=blink_color,
            )
        ax.legend(title="Eye", fontsize=7)
    else:
        # single eye: saccades in blue, blink saccades in green
        ms_scatter(
            s_ms, include_blink_sac, ax,
            sac_color=BEFORE_COLOR, blink_color=AFTER_COLOR,
        )
        if comparison or include_blink_sac == "highlight":
            ax.legend(fontsize=7)

    #  Labels & title
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Amplitude (deg)")
    ax.set_ylabel("Peak velocity (deg/s)")
    suffix = {False: " (blink excl.)", "highlight": " (blink highlight.)"}.get(
        include_blink_sac, ""
    )
    ax.set_title(f"Main Sequence{suffix}")


# =============================================================================
# Fixation duration
# =============================================================================
def _graph_fixation_duration(ax, fix_df, fix_after=None, fix_dur_min: float = 60, fix_dur_max: float = 1000):
    """
    Helperfunction: draw the fixation-duration histogram (ms) onto `ax`.
    Single histogram when fix_after is None; stacked before/after otherwise.

    Args:
        ax: Matplotlib axis to draw on.
        fix_df (pd.DataFrame): fixation events (single, or the 'before' stage).
        fix_after (pd.DataFrame, optional): fixation events for the 'after' stage. Defaults to None.
        fix_dur_min (float): lower bound (ms); shorter fixations dropped (plotting only). Defaults to 60.
        fix_dur_max (float): upper bound (ms); longer fixations dropped (plotting only). Defaults to 1000.
    """

    def prep_data(f_df):
        # convert seconds → ms
        dur = f_df["duration"].dropna() * 1000.0

        # Filter by plausible duration range
        dur = dur[(dur >= fix_dur_min) & (dur <= fix_dur_max)]
        total = len(dur)
        dropout = total - len(dur)

        # Log kept / dropped (plotting range only)
        logger.info(f"Total fixations: {total}")
        if dur.empty:
            raise ValueError(
                "No fixation durations post filtering. Check inputs or ranges."
            )
        else:
            logger.info(
                f"Only for plotting: Kept fixations ([{fix_dur_min}, {fix_dur_max}] ms): {len(dur)}"
                )
            logger.info(
                f"Only for plotting: Dropped outliers (outside [{fix_dur_min}, {fix_dur_max}] ms): "
                f"{dropout} ({dropout / total * 100:.2f}%)"
                )

        return dur

    # Get the data to plot: single figure
    if fix_after is None:
        fix_dur = prep_data(fix_df)
        if not fix_dur.empty:
            ax.hist(fix_dur, bins=40, edgecolor="black")
    # stacked figure: before vs. after preprocessing
    else:
        fix_before = prep_data(fix_df)
        fix_after = prep_data(fix_after)
        if not (fix_before.empty and fix_after.empty):
            ax.hist(
                [fix_before, fix_after],
                bins=40,
                edgecolor="black",
                stacked=True,
                color=[BEFORE_COLOR, AFTER_COLOR],
                label=["Before Preprocessing", "After Preprocessing"],
            )
            ax.legend(fontsize=7)

    #  Labels & title
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Fixation Duration")


# =============================================================================
# Fixation frequency
# =============================================================================
def _graph_fixation_frequency(ax, fix_df, fix_after=None):
    """
    Helperfunction: draw the fixation-frequency histogram (fixations per second)
    onto `ax`. Single histogramm when fix_after is None; stacked before/after
    otherwise.

    Args:
        ax: Matplotlib axis to draw on.
        fix_df (pd.DataFrame): fixation events (single, or the 'before' stage).
        fix_after (pd.DataFrame, optional): fixation events for the 'after' stage. Defaults to None.
    """
    # Get the data to plot: single figure
    if fix_after is None:
        # convert onset from seconds → ms, then group by second and count fixations
        fix_df["sec"] = fix_df["onset"].astype(float).floordiv(1000.0).astype(int)
        fix_per_sec = fix_df.groupby("sec").size()

        ax.hist(
            fix_per_sec.values,
            bins=np.arange(fix_per_sec.max() + 2) - 0.3,
            width=0.6,
            edgecolor="black",
        )
    # stacked figure: before vs. after preprocessing
    else:
        # convert onset from seconds → ms, then group by second and count fixations
        fix_df["sec"] = fix_df["onset"].astype(float).floordiv(1000.0).astype(int)
        fix_before = fix_df.groupby("sec").size()
        fix_after["sec"] = fix_after["onset"].astype(float).floordiv(1000.0).astype(int)
        fix_after = fix_after.groupby("sec").size()

        logger.info(
            f"Fixation frequency — mean/s: before {fix_before.mean():.2f}, "
            f"after {fix_after.mean():.2f}."
        )

        ax.hist(
            [fix_before.values, fix_after.values],
            bins=np.arange(max(fix_before.max(), fix_after.max()) + 1 - 0.3),
            width=0.6,
            edgecolor="black",
            stacked=True,
            color=[BEFORE_COLOR, AFTER_COLOR],
            label=["Before Preprocessing", "After Preprocessing"],
        )
        ax.legend(fontsize=7)

    #  Labels & title
    ax.set_xlim(left=-0.3)
    ax.set_xlabel("Fixations per second")
    ax.set_ylabel("Count")
    ax.set_title("Fixation Frequency")


# =============================================================================
# Saccade amplitude
# =============================================================================
def _graph_saccade_amplitude(ax, sac_df, sac_after=None, sac_amp_max: float = 40, include_blink_sac: bool | str = False):
    """
    Helperfunction: draw the saccade amplitude histogram (deg) onto `ax`.
    Single histogram when sac_after is None; stacked before/after otherwise.
    With include_blink_sac='highlight' (single only), blink saccades are
    drawn as a separate highlighted layer.

    Args:
        ax: Matplotlib axis to draw on.
        sac_df (pd.DataFrame): saccade events (single, or the 'before' stage).
        sac_after (pd.DataFrame, optional): saccade events for the 'after' stage. Defaults to None.
        sac_amp_max (float): Upper bound (deg) to drop implausibly large amplitudes (plotting only). Defaults to 40.
        include_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
    """

    def prep_data(s_df):
        #  Optionally exclude blink saccades (only when include is False)
        if include_blink_sac is False:
            s_df, _ = _exclude_blink(s_df, include_blink_sac)

        #  Select saccades with a valid amplitude, drop outliers (plotting range only)
        s = s_df.dropna(subset=["sacc_visual_angle"])
        total = len(s)
        s = s[s["sacc_visual_angle"] <= sac_amp_max]
        dropout = total - len(s)

        #  Log the numbers of items kept / dropped
        logger.info(f"Total saccades: {total}")
        logger.info(f"Only for plotting: Kept saccades (<={sac_amp_max}°): {len(s)}")
        if total > 0:
            logger.info(
                f"Only for plotting: Dropped outliers (>{sac_amp_max}°): {dropout} ({dropout / total * 100:.2f}%)"
            )
        return s

    #  Create figure: single, highlighted, or stacked before/after
    if sac_after is None:
        logger.info("Plotting saccade amplitude histogram (deg) — single histogram.")
        sac_df = prep_data(sac_df)
        if sac_df.empty:
            raise ValueError(f"No saccade amplitudes within 0–{sac_amp_max}° found.")
        if include_blink_sac == "highlight":
            saccade = sac_df.loc[sac_df["blink_saccade"] == False, "sacc_visual_angle"]
            blink = sac_df.loc[sac_df["blink_saccade"] == True, "sacc_visual_angle"]
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
            ax.hist(sac_df["sacc_visual_angle"], bins=40, edgecolor="black")

    # stacked figure: before vs. after preprocessing
    else:
        logger.info("Plotting saccade amplitude histogram (deg) — stacked before/after.")
        if include_blink_sac == "highlight":
            logger.warning(
                "include_blink_sac='highlight' is not supported for stacked before/after plots. Blink saccades will be included without highlighting."
            )
        s_before = prep_data(sac_df)
        s_after = prep_data(sac_after)
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

    #  Labels & title
    ax.set_xlabel("Saccade amplitude (deg)")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0)
    title = "Saccade Amplitude"
    if include_blink_sac is False:
        title += " (blink saccades excl.)"
    elif include_blink_sac == "highlight":
        title += " (blink saccades highlighted)"
    ax.set_title(title)


# =============================================================================
# Saccade duration
# =============================================================================
def _graph_saccade_duration(ax, sac_df, sac_after=None, sac_dur_max: float = 120, include_blink_sac: bool | str = False):
    """
    Helperfunction: draw the saccade duration histogram (ms) onto `ax`.
    Single histogram when sac_after is None; stacked before/after otherwise.
    With include_blink_sac='highlight' (single only), blink saccades are
    drawn as a separate highlighted layer.

    Args:
        ax: Matplotlib axis to draw on.
        sac_df (pd.DataFrame): saccade events (single, or the 'before' stage).
        sac_after (pd.DataFrame, optional): saccade events for the 'after' stage. Defaults to None.
        sac_dur_max (float): Upper bound (ms) to drop implausibly long saccades (plotting only). Defaults to 120.
        include_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
    """

    def prep_data(s_df):
        #  Optionally exclude blink saccades (only when include is False)
        if include_blink_sac is False:
            s_df, _ = _exclude_blink(s_df, include_blink_sac)

        #  Convert duration from seconds to ms, drop outliers (plotting range only)
        s = s_df.dropna(subset=["duration"]).copy()
        s["duration_ms"] = s["duration"] * 1000
        total = len(s)
        s = s[s["duration_ms"] <= sac_dur_max]
        dropout = total - len(s)

        #  Log the numbers of items kept / dropped
        logger.info(f"Total saccades: {total}")
        logger.info(f"Only for plotting: Kept saccades (<={sac_dur_max}ms): {len(s)}")
        if total > 0:
            logger.info(
                f"Only for plotting: Dropped outliers (>{sac_dur_max}ms): {dropout} ({dropout / total * 100:.2f}%)"
            )
        return s

    #  Create figure: single, highlighted, or stacked before/after
    if sac_after is None:
        logger.info("Plotting saccade duration histogram (ms) — single histogram.")
        sac_df = prep_data(sac_df)
        if sac_df.empty:
            raise ValueError(f"No saccade durations within 0–{sac_dur_max}ms found.")
        if include_blink_sac == "highlight":
            saccade = sac_df.loc[sac_df["blink_saccade"] == False, "duration_ms"]
            blink = sac_df.loc[sac_df["blink_saccade"] == True, "duration_ms"]
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
            ax.hist(sac_df["duration_ms"], bins=40, edgecolor="black")

    # stacked figure: before vs. after preprocessing
    else:
        logger.info("Plotting saccade duration histogram (ms) — stacked before/after.")
        if include_blink_sac == "highlight":
            logger.warning(
                "include_blink_sac='highlight' is not supported for stacked before/after plots. Blink saccades will be included without highlighting."
            )
        s_before = prep_data(sac_df)
        s_after = prep_data(sac_after)
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

    #  Labels & title
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0)
    title = "Saccade Duration"
    if include_blink_sac is False:
        title += " (blink saccades excl.)"
    elif include_blink_sac == "highlight":
        title += " (blink saccades highlighted)"
    ax.set_title(title)


# =============================================================================
# Saccade angular histogram
# =============================================================================
def _graph_saccade_angles(ax, sac_df, sac_after=None, include_blink_sac: bool | str = False, style: str = "polar"):
    """
    Helperfunction: draw the saccade direction histogram onto `ax`.
    style='polar'     -> rose plot; ax must be a polar axis (radians, 36 bins).
    style='cartesian' -> bar histogram over 0–360°; ax must be a normal axis (10° bins).
    Single histogram when sac_after is None; stacked before/after otherwise.
    With include_blink_sac='highlight' (single only), blink saccades are
    drawn as a separate highlighted layer.

    Args:
        ax: Matplotlib axis to draw on (polar for style='polar').
        sac_df (pd.DataFrame): saccade events (single, or the 'before' stage).
        sac_after (pd.DataFrame, optional): saccade events for the 'after' stage. Defaults to None.
        include_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
        style (str): 'polar' or 'cartesian'. Defaults to 'polar'.
    """
    # Histogram bins depend on the plotting style
    bins = 36 if style == "polar" else np.arange(0, 361, 10)

    def prep_data(s_df):
        #  Optionally exclude blink saccades (only when include is False)
        if include_blink_sac is False:
            s_df, _ = _exclude_blink(s_df, include_blink_sac)

        #  Compute saccade direction: radians for polar, degrees [0, 360) for cartesian
        s = s_df.copy()
        dx = s["sac_end_x"] - s["sac_start_x"]
        dy = s["sac_end_y"] - s["sac_start_y"]
        if style == "polar":
            s["angle"] = np.arctan2(dy, dx) % (2 * np.pi)
        else:
            s["angle"] = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        #  Log the number of saccades
        logger.info(f"Total saccades: {len(s)}")
        return s

    #  Create figure: single, highlighted, or stacked before/after
    if sac_after is None:
        logger.info("Plotting saccade direction histogram — single histogram.")
        sac_df = prep_data(sac_df)
        if sac_df.empty:
            raise ValueError("No saccade directions found.")
        if include_blink_sac == "highlight":
            saccade = sac_df.loc[sac_df["blink_saccade"] == False, "angle"]
            blink = sac_df.loc[sac_df["blink_saccade"] == True, "angle"]
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
            ax.hist(sac_df["angle"], bins=bins, edgecolor="black")

    # stacked figure: before vs. after preprocessing
    else:
        logger.info("Plotting saccade direction histogram — stacked before/after.")
        if include_blink_sac == "highlight":
            logger.warning(
                "include_blink_sac='highlight' is not supported for stacked before/after plots. Blink saccades will be included without highlighting."
            )
        s_before = prep_data(sac_df)
        s_after = prep_data(sac_after)
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
        anchor = (1.15, 1.1 if style == "polar" else (1.0, 1.0))
        ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=anchor)

    #  Labels & title
    if style == "polar":
        ax.set_theta_zero_location("E")  # 0° to the right
        ax.set_theta_direction(1)  # counter-clockwise
    else:
        ax.set_xlabel("Saccade direction (deg)")
        ax.set_ylabel("Count")
    title = "Saccade Directions"
    if include_blink_sac is False:
        title += " (blink saccades excl.)"
    elif include_blink_sac == "highlight":
        title += " (blink saccades highlighted)"
    ax.set_title(title)


# =============================================================================
# Main-sequence scatter primitive
# =============================================================================
def ms_scatter(sub, include_blink_sac, ax_ms, label=None, sac_color=None, blink_color=None):
    """
    Helperfunction: draw main-sequence scatter points

    Handles three cases: 
    - before/after preprocessing data (carries 'processing_stage'),
    - multiple eye data vs. single eye data (e.g., only right eye)
    - how to deal with saccades: 'highlight'/exclude/include blink saccades 

    Args:
        sub (pd.DataFrame): saccade events for one group.
        include_blink_sac (bool | str): 'highlight' marks blink saccades; else ignored here.
        ax_ms: Matplotlib axis to draw on.
        label (str, optional): legend label for the normal saccades. Defaults to None.
        sac_color (optional): colour for normal saccades. Defaults to none (auto).
        blink_color (optional): colour for blink saccades. Defaults to none (auto).
    """
    # Case: comparison data (before/after preprocessing) — split by 'processing_stage'
    if "processing_stage" in sub.columns:
        before = sub[sub["processing_stage"] == "before"]
        after = sub[sub["processing_stage"] == "after"]

        if include_blink_sac == "highlight":
            ax_ms.scatter(
                before.loc[before["blink_saccade"] == False, "sacc_visual_angle"],
                before.loc[before["blink_saccade"] == False, "peak_velocity"],
                s=6,
                color=BEFORE_COLOR,
                alpha=0.6,
                label="before preprocessing",
            )
            ax_ms.scatter(
                after.loc[after["blink_saccade"] == False, "sacc_visual_angle"],
                after.loc[after["blink_saccade"] == False, "peak_velocity"],
                s=6,
                color=AFTER_COLOR,
                alpha=0.6,
                label="after preprocessing",
            )
            flagged = before[before["blink_saccade"] == True]
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

    elif include_blink_sac == "highlight":
        normal = sub[sub["blink_saccade"] == False]
        flagged = sub[sub["blink_saccade"] == True]
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
            color=sac_color,
            label=label,
        )