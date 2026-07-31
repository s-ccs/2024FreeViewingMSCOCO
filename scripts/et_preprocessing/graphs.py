"""
graphs.py: drawing helpers for the visualisation script plotting.py

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

# colours
BEFORE_COLOR = "#0072B2"  # blue   — before preprocessing / saccades
AFTER_COLOR = "#D55E00"   # orange — after preprocessing / blink saccades (single eye)
BLINK_COLOR = "#E69F00"   # orange — blink saccades (histograms)
BLINK_MS_COLOR = "#009E73"  # green  — blink saccades (main sequence)
# Fixed colour per eye, when plotting multiple eyes in one graph (by_eye='all')
EYE_COLORS = {
    "L": "#0072B2",          # blue
    "R": "#E69F00",          # orange
    "binocular": "#009E73",  # green
}
# Dropout accounting (values dropped for plotting only)
_DROPOUT_STATS = {}

def _save_dropout(metric, total, kept, window="", stage="single"):
    """Record how many values were dropped for plotting (outside display range)."""
    _DROPOUT_STATS[f"{metric}|{stage}"] = {
        "metric": metric, "stage": stage, "window": window,
        "total": total, "kept": kept, "dropped": total - kept,
        "pct": ((total - kept) / total * 100) if total else 0.0,
    }


def reset_dropout_stats():
    """Clear the dropout accumulator (call before the plot you want to document)."""
    _DROPOUT_STATS.clear()


def get_dropout_stats() -> dict:
    """Return a copy of the recorded dropout stats (keyed by metric)."""
    return {k: dict(v) for k, v in _DROPOUT_STATS.items()}


# =============================================================================
# Helper's helper: Exclude blink saccades
# =============================================================================
def _exclude_blink(sac, include_blink_sac):
    """
    Helper function: count and optionally drop blink saccades.

    Args:
        sac (pd.DataFrame): saccade events (must contain a 'blink_saccade' column).
        include_blink_sac (bool | str): if False, blink saccades are
            dropped; otherwise the frame is returned unchanged.

    Returns:
        tuple: (saccades, number of blink saccades flagged).
    """
    mask = sac["blink_saccade"] == True
    n_flagged = int(mask.sum())
    if include_blink_sac is False: # drop blink saccades
        return sac[~mask], n_flagged
    return sac, n_flagged


def _overlap_hist(ax, before, after, bins, before_blink=None):
    """
    Helperfunction: draw an overlapping before/after histogram onto `ax`.
    'after' is filled (semi-transparent); 'before' is drawn as an outline on top,
    so the part of 'before' sticking out above the fill is what preprocessing removed.
    If `before_blink` is given, 'before' is drawn as two stacked outlines:
    non-blink (blue) at the bottom and blink saccades (green) stacked on top, so the
    blue->green band marks the blink saccades removed before the merge.

    Args:
        ax: Matplotlib axis to draw on (works for cartesian and polar axes).
        before: values of the before distribution (non-blink part if `before_blink` is given).
        after: values of the post-processing (after) distribution.
        bins: bin edges (or bin count) shared by both histograms.
        before_blink: optional blink-saccade values of the before stage; if given, 'before'
            is rendered as stacked blue (non-blink) + green (blink) outlines.
    """
    if np.isscalar(bins):
        allv = np.concatenate([np.asarray(before), np.asarray(after)]
                          + ([np.asarray(before_blink)] if before_blink is not None else []))
        bins = np.histogram_bin_edges(allv, bins=bins)
    if before_blink is None:
        ax.hist(before, bins=bins, histtype="step", color=BEFORE_COLOR, linewidth=1.8,
                zorder=3, label="before preprocessing")
    else:
        ax.hist([before, before_blink], bins=bins, histtype="step", stacked=True,
                color=[BEFORE_COLOR, BLINK_MS_COLOR], linewidth=1.8, zorder=3,
                label=["before", "blink saccades"])
    ax.hist(after, bins=bins, histtype="stepfilled", color=AFTER_COLOR, alpha=0.45,
            edgecolor="black", zorder=2, label="after preprocessing")

def compute_saccade_directions(s_df, style):
    """Add a saccade direction angle to the dataframe.

    Args:
        s_df (pd.DataFrame): Saccades with sacc_start/end_x/y columns.
        style (str): "polar" for radians [0, 2π), else "degrees" [0, 360).

    Returns:
        pd.DataFrame: Copy of s_df with an added "angle" column.
    """
    
    #  Compute saccade direction: radians for polar, degrees [0, 360) for cartesian
    s = s_df.copy()
    dx = s["sacc_end_x"] - s["sacc_start_x"]
    dy = s["sacc_end_y"] - s["sacc_start_y"]
    # Flip the sign because ET coordinate system has its origin in the top left
    # (instead of bottom left) and the y-axis needs to be flipped for the angle computation
    dy = -dy 
    if style == "polar":
        s["angle"] = np.arctan2(dy, dx) % (2 * np.pi)
    else:
        s["angle"] = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    return s

# =============================================================================
# Main sequence
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
            eye_color = EYE_COLORS.get(str(eye), "#333333")
            r, g, b, a = to_rgba(eye_color)
            blink_color = (r * 0.55, g * 0.55, b * 0.55, a)
            # plot multiple eyes: saccades in eye colour, blink saccades in darker shade
            ms_scatter(
                sub, include_blink_sac, ax, label=str(eye),
                sac_color=eye_color, blink_color=blink_color,
            )
        ax.legend(title="Eye", fontsize=7)
    else:
        # plot single eye: saccades in blue, blink saccades in green
        ms_scatter(
            s_ms, include_blink_sac, ax,
            sac_color=BEFORE_COLOR, blink_color=BLINK_MS_COLOR,
        )
        if comparison or include_blink_sac == "highlight":
            ax.legend(fontsize=7)

    #  Labels & title
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Amplitude (deg)")
    ax.set_ylabel("Peak velocity (deg/s)")
    suffix = {False: " (blink saccades excl.)", "highlight": " (blink highlight.)", True: " (blink saccades incl.)"}.get(
        include_blink_sac, ""
    )
    ax.set_title(f"Main Sequence{suffix}")


# =============================================================================
# Fixation duration
# =============================================================================
def _graph_fixation_duration(ax, fix_df, fix_after=None, fix_dur_min: float | None = 60, fix_dur_max: float | None = 1000, dropout_stats=False):
    """
    Helperfunction: draw the fixation-duration histogram (ms) onto `ax`.
    Single histogram when fix_after is None; stacked before/after otherwise.

    Args:
        ax: Matplotlib axis to draw on.
        fix_df (pd.DataFrame): fixation events (single, or the 'before' stage).
        fix_after (pd.DataFrame, optional): fixation events for the 'after' stage. Defaults to None.
        fix_dur_min (float): lower bound (ms); shorter fixations dropped (plotting only). Defaults to 60. Pass 'None' for no threshold.
        fix_dur_max (float): upper bound (ms); longer fixations dropped (plotting only). Defaults to 1000. Pass 'None' for no threshold.
        dropout_stats (boolean): whether to save the dropout statistics; only works when one dataset is passed
    """
    if fix_dur_min is None: 
        lo = 0.0
    else:
        lo = float(fix_dur_min)
    if fix_dur_max is None:
        hi = float("inf")
    else:
        hi = float(fix_dur_max)

    window = f"[{lo:.0f} - {'inf' if np.isinf(hi) else f'{hi:.0f}'}] ms"

    def prep_data(f_df, dropout_stats, stage):
        # convert seconds → ms
        dur = f_df["duration"].dropna() * 1000.0
        total = len(dur)

        # Filter by plausible duration range
        dur = dur[(dur >= lo) & (dur <= hi)]
        dropout = total - len(dur)

        # Log kept / dropped (plotting range only)
        logger.info(f"Total fixations: {total}")
        if dur.empty:
            raise ValueError("No fixation durations post filtering. Check inputs or ranges.")
        logger.info(f"Only for plotting: Kept fixations ({window}): {len(dur)}")
        logger.info(
            f"Only for plotting: Dropped outliers (outside {window}): "
            f"{dropout} ({dropout / total * 100:.2f}%)"
        )
        if dropout_stats:
            _save_dropout(metric="Fixation duration", total=total, kept=len(dur),
                          window=window, stage=stage)
        return dur

    # Get the data to plot: single figure
    if fix_after is None:
        logger.info("Plotting fixation duration histogram (ms) — single histogram.")

        # Prep the data for plotting
        fix_dur = prep_data(fix_df, dropout_stats=dropout_stats, stage="single")
        if fix_dur.empty:
            raise ValueError(
                f"No fixation durations within [{fix_dur_min}, {fix_dur_max}] ms found."
            )

        # plot
        ax.hist(fix_dur, bins=40, edgecolor="black")

    # stacked figure: before vs. after preprocessing
    else:
        logger.info("Plotting fixation duration histogram (ms) — stacked before/after.")

        # prep the data for plotting
        fix_before = prep_data(fix_df, dropout_stats=dropout_stats, stage="before")
        fix_after = prep_data(fix_after, dropout_stats=dropout_stats, stage="after")
        if fix_before.empty or fix_after.empty:
            raise ValueError(
                f"No fixation durations within [{fix_dur_min}, {fix_dur_max}] ms found."
            )

        # plot
        _overlap_hist(ax, fix_before, fix_after, 40)
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
    onto `ax`. Single histogram when fix_after is None; stacked before/after otherwise.

    Args:
        ax: Matplotlib axis to draw on.
        fix_df (pd.DataFrame): fixation events (single, or the 'before' stage).
        fix_after (pd.DataFrame, optional): fixation events for the 'after' stage. Defaults to None.
    """

    def per_sec(df):
        # onset is in SECONDS -> floor to the integer second, then count fixations per second
        sec = df["onset"].floordiv(1).astype(int)
        full = np.arange(int(df["onset"].min()), int(df["onset"].max()) + 1)
        return sec.value_counts().reindex(full, fill_value=0).sort_index()

    # single figure
    if fix_after is None:
        fix_per_sec = per_sec(fix_df)
        ax.hist(
            fix_per_sec.values,
            bins=np.arange(fix_per_sec.max() + 2) - 0.5, # center bins on integer values
            rwidth=0.6,
            edgecolor="black",
        )
    # stacked figure: before vs. after preprocessing
    else:
        logger.info("Plotting fixation frequency histogram (fixations per second) — stacked before/after.")
        # prep the data for plotting
        fix_before = per_sec(fix_df)
        fix_after = per_sec(fix_after)

        # logger.info(
        #    f"Fixation frequency — mean/s: before {fix_before.mean():.2f}, "
        #    f"after {fix_after.mean():.2f}."
        #)

        # maximum value across both datasets for consistent binning
        max_val = max(fix_before.max(), fix_after.max())
        # plot: bar-type overlap with rwidth (same bar width as the single after-processing plot)
        bins = np.arange(max_val + 2) - 0.5
        ax.hist(fix_before.values, bins=bins, rwidth=0.6, facecolor="none",
                edgecolor=BEFORE_COLOR, linewidth=1.8, zorder=3, label="before preprocessing")
        ax.hist(fix_after.values, bins=bins, rwidth=0.6, color=AFTER_COLOR, alpha=0.45,
                edgecolor="black", zorder=2, label="after preprocessing")
        ax.legend(fontsize=7)

    # Labels & title
    ax.set_xlim(left=-0.3)
    ax.set_xlabel("Fixations per second")
    ax.set_ylabel("Count")
    ax.set_title("Fixation Frequency")


# =============================================================================
# Saccade amplitude
# =============================================================================
def _graph_saccade_amplitude(ax, sac_df, sac_after=None, sac_amp_max: float | None = 40, include_blink_sac: bool | str = False, dropout_stats: bool = False):
    """
    Helperfunction: draw the saccade amplitude histogram (deg) onto `ax`.
    Single histogram when sac_after is None; stacked before/after otherwise.
    With include_blink_sac='highlight' (single only), blink saccades are
    drawn as a separate highlighted layer.

    Args:
        ax: Matplotlib axis to draw on.
        sac_df (pd.DataFrame): saccade events (single, or the 'before' stage).
        sac_after (pd.DataFrame, optional): saccade events for the 'after' stage. Defaults to None.
        sac_amp_max (float): Upper bound (deg) to drop implausibly large amplitudes (plotting only). Defaults to 40. Pass 'None' for no threshold.
        include_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
        dropout_stats (boolean): whether to save the dropout statistics; only works when one dataset is passed. Defaults to False.
    """
    if sac_amp_max is None:
        amp_max = float("inf")
    else:
        amp_max = float(sac_amp_max)
    window = f"[0 - {'inf' if np.isinf(amp_max) else f'{amp_max:.0f}'}] deg"

    def prep_data(s_df, dropout_stats, stage):
        #  Optionally exclude blink saccades (only when include is False)
        if include_blink_sac is False:
            s_df, _ = _exclude_blink(s_df, include_blink_sac)

        #  Select saccades with a valid amplitude, drop outliers (plotting range only)
        s = s_df.dropna(subset=["sacc_visual_angle"])
        total = len(s)
        s = s[s["sacc_visual_angle"] <= amp_max]
        dropout = total - len(s)

        #  Log the numbers of items kept / dropped
        logger.info(f"Total saccades: {total}")
        logger.info(f"Only for plotting: Kept saccades {window}): {len(s)}")
        if total > 0:
            logger.info(
                f"Only for plotting: Dropped outliers (outside {window}): {dropout} ({dropout / total * 100:.2f}%)"
            )

        if dropout_stats:
            _save_dropout(metric = "Saccade amplitude", total = total, kept = len(s),
                    window=f"[0 - {amp_max:.0f}] deg", stage=stage)
        return s

    #  Create figure: single, highlighted, or stacked before/after
    if sac_after is None:
        logger.info("Plotting saccade amplitude histogram (deg) — single histogram.")

        # prep the data for plotting
        sac_df = prep_data(sac_df, dropout_stats=dropout_stats, stage="single")
        if sac_df.empty:
            raise ValueError(f"No saccade amplitudes within 0–{sac_amp_max}° found.")
        if include_blink_sac == "highlight":
            # separate the blink saccades from the normal saccades for highlighting
            saccade = sac_df.loc[sac_df["blink_saccade"] == False, "sacc_visual_angle"]
            blink = sac_df.loc[sac_df["blink_saccade"] == True, "sacc_visual_angle"]

            # plot
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

        # prep the data for plotting
        s_before = prep_data(sac_df, dropout_stats=dropout_stats, stage="before")
        s_after = prep_data(sac_after, dropout_stats=dropout_stats, stage="after")
        if s_before.empty and s_after.empty:
            raise ValueError(f"No saccade amplitudes within 0–{sac_amp_max}° found.")

        # plot: after filled behind, before as outline on top; highlight blink saccades (green)
        if include_blink_sac is False:
            _overlap_hist(ax, s_before["sacc_visual_angle"].values,
                          s_after["sacc_visual_angle"].values, 40)
        else:
            before_nb = s_before.loc[~s_before["blink_saccade"], "sacc_visual_angle"].values
            before_bl = s_before.loc[s_before["blink_saccade"], "sacc_visual_angle"].values
            after_nb = s_after.loc[~s_after["blink_saccade"], "sacc_visual_angle"].values
            _overlap_hist(ax, before_nb, after_nb, 40, before_blink=before_bl)
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
    else:
        title += " (blink saccades incl.)"
    ax.set_title(title)


# =============================================================================
# Saccade duration
# =============================================================================
def _graph_saccade_duration(ax, sac_df, sac_after=None, sac_dur_max: float | None = 120, include_blink_sac: bool | str = False, dropout_stats: bool = False):
    """
    Helperfunction: draw the saccade duration histogram (ms) onto `ax`.
    Single histogram when sac_after is None; stacked before/after otherwise.
    With include_blink_sac='highlight' (single only), blink saccades are
    drawn as a separate highlighted layer.

    Args:
        ax: Matplotlib axis to draw on.
        sac_df (pd.DataFrame): saccade events (single, or the 'before' stage).
        sac_after (pd.DataFrame, optional): saccade events for the 'after' stage. Defaults to None.
        sac_dur_max (float): Upper bound (ms) to drop implausibly long saccades (plotting only). Defaults to 120. Pass 'None' for no threshold.
        include_blink_sac (bool | str): False excludes blink saccades,
            'highlight' marks them, True includes them without marking. Defaults to False.
        dropout_stats (boolean): whether to save the dropout statistics; only works when one dataset is passed. Defaults to False.
    """
    if sac_dur_max is None:
        dur_max = float("inf")
    else:
        dur_max = float(sac_dur_max)
    window = f"[0 - {'inf' if np.isinf(dur_max) else f'{dur_max:.0f}'}] deg"

    def prep_data(s_df, dropout_stats, stage):
        #  Optionally exclude blink saccades (only when include is False)
        if include_blink_sac is False:
            s_df, _ = _exclude_blink(s_df, include_blink_sac)

        #  Convert duration from seconds to ms, drop outliers (plotting range only)
        s = s_df.dropna(subset=["duration"]).copy()
        s["duration_ms"] = s["duration"] * 1000
        total = len(s)
        s = s[s["duration_ms"] <= dur_max]
        dropout = total - len(s)

        #  Log the numbers of items kept / dropped
        logger.info(f"Total saccades: {total}")
        logger.info(f"Only for plotting: Kept saccades ({window}): {len(s)}")
        if total > 0:
            logger.info(
                f"Only for plotting: Dropped outliers (outside {window}): "
                f"{dropout} ({dropout / total * 100:.2f}%)"
            )
        if dropout_stats:
            _save_dropout(metric="Saccade duration", total=total, kept=len(s),
                          window=window, stage=stage)
        return s

    #  Create figure: single, highlighted, or stacked before/after
    if sac_after is None:
        logger.info("Plotting saccade duration histogram (ms) — single histogram.")

        # prep the data for plotting
        sac_df = prep_data(sac_df, dropout_stats = dropout_stats, stage="single")
        if sac_df.empty:
            raise ValueError(f"No saccade durations within 0–{sac_dur_max}ms found.")

        if include_blink_sac == "highlight":
            # separate the blink saccades from the normal saccades for highlighting
            saccade = sac_df.loc[sac_df["blink_saccade"] == False, "duration_ms"]
            blink = sac_df.loc[sac_df["blink_saccade"] == True, "duration_ms"]

            # plot
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

        # prep the data for plotting
        s_before = prep_data(sac_df, dropout_stats=dropout_stats, stage="before")
        s_after = prep_data(sac_after, dropout_stats=dropout_stats, stage="after")
        if s_before.empty and s_after.empty:
            raise ValueError(f"No saccade durations within 0–{sac_dur_max}ms found.")

        # plot: after filled behind, before as outline on top; highlight blink saccades (green)
        if include_blink_sac is False:
            _overlap_hist(ax, s_before["duration_ms"].values,
                          s_after["duration_ms"].values, 40)
        else:
            before_nb = s_before.loc[~s_before["blink_saccade"], "duration_ms"].values
            before_bl = s_before.loc[s_before["blink_saccade"], "duration_ms"].values
            after_nb = s_after.loc[~s_after["blink_saccade"], "duration_ms"].values
            _overlap_hist(ax, before_nb, after_nb, 40, before_blink=before_bl)
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
    else:
        title += " (blink saccades incl.)"
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
        s = compute_saccade_directions(s_df, style)

        #  Log the number of saccades
        logger.info(f"Total saccades: {len(s)}")
        return s

    #  Create figure: single, highlighted, or stacked before/after
    if sac_after is None:
        logger.info("Plotting saccade direction histogram — single histogram.")

        # prep the data for plotting
        sac_df = prep_data(sac_df)
        if sac_df.empty:
            raise ValueError("No saccade directions found.")

        if include_blink_sac == "highlight":
            # separate the blink saccades from the normal saccades for highlighting
            saccade = sac_df.loc[sac_df["blink_saccade"] == False, "angle"]
            blink = sac_df.loc[sac_df["blink_saccade"] == True, "angle"]

            # plot
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

        # prep the data for plotting
        s_before = prep_data(sac_df)
        s_after = prep_data(sac_after)
        if s_before.empty and s_after.empty:
            raise ValueError("No saccade directions found.")

        # plot: after filled behind, before as outline on top; highlight blink saccades (green)
        if include_blink_sac is False:
            _overlap_hist(ax, s_before["angle"].values, s_after["angle"].values, bins)
        else:
            before_nb = s_before.loc[~s_before["blink_saccade"], "angle"].values
            before_bl = s_before.loc[s_before["blink_saccade"], "angle"].values
            after_nb = s_after.loc[~s_after["blink_saccade"], "angle"].values
            _overlap_hist(ax, before_nb, after_nb, bins, before_blink=before_bl)
        anchor = (1.15, 1.12) if style == "polar" else (1.0, 1.0)
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
    else:
        title += " (blink saccades incl.)"
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
                    color=BLINK_MS_COLOR,
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