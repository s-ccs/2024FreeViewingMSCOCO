."""
visualisation.py: Eye-tracking visualisation.

Overview:

Preprocessing comparison:
    plot_eye_trace_both_eyes() -> Horizontal eye trace before vs. after merging (Hooge et al. 2022)

Main sequence:
    plot_main_sequence()
    detect_main_sequence_outliers()
    log_main_sequence_outliers()

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
    MS_OUTLIER_MAD_THRESH,
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


# =============================================================================
# 2.2.2 Preprocessing comparison: Eye trace before vs. after merging
# =============================================================================


def plot_eye_trace_both_eyes(events_before, events_after, time_window=None):
    """
    Plot horizontal eye position trace for both eyes with fixations overlaid,
    comparing before and after the merging step.

    Parameters
    ----------
    events_before : pd.DataFrame
        Original (pre-merge) events DataFrame.
    events_after : pd.DataFrame
        Merged events DataFrame.
    time_window : tuple, optional
        (start_time, end_time) in seconds to zoom into a specific range.

    Returns
    -------
    matplotlib.figure.Figure
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

        # Build continuous trace from fixation intervals
        times, positions = [], []
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
            ax.add_patch(
                plt.Rectangle(
                    (row["onset"], y_center - 10),
                    row["duration"],
                    20,
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
            ax.add_patch(
                plt.Rectangle(
                    (row["onset"], y_center - 10),
                    row["duration"],
                    20,
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
    fig.suptitle(
        "Horizontal Eye Position: Fixation Merging Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    return fig


# =============================================================================
# 1.4 Main Sequence
# !!! Warning: not yet refactored for the events.tsv file #TBD
# =============================================================================


def plot_main_sequence(
    events_df: pd.DataFrame,
    out_path: str,
    out_file_format: str = OUT_FILE_FORMAT,
    by_eye: str = BY_EYE,
    title: str = "Main Sequence",
    x_label: str = "Saccade amplitude (deg)",
    y_label: str = "Peak velocity (deg/s)",
    drop_near_blinks: bool = MS_DROP_NEAR_BLINKS,
    drop_ms_outliers: bool = MS_DROP_OUTLIERS,
    ms_outlier_mad_thresh: float = MS_OUTLIER_MAD_THRESH,
):
    """
    Plot the main sequence (saccade amplitude vs. peak velocity) in log-log space.

    Optionally drops near-blink saccades and/or main-sequence outliers.

    Parameters
    ----------
    events_df : pd.DataFrame
    out_path : str
        Directory to save the figure. Pass None to skip saving.
    out_file_format : str
        File extension for saving, e.g. 'svg', 'pdf', 'eps'.
    by_eye : str
        One of: 'all', 'left', 'right', 'both'.
    title : str
    x_label, y_label : str
    drop_near_blinks : bool
        If True, exclude saccades flagged as near a blink.
    drop_ms_outliers : bool
        If True, detect and exclude main-sequence outliers (MAD-based).
    ms_outlier_mad_thresh : float
        MAD threshold for outlier detection (default: 4.3).

    !!! Warning: not yet refactored for the events.tsv file #TBD
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

    if drop_ms_outliers:
        s_flagged = detect_main_sequence_outliers(s, mad_thresh=ms_outlier_mad_thresh)
        log_main_sequence_outliers(s_flagged, out_path=out_path, base_name=base_name)
        s = s_flagged[~s_flagged["ms_is_outlier"]]

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
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

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


def detect_main_sequence_outliers(
    events_df: pd.DataFrame,
    mad_thresh: float = MS_DETECT_MAD_THRESH,
) -> pd.DataFrame:
    """
    Robust main-sequence outlier detection in log-log space using HuberRegressor + MAD.

    Adds columns: `ms_pred_log10v`, `ms_resid`, `ms_is_outlier`.

    Parameters
    ----------
    events_df : pd.DataFrame
        Saccade events (or full events; non-saccades are ignored internally).
    mad_thresh : float
        Threshold in MAD units (default: 3.0).

    Returns
    -------
    pd.DataFrame
        Saccade-only DataFrame with outlier columns added.
    """
    s_df = events_df[events_df["trial_type"] == "saccade"].copy()
    x = np.log10(s_df["sacc_visual_angle"].to_numpy())
    y = np.log10(s_df["peak_velocity"].to_numpy())

    model = HuberRegressor().fit(x.reshape(-1, 1), y)
    yhat = model.predict(x.reshape(-1, 1))
    resid = y - yhat

    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    is_outlier = np.abs(resid - med) > (mad_thresh * mad)

    s_df["ms_pred_log10v"] = yhat
    s_df["ms_resid"] = resid
    s_df["ms_is_outlier"] = is_outlier
    return s_df


def log_main_sequence_outliers(
    s_with_flags: pd.DataFrame,
    out_path: str,
    base_name: str,
) -> dict:
    """
    Write outlier rows to a CSV and print a summary.

    Parameters
    ----------
    s_with_flags : pd.DataFrame
        Output of detect_main_sequence_outliers().
    out_path : str
        Directory to write the CSV into.
    base_name : str
        Filename stem for the output CSV.

    Returns
    -------
    dict
        Summary with keys: n_total, n_outliers, outlier_rate, saved_to.
    """
    os.makedirs(out_path, exist_ok=True)

    outliers = s_with_flags[s_with_flags["ms_is_outlier"]]
    out_file = os.path.join(out_path, f"{base_name}_outliers.csv")
    outliers.to_csv(out_file, index=False)

    summary = {
        "n_total": int(len(s_with_flags)),
        "n_outliers": int(len(outliers)),
        "outlier_rate": float(len(outliers) / len(s_with_flags)),
        "saved_to": out_file,
    }

    print("\nMain sequence outlier summary")
    print("-" * 30)
    print(f"{'Total saccades':<18}: {summary['n_total']:>6}")
    print(f"{'Outliers':<18}: {summary['n_outliers']:>6}")
    print(f"{'Outlier rate':<18}: {summary['outlier_rate'] * 100:>5.2f}%")
    print(f"{'Saved to':<18}: {summary['saved_to']}")

    return summary


# =============================================================================
# 1.5 Plot Fixation Duration
# =============================================================================
def plot_fixation_duration(
    events_df: pd.DataFrame,
    out_path: str,
    out_file_format: str = OUT_FILE_FORMAT,
    by_eye: str = BY_EYE,
    min_ms: float = FIX_DUR_MIN_MS,
    max_ms: float = FIX_DUR_MAX_MS,
    bin_w: int = FIX_DUR_BIN_W,
    title: str = "Fixation Durations",
    x_label: str = "Fixation duration (ms)",
    y_label: str = "Count",
):
    """
    Histogram of fixation durations (ms), filtered by plausible range.

    !!! Warning: not yet refactored for the events.tsv file #TBD
    """
    if by_eye not in {"all", "left", "right", "both"}:
        raise ValueError("by_eye must be one of: 'all', 'left', 'right', 'both'")

    fix = events_df.loc[
        events_df["trial_type"] == "fixation", ["duration", "eye"]
    ].copy()
    fix = fix.dropna(subset=["duration"])
    fix["duration_ms"] = fix["duration"] * 1000.0

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        fix = fix.query("eye == @chosen_eye").copy()

    dur = fix["duration_ms"]
    dur = dur[(dur >= min_ms) & (dur <= max_ms)]
    if dur.empty:
        raise ValueError(
            "No fixation durations after filtering. Check inputs or ranges."
        )

    fig, ax = plt.subplots()
    ax.hist(dur)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "both": "Binocular only",
    }
    ax.set_title(f"{title} — {title_map[by_eye]}")
    fig.tight_layout()
    plt.show()

    out_file = (
        f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
    )
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to '{out_file}'")


# =============================================================================
# 1.6 Plot Saccade Amplitude
# =============================================================================
def saccade_amplitude(
    events_df,
    by_eye=BY_EYE,
    title="Saccade Amplitude",
    x_label="Saccade amplitude (deg)",
    y_label="Count",
    out_path=None,
    out_file_format=OUT_FILE_FORMAT,
    max_deg: float = SACC_AMP_MAX_DEG,
):
    """
    Histogram of saccade amplitudes (deg), clipped at `max_deg`°.

    !!! Warning: not yet refactored for the events.tsv file #TBD
    """
    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye").copy()

    all_amplitudes = s_df[
        "sacc_visual_angle"
    ].dropna()  # TBD: was referencing undefined `df`
    amplitudes = all_amplitudes[all_amplitudes <= max_deg]

    dropout = len(all_amplitudes[all_amplitudes > max_deg])
    print(f"Total saccades: {len(all_amplitudes)}")
    print(f"Kept saccades (<={max_deg}°): {len(amplitudes)}")
    print(
        f"Dropped outliers (>{max_deg}°): {dropout}, {(dropout / len(all_amplitudes)) * 100:.2f}%"
    )

    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "both": "Binocular only",
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(amplitudes, bins=40, edgecolor="black")
    ax.set_title(f"{title} — {title_map[by_eye]}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(left=0)
    fig.tight_layout()

    out_file = (
        f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
    )
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to '{out_file}'")
    plt.show()
    plt.close(fig)


# =============================================================================
# 1.7 Plot Saccade Duration
# =============================================================================
def saccade_duration(
    events_df,
    by_eye=BY_EYE,
    title="Saccade Duration",
    x_label="Saccade duration (ms)",
    y_label="Count",
    out_path=None,
    out_file_format=OUT_FILE_FORMAT,
    max_saccade_duration: int = SACC_DUR_MAX_MS,
):
    """
    Histogram of saccade durations (ms), with optional upper-limit clipping.

    !!! Warning: not yet refactored for the events.tsv file #TBD
    """
    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye")

    durations = (s_df["duration"] * 1000).dropna()

    if max_saccade_duration is not None:
        n_before = len(durations)
        durations = durations[durations <= max_saccade_duration]
        dropout = n_before - len(durations)
        print(
            f"Dropped {(dropout / n_before) * 100:.2f}% samples with duration > {max_saccade_duration} ms."
        )

    title_map = {
        "all": "All eyes",
        "left": "Left eye only",
        "right": "Right eye only",
        "both": "Binocular only",
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(durations, bins=40, edgecolor="black")
    ax.set_title(f"{title} — {title_map[by_eye]}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(left=0)
    fig.tight_layout()

    out_file = (
        f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
    )
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to '{out_file}'")
    plt.show()
    plt.close(fig)


# =============================================================================
# 1.8 Fixation Frequency
# =============================================================================
def fixation_frequency(
    events_df,
    by_eye=BY_EYE,
    title="Fixation frequency histogram",
    x_label="Fixation Frequency",
    y_label="Count",
    out_path=None,
    out_file_format=OUT_FILE_FORMAT,
):
    """
    Histogram of fixation frequency (fixations per second).

    !!! Warning: not yet refactored for the events.tsv file #TBD
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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    out_file = (
        f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
    )
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to '{out_file}'")
    plt.show()


# =============================================================================
# 1.9 Saccade Angular Histogram
# =============================================================================
def saccade_angular_histogram(
    events_df,
    by_eye=BY_EYE,
    title="Saccade Direction Histogram",
    cart_title="Cartesian Angular Histogram",
    out_path=None,
    out_file_format=OUT_FILE_FORMAT,
    refinement=ANG_HIST_REFINEMENT,
):
    """
    Polar and Cartesian histograms of saccade directions.

    Parameters
    ----------
    refinement : bool
        If True, weight by amplitude and exclude microsaccades (<1°).

    !!! Warning: not yet refactored for the events.tsv file #TBD
    """
    s_df = events_df[events_df["trial_type"] == "saccade"].copy()

    if by_eye != "all":
        eye_mapping = {"left": "L", "right": "R", "both": "both"}
        chosen_eye = eye_mapping[by_eye]
        s_df = s_df.query("eye == @chosen_eye").copy()

    dx = s_df["sacc_end_x"] - s_df["sacc_start_x"]
    dy = s_df["sacc_end_y"] - s_df["sacc_start_y"]
    angles_deg = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    angles_rad = np.deg2rad(angles_deg)

    # --- Polar histogram ---
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)

    if refinement:
        mask = (
            s_df["amplitude_deg"] >= ANG_HIST_MICROSACC_MIN_DEG
        )  # TBD: column may not exist in TSV
        weights = s_df.loc[mask, "amplitude_deg"]
        ax.hist(
            angles_rad[mask],
            bins=ANG_HIST_BINS_POLAR,
            weights=weights,
            edgecolor="black",
        )
    else:
        ax.hist(angles_rad, bins=ANG_HIST_BINS_POLAR, edgecolor="black")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(title)
    plt.tight_layout()

    out_file = (
        f"{out_path}/{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
    )
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to '{out_file}'")
    plt.show()

    # --- Cartesian histogram ---
    plt.figure()
    plt.hist(
        angles_deg, bins=np.arange(0, 361, ANG_HIST_BIN_WIDTH_CART), edgecolor="black"
    )
    plt.xlabel("Saccade direction (deg)")
    plt.ylabel("Count")
    plt.title(cart_title)

    out_file = f"{out_path}/{cart_title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to '{out_file}'")
    plt.show()
