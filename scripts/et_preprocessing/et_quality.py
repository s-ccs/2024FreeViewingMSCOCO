"""
et_quality.py
-------------
Eye-tracking data quality measures for the free-viewing dataset.
Runs SEPARATELY from the preprocessing pipeline (main.py); it only reads config.py, graphs.py and
plotting.py for shared settings, colours and eye mappings.

Measures
--------
1. Accuracy  — angular distance (deg) between the fixation during the fixation dot and the dot itself
               (screen centre). One value per trial and eye. Input: events.tsv with triggers
               (e.g. derivatives/custom-preprocessing/sub-XXX/ses-001/eeg/...events.tsv).
2. Precision — SD and RMS-S2S (deg) of the gaze samples within each EyeLink fixation (EFIX).
               One value per fixation and eye. Input: raw EyeLink .asc file.

Inputs
------
Both input folders must be given. Files are found by subject ID anywhere below them (recursive):
- --events-dir : 'sub-XXX*events.tsv' (files named '*_et_events.tsv' are ignored, they contain no triggers)
- --asc-dir    : 'sub-XXX*.asc'

Outputs (per subject, in <out-dir>/sub-XXX/ses-001/QUALITY_SUBDIR/)
-------
- <stem>_accuracy.tsv      one row per trial and eye
- <stem>_precision.tsv     one row per fixation and eye
- <stem>_quality.tsv       describe() per eye and measure (count, mean, std, min, 25%, 50%, 75%, max)
- plots/                   accuracy histogram, fixation positions, precision violin plot, precision map
plus one file for all subjects: <out-dir>/quality_summary.tsv
<out-dir> defaults to config.DATA_ROOT/derivatives/QUALITY_DERIVATIVE.

Usage
-----
python et_quality.py --events-dir <folder> --asc-dir <folder> [--subjects 005 006] [--steps accuracy]
                     [--out-dir <folder>] [--show_plots]

Example (local BIDS copy):
python et_quality.py --subjects 005 006 007
    --events-dir C:/.../BIDS/derivatives/custom-preprocessing
    --asc-dir    C:/.../BIDS

Logging: --log-level DEBUG | INFO | WARNING | ERROR (default: INFO)
"""

import argparse
import io
import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from config import SCREEN_DIST_M, SCREEN_RES, SCREEN_SIZE_M
from graphs import EYE_COLORS
from plotting import EYE_MAPPING, GAZE_MAPPING

logger = logging.getLogger(__name__)


# Settings
# =============================================================================
# Output location 
QUALITY_DERIVATIVE = "et-quality"
QUALITY_SUBDIR = "eeg"

# Experiment
DOT_LABEL = "Fixation dot shown" # trigger text of the fixation dot (also matches "... again due to recalibration")
STIM_LABEL = "Stimulus image shown" # trigger text of the stimulus image
GAZE_TOLERANCE_PX = 44 # gaze criterion of `wait_for_centre_gaze` in script.opensesame
IMAGE_SIZE_PX = (947, 710) # stimulus image size (w, h), centred on the screen

CX, CY = SCREEN_RES[0] / 2, SCREEN_RES[1] / 2   # screen centre = fixation dot, in EyeLink px

EYE_LABELS = {"L": "left eye", "R": "right eye", "binocular": "both eyes"}
PRECISION_LABELS = {"sd_deg": "SD (deg)", "rms_s2s_deg": "RMS-S2S (deg)"}


# =============================================================================
# Command line
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eye-tracking data quality: accuracy (fixation dot) and precision (SD, RMS-S2S).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps",
        choices=["accuracy", "precision", "all"],
        default="all",
        help="Which measures to compute. Default: all",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        metavar="ID",
        help="subject IDs, e.g. --subjects 005 006. Default: all subjects found in --events-dir and --asc-dir.",
    )
    parser.add_argument(
        "--events-dir",
        required=True,
        type=Path,
        help=("Folder with the events.tsv files (with triggers). Searched recursively for 'sub-XXX*events.tsv' "
              "(files named '*_et_events.tsv' are ignored, they contain no triggers)."),
    )
    parser.add_argument(
        "--asc-dir",
        required=True,
        type=Path,
        help="Folder with the EyeLink .asc files. Searched recursively for 'sub-XXX*.asc'.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path(config.DATA_ROOT) / "derivatives" / QUALITY_DERIVATIVE,
        type=Path,
        help=f"Output folder. Default: {Path(config.DATA_ROOT) / 'derivatives' / QUALITY_DERIVATIVE} (from config.py).",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="If set, display figures. Otherwise figures are only saved.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        type=str.upper,
        metavar="LEVEL",
        help="Logging verbosity level. Default: INFO.",
    )
    return parser.parse_args()


def _find_file(folder: Path, pattern: str, exclude: str | None = None) -> Path | None:
    """First file below `folder` (recursive) matching `pattern`, ignoring names that contain `exclude`."""
    hits = sorted(p for p in Path(folder).rglob(pattern) if exclude is None or exclude not in p.name)
    if len(hits) > 1:
        logger.warning(f"Several files match '{pattern}' in '{folder}', using '{hits[0]}'")
    return hits[0] if hits else None


def subject_paths(subject_id: str, events_dir: Path, asc_dir: Path, out_root: Path) -> dict:
    """
    Return all relevant paths for one subject.

    events_tsv : first 'sub-XXX*events.tsv' below `events_dir` (ignoring '*_et_events.tsv'), or None
    asc        : first 'sub-XXX*.asc' below `asc_dir`, or None
    out_dir    : <out_root>/sub-XXX/ses-001/QUALITY_SUBDIR
    """

    events_tsv = _find_file(events_dir, f"sub-{subject_id}*events.tsv", exclude="_et_events")
    asc = _find_file(asc_dir, f"sub-{subject_id}*.asc")
    out_dir = Path(out_root) / f"sub-{subject_id}" / config.SESSION / QUALITY_SUBDIR

    return {
        "events_tsv": events_tsv,
        "asc": asc,
        "out_dir": out_dir,
        "accuracy_tsv": out_dir / f"sub-{subject_id}_accuracy.tsv",
        "precision_tsv": out_dir / f"sub-{subject_id}_precision.tsv",
        "quality_tsv": out_dir / f"sub-{subject_id}_quality.tsv",
        "plots_dir": out_dir / config.PLOTS_SUBDIR,
    }


# =============================================================================
# pixels -> visual angle
# =============================================================================
def px_to_m(x_px, y_px, screen_res=SCREEN_RES, screen_size_m=SCREEN_SIZE_M):
    """
    Convert EyeLink pixel coordinates (origin top-left, y down)
    to metres relative to the screen centre (y up).

    Returns: (x_m, y_m)
    """
    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)

    # shift origin to the screen centre and convert px -> m
    x_m = (x_px - screen_res[0] / 2) * screen_size_m[0] / screen_res[0]
    y_m = (screen_res[1] / 2 - y_px) * screen_size_m[1] / screen_res[1]

    return x_m, y_m


def unit_vector(x_px, y_px, screen_res=SCREEN_RES, screen_size_m=SCREEN_SIZE_M, screen_dist_m=SCREEN_DIST_M):
    """
    Unit gaze-direction vector(s) for EyeLink pixel positions.
    Eye at origin, screen centre on the z-axis.

    Returns: array of shape (..., 3)
    """
    # pixel -> metres
    x_m, y_m = px_to_m(x_px, y_px, screen_res=screen_res, screen_size_m=screen_size_m)

    # 3D vector (x_m, y_m, D) from the eye to the point on the screen
    v = np.stack([x_m, y_m, np.full_like(x_m, screen_dist_m)], axis=-1)

    # normalise to length 1
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angular_distance_deg(u, v):
    """Angle in degrees between unit vectors u and v (arrays of shape (..., 3))."""
    # dot product along the last axis, clipped to [-1, 1] (rounding can push it slightly above 1)
    dot = np.clip(np.sum(u * v, axis=-1), -1, 1)
    return np.degrees(np.arccos(dot))


def px_angular_distance_deg(x1, y1, x2, y2):
    """Angle in degrees between two screen positions given in EyeLink pixels."""
    return angular_distance_deg(unit_vector(x1, y1), unit_vector(x2, y2))


# =============================================================================
# Accuracy at the fixation dot
# =============================================================================
def fixation_cross_accuracy(events: pd.DataFrame, target_x=CX, target_y=CY,
                            dot_label=DOT_LABEL, stim_label=STIM_LABEL) -> pd.DataFrame:
    """
    Accuracy at the fixation dot: angular distance between the fixation (per eye) that overlaps the
    window [dot onset, stimulus onset) the most, and the target.

    Window: per trial (same block and trial), from the LAST fixation dot (after a possible recalibration)
    to the stimulus onset. The fixation with the largest overlap is the one that met the gaze criterion;
    taking the last fixation instead would pick up anticipatory saccades just before the image appears.

    Requires the trigger rows to carry 'block' and 'trial'.

    Returns: DataFrame with one row per trial and eye (block, trial, dot_onset, stim_onset, onset, duration,
        eye, fix_avg_x, fix_avg_y, overlap, ang_dist)
    """
    # find the fixation dot and stimulus events, and the fixation events
    dots = events[events["trial_type"].str.contains(dot_label, na=False)].rename(columns={"onset": "dot_onset"})
    stims = events[events["trial_type"].str.contains(stim_label, na=False)].rename(columns={"onset": "stim_onset"})
    fix = events[events["trial_type"] == "fixation"]

    # pair every fixation dot with the stimulus of the SAME block and trial
    fixation_windows = stims[["block", "trial", "stim_onset"]].merge(dots[["block", "trial", "dot_onset"]], on=["block", "trial"])

    # several dots for one stimulus (recalibration) -> only take the last one
    fixation_windows = fixation_windows.groupby(["block", "trial", "stim_onset"], as_index=False).agg(dot_onset=("dot_onset", "max"))

    # per window: fixations overlapping [dot_onset, stim_onset), keep the one with the largest overlap per eye
    fixations_list = []
    for w in fixation_windows.itertuples(index=False):
        in_window = fix[(fix["end_time"] > w.dot_onset) & (fix["onset"] < w.stim_onset)].copy()
        in_window["overlap"] = in_window["end_time"].clip(upper=w.stim_onset) - in_window["onset"].clip(lower=w.dot_onset)
        best = in_window.sort_values("overlap").groupby("eye").tail(1)
        fixations_list.append(best.assign(block=w.block, trial=w.trial, dot_onset=w.dot_onset, stim_onset=w.stim_onset))

    acc = pd.concat(fixations_list, ignore_index=True)

    # angular distance between the fixation average and the target
    acc["ang_dist"] = px_angular_distance_deg(acc["fix_avg_x"], acc["fix_avg_y"], target_x, target_y)

    cols = ["block", "trial", "dot_onset", "stim_onset", "onset", "duration", "eye", "fix_avg_x", "fix_avg_y", "overlap", "ang_dist"]
    return acc[cols].sort_values(["block", "trial", "eye"]).reset_index(drop=True)


# =============================================================================
# Precision from the raw asc file
# =============================================================================
def read_asc_samples_and_fixations(asc_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read binocular gaze samples and EyeLink fixation events (EFIX) from an EyeLink .asc file.

    Sample lines:  time  x_L  y_L  pupil_L  x_R  y_R  pupil_R  flags   (missing values = '.')
    EFIX lines:    EFIX (End of FIXation)  eye  start  end  duration  x  y  pupil

    Returns:
        samples (pd.DataFrame): time (ms, tracker clock), x_L, y_L, x_R, y_R (px, NaN if missing)
        fixations (pd.DataFrame): eye ('L'/'R'), start, end (ms, tracker clock), duration (ms), fix_avg_x, fix_avg_y (px)
    """
    sample_lines, efix_lines = [], []
    with open(asc_path, "r", errors="replace") as f:
        for line in f:
            if line[:1].isdigit():
                sample_lines.append(line)
            elif line.startswith("EFIX"):
                efix_lines.append(line)

    samples = pd.read_csv(
        io.StringIO("".join(sample_lines)), sep=r"\s+", header=None, usecols=[0, 1, 2, 4, 5],
        names=["time", "x_L", "y_L", "x_R", "y_R"], na_values=".", engine="c",
    )
    fixations = pd.read_csv(
        io.StringIO("".join(efix_lines)), sep=r"\s+", header=None, usecols=[1, 2, 3, 4, 5, 6],
        names=["eye", "start", "end", "duration", "fix_avg_x", "fix_avg_y"],
    )

    logger.info(f"Read {len(samples)} samples and {len(fixations)} fixations from '{asc_path}'")
    return samples, fixations


def compute_precision(samples: pd.DataFrame, fixations: pd.DataFrame) -> pd.DataFrame:
    """
    Precision per fixation and eye, from the samples within each fixation (start <= time <= end):
        - SD:      sqrt(mean(d(sample_i, fixation average)^2))
        - RMS-S2S: sqrt(mean(d(sample_i, sample_i-1)^2)); pairs with a missing sample are dropped
                   (no bridging over missing samples)
    Distances are angular distances in degrees. The fixation average is EyeLink's fix_avg_x/y
    (differs from the sample mean by ~0.06 px median, checked on sub-007).

    Returns:
        pd.DataFrame: fixations with additional columns sd_deg, rms_s2s_deg
    """
    samples = samples.sort_values("time").reset_index(drop=True)
    time = samples["time"].to_numpy()
    gaze = {eye: (samples[f"x_{eye}"].to_numpy(), samples[f"y_{eye}"].to_numpy()) for eye in ["L", "R"]}

    sd, rms = [], []
    for fix in fixations.itertuples(index=False):
        # samples within the fixation window (start <= time <= end)
        start_idx = np.searchsorted(time, fix.start, side="left")
        end_idx = np.searchsorted(time, fix.end, side="right")
        x, y = gaze[fix.eye][0][start_idx:end_idx], gaze[fix.eye][1][start_idx:end_idx]

        # SD: distance to the fixation avg
        d_avg = px_angular_distance_deg(x, y, fix.fix_avg_x, fix.fix_avg_y)
        sd.append(np.sqrt(np.nanmean(d_avg ** 2)) if np.any(~np.isnan(d_avg)) else np.nan)

        # RMS-S2S: distance between consecutive samples (NaN pairs are skipped by nanmean)
        d_s2s = px_angular_distance_deg(x[1:], y[1:], x[:-1], y[:-1])
        rms.append(np.sqrt(np.nanmean(d_s2s ** 2)) if np.any(~np.isnan(d_s2s)) else np.nan)

    out = fixations.copy()
    out["sd_deg"] = sd
    out["rms_s2s_deg"] = rms
    return out


# =============================================================================
# Plots
# =============================================================================
def _save(fig, out_path, file_name, title):
    """Save `fig` into `out_path` (created if missing), or log that it was not saved."""
    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, file_name)
        fig.savefig(out_file, bbox_inches="tight")
        logger.info(f"{title} plot saved to '{out_file}'")
    else:
        logger.warning(f"{title} plot not saved — pass `out_path` to save.")


def plot_fixation_dot_accuracy(
    acc_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "all",
    title: str = "Fixation Dot Accuracy",
    tolerance_px: float | None = GAZE_TOLERANCE_PX,
    bin_width: float = 0.05,
):
    """
    Histogram of fixation-dot accuracy (deg), with the median per eye.

    Args:
        acc_df (pd.DataFrame): output of `fixation_cross_accuracy` (columns 'eye', 'ang_dist').
        out_path (str): Directory to save the figure (created if missing). Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular' or 'both'. Defaults to 'all'.
        title (str): Defaults to 'Fixation Dot Accuracy'.
        tolerance_px (float, optional): gaze criterion (px from the screen centre), drawn in degrees as a
            dotted line. Pass None to omit. Defaults to GAZE_TOLERANCE_PX.
        bin_width (float): histogram bin width in degrees. Defaults to 0.05.
    """
    if by_eye not in GAZE_MAPPING:
        raise ValueError(f"by_eye must be one of: {list(GAZE_MAPPING)}; got '{by_eye}' instead.")

    # optionally filter by eye
    acc = acc_df.dropna(subset=["ang_dist"])
    if by_eye != "all":
        acc = acc[acc["eye"] == EYE_MAPPING[by_eye]]
    if acc.empty:
        raise ValueError("No accuracy values to plot. Check inputs or the `by_eye` selection.")

    # one outline histogram per eye, median as dashed line
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, acc["ang_dist"].max() + bin_width, bin_width)
    for eye, sub in acc.groupby("eye"):
        eye_color = EYE_COLORS.get(str(eye), "#333333")
        median = sub["ang_dist"].median()
        ax.hist(sub["ang_dist"], bins=bins, histtype="step", linewidth=1.8, color=eye_color,
                label=f"{EYE_LABELS.get(eye, eye)} (median {median:.2f}°, n = {len(sub)})")
        ax.axvline(median, color=eye_color, linestyle="--", linewidth=1)

    # gaze criterion of the experiment, converted to degrees
    if tolerance_px is not None:
        tolerance_deg = float(px_angular_distance_deg(CX, CY, CX + tolerance_px, CY))
        ax.axvline(tolerance_deg, color="grey", linestyle=":", linewidth=1.2,
                   label=f"gaze criterion {tolerance_px:.0f} px ≈ {tolerance_deg:.2f}°")

    # Labels & title
    ax.set_xlabel("Distance to fixation dot (deg)")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"{title} — {GAZE_MAPPING[by_eye]}")
    ax.legend(frameon=False)
    fig.tight_layout()

    # Save & show
    _save(fig, out_path, f"{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}", title)
    plt.show()
    return fig


def plot_fixation_dot_positions(
    acc_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "all",
    title: str = "Fixation Positions around Fixation Dot",
    tolerance_px: float | None = GAZE_TOLERANCE_PX,
):
    """
    Scatter plot of the fixation positions (fix_avg_x, fix_avg_y) relative to the fixation dot.
    Axes in px, y inverted so that the plot looks like the screen.

    Args:
        acc_df (pd.DataFrame): output of `fixation_cross_accuracy` (columns 'eye', 'fix_avg_x', 'fix_avg_y').
        out_path (str): Directory to save the figure (created if missing). Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular' or 'both'. Defaults to 'all'.
        title (str): Defaults to 'Fixation Positions around Fixation Dot'.
        tolerance_px (float, optional): gaze criterion radius in px, drawn as a dotted circle.
            Pass None to omit. Defaults to GAZE_TOLERANCE_PX.
    """
    if by_eye not in GAZE_MAPPING:
        raise ValueError(f"by_eye must be one of: {list(GAZE_MAPPING)}; got '{by_eye}' instead.")

    # optionally filter by eye
    pos = acc_df.dropna(subset=["fix_avg_x", "fix_avg_y"])
    if by_eye != "all":
        pos = pos[pos["eye"] == EYE_MAPPING[by_eye]]
    if pos.empty:
        raise ValueError("No fixation positions to plot. Check inputs or the `by_eye` selection.")

    # positions relative to the fixation dot (screen centre)
    pos = pos.assign(dx=pos["fix_avg_x"] - CX, dy=pos["fix_avg_y"] - CY)

    # one colour per eye
    fig, ax = plt.subplots(figsize=(6, 6))
    for eye, sub in pos.groupby("eye"):
        ax.scatter(sub["dx"], sub["dy"], s=12, alpha=0.6,
                   color=EYE_COLORS.get(str(eye), "#333333"), label=f"{EYE_LABELS.get(eye, eye)} (n = {len(sub)})")

    # gaze criterion and fixation dot
    if tolerance_px is not None:
        ax.add_patch(plt.Circle((0, 0), tolerance_px, fill=False, color="grey", linestyle=":",
                                linewidth=1.2, label=f"gaze criterion ({tolerance_px:.0f} px)"))
    ax.plot(0, 0, marker="+", color="black", markersize=14, markeredgewidth=2, label="fixation dot")

    # symmetric limits, y inverted (EyeLink: y points down)
    lim = max(tolerance_px or 0, pos[["dx", "dy"]].abs().max().max()) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)
    ax.set_aspect("equal")

    # Labels & title
    ax.set_xlabel("x relative to fixation dot (px)")
    ax.set_ylabel("y relative to fixation dot (px, + = down)")
    ax.set_title(f"{title} — {GAZE_MAPPING[by_eye]}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    # Save & show
    _save(fig, out_path, f"{title.lower().replace(' ', '_')}-{by_eye}Eyes.{out_file_format}", title)
    plt.show()
    return fig


def plot_precision_violin(
    prec_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "all",
    title: str = "Precision",
    measures: tuple = ("sd_deg", "rms_s2s_deg"),
    log_scale: bool = True,
):
    """
    Violin plot of precision per fixation, one panel per measure, one violin per eye.
    Inside each violin: median (white dot) and interquartile range (black bar).

    Args:
        prec_df (pd.DataFrame): output of `compute_precision` (columns 'eye', 'sd_deg', 'rms_s2s_deg').
        out_path (str): Directory to save the figure (created if missing). Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular' or 'both'. Defaults to 'all'.
        title (str): Defaults to 'Precision'.
        measures (tuple): precision columns to plot, one panel each. Defaults to ('sd_deg', 'rms_s2s_deg').
        log_scale (bool): if True, the violins are computed and drawn on a log10 axis (precision values are
            right-skewed). Defaults to True.
    """
    if by_eye not in GAZE_MAPPING:
        raise ValueError(f"by_eye must be one of: {list(GAZE_MAPPING)}; got '{by_eye}' instead.")

    # optionally filter by eye
    prec = prec_df.copy()
    if by_eye != "all":
        prec = prec[prec["eye"] == EYE_MAPPING[by_eye]]
    if prec.empty:
        raise ValueError("No precision values to plot. Check inputs or the `by_eye` selection.")

    eyes = sorted(prec["eye"].dropna().unique())

    # one panel per measure
    fig, axs = plt.subplots(1, len(measures), figsize=(4 * len(measures), 4.5), squeeze=False)

    for ax, measure in zip(axs[0], measures):
        data = [prec.loc[prec["eye"] == eye, measure].dropna() for eye in eyes]
        if log_scale:
            data = [d[d > 0] for d in data]
            plotting_data = [np.log10(d) for d in data]
        else:
            plotting_data = data

        # violins, coloured per eye
        parts = ax.violinplot(plotting_data, showextrema=False, widths=0.8)
        for body, eye in zip(parts["bodies"], eyes):
            body.set_facecolor(EYE_COLORS.get(str(eye), "#333333"))
            body.set_edgecolor("black")
            body.set_alpha(1)

        # median and IQR
        for i, d in enumerate(plotting_data, start=1):
            q1, median, q3 = np.percentile(d, [25, 50, 75])
            ax.vlines(i, q1, q3, color="black", linewidth=5)
            ax.scatter(i, median, color="white", s=25, zorder=3)

        # x labels with median per eye (in degrees)
        ax.set_xticks(range(1, len(eyes) + 1))
        ax.set_xticklabels([f"{EYE_LABELS.get(eye, eye)}\nmedian {d.median():.3f}°" for eye, d in zip(eyes, data)])

        # y axis: show degrees, not log10 values
        if log_scale:
            lo, hi = ax.get_ylim()
            ticks = []
            for t in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]:
                position = np.log10(t)            # where t lies on the log10 axis
                if lo <= position <= hi:          # if it is inside the visible range
                    ticks.append(t)               # keep it as a tick
            ax.set_yticks(np.log10(ticks))
            ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_ylabel(PRECISION_LABELS.get(measure, measure) + (" (log scale)" if log_scale else ""))

    # Labels & title
    fig.suptitle(f"{title} — {GAZE_MAPPING[by_eye]}")
    fig.tight_layout()

    # Save & show
    _save(fig, out_path, f"{title.lower().replace(' ', '_')}_violin-{by_eye}Eyes.{out_file_format}", title)
    plt.show()
    return fig


def plot_precision_spatial(
    prec_df: pd.DataFrame,
    out_path: str = None,
    out_file_format: str = "svg",
    by_eye: str = "all",
    title: str = "Precision by Screen Position",
    measure: str = "rms_s2s_deg",
    cell_px: int = 120,
    min_fixations: int = 10,
    image_size_px: tuple | None = IMAGE_SIZE_PX,
):
    """
    Heat map of the median precision per screen cell, based on the fixation position (fix_avg_x, fix_avg_y).
    One panel per eye. Cells with fewer than `min_fixations` fixations stay empty. Grid centred on the screen centre.

    Args:
        prec_df (pd.DataFrame): output of `compute_precision` (columns 'eye', 'fix_avg_x', 'fix_avg_y', measure).
        out_path (str): Directory to save the figure (created if missing). Pass None to skip saving (default).
        out_file_format (str): File extension for saving, e.g. 'svg', 'pdf'. Defaults to 'svg'.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular' or 'both'. Defaults to 'all'.
        title (str): Defaults to 'Precision by Screen Position'.
        measure (str): precision column to plot, 'sd_deg' or 'rms_s2s_deg'. Defaults to 'rms_s2s_deg'.
        cell_px (int): cell size in px. Defaults to 120.
        min_fixations (int): minimum number of fixations per cell. Defaults to 10.
        image_size_px (tuple, optional): size of the centred stimulus image (w, h) in px, drawn as a dashed
            rectangle. Pass None to omit. Defaults to IMAGE_SIZE_PX.
    """
    if by_eye not in GAZE_MAPPING:
        raise ValueError(f"by_eye must be one of: {list(GAZE_MAPPING)}; got '{by_eye}' instead.")

    # optionally filter by eye; keep only fixations on the screen
    prec = prec_df.dropna(subset=["fix_avg_x", "fix_avg_y", measure])
    if by_eye != "all":
        prec = prec[prec["eye"] == EYE_MAPPING[by_eye]]
    prec = prec[prec["fix_avg_x"].between(0, SCREEN_RES[0]) & prec["fix_avg_y"].between(0, SCREEN_RES[1])]
    if prec.empty:
        raise ValueError("No precision values to plot. Check inputs or the `by_eye` selection.")

    eyes = sorted(prec["eye"].unique())

    # cell index of every fixation, grid centred on the screen centre
    n_x = 2 * int(np.ceil(CX / cell_px))
    n_y = 2 * int(np.ceil(CY / cell_px))
    x0, y0 = CX - n_x / 2 * cell_px, CY - n_y / 2 * cell_px      # top-left corner of the grid
    prec = prec.assign(cell_x=((prec["fix_avg_x"] - x0) // cell_px).astype(int),
                       cell_y=((prec["fix_avg_y"] - y0) // cell_px).astype(int))

    # median per cell and eye (only cells with enough fixations)
    cells = prec.groupby(["eye", "cell_y", "cell_x"])[measure].agg(["median", "count"]).reset_index()
    cells = cells[cells["count"] >= min_fixations]
    if cells.empty:
        raise ValueError(f"No screen cell has ≥ {min_fixations} fixations. Lower `min_fixations` or increase `cell_px`.")
    vmin, vmax = cells["median"].quantile([0.02, 0.98])     # shared colour scale for all panels

    fig, axs = plt.subplots(1, len(eyes), figsize=(5.5 * len(eyes), 3.8), squeeze=False)

    for ax, eye in zip(axs[0], eyes):
        grid = np.full((n_y, n_x), np.nan)
        c = cells[cells["eye"] == eye]
        grid[c["cell_y"], c["cell_x"]] = c["median"]

        im = ax.imshow(grid, extent=[x0, x0 + n_x * cell_px, y0 + n_y * cell_px, y0], cmap="viridis",
                       vmin=vmin, vmax=vmax, aspect="equal")

        # stimulus image area
        if image_size_px is not None:
            w, h = image_size_px
            ax.add_patch(plt.Rectangle((CX - w / 2, CY - h / 2), w, h,
                                       fill=False, edgecolor="black", linestyle="--", linewidth=1))

        ax.set_xlim(0, SCREEN_RES[0])
        ax.set_ylim(SCREEN_RES[1], 0)    # EyeLink y points down
        ax.set_title(f"{EYE_LABELS.get(eye, eye)} (n = {int(c['count'].sum())} fixations)")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    # Labels & title
    fig.colorbar(im, ax=axs[0].tolist(), label=f"median {PRECISION_LABELS.get(measure, measure)}", shrink=0.85)
    fig.suptitle(f"{title} — {GAZE_MAPPING[by_eye]}\n(cells of {cell_px} px with ≥ {min_fixations} fixations)")

    # Save & show
    _save(fig, out_path, f"{title.lower().replace(' ', '_')}_{measure}-{by_eye}Eyes.{out_file_format}", title)
    plt.show()
    return fig


# =============================================================================
# Per-subject run
# =============================================================================
def run_subject(subject_id: str, steps: str, events_dir: Path, asc_dir: Path, out_root: Path) -> pd.DataFrame | None:
    """
    Compute the quality measures for one subject, save TSVs and plots.

    Returns: summary DataFrame (one row per eye and measure, with column 'subject') or None if nothing could be computed.
    """
    paths = subject_paths(subject_id, events_dir, asc_dir, out_root)
    summaries = []

    # --- Accuracy -----------------------------------------------------------
    if steps in ("accuracy", "all"):
        if paths["events_tsv"] is None:
            logger.warning(f"Accuracy skipped — no 'sub-{subject_id}*events.tsv' found in '{events_dir}'")
        else:
            events = pd.read_csv(paths["events_tsv"], sep="\t", low_memory=False)
            acc = fixation_cross_accuracy(events)
            os.makedirs(paths["out_dir"], exist_ok=True)
            acc.to_csv(paths["accuracy_tsv"], sep="\t", index=False)
            logger.info(f"Accuracy: {len(acc)} rows saved to '{paths['accuracy_tsv']}'")
            summaries.append(acc.groupby("eye")[["ang_dist"]].describe())

            plot_fixation_dot_accuracy(acc, out_path=paths["plots_dir"], out_file_format=config.OUT_FILE_FORMAT)
            plot_fixation_dot_positions(acc, out_path=paths["plots_dir"], out_file_format=config.OUT_FILE_FORMAT)
            plt.close("all")

    # --- Precision ----------------------------------------------------------
    if steps in ("precision", "all"):
        if paths["asc"] is None:
            logger.warning(f"Precision skipped — no 'sub-{subject_id}*.asc' found in '{asc_dir}'")
        else:
            samples, fixations = read_asc_samples_and_fixations(paths["asc"])
            prec = compute_precision(samples, fixations)
            os.makedirs(paths["out_dir"], exist_ok=True)
            prec.to_csv(paths["precision_tsv"], sep="\t", index=False)
            logger.info(f"Precision: {len(prec)} rows saved to '{paths['precision_tsv']}'")
            summaries.append(prec.groupby("eye")[["sd_deg", "rms_s2s_deg"]].describe())

            plot_precision_violin(prec, out_path=paths["plots_dir"], out_file_format=config.OUT_FILE_FORMAT)
            plot_precision_spatial(prec, out_path=paths["plots_dir"], out_file_format=config.OUT_FILE_FORMAT)
            plt.close("all")

    if not summaries:
        return None

    # describe() gives one row per eye and two-level columns (measure, statistic);
    # stack the measures into rows -> one row per eye and measure (TSV-friendly)
    summary = (pd.concat(summaries, axis=1)
               .stack(level=0, future_stack=True)
               .rename_axis(["eye", "measure"])
               .reset_index())
    summary.to_csv(paths["quality_tsv"], sep="\t", index=False)
    logger.info(f"Summary saved to '{paths['quality_tsv']}'")
    return summary.assign(subject=subject_id)


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress interactive display unless --show_plots is passed.
    # 'Agg' renders to files only and never loads a GUI toolkit (Qt/Tk).
    if not args.show_plots:
        plt.switch_backend("Agg")
        plt.show = lambda: None

    out_root = args.out_dir

    # Overview
    print(f"\n{'=' * 69}")
    print(f"Quality Analysis steps  : {args.steps}")
    print(f"Subjects                : {args.subjects}")
    print(f"Events input            : {args.events_dir}")
    print(f"ASC input               : {args.asc_dir}")
    print(f"Output             : {out_root}")
    print(f"{'=' * 69}")

    if args.subjects == ["all"]:
        # all subject IDs that have an events file or an .asc file in the input folders
        names = [p.name for p in Path(args.events_dir).rglob("sub-*events.tsv") if "_et_events" not in p.name]
        names += [p.name for p in Path(args.asc_dir).rglob("sub-*.asc")]
        subjects = sorted({m.group(1) for x in names if (m := re.search(r"sub-(\d+)", x))})
    else:
        subjects = args.subjects

    all_summaries, n_fail = [], 0
    for subject_id in subjects:
        print(f"\n── sub-{subject_id} {'─' * (52 - len(subject_id))}")
        try:
            summary = run_subject(subject_id, args.steps, args.events_dir, args.asc_dir, out_root)
        except Exception:
            logger.exception(f"sub-{subject_id} failed")
            summary = None
        if summary is None:
            n_fail += 1
        else:
            all_summaries.append(summary)

    # one file for all subjects
    if all_summaries:
        out_file = Path(out_root) / "quality_summary.tsv"
        os.makedirs(out_file.parent, exist_ok=True)
        combined = pd.concat(all_summaries, ignore_index=True)
        combined = combined[["subject"] + [c for c in combined.columns if c != "subject"]]
        combined.to_csv(out_file, sep="\t", index=False)
        logger.info(f"Summary of all subjects saved to '{out_file}'")

    print(f"\n{'=' * 64}")
    print(f"Done: {len(all_summaries)} subject(s) successfully processed, {n_fail} subject(s)failed.")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
