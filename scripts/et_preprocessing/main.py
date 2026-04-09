"""
Usage:
- Full pipeline + subs specified in config.py: python main.py
- Full pipeline + single subject: python main.py --subjects 007
- Only preprocessing: python main.py --steps preprocessing --overwrite
- Only visualisation + for specific subs + show plots interactively: python main.py --steps visualisation --subjects 005 006 --show_plots

TBD: Single vs summarized plots, add option 2 args!
Oder einfach die Fkt zur Verfügung stellen

"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import config
from preprocessing import (
    load_subject_tsv,
    merge_events,
)
from visualisation import (
    plot_eye_trace_both_eyes,
    plot_main_sequence,
    plot_fixation_duration,
    saccade_amplitude,
    saccade_duration,
    fixation_frequency,
    saccade_angular_histogram,
)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Free-viewing eye-tracking pipeline: preprocessing and visualisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--steps",
        choices=["preprocessing", "visualisation", "both"],
        default="both",
        help=(
            "Which pipeline steps to run. "
            "'preprocessing' and 'visualisation'; default: both"
        ),
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        default=config.SUBJECTS,
        metavar="ID",
        help=(
            "subject IDs to process, e.g. --subjects 005 006 007. "
            f"Default: {config.SUBJECTS} (from config.py)."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "If set, re-run preprocessing even if a merged TSV already exists for a subject. Otherwise, processed subjects are skipped."
        ),
    )

    parser.add_argument(
        "--show_plots",
        action="store_true",
        help=("If set, display figures. Otherwise figures are saved to disk silently."),
    )

    return parser.parse_args()


def subject_paths(subject_id: str) -> dict:
    """
    Return all relevant paths for one subject, derived from DATA_ROOT.

    Template
    ------
    DATA_ROOT/
    └── sub-XXX/
        └── ses-001/
            └── misc/
                ├── sub-XXX_ses-001_task-freeviewing_et_events.tsv
                ├── sub-XXX_ses-001_task-freeviewing_et_events_merged.tsv
                └── plots/
    """
    misc_dir = config.DATA_ROOT / f"{subject_id}" / config.SESSION / config.MISC_SUBDIR
    stem = f"{subject_id}_{config.SESSION}_task-{config.TASK}"
    return {
        "misc_dir": misc_dir,
        "raw_tsv": misc_dir / f"{stem}_et_events.tsv",
        "merged_tsv": misc_dir / f"{stem}_et_events_merged.tsv",
        "plots_dir": misc_dir / config.PLOTS_SUBDIR,
    }


def run_preprocessing(subject_id: str, overwrite: bool) -> bool:
    """
    1. Load events: sub-XXX_ses-001_task-freeviewing_et_events.tsv
    2. Eun the two-stage merge + save to merged TSV: sub-XXX_ses-001_task-freeviewing_et_events_merged.tsv
    3. pre/post-merge eye trace comparison figure.

    Args:
        subject_id (str): subject ID (zero-padded)
        overwrite (bool): if False -> skips subs when merged.tsv exists
        show_plots (bool)

    Returns:
        bool: preprocessing ran through
    """
    paths = subject_paths(subject_id)

    # check if the file is already processed/ --overwrite flag is set; if not: skip preprocessing
    if paths["merged_tsv"].exists() and not overwrite:
        print(
            f"[Info] Skipping Preprocessing for {subject_id} since merged file already exists: {paths['merged_tsv'].name}"
        )
        print(f"Set flag --overwrite to reprocess.")
        return True

    # 1. Load events
    print(f"    Loading events TSV from {paths['misc_dir']} ...")
    try:
        events_raw = load_subject_tsv(
            folder_path=paths["misc_dir"],
            subject_id=subject_id,
            window_ms=config.BLINK_WINDOW_MS,
        )
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        return False

    # 2. Hooge et al. (2022) merger
    print(
        f"\n[Info] Merging events  "
        f"(a_min={config.A_MIN}°, "
        f"t_min_fix={config.T_MIN_FIX * 1000:.0f} ms, "
        f"blink_window={config.BLINK_WINDOW_MS:.0f} ms)..."
    )
    events_merged = merge_events(
        events_raw,
        a_min=config.A_MIN,
        t_min_fix=config.T_MIN_FIX,
    )

    # Save
    os.makedirs(paths["misc_dir"], exist_ok=True)
    events_merged.to_csv(paths["merged_tsv"], sep="\t", index=False)
    print(f"\n[Info] Saved: {paths['merged_tsv']}")

    # 3. pre/post-merge eye trace comparison
    os.makedirs(paths["plots_dir"], exist_ok=True)
    print(f"[Info] Plotting eye trace comparison...")
    fig = plot_eye_trace_both_eyes(events_raw, events_merged)
    fig.savefig(
        paths["plots_dir"] / f"sub-{subject_id}_eye_trace_merge_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"\n[Info] Saved: {paths['plots_dir']}")
    plt.close(fig)

    return True


def run_visualisation(subject_id: str) -> bool:
    """
    1. Load the merged events TSV: sub-XXX_ses-001_task-freeviewing_et_events.tsv
    2. Plot analysis figures for the subject

    Args:
        subject_id (str): subject ID (zero-padded)

    Returns:
        bool: preprocessing ran through
    """
    paths = subject_paths(subject_id)

    if not paths["merged_tsv"].exists():
        print(f"    [skip] No merged file found: {paths['merged_tsv'].name}")
        print(f"           Run --steps preprocessing first.")
        return False

    # --- Load merged events ---
    print(f"    Loading merged events from {paths['misc_dir']} ...")
    events = pd.read_csv(paths["merged_tsv"], sep="\t")

    out_path = str(paths["plots_dir"])
    os.makedirs(out_path, exist_ok=True)

    # --- Main sequence ---
    print(f"    Plotting main sequence...")
    plot_main_sequence(
        events_df=events,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        drop_near_blinks=config.MS_DROP_NEAR_BLINKS,
        drop_ms_outliers=config.MS_DROP_OUTLIERS,
        ms_outlier_mad_thresh=config.MS_OUTLIER_MAD_THRESH,
    )

    # Fixation duration
    print(f"\nPlotting fixation duration...")
    plot_fixation_duration(
        events_df=events,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        min_ms=config.FIX_DUR_MIN_MS,
        max_ms=config.FIX_DUR_MAX_MS,
        bin_w=config.FIX_DUR_BIN_W,
    )

    # Saccade amplitude
    print(f"\nPlotting saccade amplitude...")
    saccade_amplitude(
        events_df=events,
        by_eye=config.BY_EYE,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        max_deg=config.SACC_AMP_MAX_DEG,
    )

    # Saccade duration
    print(f"\nPlotting saccade duration...")
    saccade_duration(
        events_df=events,
        by_eye=config.BY_EYE,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        max_saccade_duration=config.SACC_DUR_MAX_MS,
    )

    # Fixation freq
    print(f"\nPlotting fixation frequency...")
    fixation_frequency(
        events_df=events,
        by_eye=config.BY_EYE,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
    )

    # Saccade angular histogram
    print(f" Plotting saccade angular histogram...")
    saccade_angular_histogram(
        events_df=events,
        by_eye=config.BY_EYE,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        refinement=config.ANG_HIST_REFINEMENT,
    )

    print(f"Figures saved: {out_path}")
    return True


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    # Monkey Patching: Suppress interactive display unless --show_plots is passed.
    # Replaces plt.show() GLOBALLY, also inside visualisation.py.
    if not args.show_plots:
        plt.show = lambda: None

    # Pipeline Overview
    print(f"\n{'=' * 60}")
    print(f"\tPipeline steps : {args.steps}")
    print(f"\tSubjects : {args.subjects}")
    print(f"\tOverwrite : {args.overwrite}")
    print(f"\tShow plots : {args.show_plots}")
    print(f"\tBIDS root : {config.DATA_ROOT}")
    print(f"\tOutput subdir : .../{config.SESSION}/{config.MISC_SUBDIR}/")
    print(f"\tFigures subdir : .../{config.MISC_SUBDIR}/{config.PLOTS_SUBDIR}/")
    print(f"{'=' * 60}\n")

    n_ok = 0
    n_fail = 0

    if args.subjects == ["all"]:
        subjects = os.listdir(config.DATA_ROOT)
        subjects = [x for x in subjects if "sub-" in x]
    else:
        subjects = args.subjects

    for subject_id in subjects:
        print(f"\n── sub-{subject_id} {'─' * (48 - len(subject_id))}")
        ok = True

        if args.steps in ("preprocessing", "both"):
            print(f"Run Step: [preprocessing]")
            success = run_preprocessing(
                subject_id,
                overwrite=args.overwrite,
            )
            ok = ok and success

        if args.steps in ("visualisation", "both"):
            print("Run Step: [visualisation]")
            success = run_visualisation(subject_id)
            ok = ok and success

        if ok:
            n_ok += 1
        else:
            n_fail += 1

    # Pipeline Summary
    print(f"\n{'=' * 60}")
    print(f"\tCompleted : {n_ok}/{len(subjects)} subject(s)")
    if n_fail:
        print(f"\tFailed : {n_fail}{len(subjects)}  subject(s)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
