"""
Usage:
- Full pipeline + subs specified in config.py: python main.py
- Full pipeline + single subject: python main.py --subjects 007
- Only preprocessing: python main.py --steps preprocessing --overwrite
- Only visualisation + for specific subs + show plots interactively: python main.py --steps visualisation --subjects 005 006 --show_plots

Logging:
- Default log level is INFO. To change verbosity, pass --log-level followed by one of:
  DEBUG, INFO, WARNING, ERROR — e.g. python main.py --log-level WARNING
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config
from preprocessing import (
    load_subject_tsv,
    merge_events,
)
from plotting import (
    plot_eye_trace_both_eyes,
    plot_main_sequence,
    plot_fixation_duration,
    plot_saccade_amplitude,
    plot_saccade_duration,
    plot_fixation_frequency,
    plot_saccade_angles,
    plot_summary,
)
from report import generate_report

logger = logging.getLogger(__name__)


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
        choices=["preprocessing", "visualisation", "all"],
        default="all",
        help=(
            "Which pipeline steps to run. "
            "'preprocessing', 'visualisation'; default: all"
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
            "If set, re-run preprocessing even if a merged TSV already exists for a subject. "
            "Otherwise, processed subjects are skipped."
        ),
    )

    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="If set, display figures. Otherwise figures are saved to disk silently.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        type=str.upper,
        metavar="LEVEL",
        help="Logging verbosity level. One of: DEBUG, INFO, WARNING, ERROR. Default: INFO.",
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
    misc_dir = (
        config.DATA_ROOT / f"sub-{subject_id}" / config.SESSION / config.MISC_SUBDIR
    )
    stem = f"sub-{subject_id}_{config.SESSION}_task-{config.TASK}"
    return {
        "misc_dir": misc_dir,
        "raw_tsv": misc_dir / f"{stem}_et_events.tsv",
        "merged_tsv": misc_dir / f"{stem}_et_events_merged.tsv",
        "plots_dir": misc_dir / config.PLOTS_SUBDIR,
    }


def run_preprocessing(subject_id: str, overwrite: bool) -> bool:
    """
    1. Load events: sub-XXX_ses-001_task-freeviewing_et_events.tsv
    2. Run the two-stage merge + save to merged TSV: sub-XXX_ses-001_task-freeviewing_et_events_merged.tsv
    3. pre/post-merge eye trace comparison figure.

    Args:
        subject_id (str): subject ID (zero-padded)
        overwrite (bool): if False -> skips subs when merged.tsv exists

    Returns:
        bool: preprocessing ran through
    """
    paths = subject_paths(subject_id)

    if paths["merged_tsv"].exists() and not overwrite:
        logger.info(
            f"Skipping preprocessing for {subject_id} — merged file already exists: "
            f"{paths['merged_tsv'].name}. Use --overwrite to reprocess."
        )
        return True

    # 1. Load events
    logger.info(f"Loading events TSV from dir:{paths['misc_dir']} ...")
    try:
        events_raw = load_subject_tsv(
            folder_path=paths["misc_dir"],
            subject_id=subject_id,
            window_ms=config.BLINK_WINDOW_MS,
        )
    except FileNotFoundError as e:
        logger.error(e)
        return False

    # 2. Hooge et al. (2022) merger
    logger.info(
        f"Merging events  "
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
    logger.info(f"--> Saved merged TSV: {paths['merged_tsv']}")

    # 3. pre/post-merge eye trace comparison
    os.makedirs(paths["plots_dir"], exist_ok=True)
    logger.info("Plotting eye trace comparison...")
    plot_eye_trace_both_eyes(
        events_before=events_raw,
        events_after=events_merged,
        out_path=paths["plots_dir"],
        out_file_format="svg",
        title="Eye Trace Merge Comparison",
        time_window=(180, 200),
    )

    if config.REPORT:
        logger.info("Generating Subject Report...")
        generate_report(
            events_raw=events_raw,
            events_merged=events_merged,
            subject_id=f"sub-{subject_id}",
            out_path=str(paths["misc_dir"]),
            by_eye=config.BY_EYE,
            sac_amp_max=config.SACC_AMP_MAX_DEG,
            fix_dur_min=config.FIX_DUR_MIN_MS,
            fix_dur_max=config.FIX_DUR_MAX_MS,
            sac_dur_max=config.SACC_DUR_MAX_MS,
            drop_near_blinks=config.DROP_NEAR_BLINKS,
        )

    return True


def run_visualisation(subject_id: str) -> bool:
    """
    1. Load the merged events TSV: sub-XXX_ses-001_task-freeviewing_et_events.tsv
    2. Plot analysis figures for the subject

    Args:
        subject_id (str): subject ID (zero-padded)

    Returns:
        bool: visualisation ran through
    """
    paths = subject_paths(subject_id)

    if not paths["merged_tsv"].exists():
        logger.warning(
            f"No merged file found for {subject_id}: {paths['merged_tsv'].name}. "
            f"Run --steps preprocessing first."
        )
        return False

    logger.info(f"Loading merged events from dir: {paths['misc_dir']} ...")
    events = pd.read_csv(paths["merged_tsv"], sep="\t")

    out_path = str(paths["plots_dir"])
    os.makedirs(out_path, exist_ok=True)

    logger.info("Plotting main sequence...")
    plot_main_sequence(
        events_df=events,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        drop_near_blinks=config.DROP_NEAR_BLINKS,
    )

    logger.info("Plotting fixation duration...")
    plot_fixation_duration(
        events_df=events,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        fix_dur_min=config.FIX_DUR_MIN_MS,
        fix_dur_max=config.FIX_DUR_MAX_MS,
    )

    logger.info("Plotting saccade amplitude...")
    plot_saccade_amplitude(
        events_df=events,
        by_eye=config.BY_EYE,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        sac_amp_max=config.SACC_AMP_MAX_DEG,
    )

    logger.info("Plotting saccade duration...")
    plot_saccade_duration(
        events_df=events,
        by_eye=config.BY_EYE,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        sac_dur_max=config.SACC_DUR_MAX_MS,
    )

    logger.info("Plotting fixation frequency...")
    plot_fixation_frequency(
        events_df=events,
        by_eye=config.BY_EYE,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
    )

    logger.info("Plotting saccade angular histogram...")
    plot_saccade_angles(
        events_df=events,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        title="Saccade Direction Histogram",
        style=None,
    )

    logger.info("Plotting summary...")
    plot_summary(
        events_df=events,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        title="Summary",
        fix_dur_min=config.FIX_DUR_MIN_MS,
        fix_dur_max=config.FIX_DUR_MAX_MS,
        sac_amp_max=config.SACC_AMP_MAX_DEG,
        sac_dur_max=config.SACC_DUR_MAX_MS,
        drop_near_blinks=config.DROP_NEAR_BLINKS,
    )

    logger.info(f"--> All figures saved to: {out_path}")
    return True


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Monkey Patching: Suppress interactive display unless --show_plots is passed.
    # Replaces plt.show() GLOBALLY, also inside visualisation.py.
    if not args.show_plots:
        plt.show = lambda: None

    # Pipeline Overview
    print(f"\n{'=' * 60}")
    print(f"Pipeline steps : {args.steps}")
    print(f"Subjects       : {args.subjects}")
    print(f"Overwrite      : {args.overwrite}")
    print(f"Show plots     : {args.show_plots}")
    print(f"Log level      : {args.log_level}")
    print(
        f"BIDS root      : {config.DATA_ROOT}"
    )  # TBD make the '\' and '/' einheitlich
    print(f"Output subdir  : .../{config.SESSION}/{config.MISC_SUBDIR}/")
    print(f"Figures subdir : .../{config.MISC_SUBDIR}/{config.PLOTS_SUBDIR}/")
    print(f"{'=' * 60}")

    n_ok = 0
    n_fail = 0

    if args.subjects == ["all"]:
        subjects = os.listdir(config.DATA_ROOT)
        subjects = [m.group(1) for x in subjects if (m := re.search(r"sub-(\d+)", x))]
    else:
        subjects = args.subjects

    for subject_id in subjects:
        print(f"\n── sub-{subject_id} {'─' * (52 - len(subject_id))}")
        ok = True

        if args.steps in ("preprocessing", "all"):
            logger.info(f"Running Preprocessing...")
            success = run_preprocessing(
                subject_id,
                overwrite=args.overwrite,
            )
            ok = ok and success

        if args.steps in ("visualisation", "all"):
            logger.info(f"Running Visualisation...")
            success = run_visualisation(subject_id)
            ok = ok and success

        if ok:
            n_ok += 1
        else:
            n_fail += 1

    # Pipeline Summary
    print(f"\n{'=' * 60}")
    print(f"Completed : {n_ok}/{len(subjects)} subject(s)")
    if n_fail:
        print(f"Failed    : {n_fail}/{len(subjects)} subject(s)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
