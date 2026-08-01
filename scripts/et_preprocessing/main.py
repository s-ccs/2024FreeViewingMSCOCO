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
    merge_fixation_candidates,
    annotate_blink_saccades_in_df
)
from plotting import (
    plot_eye_trace_pre_post_processing,
    plot_main_sequence,
    plot_fixation_duration,
    plot_saccade_amplitude,
    plot_saccade_duration,
    plot_fixation_frequency,
    plot_saccade_angles,
    plot_summary,
    plot_summary_comparison,
)
from report import generate_report

logger = logging.getLogger(__name__)

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

    EXAMPLE
    ------
    DATA_ROOT/
    └── derivatives/
        └── INPUT_DERIVATIVE/
            └── sub-XXX/
                └── ses-001/
                    └── INPUT_SUBDIR/
                        ├── sub-XXX_ses-001_task-freeviewing_run-1_{INPUT_SUFFIX}.tsv
        └── OUTPUT_DERIVATIVE/
            └── sub-XXX/
                └── ses-001/
                    └── OUTPUT_SUBDIR/
                        ├── sub-XXX_ses-001_task-freeviewing_run-1_{OUTPUT_SUFFIX}.tsv
                        └── PLOTS_SUBDIR/
    """
    if not config.INPUT_DERIVATIVE:
        in_dir = Path(config.DATA_ROOT)/f"sub-{subject_id}"/config.SESSION/config.INPUT_SUBDIR
    else:
        in_dir = Path(config.DATA_ROOT)/"derivatives"/config.INPUT_DERIVATIVE/f"sub-{subject_id}"/config.SESSION/config.INPUT_SUBDIR

    out_dir = Path(config.DATA_ROOT)/"derivatives"/config.OUTPUT_DERIVATIVE/f"sub-{subject_id}"/config.SESSION/config.OUTPUT_SUBDIR

    if config.RUN:
        stem = f"sub-{subject_id}_{config.SESSION}_task-{config.TASK}_run-{config.RUN}"
    else:
        stem = f"sub-{subject_id}_{config.SESSION}_task-{config.TASK}"

    return {
        "in_dir": in_dir,
        "out_dir": out_dir,
        "in_tsv": in_dir / f"{stem}_{config.INPUT_SUFFIX}.tsv",
        "out_tsv": out_dir / f"{stem}_{config.OUTPUT_SUFFIX}.tsv",
        "plots_dir": out_dir / config.PLOTS_SUBDIR
    }


def run_preprocessing(subject_id: str, overwrite: bool) -> bool:
    """
    1. Load events: sub-XXX_ses-001_task-freeviewing_et_events.tsv
    2. Run the two-stage merge + save to merged TSV: sub-XXX_ses-001_task-freeviewing_et_events_merged.tsv
    3. pre/post-merge eye trace comparison figure.
    4. report generation

    Args:
        subject_id (str): subject ID (zero-padded)
        overwrite (bool): if False -> skips subs when merged.tsv exists

    Returns:
        bool: preprocessing ran through
    """
    paths = subject_paths(subject_id)

    if os.path.exists(paths["out_tsv"]) and not overwrite:
        logger.info(
            f"Skipping preprocessing for {subject_id} — merged file already exists: "
            f"{paths['out_tsv']}. Use --overwrite to reprocess."
        )
        return True

    # 1. Load events
    logger.info(f"Loading events TSV from dir:{paths['in_dir']} ...")
    try:
        events_raw = load_subject_tsv(
            filepath=paths["in_tsv"],
            subject_id=subject_id,
        )
    except FileNotFoundError as e:
        logger.error(e)
        return False

    # 2. Annotate blink saccades
    logger.info(f"Annotating blink saccades (window={config.BLINK_WINDOW_MS} ms")
    events_raw = annotate_blink_saccades_in_df(events_raw, window_ms=config.BLINK_WINDOW_MS)

    # 3a. Hooge et al. (2022), Stage 1
    # Saccades which are below a certain amplitude and duration are dropped
    # and the surrounding fixations are merged
    logger.info(
        f"Merging events  "
        f"(a_min={config.A_MIN}°, "
        f"t_min_fix={config.T_MIN_FIX * 1000:.0f} ms, "
    )
    if isinstance(config.MERGE_THRESHOLD, (int, float)):
        merge_threshold = config.MERGE_THRESHOLD
    else:
        logger.warning(f"MERGE_THRESHOLD is not a number: {type(config.MERGE_THRESHOLD)}. MERGE_THRESHOLD will be set to 'None'")
        merge_threshold = None
        
    events_merged = merge_fixation_candidates(
        events_raw,
        a_min=config.A_MIN,
        merge_threshold=merge_threshold
    )

    # 3b. Hooge et al. (2022), Stage 2
    # Fixations shorter than the specified threshold are dropped from the events dataframe.
    if config.T_MIN_FIX is not None:
        t_min_fix_s = config.T_MIN_FIX / 1000.0
        n_fix_before = int((events_merged["trial_type"] == "fixation").sum())

        idx_drop_fix = events_merged.index[
            (events_merged["trial_type"] == "fixation")
            & (events_merged["duration"] < t_min_fix_s)
        ]
        events_merged = events_merged.drop(idx_drop_fix).reset_index(drop=True)

        pct = len(idx_drop_fix) / max(n_fix_before, 1) * 100
        logger.info(
            f"Stage 2: dropped {len(idx_drop_fix)}/{n_fix_before} fixations "
            f"({pct:.1f}%, both eyes) shorter than {config.T_MIN_FIX:.0f} ms."
        )
        if pct > 10:
            logger.warning(
                f"Stage 2 removed {pct:.1f}% of fixations. Above ~10% this usually means "
                f"Stage 1 is not merging — check the merge counts above."
            )

    # Save
    os.makedirs(paths["out_dir"], exist_ok=True)
    events_merged.to_csv(paths["out_tsv"], sep="\t", index=False)
    logger.info(f"--> Saved merged TSV: {paths['out_tsv']}")

    # 3. pre/post-merge eye trace comparison
    os.makedirs(paths["plots_dir"], exist_ok=True)
    logger.info("Plotting eye trace comparison...")
    plot_eye_trace_pre_post_processing(
        events_before=events_raw,
        events_after=events_merged,
        out_path=paths["plots_dir"],
        out_file_format="svg",
        title="Eye Trace Merge Comparison",
        window_size=20,
        top_n=3,
    )

    # 4. before/after summary comparison (needs both raw + merged data)
    logger.info("Plotting before/after summary comparison...")
    fig = plot_summary_comparison(
        events_before=events_raw,
        events_after=events_merged,
        out_path=str(paths["plots_dir"]),
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        fix_dur_min=config.FIX_DUR_MIN_MS,
        fix_dur_max=config.FIX_DUR_MAX_MS,
        sac_amp_max=config.SAC_AMP_MAX_DEG,
        sac_dur_max=config.SAC_DUR_MAX_MS,
        include_blink_sac=config.INCLUDE_BLINK_SAC,
    )
    plt.close(fig)

    if config.REPORT:
        logger.info("Generating Subject Report...")
        generate_report(
            events_raw=events_raw,
            events_merged=events_merged,
            subject_id=f"sub-{subject_id}",
            out_path=str(paths["out_dir"]),
            by_eye=config.BY_EYE,
            sac_amp_max=config.SAC_AMP_MAX_DEG,
            fix_dur_min=config.FIX_DUR_MIN_MS,
            fix_dur_max=config.FIX_DUR_MAX_MS,
            sac_dur_max=config.SAC_DUR_MAX_MS,
            include_blink_sac=config.INCLUDE_BLINK_SAC,
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

    if not os.path.exists(paths["out_tsv"]):
        logger.warning(
            f"No merged file found for {subject_id}: {paths['out_tsv'].name}. "
            f"Run --steps preprocessing first."
        )
        return False

    logger.info(f"Loading merged events from dir: {paths['out_dir']} ...")
    events = pd.read_csv(paths["out_tsv"], sep="\t")
    # Load raw via load_subject_tsv (not plain read_csv) so blink saccades get annotated;
    # the before/after comparison needs the 'blink_saccade' column to highlight them.
    events_raw = load_subject_tsv(
        filepath=paths["in_tsv"],
        subject_id=subject_id,
    )
    events_raw = annotate_blink_saccades_in_df(
        events_raw, window_ms=config.BLINK_WINDOW_MS
    )

    out_path = str(paths["plots_dir"])
    os.makedirs(out_path, exist_ok=True)

    logger.info("Plotting main sequence...")
    fig = plot_main_sequence(
        events_df=events,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        include_blink_sac=config.INCLUDE_BLINK_SAC,
    )
    plt.close(fig)

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
        sac_amp_max=config.SAC_AMP_MAX_DEG,
        include_blink_sac=config.INCLUDE_BLINK_SAC
    )

    logger.info("Plotting saccade duration...")
    plot_saccade_duration(
        events_df=events,
        by_eye=config.BY_EYE,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        sac_dur_max=config.SAC_DUR_MAX_MS,
        include_blink_sac=config.INCLUDE_BLINK_SAC
    )

    logger.info("Plotting fixation frequency...")
    fig = plot_fixation_frequency(
        events_df=events,
        by_eye=config.BY_EYE,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
    )
    plt.close(fig)

    logger.info("Plotting saccade angular histogram...")
    plot_saccade_angles(
        events_df=events,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        title="Saccade Direction Histogram",
        style=None,
        include_blink_sac=config.INCLUDE_BLINK_SAC,
    )

    logger.info("Plotting summary...")
    fig = plot_summary(
        events_df=events,
        out_path=out_path,
        out_file_format=config.OUT_FILE_FORMAT,
        by_eye=config.BY_EYE,
        title="Summary",
        fix_dur_min=config.FIX_DUR_MIN_MS,
        fix_dur_max=config.FIX_DUR_MAX_MS,
        sac_amp_max=config.SAC_AMP_MAX_DEG,
        sac_dur_max=config.SAC_DUR_MAX_MS,
        include_blink_sac=config.INCLUDE_BLINK_SAC,
        dropout_stats=config.DROPOUT_STATS,
    )
    plt.close(fig)
    
    logger.info("Plotting summary comparison...")
    fig = plot_summary_comparison(
        events_before=events_raw,
        events_after=events,
        out_path=None,
        by_eye=config.BY_EYE,
        fix_dur_min=config.FIX_DUR_MIN_MS,
        fix_dur_max=config.FIX_DUR_MAX_MS,
        sac_amp_max=config.SAC_AMP_MAX_DEG,
        sac_dur_max=config.SAC_DUR_MAX_MS,
        include_blink_sac=config.INCLUDE_BLINK_SAC,
        dropout_stats=config.DROPOUT_STATS,
    )
    plt.close(fig)

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
    print(f"\n{'=' * 64}")
    print(f"Pipeline steps     : {args.steps}")
    print(f"Subjects           : {args.subjects}")
    print(f"Overwrite          : {args.overwrite}")
    print(f"Show plots         : {args.show_plots}")
    print(f"Log level          : {args.log_level}")
    print(f"BIDS root          : {config.DATA_ROOT}")
    print(f"Input derivative   : .../{config.INPUT_DERIVATIVE}")
    print(f"Output derivative  : .../{config.OUTPUT_DERIVATIVE}")
    print(f"{'=' * 64}")

    n_ok = 0
    n_fail = 0

    if args.subjects == ["all"]:
        subjects = os.listdir(config.DATA_ROOT / "derivatives" / config.INPUT_DERIVATIVE)
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
