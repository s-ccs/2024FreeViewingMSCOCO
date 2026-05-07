"""
report.py: HTML report generation

Overview:
    compute_stats()
    generate_report()

Helpers:
    _fig_to_base64()
    _render_table()
    _render_html()
"""

import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plotting import plot_summary, plot_eye_trace_both_eyes
from config import BLINK_WINDOW_MS

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: make figure HTML style
# =============================================================================
def _fig_to_base64(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =============================================================================
# Stats
# =============================================================================
def compute_stats(events_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics from an events DataFrame.

    Args:
        events_df (pd.DataFrame): Merged events dataframe.

    Returns:
        dict: Nested dict of stats, grouped by event type.
    """
    fix = events_df[events_df["trial_type"] == "fixation"].copy()
    sac = events_df[events_df["trial_type"] == "saccade"].copy()
    blink = events_df[events_df["trial_type"] == "blink"].copy()

    dur_ms = fix["duration"].dropna() * 1000  # convert to ms
    sac_dur_ms = sac["duration"].dropna() * 1000  # convert to ms
    exp_duration = events_df["end_time"].max() - events_df["onset"].min()

    n_near_blink = sac["near_blink"].sum()
    ratio_near_blink = n_near_blink / len(sac) * 100

    stats = {
        "session": {
            "Duration (s)": f"{exp_duration:.1f}",
            "Total events": len(events_df),
        },
        "fixations": {
            "Count": len(fix),
            "Mean duration (ms)": f"{dur_ms.mean():.1f}",
            "Median duration (ms)": f"{dur_ms.median():.1f}",
            "Std duration (ms)": f"{dur_ms.std():.1f}",
            "Min duration (ms)": f"{dur_ms.min():.1f}",
            "Max duration (ms)": f"{dur_ms.max():.1f}",
        },
        "saccades": {
            "Count": len(sac),
            "Near-blink flagged": f"{n_near_blink} ({ratio_near_blink:.1f}%) [window ±{BLINK_WINDOW_MS:.0f} ms]",
            "Mean amplitude (deg)": f"{sac['sacc_visual_angle'].mean():.2f}",
            "Median amplitude (deg)": f"{sac['sacc_visual_angle'].median():.2f}",
            "Mean peak velocity (deg/s)": f"{sac['peak_velocity'].mean():.1f}",
            "Median peak velocity (deg/s)": f"{sac['peak_velocity'].median():.1f}",
            "Mean duration (ms)": f"{sac_dur_ms.mean():.1f}",
            "Median duration (ms)": f"{sac_dur_ms.median():.1f}",
        },
        "blinks": {
            "Count": len(blink),
        },
    }

    return stats


# =============================================================================
# HTML rendering
# =============================================================================
def _render_table(section_title: str, rows: dict) -> str:
    """Render a dict as a simple key-value stats block."""
    rows_html = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows.items())
    return f"""
    <div class="stat-block">
        <h3>{section_title}</h3>
        <table>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>"""


def _render_comparison_table(section_title: str, before: dict, after: dict) -> str:
    """Render a before/after comparison table with delta column."""
    rows_html = ""
    for key in after:
        val_before = before.get(key, "—")
        val_after = after[key]
        # Compute delta only for plain numeric values
        try:
            delta = float(after[key]) - float(before[key])
            delta_str = f"{delta:+.1f}"
        except (ValueError, TypeError):
            delta_str = "—"
        rows_html += f"""
        <tr>
            <td>{key}</td>
            <td>{val_before}</td>
            <td>{val_after}</td>
            <td style="font-weight:bold">{delta_str}</td>
        </tr>"""

    return f"""
    <div class="stat-block wide">
        <details>
            <summary>{section_title}</summary>
            <table>
                <thead>
                    <tr>
                        <th></th>
                        <th style="text-align: left">Before</th>
                        <th style="text-align: left">After</th>
                        <th style="text-align: right">Δ</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </details>
    </div>"""


def _render_html(
    subject_id: str,
    stats_before: dict,
    stats_after: dict,
    eye_trace_plot: str,
    summary_plot: str,
) -> str:
    """
    Assemble the full HTML report string.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    tables = (
        _render_table("Session Info", stats_after["session"])
        + _render_comparison_table(
            "Fixations", stats_before["fixations"], stats_after["fixations"]
        )
        + _render_comparison_table(
            "Saccades", stats_before["saccades"], stats_after["saccades"]
        )
        + _render_comparison_table(
            "Blinks", stats_before["blinks"], stats_after["blinks"]
        )
    )

    return f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>ET Report — {subject_id.replace("sub-", "Subject ")}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1100px;
                    margin: 40px auto;
                    color: #222;
                    background: #fafafa;
                }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 6px; }}
                h2 {{ color: #34495e; margin-top: 36px; }}
                h3 {{ color: #555; margin-bottom: 6px; }}
                .meta {{ color: #888; font-size: 0.9em; margin-bottom: 32px; }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .stat-block {{
                    background: #fff;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    padding: 16px 20px;
                }}
                table {{ width: 100%; border-collapse: collapse; }}
                td {{
                    padding: 4px 8px;
                    font-size: 0.9em;
                    border-bottom: 1px solid #f0f0f0;
                }}
                td:first-child {{ color: #555; }}
                td:last-child {{ text-align: right; font-weight: bold; }}
                .stat-block.wide {{
                    grid-column: 1 / -1;
                }}
                thead th {{
                    text-align: right;
                    color: #555;
                    font-size: 0.85em;
                    padding: 4px 8px;
                    border-bottom: 2px solid #ddd;
                }}
                thead th:first-child {{
                    text-align: left;
                }}
                .plot-block {{
                    background: #fff;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    padding: 20px;
                    text-align: center;
                }}
                img {{ max-width: 100%; height: auto; }}
                details > summary {{
                    cursor: pointer;
                    font-size: 1em;
                    font-weight: bold;
                    color: #555;
                    margin-bottom: 6px;
                    list-style: none;
                    user-select: none;
                }}
                details > summary::before {{
                    content: "▶  ";
                    font-size: 0.75em;
                    color: #aaa;
                    display: inline-block;
                }}
                details[open] > summary::before {{
                    content: "▼  ";
                }}
                details > table {{
                    margin-top: 8px;
                }}
            </style>
        </head>
        <body>
            <h1>Eye-Tracking Report — {subject_id.replace("sub-", "Subject ")}</h1>
            <p class="meta">Generated: {timestamp}</p>
            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                {tables}
            </div>
            <h2>Eye Trace Merge Comparison</h2>
            <div class="plot-block">
                <img src="data:image/png;base64,{eye_trace_plot}" alt="Eye Trace Merge Comparison">
            </div>
            <h2>Summary Plot After Preprocessing</h2>
            <div class="plot-block">
                <img src="data:image/png;base64,{summary_plot}" alt="Summary plot">
            </div>
        </body>
        </html>"""


# =============================================================================
# generate Report
# =============================================================================
def generate_report(
    events_raw: pd.DataFrame,
    events_merged: pd.DataFrame,
    subject_id: str,
    out_path: str,
    by_eye: str = "both",
    sac_amp_max: float = 40,
    fix_dur_min: float = 60,
    fix_dur_max: float = 1000,
    sac_dur_max: float = 120,
    drop_near_blinks: bool = False,
):
    """
    Generate a self-contained HTML report for one subject, containing summary
    statistics and the summary plot.

    Args:
        events_df (pd.DataFrame): Merged events dataframe.
        subject_id (str): Subject ID string, used in the report title and filename.
        out_path (str): Directory to save the report HTML file.
        by_eye (str): One of: 'all', 'left', 'right', 'both'. Defaults to 'both'.
        sac_amp_max (float, optional): Upper bound for saccade amplitude (deg). Defaults to 40.
        fix_dur_min (float, optional): Lower bound for fixation duration (ms). Defaults to 60.
        fix_dur_max (float, optional): Upper bound for fixation duration (ms). Defaults to 1000.
        sac_dur_max (float, optional): Upper bound for saccade duration (ms). Defaults to 120.
        drop_near_blinks (bool, optional): Exclude near-blink saccades from main sequence. Defaults to False.
    """
    logger.info(f"Computing stats...")
    stats_before = compute_stats(events_raw)
    stats_after = compute_stats(events_merged)

    logger.info(f"Generating summary plot...")
    fig = plot_summary(
        events_df=events_merged,
        out_path=None,
        by_eye=by_eye,
        fix_dur_min=fix_dur_min,
        fix_dur_max=fix_dur_max,
        sac_amp_max=sac_amp_max,
        sac_dur_max=sac_dur_max,
        drop_near_blinks=drop_near_blinks,
    )
    summary_plot = _fig_to_base64(fig)
    plt.close(fig)

    logger.info("Generating eye trace comparison plot...")
    fig = plot_eye_trace_both_eyes(
        events_before=events_raw,
        events_after=events_merged,
        out_path=None,
        title="Eye Trace Merge Comparison",
        time_window=(180, 200),
    )
    eye_trace_plot = _fig_to_base64(fig)
    plt.close(fig)

    logger.info(f"[{subject_id}] Rendering HTML report...")
    html = _render_html(
        subject_id, stats_before, stats_after, eye_trace_plot, summary_plot
    )

    out_file = Path(out_path) / f"{subject_id}_report.html"
    out_file.write_text(html, encoding="utf-8")
    logger.info(f"[{subject_id}] Report saved to '{out_file}'")
