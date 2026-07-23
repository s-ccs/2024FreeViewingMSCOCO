"""
report.py: HTML report generation.

Structure of the generated report:
    Left sidebar navigation + table of contents, and five sections:
        1. Overview          -> meta + summary statistics (before/after)
        2. Blink             -> blink-saccade flagging, main sequence (highlighted), window
        3. Fixation Merger   -> eye-trace carousel, algorithm, thresholds, data changes
        4. Summary Plots     -> 2x3 before/after comparison, 2x3 after merge, dropout note
        5. Config            -> collapsible dump of config.py values

Public:
    compute_stats(), generate_report()
Helpers:
    _fig_to_base64(), _render_table(), _render_comparison_table(),
    _render_carousel(), _render_config()
"""

import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config
from config import BLINK_WINDOW_MS, A_MIN
from plotting import (
    plot_summary,
    plot_summary_comparison,
    plot_main_sequence,
    plot_eye_trace_pre_post_processing,
)
from graphs import reset_dropout_stats, get_dropout_stats

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: figure -> base64 PNG
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
        events_df (pd.DataFrame): Events dataframe.

    Returns:
        dict: Nested dict of stats, grouped by event type.
    """
    fix = events_df[events_df["trial_type"] == "fixation"].copy()
    sac = events_df[events_df["trial_type"] == "saccade"].copy()
    blink = events_df[events_df["trial_type"] == "blink"].copy()

    dur_ms = fix["duration"].dropna() * 1000
    sac_dur_ms = sac["duration"].dropna() * 1000
    exp_duration = events_df["end_time"].max() - events_df["onset"].min()

    n_blink_saccade = int(sac["blink_saccade"].sum()) if "blink_saccade" in sac else 0
    ratio_blink_saccade = n_blink_saccade / len(sac) * 100 if len(sac) else 0.0

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
            "Blink saccade flagged": f"{n_blink_saccade} ({ratio_blink_saccade:.1f}%) [window ±{BLINK_WINDOW_MS:.0f} ms]",
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
# HTML fragment helpers
# =============================================================================
def _render_table(section_title: str, rows: dict) -> str:
    """Render a dict as a simple key-value stats block."""
    rows_html = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows.items())
    return f"""
    <div class="stat-block">
        <h3>{section_title}</h3>
        <table><tbody>{rows_html}</tbody></table>
    </div>"""


def _render_comparison_table(section_title: str, before: dict, after: dict) -> str:
    """Render a before/after comparison table (collapsible) with a delta column."""
    rows_html = ""
    for key in after:
        val_before = before.get(key, "—")
        val_after = after[key]
        try:
            delta = float(after[key]) - float(before[key])
            delta_str = f"{delta:+.1f}"
        except (ValueError, TypeError):
            delta_str = "—"
        rows_html += f"""
        <tr><td>{key}</td><td>{val_before}</td><td>{val_after}</td>
            <td style="font-weight:bold">{delta_str}</td></tr>"""

    return f"""
    <div class="stat-block wide">
        <details>
            <summary>{section_title}</summary>
            <table>
                <thead><tr><th></th><th style="text-align:left">Before</th>
                    <th style="text-align:left">After</th><th style="text-align:right">Δ</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </details>
    </div>"""


def _render_carousel(carousel_id: str, slides: list) -> str:
    """Render an image carousel from a list of (label, base64_png) tuples."""
    if not slides:
        return "<p class='plot-caption'>No eye-trace figures available.</p>"

    track = ""
    dots = ""
    for i, (label, b64) in enumerate(slides):
        track += f"""
            <div class="carousel-slide">
                <p class="slide-label">{label}</p>
                <img src="data:image/png;base64,{b64}" alt="{label}">
            </div>"""
        active = " active" if i == 0 else ""
        dots += (
            f'<span class="dot{active}" onclick="goToSlide(\'{carousel_id}\', {i})" '
            f'title="{label}"></span>'
        )

    return f"""
    <div class="carousel" id="{carousel_id}">
        <button class="carousel-btn prev" onclick="moveCarousel('{carousel_id}', -1)">&#8592;</button>
        <div class="carousel-track-wrapper"><div class="carousel-track">{track}</div></div>
        <button class="carousel-btn next" onclick="moveCarousel('{carousel_id}', 1)">&#8594;</button>
        <div class="carousel-dots">{dots}</div>
    </div>"""

def _render_dropout_table(stats: dict) -> str:
    """Render plotting-only dropout counts. Columns adapt to the recorded stages."""
    by_metric = {}
    for rec in stats.values():
        by_metric.setdefault(rec["metric"], {})[rec.get("stage", "single")] = rec

    order = ["before", "after"]
    stages = [s for s in order if any(s in d for d in by_metric.values())]
    head = "".join(f'<th style="text-align:left">Dropped ({s})</th>' for s in stages)

    rows = ""
    for metric, d in by_metric.items():
        window = next(iter(d.values()))["window"]
        cells = "".join(
            (f"<td>{d[s]['dropped']} / {d[s]['total']} ({d[s]['pct']:.1f}%)</td>" if s in d else "<td>—</td>")
            for s in stages
        )
        rows += f"<tr><td>{metric}</td><td>{window}</td>{cells}</tr>"

    return f"""
    <div class="stat-block wide"><table>
        <thead><tr><th>Metric</th><th style="text-align:left">Plotting range</th>{head}</tr></thead>
        <tbody>{rows}</tbody></table></div>"""

def _render_config() -> str:
    """Render all upper-case config.py constants as a collapsible table."""
    EXCLUDE = {"DATA_ROOT"}  # hide data root path for privacy
    items = {
        k: getattr(config, k)
        for k in dir(config)
        if k.isupper() and not k.startswith("_") and k not in EXCLUDE
    }
    rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(items.items())
    )
    return f"""
    <div class="stat-block wide">
        <details>
            <summary>config.py values</summary>
            <table><tbody>{rows}</tbody></table>
        </details>
    </div>"""


# =============================================================================
# Static CSS / JS  (plain strings — no f-string brace escaping needed)
# =============================================================================
_CSS = """
* { box-sizing: border-box; }
body { font-family: Arial, sans-serif; color: #222; background: #fafafa; margin: 0; }
.layout { display: flex; align-items: flex-start; }
.sidebar {
    position: sticky; top: 0; align-self: flex-start;
    width: 220px; min-height: 100vh; padding: 28px 16px;
    background: #fff; border-right: 1px solid #ddd;
}
.sidebar .brand { font-weight: bold; color: #2c3e50; margin-bottom: 4px; }
.sidebar .toc-title { font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.08em; color: #999; margin: 24px 0 8px; }
.sidebar nav a { display: block; padding: 7px 10px; margin: 2px 0; border-radius: 5px;
    color: #34495e; text-decoration: none; font-size: 0.9em; }
.sidebar nav a:hover { background: #f0f0f0; }
.sidebar nav a.active { background: #2c3e50; color: #fff; }
.content { flex: 1; max-width: 1000px; margin: 0 auto; padding: 32px 44px 80px; }
h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 6px; }
h2 { color: #34495e; margin-top: 8px; }
h3 { color: #555; margin-bottom: 6px; }
.meta { color: #888; font-size: 0.9em; margin-bottom: 8px; }
section { scroll-margin-top: 16px; margin-bottom: 56px; padding-top: 8px; }
section > .sec-num { color: #aab; font-weight: bold; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 8px; }
.stat-block { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 16px 20px; }
.stat-block.wide { grid-column: 1 / -1; }
table { width: 100%; border-collapse: collapse; }
td { padding: 4px 8px; font-size: 0.9em; border-bottom: 1px solid #f0f0f0; }
td:first-child { color: #555; }
td:last-child { text-align: right; font-weight: bold; }
thead th { text-align: right; color: #555; font-size: 0.85em; padding: 4px 8px; border-bottom: 2px solid #ddd; }
thead th:first-child { text-align: left; }
.plot-block { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 20px; text-align: center; margin-bottom: 20px; }
.plot-block h3 { text-align: left; }
img { max-width: 100%; height: auto; }
.plot-caption { color: #555; font-size: 0.9em; margin: 4px 0 16px; }
.note { background: #fff8e6; border: 1px solid #f0d98a; border-radius: 6px; padding: 12px 16px; font-size: 0.88em; color: #6b5b1f; margin: 12px 0; }
.formula { background: #f4f6f8; border-left: 3px solid #2c3e50; padding: 8px 14px; font-family: "Courier New", monospace; font-size: 0.9em; margin: 12px 0; }
details > summary { cursor: pointer; font-weight: bold; color: #555; margin-bottom: 6px; list-style: none; user-select: none; }
details > summary::before { content: "\\25B6  "; font-size: 0.75em; color: #aaa; }
details[open] > summary::before { content: "\\25BC  "; }
/* carousel */
.carousel { position: relative; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 40px 60px 24px; margin-bottom: 20px; }
.carousel-track-wrapper { overflow: hidden; }
.carousel-track { display: flex; transition: transform 0.3s ease; }
.carousel-slide { min-width: 100%; text-align: center; }
.carousel-btn { position: absolute; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.9);
    border: 1px solid #ccc; border-radius: 50%; width: 36px; height: 36px; cursor: pointer; font-size: 1.1em; z-index: 10; }
.carousel-btn.prev { left: 10px; }
.carousel-btn.next { right: 10px; }
.carousel-dots { text-align: center; margin-top: 12px; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #ccc; margin: 0 4px; cursor: pointer; transition: background 0.2s; }
.dot.active { background: #2c3e50; }
.slide-label { font-size: 0.85em; color: #888; margin-bottom: 8px; }
"""

_JS = """
function moveCarousel(id, direction) {
    const carousel = document.getElementById(id);
    const track = carousel.querySelector('.carousel-track');
    const slides = track.querySelectorAll('.carousel-slide');
    const dots = carousel.querySelectorAll('.dot');
    let current = parseInt(carousel.dataset.current || 0);
    current = (current + direction + slides.length) % slides.length;
    carousel.dataset.current = current;
    track.style.transform = 'translateX(-' + (current * 100) + '%)';
    dots.forEach(function(d, i) { d.classList.toggle('active', i === current); });
}
function goToSlide(id, index) {
    const carousel = document.getElementById(id);
    const track = carousel.querySelector('.carousel-track');
    const dots = carousel.querySelectorAll('.dot');
    carousel.dataset.current = index;
    track.style.transform = 'translateX(-' + (index * 100) + '%)';
    dots.forEach(function(d, i) { d.classList.toggle('active', i === index); });
}
// Sidebar scrollspy: highlight the section currently in view
document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('.sidebar nav a');
    const map = {};
    links.forEach(function(a) { map[a.getAttribute('href').slice(1)] = a; });
    const obs = new IntersectionObserver(function(entries) {
        entries.forEach(function(e) {
            if (e.isIntersecting) {
                links.forEach(function(a) { a.classList.remove('active'); });
                if (map[e.target.id]) map[e.target.id].classList.add('active');
            }
        });
    }, { rootMargin: '-20% 0px -70% 0px' });
    document.querySelectorAll('section').forEach(function(s) { obs.observe(s); });
});
"""


# =============================================================================
# HTML assembly
# =============================================================================
def _render_html(
    subject_id,
    timestamp,
    stats_before,
    stats_after,
    ms_highlight_plot,
    eye_trace_slides,
    summary_comparison_plot,
    summary_after_plot,
    merge_info,
    dropout_html,
    config_html,
) -> str:
    """Assemble the full HTML report string."""
    title = subject_id.replace("sub-", "Subject ")

    stats_grid = (
        _render_table("Session Info", stats_after["session"])
        + _render_comparison_table("Fixations", stats_before["fixations"], stats_after["fixations"])
        + _render_comparison_table("Saccades", stats_before["saccades"], stats_after["saccades"])
        + _render_comparison_table("Blinks", stats_before["blinks"], stats_after["blinks"])
    )
    carousel = _render_carousel("carousel-eyetrace", eye_trace_slides)

    nav = """
        <nav>
            <a href="#sec-overview">1 · Overview</a>
            <a href="#sec-blink">2 · Blink</a>
            <a href="#sec-merger">3 · Eye-movement selection</a>
            <a href="#sec-summary">4 · Summary Plots</a>
            <a href="#sec-config">5 · Config</a>
        </nav>"""

    body = f"""
        <section id="sec-overview">
            <h1>Eye-Tracking Report — {title}</h1>
            <p class="meta">Generated: {timestamp}</p>
            <h2><span class="sec-num">1</span> · Overview</h2>
            <p class="plot-caption">Summary statistics before vs. after preprocessing
                (micro-saccade removal + fixation merge).</p>
            <div class="stats-grid">{stats_grid}</div>
        </section>

        <section id="sec-blink">
            <h2><span class="sec-num">2</span> · Blink Saccades</h2>
            <p class="plot-caption">
                Saccades that fall within a time window of ±{merge_info['blink_window']:.0f} ms
                around a blink event are flagged as <strong>blink saccades</strong>
                (they are often lid-movement artefacts). Flagged: <strong>{merge_info['blink_flagged']}</strong>.
                Whether they are kept, dropped or highlighted is controlled by
                <code>INCLUDE_BLINK_SAC</code> in the config or in the parameters of the plotting function.
            </p>
            <div class="plot-block">
                <h3>Main Sequence before processing (blink saccades highlighted)</h3>
                <img src="data:image/png;base64,{ms_highlight_plot}" alt="Main sequence before processing with highlighted blink saccades">
            </div>
        </section>

        <section id="sec-merger">
            <h2><span class="sec-num">3</span> · Fixation Merger</h2>
            <p class="plot-caption">
                Two-step procedure following Hooge et al. (2022): first, implausibly small
                <em>and</em> short saccades are dropped; then consecutive
                fixations of the same eye that are now no longer separated by such a saccade are merged.
                In the second step, fixations with a duration below a certain threshold e.g. 60 ms are dropped.
            </p>
            <div class="formula">
                drop saccade &nbsp;if&nbsp; amplitude &lt; a_min ({merge_info['a_min']}°)
                &nbsp;AND&nbsp; duration &lt; T_min<br>
                T_min (ms) = 2.2 · a_min + 27 = {merge_info['t_min_ms']:.1f} ms
            </div>
            <p class="plot-caption"><strong>Effect on the data:</strong>
                saccades {merge_info['n_sac_before']} → {merge_info['n_sac_after']}
                ({merge_info['d_sac']:+d}),
                fixations {merge_info['n_fix_before']} → {merge_info['n_fix_after']}
                ({merge_info['d_fix']:+d}). Dropping saccades below threshold removes saccades;
                merging then reduces the fixation count (fewer, longer fixations).</p>
            <h3>Eye-trace: before vs. after merge</h3>
            <p class="plot-caption">Top-3 time windows per eye with the most merges. Fixations before
                merging in <span style="color:blue"><strong>blue</strong></span>,
                after merging in <span style="color:orange"><strong>orange</strong></span>.</p>
            {carousel}
        </section>

        <section id="sec-summary">
            <h2><span class="sec-num">4</span> · Summary Plots</h2>
            <div class="plot-block">
                <h3>Summary plots: before vs. after preprocessing</h3>
                <img src="data:image/png;base64,{summary_comparison_plot}" alt="Summary comparison before/after">
            </div>
            <div class="plot-block">
                <h3>Summary plots: after preprocessing</h3>
                <img src="data:image/png;base64,{summary_after_plot}" alt="Summary after merge">
            </div>
            <h3>Values dropped for display only</h3>
            <div class="note">
                <strong>Note:</strong> in the plots above, values outside the plotting range
                (e.g. implausibly long fixations or large amplitudes) are dropped <strong>for display
                only</strong>. This is a plotting design decision and does <strong>not</strong> alter the
                underlying event data.
            </div>
            {dropout_html}
        </section>

        <section id="sec-config">
            <h2><span class="sec-num">5</span> · Config</h2>
            <p class="plot-caption">Configuration used for this run (<code>config.py</code>).</p>
            <div class="stats-grid">{config_html}</div>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ET Report — {title}</title>
    <style>{_CSS}</style>
</head>
<body>
    <div class="layout">
        <aside class="sidebar">
            <div class="brand">ET Report</div>
            <div class="meta">{title}</div>
            <div class="toc-title">Contents</div>
            {nav}
        </aside>
        <main class="content">
            {body}
        </main>
    </div>
    <script>{_JS}</script>
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
    by_eye: str = "right",
    sac_amp_max: float = 40,
    fix_dur_min: float = 60,
    fix_dur_max: float = 1000,
    sac_dur_max: float = 120,
    include_blink_sac: bool | str = True,
):
    """
    Generate a self-contained, navigable HTML report for one subject.

    Args:
        events_raw (pd.DataFrame): Original (pre-merge) events dataframe.
        events_merged (pd.DataFrame): Merged (post-merge) events dataframe.
        subject_id (str): Subject ID string, used in the report title and filename.
        out_path (str): Directory to save the report HTML file.
        by_eye (str): One of: 'all', 'left', 'right', 'binocular'. Defaults to 'right'.
        sac_amp_max (float, optional): Upper bound for saccade amplitude (deg). Defaults to 40.
        fix_dur_min (float, optional): Lower bound for fixation duration (ms). Defaults to 60.
        fix_dur_max (float, optional): Upper bound for fixation duration (ms). Defaults to 1000.
        sac_dur_max (float, optional): Upper bound for saccade duration (ms). Defaults to 120.
        include_blink_sac (bool | str): Blink-saccade handling for the summary main sequence.
    """
    logger.info("Computing stats...")
    stats_before = compute_stats(events_raw)
    stats_after = compute_stats(events_merged)

    # --- Section 2: main sequence with highlighted blink saccades ---
    logger.info("Generating main sequence (highlighted blinks)...")
    fig = plot_main_sequence(
        events_df=events_raw,
        out_path=None,
        by_eye=by_eye,
        include_blink_sac="highlight",
    )
    ms_highlight_plot = _fig_to_base64(fig)
    plt.close(fig)

    # --- Section 3: eye-trace carousel ---
    logger.info("Generating eye-trace comparison...")
    figs = plot_eye_trace_pre_post_processing(
        events_before=events_raw,
        events_after=events_merged,
        out_path=None,
        title="Eye Trace Merge Comparison",
        top_n=3,
    )
    eye_trace_slides = []
    for key, fig in figs.items():
        rank, eye = key.split("-")
        eye_name = "Left" if eye == "L" else "Right"
        suffix = " (most merges)" if rank == "0" else ""
        eye_trace_slides.append(
            (f"{eye_name} Eye — Rank {int(rank) + 1}{suffix}", _fig_to_base64(fig))
        )
        plt.close(fig)

    # --- Section 4: before/after comparison + after-merge summary ---
    logger.info("Generating summary comparison...")
    fig = plot_summary_comparison(
        events_before=events_raw,
        events_after=events_merged,
        out_path=None,
        by_eye=by_eye,
        fix_dur_min=fix_dur_min,
        fix_dur_max=fix_dur_max,
        sac_amp_max=sac_amp_max,
        sac_dur_max=sac_dur_max,
        include_blink_sac=include_blink_sac,
    )
    summary_comparison_plot = _fig_to_base64(fig)
    plt.close(fig)

    logger.info("Generating summary (after merge)...")
    reset_dropout_stats()
    fig = plot_summary(
        events_df=events_merged,
        out_path=None,
        by_eye=by_eye,
        fix_dur_min=fix_dur_min,
        fix_dur_max=fix_dur_max,
        sac_amp_max=sac_amp_max,
        sac_dur_max=sac_dur_max,
        include_blink_sac=False,
        dropout_stats=True,
    )
    summary_after_plot = _fig_to_base64(fig)
    plt.close(fig)
    dropout_stats = get_dropout_stats()

    # --- prose numbers ---
    n_fix_before = int(stats_before["fixations"]["Count"])
    n_fix_after = int(stats_after["fixations"]["Count"])
    n_sac_before = int(stats_before["saccades"]["Count"])
    n_sac_after = int(stats_after["saccades"]["Count"])
    merge_info = {
        "a_min": A_MIN,
        "t_min_ms": 2.2 * A_MIN + 27,
        "blink_window": BLINK_WINDOW_MS,
        "blink_flagged": stats_after["saccades"]["Blink saccade flagged"],
        "n_fix_before": n_fix_before,
        "n_fix_after": n_fix_after,
        "d_fix": n_fix_after - n_fix_before,
        "n_sac_before": n_sac_before,
        "n_sac_after": n_sac_after,
        "d_sac": n_sac_after - n_sac_before,
    }

    logger.info(f"[{subject_id}] Rendering HTML report...")
    html = _render_html(
        subject_id=subject_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        stats_before=stats_before,
        stats_after=stats_after,
        ms_highlight_plot=ms_highlight_plot,
        eye_trace_slides=eye_trace_slides,
        summary_comparison_plot=summary_comparison_plot,
        summary_after_plot=summary_after_plot,
        merge_info=merge_info,
        config_html=_render_config(),
        dropout_html=_render_dropout_table(dropout_stats),
    )

    out_file = Path(out_path) / f"{subject_id}_report.html"
    out_file.write_text(html, encoding="utf-8")
    logger.info(f"[{subject_id}] Report saved to '{out_file}'")
    return str(out_file)
