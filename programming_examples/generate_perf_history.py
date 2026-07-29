#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Generates programming_examples/perf-history.html: per-nightly TTFT (ms) and
# decode tok/s for each LLM, plotted over time with Chart.js. Data comes from
# the append-only history.ndjson accumulated on the `perf-history` branch by
# llms/bench/append_history.py.
#
# Usage:
#   python3 generate_perf_history.py --history history.ndjson --output perf-history.html
#
# Each history.ndjson line is a flat record:
#   {"date","timestamp_utc","run_id","air_sha","aie_hash","peano","model",
#    "ttft_ms","decode_tokens_per_sec","context_len","verify_status"}
#
# The x-axis is one point per nightly (labeled by date; the built commit SHA is
# shown in the tooltip). A datapoint whose verify_status == "fail" is drawn with
# a red marker so regressions in correctness stand out on the throughput lines.

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Reused so model ordering/labels match the operator+LLM dashboard.
try:
    from generate_readme import LLM_HF_MODELS
except ImportError:  # pragma: no cover - allow running from another cwd
    import sys

    sys.path.insert(0, str(SCRIPT_DIR))
    from generate_readme import LLM_HF_MODELS

FAIL_COLOR = "#d32f2f"

# Plot only the most recent N unique dates; the full history.ndjson is kept
# untouched. ~30 nightlies (about a month) keeps date labels legible at the
# page's 1000px chart width — beyond ~45 the labels collide.
DEFAULT_WINDOW = 30

# Distinct-enough palette for up to ~12 model lines; cycled if exceeded.
# Deliberately excludes orange/red/brown hues so no model line is confusable
# with the red ✕ used to flag failed-verify points (see _series).
PALETTE = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#e377c2",  # magenta
    "#7f7f7f",  # gray
    "#393b79",  # indigo
    "#1b9e77",  # teal
    "#637939",  # dark green
    "#756bb1",  # slate
    "#31a354",  # emerald
]


def load_history(history_path):
    """Read history.ndjson into a list of row dicts (empty if absent/empty)."""
    p = Path(history_path) if history_path else None
    if not p or not p.is_file():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _model_order(rows):
    """Models sorted with dashboard-known ones first (registry order), then any
    extras alphabetically — so the legend is stable across runs."""
    present = {r.get("model", "") for r in rows if r.get("model")}
    known = [m for m in LLM_HF_MODELS if m in present]
    extra = sorted(present - set(known))
    return known + extra


def _series(rows, models, labels, metric):
    """Build one Chart.js dataset per model for the given metric key.

    Values are aligned to `labels` (the sorted unique dates); a missing date is
    null so Chart.js leaves a gap. Per-point colors flag failed-verify runs red.
    """
    # (model, date) -> most recent row for that pair
    by_key = {}
    for r in rows:
        by_key[(r.get("model"), r.get("date"))] = r

    datasets = []
    for i, model in enumerate(models):
        color = PALETTE[i % len(PALETTE)]
        data, point_colors, metas = [], [], []
        point_styles, point_radii, hover_radii = [], [], []
        for date in labels:
            r = by_key.get((model, date))
            if r is None:
                data.append(None)
                point_colors.append(color)
                point_styles.append("circle")
                point_radii.append(4)
                hover_radii.append(6)
                metas.append(None)
                continue
            val = r.get(metric)
            data.append(val)
            failed = r.get("verify_status") == "fail"
            # Failed verify is encoded by SHAPE (a red ✕), not color alone, so
            # it is unambiguous against any line hue and colorblind-friendly.
            point_colors.append(FAIL_COLOR if failed else color)
            point_styles.append("crossRot" if failed else "circle")
            point_radii.append(8 if failed else 4)
            hover_radii.append(10 if failed else 6)
            metas.append(
                {
                    "sha": (r.get("air_sha") or "")[:7],
                    "verify": r.get("verify_status") or "",
                }
            )
        datasets.append(
            {
                "label": model,
                "data": data,
                "borderColor": color,
                "backgroundColor": color,
                "pointBackgroundColor": point_colors,
                "pointBorderColor": point_colors,
                "pointStyle": point_styles,
                "pointRadius": point_radii,
                "pointHoverRadius": hover_radii,
                "pointBorderWidth": 3,
                "spanGaps": False,
                "tension": 0.2,
                # Non-reserved key: Chart.js has historically used dataset._meta
                # internally, so keep our per-point metadata under our own name.
                "verifyMeta": metas,
            }
        )
    return datasets


HTML_TEMPLATE = """\
<!doctype html>
<html lang="en-US">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLIR-AIR — LLM Performance History</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         max-width: 1000px; margin: 0 auto; padding: 24px; color: #1b1f23; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .sub {{ color: #586069; margin-top: 0; }}
  .chart-box {{ position: relative; height: 380px; margin: 32px 0; }}
  a {{ color: #0366d6; }}
  .legend-note {{ color: #586069; font-size: 0.85rem; }}
</style>
</head>
<body>
<p><a href="index.html">&larr; Back to Programming Examples dashboard</a></p>
<h1>LLM Performance History (NPU2)</h1>
<p class="sub">Per-nightly end-to-end inference performance on the AMD Ryzen AI (Krackan Point, NPU2)
benchmark runner. Each point is one nightly build, labeled by date; hover to see the commit SHA and
verify status. A <span style="color:{fail_color}; font-weight:700;">red &#10007; marker</span> flags a nightly
whose correctness verify failed (the marker shape, not just its color, signals the failure).</p>
<p class="legend-note">Click a model in a legend to toggle its line.{window_note}</p>

<div class="chart-box"><canvas id="ttft"></canvas></div>
<div class="chart-box"><canvas id="decode"></canvas></div>

<script>
const LABELS = {labels_json};
const TTFT_DATASETS = {ttft_json};
const DECODE_DATASETS = {decode_json};

function makeChart(canvasId, datasets, yLabel) {{
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {{
    type: 'line',
    data: {{ labels: LABELS, datasets: datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'nearest', intersect: false }},
      plugins: {{
        title: {{ display: true, text: yLabel }},
        tooltip: {{
          callbacks: {{
            afterLabel: function(item) {{
              const meta = item.dataset.verifyMeta && item.dataset.verifyMeta[item.dataIndex];
              if (!meta) return '';
              let s = 'commit ' + meta.sha;
              if (meta.verify) s += '  (verify: ' + meta.verify + ')';
              return s;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'nightly (date)' }} }},
        y: {{ title: {{ display: true, text: yLabel }}, beginAtZero: true }}
      }}
    }}
  }});
}}

makeChart('ttft', TTFT_DATASETS, 'TTFT (ms) — lower is better');
makeChart('decode', DECODE_DATASETS, 'Decode throughput (tok/s) — higher is better');
</script>
</body>
</html>
"""


EMPTY_TEMPLATE = """\
<!doctype html>
<html lang="en-US">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLIR-AIR — LLM Performance History</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         max-width: 1000px; margin: 0 auto; padding: 24px; color: #1b1f23; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .sub {{ color: #586069; }}
  a {{ color: #0366d6; }}
</style>
</head>
<body>
<p><a href="index.html">&larr; Back to Programming Examples dashboard</a></p>
<h1>LLM Performance History (NPU2)</h1>
<p class="sub">No nightly performance history has been recorded yet. Charts will appear here once the
nightly LLM benchmark has published its first datapoints.</p>
</body>
</html>
"""


def generate_html(rows, window=DEFAULT_WINDOW):
    """Render the full self-contained HTML page from history rows.

    Always returns a valid page: with no rows it renders an empty-state page so
    the dashboard's link to perf-history.html never 404s (the link is emitted
    whenever the benchmark section renders, which can precede the first history
    write).

    Only the most recent `window` unique dates are plotted (all rows stay in
    history.ndjson); window <= 0 plots the full history.
    """
    if not rows:
        return EMPTY_TEMPLATE.format()
    all_labels = sorted({r.get("date") for r in rows if r.get("date")})
    labels = all_labels[-window:] if window and window > 0 else all_labels
    window_note = (
        f" Showing the most recent {len(labels)} of {len(all_labels)} nightlies."
        if len(labels) < len(all_labels)
        else ""
    )
    models = _model_order(rows)
    ttft = _series(rows, models, labels, "ttft_ms")
    decode = _series(rows, models, labels, "decode_tokens_per_sec")
    return HTML_TEMPLATE.format(
        fail_color=FAIL_COLOR,
        window_note=window_note,
        labels_json=json.dumps(labels),
        ttft_json=json.dumps(ttft),
        decode_json=json.dumps(decode),
    )


def generate_embed_md(rows, window=DEFAULT_WINDOW):
    """Render the charts as a Markdown page for embedding in the MkDocs site.

    Reuses the standalone HTML but drops the page-level `<body>` styling,
    back-link, and duplicate `<h1>` so the charts render inside the Material
    theme chrome instead of restyling the whole page.
    """
    if not rows:
        return (
            "# LLM Performance History\n\n"
            "No nightly performance history has been recorded yet. Charts will "
            "appear here once the nightly LLM benchmark has published its first "
            "datapoints.\n"
        )
    import re

    html = generate_html(rows, window)
    cdn_m = re.search(r'<script src="https://cdn[^>]*></script>', html)
    body_m = re.search(r"<body>(.*?)</body>", html, re.S)
    if not cdn_m or not body_m:
        # Template shape changed unexpectedly; fall back to the standalone page
        # rather than raising, so the docs build never breaks on this page.
        return "# LLM Performance History\n\n" + html
    body = body_m.group(1)
    body = re.sub(r'<p><a href="index.html">.*?</a></p>\s*', "", body, flags=re.S)
    body = re.sub(r"<h1>.*?</h1>\s*", "", body, count=1, flags=re.S)
    # Minimal, scoped CSS: only size the chart containers. The standalone page's
    # <style> is deliberately NOT reused — its `body`/`h1`/`a` global selectors
    # would restyle the surrounding Material page chrome.
    style = "<style>.chart-box { position: relative; height: 380px; margin: 32px 0; }</style>"
    return f"# LLM Performance History\n\n{cdn_m.group(0)}\n{style}\n{body}\n"


def main():
    ap = argparse.ArgumentParser(description="Generate the LLM perf history page.")
    ap.add_argument("--history", required=True, help="Path to history.ndjson")
    ap.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "perf-history.html",
        help="Output path (default: programming_examples/perf-history.html)",
    )
    ap.add_argument(
        "--embed",
        action="store_true",
        help="Emit a Markdown page for embedding in the MkDocs site instead of "
        "a standalone HTML page.",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Plot only the most recent N unique dates (<=0 for all). "
        f"Default: {DEFAULT_WINDOW}.",
    )
    args = ap.parse_args()

    rows = load_history(args.history)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Always write a page (empty-state when there are no rows) so the site's
    # link to this page is stable and never 404s.
    args.output.write_text(
        generate_embed_md(rows, args.window)
        if args.embed
        else generate_html(rows, args.window)
    )
    print(f"Generated {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
