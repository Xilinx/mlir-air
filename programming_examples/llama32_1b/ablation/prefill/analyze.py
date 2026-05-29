"""Read prefill results JSON and emit a markdown report.

Sections:
- Validation badge (per cell × kernel-group)
- Single-layer per-call medians (per cell × kernel-group)
- 16-layer total wall (per cell, with comparison to profile.md's 1.27s)
- Marginal deltas (A→B, B→C, C→D, A→D — per kernel-group AND aggregated)
- Per-launch breakdown extracted from Cell C's single-layer timing data
"""

import argparse
import json
import os
import time

PROFILE_MD_HEADLINE_S = 1.27  # production prefill from profile.md


def report(results):
    cells = results["cells"]
    out = []
    out.append("# Prefill Ablation — Report\n")
    out.append(
        f"Trials: {results['trials']}, config: seq={results['config']['seq_len']}, "
        f"emb={results['config']['emb_dim']}, hidden={results['config']['hidden_dim']}\n"
    )

    # Validation table
    out.append("## Validation\n")
    out.append("| Cell | rms_gemms_rope | o_ffn |")
    out.append("|------|----------------|-------|")
    for c in ("A", "B", "C", "D"):
        rg = cells.get(c, {}).get("rms_gemms_rope", {}).get("validation", "—")
        of = cells.get(c, {}).get("o_ffn", {}).get("validation", "—")
        out.append(f"| {c} | {rg} | {of} |")
    out.append("")

    # Single-layer per-call timing table
    out.append("## Single-layer per-call medians (ms)\n")
    out.append("| Cell | rms_gemms_rope | o_ffn |")
    out.append("|------|----------------|-------|")
    for c in ("A", "B", "C", "D"):
        rg_s = (
            cells.get(c, {})
            .get("rms_gemms_rope", {})
            .get("single_layer", {})
            .get("median_s")
        )
        of_s = cells.get(c, {}).get("o_ffn", {}).get("single_layer", {}).get("median_s")
        rg_str = f"{rg_s*1000:.2f}" if rg_s is not None else "—"
        of_str = f"{of_s*1000:.2f}" if of_s is not None else "—"
        out.append(f"| {c} | {rg_str} | {of_str} |")
    out.append("")

    # 16-layer headline table
    out.append("## 16-layer total wall (s) — comparable to profile.md's 1.27 s\n")
    out.append("| Cell | Median (s) | Min (s) | Max (s) | vs profile.md |")
    out.append("|------|------------|---------|---------|---------------|")
    for c in ("A", "B", "C", "D"):
        e = cells.get(c, {}).get("16_layer", {})
        if not e:
            out.append(f"| {c} | — | — | — | — |")
            continue
        md = e["median_s"]
        mn = e["min_s"]
        mx = e["max_s"]
        ratio = md / PROFILE_MD_HEADLINE_S
        out.append(f"| {c} | {md:.3f} | {mn:.3f} | {mx:.3f} | {ratio:.2f}× |")
    out.append("")

    # Marginal deltas (16-layer total)
    out.append("## Marginal deltas (16-layer total)\n")

    def m(c):
        return cells.get(c, {}).get("16_layer", {}).get("median_s")

    pairs = [
        ("A→B (= #2 per-layer weight BOs)", "A", "B"),
        ("B→C (= #3 shared intermediate BOs)", "B", "C"),
        ("C→D (= #1 multi-launch merging, isolated)", "C", "D"),
        ("A→D (= total dispatch-related speedup)", "A", "D"),
    ]
    out.append("| Comparison | Δ s | Speedup |")
    out.append("|------------|-----|---------|")
    for label, a, b in pairs:
        ma, mb = m(a), m(b)
        if ma is None or mb is None:
            out.append(f"| {label} | — | — |")
            continue
        out.append(f"| {label} | {ma - mb:+.3f} | {ma/mb:.2f}× |")
    out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.results_json) as f:
        results = json.load(f)
    text = report(results)
    out = args.out or f"report_prefill_{int(time.time())}.md"
    with open(out, "w") as f:
        f.write(text)
    print(f"Wrote {out}\n")
    print(text)


if __name__ == "__main__":
    main()
