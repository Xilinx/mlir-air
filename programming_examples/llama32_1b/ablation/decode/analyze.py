"""Generate a markdown report from a Plan 2 results.json.

Usage:
  python3 analyze.py results.json > report.md
"""

import json
import sys


def fmt_ms(s):
    return f"{s * 1000:.2f} ms"


def main():
    if len(sys.argv) < 2:
        print("usage: python3 analyze.py results.json", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        r = json.load(f)

    print("# Plan 2 (full decode) ablation report")
    print()
    print(
        f"- current_pos: **{r['current_pos']}** (after a {r['prompt_len']}-token prefill)"
    )
    print(
        f"- trials per cell: **{r['trials']}** (drop trial 1 as warmup, median of remaining)"
    )
    print(f"- per timed trial: ONE decode token through 16 layers + LM head + argmax")
    print()

    cells = r["cells"]
    cell_order = ["A", "B", "C", "D"]
    cell_labels = {
        "A": "Naive no-merge",
        "B": "+ per-layer weight BOs (#2)",
        "C": "+ shared intermediate BOs (#3)",
        "D": "+ multi-launch merging (#1) [production]",
    }

    print("## Per-token total wall time")
    print()
    print("| Cell | Median | Range | Δ vs prev | Speedup vs prev |")
    print("|------|--------|-------|-----------|-----------------|")

    prev_median = None
    baseline = None
    for c in cell_order:
        if c not in cells:
            continue
        d = cells[c]
        if "median_total_s" not in d:
            print(f"| {c} {cell_labels[c]} | — | VALIDATION FAIL | — | — |")
            continue
        med = d["median_total_s"]
        rng = f"[{fmt_ms(d['min_total_s'])}, {fmt_ms(d['max_total_s'])}]"
        if prev_median is None:
            delta = "—"
            speed = "(baseline)"
            baseline = med
        else:
            delta = f"{(prev_median - med) * 1000:+.2f} ms"
            speed = f"{prev_median / med:.2f}×" if med > 0 else "—"
        print(
            f"| **{c}** {cell_labels[c]} | {fmt_ms(med)} | {rng} | {delta} | {speed} |"
        )
        prev_median = med

    if baseline is not None and "D" in cells and "median_total_s" in cells["D"]:
        a_to_d = baseline / cells["D"]["median_total_s"]
        print()
        print(f"**A → D total speedup: {a_to_d:.2f}×**")
    print()

    print("## Per-kernel-group medians (single call)")
    print()
    print("| Cell | rms_gemv_rope median | o_gemv_ffn median |")
    print("|------|----------------------|-------------------|")
    for c in cell_order:
        if c not in cells or "rms_gemv_rope_per_call_median_s" not in cells[c]:
            continue
        d = cells[c]
        print(
            f"| {c} | {fmt_ms(d['rms_gemv_rope_per_call_median_s'])} "
            f"| {fmt_ms(d['o_gemv_ffn_per_call_median_s'])} |"
        )
    print()

    print("## Component breakdown (Cell D, fixed costs)")
    print()
    if "D" in cells and "cpu_attn_total_median_s" in cells["D"]:
        d = cells["D"]
        print(
            f"- CPU attention floor (sum across 16 layers): **{fmt_ms(d['cpu_attn_total_median_s'])}**"
        )
        print(
            f"- LM head (production-merged, invariant): **{fmt_ms(d['lm_head_median_s'])}**"
        )
        print(f"- Total per-token wall: **{fmt_ms(d['median_total_s'])}**")
    print()

    print("## Validation")
    print()
    print("| Cell | Validation |")
    print("|------|------------|")
    for c in cell_order:
        if c not in cells:
            print(f"| {c} | (not run) |")
            continue
        v = cells[c].get("validation", "?")
        print(f"| {c} | {v} |")


if __name__ == "__main__":
    main()
