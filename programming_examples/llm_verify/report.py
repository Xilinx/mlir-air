# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report accumulator + JSON / markdown dumpers.

Two layouts produced from the same Report instance:

    `make verify`     Top-k token-level inclusion gate. Records are added
                      via add_topk(pair, record); the markdown dumps a
                      Prompts table + per-pair top-k tables with agreed-
                      prefix sub-lines. has_failure() reflects the gate.

    `make diagnosis`  Per-layer ffn_out cosine + max_abs (NPU vs HF bf16).
                      Records are added via add(record); the markdown
                      dumps one informational table with one row per
                      probed layer. Diagnosis never fails the run —
                      the verify gate is the only correctness signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from comparators import ComparisonRecord, TopKCheckRecord


class Report:
    def __init__(self, config: dict):
        self.config: dict = dict(config)
        self.records: list[ComparisonRecord] = []
        self.topk_checks: list[tuple[str, TopKCheckRecord]] = []
        self.prompts: list[str] = []

    def add(self, record: ComparisonRecord) -> None:
        self.records.append(record)

    def add_topk(self, pair: str, record: TopKCheckRecord) -> None:
        self.topk_checks.append((pair, record))

    def set_prompts(self, prompts: list[str]) -> None:
        self.prompts = list(prompts)

    def summary(self) -> dict:
        topk_passed = sum(1 for _, r in self.topk_checks if r.status == "OK")
        topk_failed = sum(1 for _, r in self.topk_checks if r.status == "FAIL")
        return {
            "n_layer_records": len(self.records),
            "topk_passed": topk_passed,
            "topk_failed": topk_failed,
        }

    def has_failure(self) -> bool:
        # Only the verify-mode top-k gate signals failure. Diagnosis is
        # informational; per-layer cosine numbers are inspected by humans,
        # not gated.
        for pair, rec in self.topk_checks:
            if pair == "npu_vs_hf" and rec.status == "FAIL":
                return True
        return False

    def dump_json(self, path: str | Path) -> None:
        topk_view: Optional[list[dict]] = None
        if self.topk_checks:
            topk_view = [
                {"pair": pair, **rec.to_dict()} for pair, rec in self.topk_checks
            ]
        data = {
            "config": self.config,
            "prompts": self.prompts or None,
            "per_layer": [r.to_dict() for r in self.records],
            "topk_checks": topk_view,
            "summary": self.summary(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def dump_markdown(self, path: str | Path) -> None:
        s = self.summary()
        verdict = "FAIL" if self.has_failure() else "PASS"
        lines: list[str] = []
        lines.append("# Verify report")
        cfg_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
        lines.append(f"\nConfig: {cfg_str}")
        lines.append(f"\nResult: **{verdict}**")
        if self.topk_checks:
            lines.append(
                f"\nTop-k token gate: {s['topk_passed']} PASS / "
                f"{s['topk_failed']} FAIL "
                f"(across {len(self.topk_checks)} prompt-pair checks)"
            )
        if self.prompts:
            lines.append("\n## Prompts\n")
            lines.append("| # | Prompt |\n|--:|--------|")
            for pi, p in enumerate(self.prompts):
                cell = p.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
                lines.append(f"| {pi} | {cell} |")

        # ---- Diagnosis: per-layer ffn_out (NPU vs HF) -----------------------
        ffn_records = [r for r in self.records if r.name == "ffn_out"]
        if ffn_records:
            lines.append(
                "\n## Per-layer hidden state (ffn_out, NPU vs HF bf16)\n"
                "_Informational — diagnosis does not fail the run; "
                "`make verify` is the gate._\n"
            )
            lines.append("| Layer | cos_p5 | cos_min | cos_median | max_abs |")
            lines.append("|------:|-------:|--------:|-----------:|--------:|")
            for r in ffn_records:
                lines.append(
                    f"| {r.layer} | {r.cosine['p5']:.6f} "
                    f"| {r.cosine['min']:.6f} | {r.cosine['median']:.6f} "
                    f"| {r.errors['max_abs']:.4g} |"
                )

        # ---- Verify: top-k inclusion (per-pair tables) ----------------------
        if self.topk_checks:
            by_pair: dict[str, list] = {}
            for pair, rec in self.topk_checks:
                by_pair.setdefault(pair, []).append(rec)

            def _format_choice(text, token_id, rank):
                """Render one side's chosen token as `"text" (#rank)` or `(✗)`."""
                label = text if text is not None else f"id={token_id}"
                if rank is not None:
                    return f"{label} (#{rank})"
                return f"{label} (✗)"

            for pair, recs in by_pair.items():
                pair_passed = sum(1 for r in recs if r.status == "OK")
                pair_failed = sum(1 for r in recs if r.status == "FAIL")
                k = recs[0].k if recs else "?"
                test_side, ref_side = (s.upper() for s in pair.split("_vs_"))
                lines.append(
                    f"\n## Top-k token inclusion — {pair} "
                    f"(k={k}, {pair_passed}/{len(recs)} PASS)\n"
                )
                lines.append(
                    f"| # | Prompt | Steps | Diverge step "
                    f"| {test_side} choice (rank in {ref_side}) "
                    f"| {ref_side} choice (rank in {test_side}) | Status |"
                )
                lines.append(
                    "|--:|--------|------:|-------------:"
                    "|---------|---------|:-------|"
                )
                for r in recs:
                    if r.divergence_step is None:
                        div_cell = "—"
                        test_cell = "(all match)"
                        ref_cell = "(all match)"
                    else:
                        div_cell = str(r.divergence_step)
                        test_cell = _format_choice(
                            r.test_chosen_text_at_div,
                            r.test_chosen_at_div,
                            r.test_chosen_rank_in_ref,
                        )
                        ref_cell = _format_choice(
                            r.ref_chosen_text_at_div,
                            r.ref_chosen_at_div,
                            r.ref_chosen_rank_in_test,
                        )
                    prompt_cell = r.prompt_text.replace("|", "\\|")
                    lines.append(
                        f"| {r.prompt_idx} | {prompt_cell} | {r.n_steps} "
                        f"| {div_cell} | {test_cell} | {ref_cell} | {r.status} |"
                    )
                for r in recs:
                    if r.agreed_prefix_text and r.agreed_prefix_text != '""':
                        lines.append(
                            f"\n*Prompt {r.prompt_idx} agreed prefix "
                            f"(steps 0-{r.divergence_step - 1}):* "
                            f"{r.agreed_prefix_text}"
                        )
                for r in recs:
                    if r.fail_reason:
                        lines.append(f"\n*Prompt {r.prompt_idx} FAIL:* {r.fail_reason}")
                if pair_failed:
                    lines.append(
                        f"\n_{pair_failed}/{len(recs)} prompts failed top-{k} "
                        "inclusion at first divergence._"
                    )

        Path(path).write_text("\n".join(lines) + "\n")
