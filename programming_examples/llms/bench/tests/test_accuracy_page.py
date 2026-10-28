# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An accuracy record through append_history's flattening, the docs loader and
the table generate_readme.py renders."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # programming_examples/

import append_history  # noqa: E402
import generate_readme  # noqa: E402


def record(day, same_top, kld, ctx=2048, scored_from=1024, kind="dequant"):
    return {
        "model": "lfm2_1_2b_q4nx",
        "runner": "krackan-offline",
        "timestamp_utc": f"2026-11-{day:02d}T00:00:00Z",
        "reference": {"id": "LiquidAI/LFM2-1.2B", "kind": kind},
        "dataset": "wikitext-2-test",
        "ctx": ctx,
        "scored_from": scored_from,
        "path": "decode",
        "toolchain": {},
        "metrics": {
            "n_tokens": 4092,
            "same_top": same_top,
            "same_top_se": 0.002,
            "kld_mean": kld,
            "kld_se": 0.0001,
            "ln_ppl_ratio": 0.001,
            "ln_ppl_ratio_se": 0.002,
            "kld_percentiles": {"p99": 0.01},
        },
    }


class AccuracyPageTest(unittest.TestCase):
    def render(self, recs):
        with tempfile.TemporaryDirectory() as d:
            hist = Path(d, "accuracy_history.ndjson")
            rows = []
            for i, r in enumerate(recs):
                rows += [
                    {**row, "source": "offline"}
                    for row in append_history._flat_accuracy_rows([r], i)
                ]
            hist.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return generate_readme.render_llm_accuracy(
                generate_readme.load_llm_accuracy_history(hist)
            )

    def test_rows_and_depth_label(self):
        page = self.render(
            [
                record(1, 0.98, 0.001),
                record(1, 0.98, 0.001, ctx=8192, scored_from=8192 - 257),
            ]
        )
        self.assertIn("krackan-offline", page)
        self.assertIn("| decode |", page)
        self.assertIn("| decode @ 8k, last 256 |", page)
        self.assertNotIn("&#9888;</span>", page)

    def test_regression_is_marked(self):
        page = self.render(
            [record(1, 0.98, 0.001), record(2, 0.98, 0.001), record(3, 0.90, 0.01)]
        )
        self.assertIn("&#9888;</span>", page)

    def test_windows_stay_apart(self):
        page = self.render(
            [
                record(1, 0.98, 0.001, ctx=1024, scored_from=512),
                record(1, 0.90, 0.010, ctx=8192, scored_from=4096),
                record(1, 0.95, 0.005, ctx=8192, scored_from=8192 - 257),
            ]
        )
        for label in ("decode @ 1k", "decode @ 8k", "decode @ 8k, last 256"):
            self.assertIn(f"| {label} |", page)

    def test_no_rows_no_section(self):
        self.assertEqual(generate_readme.render_llm_accuracy([]), "")


if __name__ == "__main__":
    unittest.main()
