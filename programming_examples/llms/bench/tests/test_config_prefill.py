# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Models profiled once per native configuration (the affine-Q4 deployments):
their per-configuration TTFTs make the module's prefill curve, and their
single-point rows leave the table once both curves exist."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[3]


def _load(path):
    # By file path: programming_examples/ itself must stay off sys.path.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_readme = _load(EXAMPLES / "generate_readme.py")


def _row(model, ttft, decode, verify="pass"):
    return {
        "model": model,
        "timestamp_utc": "2026-11-02T04:00:00Z",
        "verify_status": verify,
        "metrics": {"ttft_ms": ttft, "decode_tokens_per_sec": decode},
    }


def _sweep(model):
    return {
        "model": model,
        "points": [{"context_len": 2048, "decode_tokens_per_sec": 30.0}],
    }


class ConfigPrefillTest(unittest.TestCase):
    def test_curve_from_configurations(self):
        recs = [
            _row("granite33_2b_q4nx_p1024_c2048", 1166.3162, 32.36),
            _row("granite33_2b_q4nx_p512_c1024", 610.27, 35.32, verify="fail"),
            _row("llama32_1b_q4nx", 120.0, 60.0),
        ]
        (curve,) = generate_readme.config_prefill_curves(recs)
        self.assertEqual(curve["model"], "granite33_2b_q4nx")
        self.assertEqual(
            [(p["prefill_len"], p["ttft_ms"]) for p in curve["points"]],
            [(512, 610.27), (1024, 1166.3162)],
        )
        # one failing configuration fails the curve
        self.assertEqual(curve["verify_status"], "fail")

    def test_skipped_configuration_is_not_a_pass(self):
        recs = [
            _row("granite33_2b_q4nx_p512_c1024", 610.27, 35.32),
            _row("granite33_2b_q4nx_p1024_c2048", 1166.3, 32.36, verify="skip"),
        ]
        (curve,) = generate_readme.config_prefill_curves(recs)
        self.assertEqual(curve["verify_status"], "skip")

    def test_measured_sweep_wins(self):
        recs = [_row("granite33_2b_q4nx_p512_c1024", 610.27, 35.32)]
        have = [{"model": "granite33_2b_q4nx", "points": []}]
        self.assertEqual(generate_readme.config_prefill_curves(recs, have), [])

    def test_rows_leave_once_both_curves_exist(self):
        recs = [
            _row("granite33_2b_q4nx_p512_c1024", 610.27, 35.32),
            _row("phi2_2_7b_q4nx_p512_c1024", 1857.30390297, 10.856264),
        ]
        scalar = _render(recs, [_sweep("granite33_2b_q4nx")])
        self.assertNotIn("Granite-3.3-2B (P512/C1024)", scalar)
        # Phi-2 has no decode curve: its row stays, TTFT on the curve, decode rounded
        self.assertIn("Phi-2 (P512/C1024)", scalar)
        self.assertIn("| ↓ | 10.86 |", scalar)


def _render(recs, sweeps):
    """The single-point table of the page rendered from perf.json-shaped recs."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(recs, f)
    page = generate_readme.render_llm_benchmark(f.name, sweep_recs=sweeps)
    Path(f.name).unlink()
    start = page.index("Single-point rows")
    return page[start : page.index("###", start)]


if __name__ == "__main__":
    unittest.main()
