# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""bench/accuracy_regression.py on synthetic histories."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import accuracy_regression as ar  # noqa: E402


def row(day, top, kld, model="m", path="decode", top_se=0.004, kld_se=0.0005):
    return {
        "model": model,
        "ref_kind": "dequant",
        "path": path,
        "ctx": 2048,
        "scored_from": 1024,
        "timestamp_utc": f"2026-11-{day:02d}T00:00:00Z",
        "same_top": top,
        "same_top_se": top_se,
        "kld_mean": kld,
        "kld_se": kld_se,
    }


class RegressionTest(unittest.TestCase):
    def test_noise_is_not_flagged(self):
        rows = [row(1, 0.932, 0.031), row(2, 0.935, 0.030), row(3, 0.928, 0.033)]
        self.assertEqual(ar.regressions(rows), [])

    def test_same_top_drop_is_flagged(self):
        rows = [row(1, 0.932, 0.031), row(2, 0.935, 0.030), row(3, 0.90, 0.034)]
        self.assertEqual([f["metric"] for f in ar.regressions(rows)], ["same_top"])

    def test_kld_doubling_is_flagged(self):
        rows = [row(1, 0.98, 0.003), row(2, 0.98, 0.003), row(3, 0.978, 0.009)]
        self.assertEqual([f["metric"] for f in ar.regressions(rows)], ["kld_mean"])

    def test_small_relative_kld_rise_is_not_flagged(self):
        # Many standard errors, but well under 2x: not a regression.
        rows = [row(1, 0.93, 0.030), row(2, 0.93, 0.030), row(3, 0.93, 0.036)]
        self.assertEqual(ar.regressions(rows), [])

    def test_republished_measurement_is_one_run(self):
        good = [row(1, 0.93, 0.03), row(2, 0.93, 0.03)]
        bad = row(3, 0.80, 0.30)
        copies = [dict(bad, run_id=i) for i in range(4)]
        found = ar.regressions(good + copies)
        self.assertEqual(sorted(f["metric"] for f in found), ["kld_mean", "same_top"])

    def test_needs_history(self):
        self.assertEqual(ar.regressions([row(1, 0.93, 0.03), row(2, 0.5, 0.5)]), [])

    def test_series_are_separate(self):
        rows = [
            row(1, 0.93, 0.03),
            row(2, 0.93, 0.03),
            row(3, 0.93, 0.03),
            row(4, 0.50, 0.50, path="prefill"),
        ]
        self.assertEqual(ar.regressions(rows), [])


if __name__ == "__main__":
    unittest.main()
