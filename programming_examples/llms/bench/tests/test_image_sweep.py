# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An image-turn sweep (<model>.image_sweep.json) through append_history.py and
the LLM page (generate_readme.py)."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
EXAMPLES = BENCH.parents[1]


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_readme = _load(EXAMPLES / "generate_readme.py")

MODEL = "gemma4_e2b_vision"


def point(budget, ttft, tok_s, status="ok"):
    return {
        "image_budget": budget,
        "image_tokens": budget - 6,
        "prompt_len": budget + 9,
        "ttft_ms": ttft,
        "vision_ms": ttft * 0.7 if ttft else None,
        "prefill_ms": ttft * 0.3 if ttft else None,
        "decode_tokens_per_sec": tok_s,
        "status": status,
    }


def record(points, ts="2026-11-02T00:00:00Z"):
    return {
        "model": MODEL,
        "axis": "image_tokens",
        "metric": "ttft_ms",
        "timestamp_utc": ts,
        "runner": "krackan",
        "toolchain": {"mlir_air_sha": "abc"},
        "verify_status": "pass",
        "points": points,
    }


class ImageSweepTest(unittest.TestCase):
    def append(self, d, recs, run_id):
        src = Path(d, f"image_sweep_{run_id}.json")
        src.write_text(json.dumps(recs))
        subprocess.run(
            [
                sys.executable,
                BENCH / "append_history.py",
                "--image-sweep",
                src,
                "--history",
                Path(d, "image_sweep_history.ndjson"),
                "--run-id",
                str(run_id),
            ],
            check=True,
            capture_output=True,
        )
        return Path(d, "image_sweep_history.ndjson")

    def test_append_flattens_one_row_per_budget(self):
        with tempfile.TemporaryDirectory() as d:
            hist = self.append(
                d, [record([point(70, 367.0, 32.5), point(280, 975.0, 32.3)])], 1
            )
            rows = [json.loads(l) for l in hist.read_text().splitlines()]
        self.assertEqual([r["image_budget"] for r in rows], [70, 280])
        self.assertEqual(rows[0]["model"], MODEL)
        self.assertEqual(rows[0]["ttft_ms"], 367.0)
        self.assertEqual(rows[1]["decode_tokens_per_sec"], 32.3)
        self.assertEqual(rows[0]["air_sha"], "abc")

    def test_history_keeps_latest_run_budgets(self):
        with tempfile.TemporaryDirectory() as d:
            self.append(
                d, [record([point(70, 400.0, 32.0), point(1120, 5000.0, 31.0)])], 1
            )
            hist = self.append(
                d,
                [record([point(70, 367.0, 32.5)], ts="2026-11-03T00:00:00Z")],
                2,
            )
            recs = generate_readme.load_llm_image_sweep_history(hist)
        self.assertEqual(len(recs), 1)
        pts = recs[0]["points"]
        # 1120 was not swept by the latest run, so it leaves the table.
        self.assertEqual([p["image_budget"] for p in pts], [70])
        self.assertEqual(pts[0]["ttft_ms"], 367.0)
        self.assertEqual(recs[0]["timestamp_utc"][:10], "2026-11-03")

    def test_table_cells(self):
        recs = [
            record(
                [
                    point(70, 367.2, 32.51),
                    point(280, None, None, status="error"),
                    point(1120, 4862.1, 31.63),
                ]
            )
        ]
        table = generate_readme.render_llm_image_sweep(recs)
        row = next(l for l in table.splitlines() if MODEL in l)
        right = '<td style="text-align: right;">'
        self.assertIn(f"{right}367</td>{right}✗</td>{right}4862</td>", row)
        self.assertIn(f"{right}31.6–32.5</td>", row)
        self.assertIn('colspan="3">Number of image tokens</th>', table)
        self.assertIn(">70</th>", table)
        self.assertIn(">1120</th>", table)
        self.assertIn("✗ failed.", table)

    def test_measured_date_ignores_failed_points(self):
        with tempfile.TemporaryDirectory() as d:
            self.append(d, [record([point(70, 367.0, 32.5)])], 1)
            hist = self.append(
                d,
                [
                    record(
                        [
                            point(70, None, None, status="error"),
                            point(1120, None, None, status="error"),
                        ],
                        ts="2026-11-03T00:00:00Z",
                    )
                ],
                2,
            )
            recs = generate_readme.load_llm_image_sweep_history(hist)
        # 70 keeps the Nov 2 number and date; the failed run does not move it.
        pts = {p["image_budget"]: p for p in recs[0]["points"]}
        self.assertEqual(pts[70]["ttft_ms"], 367.0)
        self.assertEqual(recs[0]["timestamp_utc"][:10], "2026-11-02")

    def test_newer_verify_from_history(self):
        rec = dict(record([point(70, 367.0, 32.5)]), verify_utc="2026-11-02")
        later_fail = {(MODEL, "krackan-point"): ("2026-11-04T00:00:00Z", "fail")}
        older_fail = {(MODEL, "krackan-point"): ("2026-11-01T00:00:00Z", "fail")}
        fail_cell = generate_readme._VERIFY_CELL["fail"]
        pass_cell = generate_readme._VERIFY_CELL["pass"]
        row = lambda t: next(l for l in t.splitlines() if MODEL in l)
        newer = generate_readme.render_llm_image_sweep([rec], verify=later_fail)
        older = generate_readme.render_llm_image_sweep([rec], verify=older_fail)
        self.assertIn(fail_cell, row(newer))
        self.assertIn(pass_cell, row(older))

    def test_no_points_no_table(self):
        self.assertEqual(generate_readme.render_llm_image_sweep([]), "")

    def test_image_model_leaves_single_point_table(self):
        null = {
            "model": MODEL,
            "timestamp_utc": "2026-11-03T00:00:00Z",
            "run_id": 7,
            "verify_status": "pass",
            "ttft_ms": None,
            "decode_tokens_per_sec": None,
            "context_len": None,
        }
        other = dict(null, model="qwen3_0_6b", ttft_ms=50.0)
        with tempfile.TemporaryDirectory() as d:
            hist = Path(d, "history.ndjson")
            hist.write_text("\n".join(json.dumps(r) for r in (null, other)) + "\n")
            page = generate_readme.render_llm_benchmark(
                None,
                history_path=hist,
                image_sweep_recs=[record([point(70, 367.0, 32.5)])],
            )
        scalar = page.split("### Image + text")[0]
        self.assertNotIn(f"llms/{MODEL}/", scalar)
        self.assertIn("llms/qwen3_0_6b/", scalar)
        self.assertIn("### Image + text", page)


if __name__ == "__main__":
    unittest.main()
