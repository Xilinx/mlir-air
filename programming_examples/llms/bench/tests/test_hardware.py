# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hardware a benchmark row was measured on, from append_history.py to the
page's Hardware column and the history charts."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXAMPLES = HERE.parents[1]


def _load(path):
    # By file path: programming_examples/ itself must stay off sys.path.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_readme = _load(EXAMPLES / "generate_readme.py")
generate_perf_history = _load(EXAMPLES / "generate_perf_history.py")

NIGHTLY = generate_readme.hardware_label("krackan-point-runner")
HALO = generate_readme.hardware_label("strix-halo")


def row(model, tps, hardware=None, day=1, source=None):
    r = {
        "model": model,
        "date": f"2026-11-{day:02d}",
        "timestamp_utc": f"2026-11-{day:02d}T00:00:00Z",
        "decode_tokens_per_sec": tps,
        "ttft_ms": 100.0,
        "context_len": 24,
    }
    if hardware:
        r.update(hardware=hardware, source="offline", runner="box")
    elif source:
        r["source"] = source
    return r


def write(d, name, rows):
    path = Path(d, name)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class RegistryTest(unittest.TestCase):
    def test_one_nightly_entry_and_unique_labels(self):
        hw = generate_readme.HARDWARE
        self.assertEqual(sum(1 for v in hw.values() if v.get("nightly")), 1)
        labels = [v["label"] for v in hw.values()]
        self.assertEqual(len(labels), len(set(labels)))


class AppendHistoryTest(unittest.TestCase):
    def test_hardware_is_stored_on_every_row(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "perf.json")
            src.write_text(
                json.dumps([{"model": "qwen3_8b_q4nx", "runner": "box", "metrics": {}}])
            )
            hist = Path(d, "history.ndjson")
            subprocess.run(
                [
                    sys.executable,
                    HERE / "append_history.py",
                    "--perf",
                    src,
                    "--history",
                    hist,
                    "--run-id",
                    "offline-box-1",
                    "--source",
                    "offline",
                    "--hardware",
                    "strix-halo",
                ],
                check=True,
                capture_output=True,
            )
            (r,) = [json.loads(line) for line in hist.read_text().splitlines()]
            self.assertEqual(r["hardware"], "strix-halo")


class PageTest(unittest.TestCase):
    def render(self, history, sweep=()):
        with tempfile.TemporaryDirectory() as d:
            sweep_recs = generate_readme.load_llm_sweep_history(
                write(d, "sweep.ndjson", sweep)
            )
            return generate_readme.render_llm_benchmark(
                None,
                history_path=write(d, "history.ndjson", history),
                sweep_recs=sweep_recs,
            )

    def test_one_row_per_model_and_hardware(self):
        page = self.render(
            [
                row("qwen3_8b_q4nx", 9.0),
                row("qwen3_8b_q4nx", 14.0, hardware="strix-halo", day=2),
            ]
        )
        self.assertIn(f"| {NIGHTLY} | 24 | 100.0 | 9.0 |", page)
        self.assertIn(f"| {HALO} | 24 | 100.0 | 14.0 |", page)

    def test_rows_without_the_field_are_the_nightly_runners(self):
        page = self.render([row("qwen3_4b_q4nx", 22.0)])
        self.assertIn(f"| {NIGHTLY} |", page)
        self.assertIn(f"- **{NIGHTLY}**", page)
        self.assertNotIn(HALO, page)

    def test_legend_lists_the_hardware_on_the_page_nightly_first(self):
        page = self.render([row("a", 1.0, hardware="strix-halo"), row("b", 1.0)])
        self.assertLess(page.index(f"- **{NIGHTLY}**"), page.index(f"- **{HALO}**"))

    def test_a_nightly_curve_does_not_hide_another_hardwares_row(self):
        sweep = [
            dict(row("qwen3_4b_q4nx", 20.0), context_len=1024, run_id=1),
        ]
        page = self.render(
            [
                row("qwen3_4b_q4nx", 22.0),
                row("qwen3_4b_q4nx", 30.0, hardware="strix-halo"),
            ],
            sweep,
        )
        self.assertIn(f"| {HALO} | 24 | 100.0 | 30.0 |", page)
        self.assertIn(f"| {NIGHTLY} | 24 | 100.0 | ↓ |", page)


class HistoryChartTest(unittest.TestCase):
    def test_one_line_per_model_and_hardware(self):
        rows = [
            row("qwen3_8b_q4nx", 9.0),
            row("qwen3_8b_q4nx", 14.0, hardware="strix-halo"),
        ]
        models = generate_perf_history._model_order(rows)
        lines = generate_perf_history._series(
            rows, models, ["2026-11-01"], "decode_tokens_per_sec"
        )
        self.assertEqual(
            [(l["label"], l["data"]) for l in lines],
            [("qwen3_8b_q4nx", [9.0]), (f"qwen3_8b_q4nx ({HALO})", [14.0])],
        )


if __name__ == "__main__":
    unittest.main()
