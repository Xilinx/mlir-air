# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A task score record through append_history's flattening, the docs loader
and the table generate_readme.py renders."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import append_history  # noqa: E402


def _load(path):
    # By file path: programming_examples/ itself must stay off sys.path.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_readme = _load(HERE.parents[1] / "generate_readme.py")


def record(day, task, npu, p, shots=5):
    return {
        "model": "qwen3_4b_q4nx",
        "runner": "krackan-offline",
        "timestamp_utc": f"2026-11-{day:02d}T00:00:00Z",
        "reference": {"id": "Qwen/Qwen3-4B", "revision": "abc"},
        "limit": 100,
        "n_shot": {task: shots},
        "toolchain": {},
        "tasks": {
            task: {
                "n": 100,
                "npu": npu,
                "ref": 0.88,
                "npu_only_right": 0,
                "ref_only_right": 9,
                "mcnemar_p": p,
            }
        },
    }


class TasksPageTest(unittest.TestCase):
    def render(self, recs):
        with tempfile.TemporaryDirectory() as d:
            hist = Path(d, "tasks_history.ndjson")
            rows = []
            for i, r in enumerate(recs):
                rows += [
                    {**row, "source": "offline"}
                    for row in append_history._flat_tasks_rows([r], i)
                ]
            hist.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return generate_readme.render_llm_tasks(
                generate_readme.load_llm_tasks_history(hist)
            )

    def test_newest_row_per_task(self):
        page = self.render(
            [
                record(1, "gsm8k", 0.70, 0.5),
                record(2, "gsm8k", 0.79, 0.004),
                record(1, "mmlu", 0.72, 0.47, shots=0),
            ]
        )
        self.assertIn("krackan-offline", page)
        self.assertIn("| gsm8k (5-shot) | 100 | 79.0 | 88.0 | 0 / 9 | 0.004 |", page)
        self.assertNotIn("| 70.0 |", page)
        self.assertIn("| mmlu (0-shot) |", page)

    def test_small_p(self):
        page = self.render([record(1, "gsm8k", 0.41, 1.2e-4)])
        self.assertIn("| < 0.001 |", page)

    def test_no_rows_no_section(self):
        self.assertEqual(generate_readme.render_llm_tasks([]), "")


if __name__ == "__main__":
    unittest.main()
