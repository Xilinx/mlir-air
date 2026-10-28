# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A task score record (verify/lm_eval_npu.py) through append_history.py."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

RECORD = {
    "model": "qwen3_4b_q4nx",
    "runner": "krackan-offline",
    "timestamp_utc": "2026-11-01T00:00:00Z",
    "reference": {"id": "Qwen/Qwen3-4B", "revision": "abc"},
    "limit": 100,
    "n_shot": {"gsm8k": 5, "mmlu": 0},
    "toolchain": {"mlir_air_sha": "deadbeef"},
    "tasks": {
        "gsm8k": {
            "n": 100,
            "npu": 0.79,
            "ref": 0.88,
            "npu_only_right": 0,
            "ref_only_right": 9,
            "mcnemar_p": 0.004,
        },
        "mmlu": {
            "n": 570,
            "npu": 0.72,
            "ref": 0.73,
            "npu_only_right": 21,
            "ref_only_right": 27,
            "mcnemar_p": 0.47,
        },
    },
}


class TasksHistoryTest(unittest.TestCase):
    def append(self, d, run_id):
        subprocess.run(
            [
                sys.executable,
                HERE / "append_history.py",
                "--tasks",
                Path(d, "tasks.json"),
                "--history",
                Path(d, "tasks_history.ndjson"),
                "--run-id",
                run_id,
                "--source",
                "offline",
            ],
            check=True,
            capture_output=True,
        )
        text = Path(d, "tasks_history.ndjson").read_text()
        return [json.loads(line) for line in text.splitlines()]

    def test_one_row_per_task_and_no_duplicate_run(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "tasks.json").write_text(json.dumps([RECORD]))
            rows = self.append(d, "offline-x-1")
            self.assertEqual(len(rows), 2)
            gsm8k = next(r for r in rows if r["task"] == "gsm8k")
            self.assertEqual(gsm8k["n_shot"], 5)
            self.assertEqual(gsm8k["ref_only_right"], 9)
            self.assertEqual(gsm8k["air_sha"], "deadbeef")
            self.assertEqual(gsm8k["source"], "offline")
            self.assertEqual(len(self.append(d, "offline-x-1")), 2)


if __name__ == "__main__":
    unittest.main()
