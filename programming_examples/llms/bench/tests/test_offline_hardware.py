# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hardware an offline row was measured on, from append_history.py to the
page's offline footnote."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load(path):
    # By file path: programming_examples/ itself must stay off sys.path.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_readme = _load(HERE.parents[1] / "generate_readme.py")


class OfflineHardwareTest(unittest.TestCase):
    def test_append_history_stores_hardware(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "accuracy.json")
            src.write_text(
                json.dumps(
                    [
                        {
                            "model": "qwen3_8b_q4nx",
                            "runner": "halo",
                            "timestamp_utc": "2026-11-01T00:00:00Z",
                            "reference": {"id": "Qwen/Qwen3-8B", "kind": "bf16"},
                            "path": "decode",
                            "metrics": {},
                        }
                    ]
                )
            )
            hist = Path(d, "accuracy_history.ndjson")
            subprocess.run(
                [
                    sys.executable,
                    HERE / "append_history.py",
                    "--accuracy",
                    src,
                    "--history",
                    hist,
                    "--run-id",
                    "offline-halo-1",
                    "--source",
                    "offline",
                    "--hardware",
                    "Strix Halo, NPU2",
                ],
                check=True,
                capture_output=True,
            )
            (row,) = [json.loads(line) for line in hist.read_text().splitlines()]
            self.assertEqual(row["hardware"], "Strix Halo, NPU2")
            self.assertEqual(row["source"], "offline")

    def test_footnote_names_each_machine_with_its_hardware(self):
        recs = [
            {"source": "offline", "runner": "halo", "hardware": "Strix Halo, NPU2"},
            {"source": "offline", "runner": "kp-offline"},
            {"source": "nightly", "runner": "xcodownstairs7"},
        ]
        note = generate_readme._offline_legend(recs)
        self.assertIn("halo (Strix Halo, NPU2)", note)
        # Rows from before the field existed came from a Krackan Point machine.
        self.assertIn("kp-offline (Krackan Point, NPU2)", note)
        self.assertNotIn("xcodownstairs7", note)

    def test_no_offline_rows_no_footnote(self):
        self.assertEqual(generate_readme._offline_legend([{"source": "nightly"}]), "")


if __name__ == "__main__":
    unittest.main()
