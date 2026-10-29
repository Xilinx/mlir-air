# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Synthetic host fixtures for strict registry selection; no device measurements."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_harness import ROOT, load

evidence = load("evidence", ROOT / "programming_examples/kernel_registry/evidence.py")


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = dict(
            operation="gemm",
            shape=[16, 32, 64],
            input_dtype="bf16",
            output_dtype="bf16",
            compute="bfp16",
            precision="high",
            layout="row-major",
            target="npu2",
            toolchain="synthetic-test-only",
            timing_boundary="one warm kernel call",
            rtol=0.016,
            atol=0.0015,
        )

    def write(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return dict(path=name, sha256=hashlib.sha256(data).hexdigest())

    def record(self, name, status="PASS", latency=2.0):
        files = {
            k: self.write(name + "-" + k, (name + k).encode())
            for k in ("source", "artifact", "reference")
        }
        bindings = {k: v["sha256"] for k, v in files.items()}
        validation = dict(
            status=status,
            compared_elements=1024,
            mismatch_count=0 if status == "PASS" else 1,
            finite=True,
            rtol=0.016,
            atol=0.0015,
            bindings=bindings,
        )
        timing = dict(
            samples_seconds=[latency, latency * 1.1],
            boundary=self.key["timing_boundary"],
            bindings=bindings,
        )
        files["validation"] = self.write(
            name + "-validation.json", json.dumps(validation).encode()
        )
        files["timing"] = self.write(name + "-timing.json", json.dumps(timing).encode())
        return dict(
            schema_version=1,
            id=name,
            kind="primitive",
            key=copy.deepcopy(self.key),
            abi="synthetic A,B,C row-major fixture",
            output_shape=[16, 64],
            validation=validation,
            timing=timing,
            files=files,
        )

    def update_report(self, row, name):
        row["files"][name] = self.write(
            row["id"] + "-" + name + ".json", json.dumps(row[name]).encode()
        )

    def test_selects_slower_pass_over_fast_fail_and_unknown(self):
        rows = [
            self.record("fast-fail", "FAIL", 0.1),
            self.record("unknown", "UNKNOWN", 0.01),
            self.record("pass", "PASS", 2.0),
            self.record("slow-pass", "PASS", 3.0),
        ]
        self.assertEqual(
            evidence.select_verified(rows, self.key, self.root)["id"], "pass"
        )
        self.assertIsNone(evidence.select_verified(rows[:2], self.key, self.root))

    def test_exact_contract_prevents_incompatible_reuse(self):
        row = self.record("pass")
        for change in (
            dict(compute="native-bf16"),
            dict(shape=[32, 32, 64]),
            dict(toolchain="different"),
            dict(timing_boundary="complete request"),
            dict(atol=0.1),
        ):
            query = dict(self.key, **change)
            self.assertIsNone(evidence.select_verified([row], query, self.root))

    def test_changed_artifact_and_cross_candidate_reports_rejected(self):
        row = self.record("pass")
        (self.root / row["files"]["artifact"]["path"]).write_bytes(b"different build")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evidence.validate_record(row, self.root)
        row["files"]["artifact"] = self.write("new-artifact", b"different build")
        with self.assertRaisesRegex(ValueError, "bind this"):
            evidence.validate_record(row, self.root)

    def test_invalid_gate_or_timing_cannot_enter_ranking(self):
        for change in (
            dict(compared_elements=10),
            dict(finite=False),
            dict(mismatch_count=1),
            dict(atol=1.0),
        ):
            row = self.record("pass")
            row["validation"].update(change)
            self.update_report(row, "validation")
            with self.assertRaises(ValueError):
                evidence.validate_record(row, self.root)
        row = self.record("pass")
        row["timing"]["samples_seconds"] = [float("nan")]
        self.update_report(row, "timing")
        with self.assertRaises(ValueError):
            evidence.validate_record(row, self.root)

    def test_missing_timing_and_megakernel_graph(self):
        row = self.record("untimed")
        row["timing"] = None
        row["files"].pop("timing")
        self.assertIsNone(evidence.select_verified([row], self.key, self.root))
        row["kind"] = "megakernel"
        with self.assertRaisesRegex(ValueError, "fused graph"):
            evidence.validate_record(row, self.root)
        row["graph"] = [{"op": "gemm", "inputs": ["A", "B"], "output": "C"}]
        evidence.validate_record(row, self.root)

    def test_path_escape_and_conflicting_ids_rejected(self):
        row = self.record("pass")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evidence.select_verified([row, row], self.key, self.root)
        row["files"]["source"]["path"] = "../outside"
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            evidence.validate_record(row, self.root)


if __name__ == "__main__":
    unittest.main()
