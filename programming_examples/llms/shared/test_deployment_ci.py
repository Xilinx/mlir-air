# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host tests for CI failure propagation, timing protocol and lit discovery."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from . import deployment_ci as ci
from .deployment import LLMS, MODELS, configuration, save_json, spec

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(LLMS.parent)
]
from air_examples.llms.bench.model_ids import verification_status


def timing(model="phi2_2_7b_q4nx", p=512):
    row = dict(
        tokens=[0] * 128, ttft_seconds=2.0, decode_seconds=4.0, request_seconds=6.0
    )
    return dict(
        model=model,
        configuration=configuration(model, p),
        outputs=128,
        warmups=[copy.deepcopy(row)],
        requests=[copy.deepcopy(row) for _ in range(3)],
        ttft_boundary="first token selected; physical KV handoff charged after t1",
    )


class DeploymentCITests(unittest.TestCase):
    def test_raw_sample_metrics(self):
        t = timing()
        t["requests"][0].update(
            ttft_seconds=1.0, decode_seconds=9.0, request_seconds=10.0
        )
        r = ci.performance_record(t["model"], t["configuration"], t)
        self.assertEqual(r["metrics"]["ttft_ms"], 2000.0)
        self.assertEqual(r["metrics"]["prefill_tokens_per_sec"], 256.0)
        self.assertEqual(r["metrics"]["decode_tokens_per_sec"], 381 / 17)
        self.assertEqual(r["model"], "phi2_2_7b_q4nx_p512_c1024")

    def test_incomplete_or_wrong_workload_rejected(self):
        for kind in (
            "warmup",
            "measurements",
            "outputs",
            "nan",
            "boundary",
            "identity",
        ):
            t = timing()
            if kind == "warmup":
                t["warmups"] = []
            elif kind == "measurements":
                t["requests"] = t["requests"][:2]
            elif kind == "outputs":
                t["requests"][0]["tokens"].pop()
            elif kind == "nan":
                t["requests"][0]["decode_seconds"] = float("nan")
            elif kind == "boundary":
                t["requests"][0]["request_seconds"] += 1
            else:
                t["configuration"]["capacity"] = 2048
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                ci.performance_record(t["model"], configuration(t["model"], 512), t)

    def pipeline(self, tmp, stage, failure=None):
        args = argparse.Namespace(
            model="phi2_2_7b_q4nx",
            prefill=512,
            work_dir=Path(tmp),
            stage=stage,
            checkpoint=None,
            q4_dir=None,
            reference_python=sys.executable,
        )
        calls = []

        def runner(argv, log):
            command = argv[4]
            calls.append(command)
            if command == failure:
                raise subprocess.CalledProcessError(1, argv)
            if command == "weights":
                save_json(
                    Path(argv[argv.index("--build-dir") + 1]) / "weights/manifest.json",
                    {},
                )
            elif command == "verify":
                out = Path(argv[argv.index("--out") + 1])
                save_json(
                    out / "gate.json",
                    dict(status="FAIL" if failure == "gate" else "OK"),
                )
                save_json(
                    out / "reset.json", dict(bitwise_logits_equal=failure != "reset")
                )
                self.assertIn("--check-reset", argv)
            elif command == "profile":
                out = Path(argv[argv.index("--out") + 1])
                t = timing()
                if failure == "timing":
                    t["requests"] = []
                save_json(out / "timing.json", t)

        with patch.object(
            ci,
            "source_identity",
            return_value=dict(commit="fixture", source_sha256="1" * 64),
        ):
            if failure:
                with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    ci.execute(args, runner)
            else:
                ci.execute(args, runner)
        return calls

    def test_profile_is_self_contained_and_gated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self.pipeline(tmp, "profile"),
                ["weights", "compile", "reference", "verify", "profile"],
            )
            result = json.loads(next(Path(tmp).glob("*.perf.json")).read_text())
            self.assertEqual(result["gate"]["status"], "OK")

    def test_failures_remove_stale_perf_and_block_later_stages(self):
        for failure in (
            "weights",
            "compile",
            "reference",
            "verify",
            "gate",
            "reset",
            "profile",
            "timing",
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                stale = Path(tmp) / "phi2_2_7b_q4nx_p512_c1024.perf.json"
                stale.write_text('{"old": true}')
                calls = self.pipeline(tmp, "profile", failure)
                self.assertFalse(stale.exists())
                if failure not in ("profile", "timing"):
                    self.assertNotIn("profile", calls)
                status = json.loads(
                    next(Path(tmp).glob("evidence/*/status.json")).read_text()
                )
                self.assertEqual(status["status"], "FAIL")

    def test_verify_never_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotIn("profile", self.pipeline(tmp, "verify"))
            self.assertFalse(list(Path(tmp).glob("*.perf.json")))

    def test_all_native_configs_have_matching_lit_and_preseed(self):
        # Seeded on the nightly runner, or on the offline machine for 7-8B models.
        seeded = [
            line
            for name in ("hf_models.txt", "hf_models_offline.txt")
            for line in (LLMS / name).read_text().splitlines()
        ]
        cases = []
        names = set()
        for model in MODELS:
            s = spec(model)
            self.assertIn(s["checkpoint"] + "@" + s["revision"], seeded)
            for p, _ in s["configurations"]:
                name = ci.perf_id(model, configuration(model, p))
                names.add(name)
                for stage in ("verify", "profile"):
                    text = (LLMS / model / f"run_npu2_{stage}_p{p}.lit").read_text()
                    self.assertIn("PERF_MODEL: " + name, text)
                    self.assertIn(f"ci-{stage} P={p}", text)
                cases.append(
                    f'<testcase classname="llms/{model}" name="run_npu2_verify_p{p}.lit"/>'
                )
        self.assertEqual(len(names), 19)
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "verify.xml"
            xml.write_text("<testsuite>" + "".join(cases) + "</testsuite>")
            self.assertEqual(
                verification_status(xml, LLMS), {name: "pass" for name in names}
            )


if __name__ == "__main__":
    unittest.main()
