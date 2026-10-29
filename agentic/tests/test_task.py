# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Task intent, native-context and shared-adapter compatibility tests."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_harness import PACKAGE, ROOT, task


class TaskRequirementsTests(unittest.TestCase):
    def test_spec_alone_does_not_establish_shared_cli_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = Path("programming_examples/llms/phi2_2_7b_q4nx")
            (root / model).mkdir(parents=True)
            for name in ("model.json", "inputs.json"):
                shutil.copyfile(ROOT / model / name, root / model / name)
            shared = root / "programming_examples/llms/shared"
            shared.mkdir()
            (shared / "deployment.py").write_text("MODELS = ()\n")
            result = task.prepare("microsoft/phi-2", root=root)
            self.assertEqual(result["execution"]["status"], "ADAPTATION_REQUIRED")
            self.assertTrue(
                any("register/review" in r for r in result["execution"]["reasons"])
            )

    def test_supported_subset_preserves_inputs_and_uses_shared_adapter(self):
        requested = {"configurations": [{"prefill": 1024, "capacity": 2048}]}
        result = task.prepare("microsoft/phi-2", requirements=requested)
        self.assertEqual(result["mode"], "reproduce")
        self.assertEqual(len(result["configurations"]), 1)
        self.assertEqual(result["configurations"][0]["prefill"], 1024)
        self.assertEqual(result["execution"]["status"], "SHARED_ADAPTER_AVAILABLE")
        self.assertEqual(result["explicit_requirements"], requested)
        self.assertEqual(len(result["bindings"]["input_ids_sha256"]), 64)

    def test_precision_and_acceptance_are_not_silently_replaced(self):
        requested = {
            "mode": "optimize",
            "arithmetic": {"logical_weights": "BF16", "prefill_gemm": "BF16"},
            "gate": {"comparison": "exact_tokens"},
            "budget": "90 minutes",
        }
        original = copy.deepcopy(requested)
        result = task.prepare("microsoft/phi-2", requirements=requested)
        self.assertEqual(requested, original)
        self.assertEqual(result["arithmetic"]["logical_weights"], "BF16")
        self.assertIsNone(result["arithmetic"]["macro"])
        self.assertEqual(result["gate"]["comparison"], "exact_tokens")
        self.assertEqual(result["gate"]["outputs"], 32)
        self.assertEqual(result["execution"]["status"], "ADAPTATION_REQUIRED")
        self.assertEqual(result["outcome"]["model_gate"], "NOT_RUN")
        self.assertEqual(task.DEFAULTS["arithmetic"]["prefill_gemm"], "BFP16")

    def test_first_token_timing_requires_a_different_profiler(self):
        result = task.prepare(
            "microsoft/phi-2",
            requirements={"timing": {"boundary": "first_token", "outputs": 1}},
        )
        self.assertEqual(result["gate"]["outputs"], 32)
        self.assertEqual(result["timing"]["outputs"], 1)
        self.assertEqual(result["execution"]["status"], "ADAPTATION_REQUIRED")
        with self.assertRaisesRegex(ValueError, "outputs=1"):
            task.prepare(
                "microsoft/phi-2", requirements={"timing": {"boundary": "first_token"}}
            )

    def test_new_shape_and_unknown_context_are_separate(self):
        result = task.prepare(
            "microsoft/phi-2",
            requirements={"configurations": [{"prefill": 256, "capacity": 1024}]},
        )
        self.assertEqual(result["configurations"][0]["status"], "REQUIRED")
        self.assertEqual(result["execution"]["status"], "ADAPTATION_REQUIRED")
        result = task.prepare(
            "microsoft/phi-2",
            requirements={"configurations": [{"prefill": 2048, "capacity": 4096}]},
        )
        self.assertEqual(
            result["configurations"][0]["status"], "CONTEXT_REVIEW_REQUIRED"
        )
        self.assertEqual(result["execution"]["status"], "ADAPTATION_REQUIRED")
        with self.assertRaisesRegex(ValueError, "no supported configuration"):
            task.prepare(
                "microsoft/phi-2",
                native_context=2048,
                requirements={"configurations": [{"prefill": 2048, "capacity": 4096}]},
            )

    def test_small_native_model_can_choose_a_smaller_workload(self):
        result = task.prepare(
            "test/small",
            revision="a" * 40,
            config={"max_position_embeddings": 512},
            requirements={"configurations": [{"prefill": 128, "capacity": 512}]},
        )
        self.assertEqual(result["mode"], "deploy")
        self.assertEqual(result["configurations"][0]["status"], "REQUIRED")
        self.assertEqual(result["execution"]["status"], "ADAPTATION_REQUIRED")

    def test_new_hub_config_is_fetched_at_the_resolved_revision(self):
        with patch.object(
            task,
            "read_hub",
            side_effect=[{"sha": "b" * 40}, {"max_position_embeddings": 4096}],
        ) as hub:
            result = task.prepare("test/new-model")
        self.assertEqual(
            hub.call_args_list[1].args[0],
            "https://huggingface.co/test/new-model/resolve/"
            + "b" * 40
            + "/config.json",
        )
        self.assertEqual(result["revision"], "b" * 40)
        self.assertIsNone(result["bindings"]["model_module"])
        self.assertEqual(
            result["outcome"], {"model_gate": "NOT_RUN", "performance": "NOT_RUN"}
        )

    def test_invalid_requirements_and_unfittable_outputs_are_rejected(self):
        invalid = [
            {"precision": "BF16"},
            {"arithmetic": {"prefil_gemm": "BF16"}},
            {"batch_size": True},
            {"batch_size": 0},
            {"mode": "anything"},
            {"gate": {"full_finite_logits": False}},
            {"timing": {"warmup_seconds_min": float("nan")}},
            {"gate": {"outputs": 10000}},
            {"configurations": []},
            {"configurations": [{"prefill": 512, "capacity": 520}]},
            {"configurations": [{"prefill": 512, "capacity": 1024}] * 2},
        ]
        for requested in invalid:
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                task.prepare("microsoft/phi-2", requirements=requested)

    def test_cli_honors_requirements_from_another_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="task with spaces ") as tmp:
            root = Path(tmp)
            requirements = root / "requirements.json"
            requirements.write_text(
                json.dumps({"configurations": [{"prefill": 512, "capacity": 1024}]})
            )
            output = root / "output/task.json"
            run = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE / "tools/task.py"),
                    "microsoft/phi-2",
                    "--requirements",
                    str(requirements),
                    "--out",
                    str(output),
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(
                len(json.loads(output.read_text(encoding="utf-8"))["configurations"]), 1
            )
            self.assertIn("SHARED_ADAPTER_AVAILABLE", run.stdout)


if __name__ == "__main__":
    unittest.main()
