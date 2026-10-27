# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host tests for public task entry, clients and production capture integration."""

import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.parse import unquote

import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task = load("task", PACKAGE / "tools/task.py")
sync = load("sync", PACKAGE / "tools/sync_package.py")
profile = load("profile", PACKAGE / "tools/air_source_profile.py")
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(ROOT / "programming_examples")
]
from air_examples.llms.shared.deployment import configuration, spec, require_bfp16
from air_examples.llms.shared.deployment_cli import compare_capture


class HarnessTests(unittest.TestCase):
    def test_all_shipped_tasks_offline_and_native_cohort(self):
        count = 0
        with patch.object(
            task, "read_hub", side_effect=AssertionError("unexpected network")
        ):
            for path in sorted(
                (ROOT / "programming_examples/llms").glob("*/model.json")
            ):
                model = json.loads(path.read_text(encoding="utf-8"))
                if "configurations" not in model or "bf16_matrices" not in model:
                    continue
                result = task.prepare(model["checkpoint"])
                required = [
                    (r["prefill"], r["capacity"])
                    for r in result["configurations"]
                    if r["status"] == "REQUIRED"
                ]
                self.assertEqual(required, [tuple(x) for x in model["configurations"]])
                self.assertEqual(result["revision"], model["revision"])
                self.assertEqual(result["arithmetic"]["prefill_gemm"], "BFP16")
                self.assertEqual(result["outcome"]["model_gate"], "NOT_RUN")
                count += len(required)
        self.assertGreaterEqual(count, 19)

    def test_unknown_model_offline_requires_architecture_work(self):
        with patch.object(
            task, "read_hub", side_effect=AssertionError("unexpected network")
        ):
            result = task.prepare(
                "test/new-model",
                revision="a" * 40,
                config={
                    "max_position_embeddings": 4096,
                    "architectures": ["CustomModel"],
                },
            )
            self.assertIsNone(result["bindings"]["model_module"])
            self.assertEqual(result["bindings"]["architecture_review"], "REQUIRED")
            self.assertEqual(
                [r["status"] for r in result["configurations"]],
                ["REQUIRED"] * 3 + ["UNSUPPORTED_NATIVE_CONTEXT"],
            )
            with self.assertRaisesRegex(ValueError, "native context"):
                task.prepare("test/new-model", revision="a" * 40, config={})
            with self.assertRaisesRegex(ValueError, "40-character"):
                task.prepare("test/new-model", revision="main", config={})

    def test_task_cli_does_not_overwrite_an_existing_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "task.json"
            command = [
                sys.executable,
                str(PACKAGE / "tools/task.py"),
                "microsoft/phi-2",
                "--out",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True)
            original = output.read_bytes()
            self.assertNotEqual(
                subprocess.run(command, capture_output=True).returncode, 0
            )
            self.assertEqual(output.read_bytes(), original)

    def test_clients_resolve_after_relocation(self):
        with tempfile.TemporaryDirectory(prefix="air relocated ") as tmp:
            root = Path(tmp)
            package = root / "agentic"
            shutil.copytree(
                PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__")
            )
            generated = sync.generated(package)
            for path, text in generated.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8", newline="\n")
            for client in (".agents", ".claude"):
                entries = list((root / client / "skills").glob("*/SKILL.md"))
                self.assertEqual(len(entries), 17)
                for entry in entries:
                    target = re.search(
                        r"\]\(([^)]+)\)", entry.read_text(encoding="utf-8")
                    ).group(1)
                    self.assertEqual(
                        (entry.parent / target).resolve(),
                        package / "skills" / entry.parent.name / "SKILL.md",
                    )

    def test_package_metadata_links_and_generated_files(self):
        skills = list(PACKAGE.glob("skills/*/SKILL.md"))
        self.assertEqual(len(skills), 17)
        for skill in skills:
            metadata = yaml.safe_load(
                skill.read_text(encoding="utf-8").split("---", 2)[1]
            )
            self.assertEqual(metadata["name"], skill.parent.name)
            self.assertIsInstance(metadata["description"], str)
        documents = [
            *PACKAGE.rglob("*.md"),
            ROOT / "AGENTS.md",
            ROOT / "docs/ai_skills.md",
        ]
        for md in documents:
            for target in re.findall(
                r"\[[^\]]*\]\(([^\s)]+)\)", md.read_text(encoding="utf-8")
            ):
                if "://" not in target and not target.startswith("#"):
                    self.assertTrue(
                        (md.parent / unquote(target.split("#")[0])).exists(),
                        f"{md}: {target}",
                    )
        for path, expected in sync.generated().items():
            self.assertEqual(path.read_text(encoding="utf-8"), expected, str(path))

    def test_source_inspection_and_bfp16_contract(self):
        self.assertEqual(profile.inspect(ROOT)["status"], "INTERFACES_PRESENT")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                profile.inspect(Path(tmp))["status"], "INTEGRATION_REVIEW_REQUIRED"
            )
        require_bfp16({"gemm_bfp16": True})
        for bad in ({}, {"gemm_bfp16": False}, {"gemm_bfp16": "true"}):
            with self.assertRaises(ValueError):
                require_bfp16(bad)

    def test_production_capture_wrapper_rejects_invalid_evidence(self):
        model = "phi2_2_7b_q4nx"
        selected = configuration(model, 512)
        z = np.zeros((32, spec(model)["vocab_size"]), dtype=np.float32)
        z[:, 7] = 1
        valid = dict(
            logits=z,
            chosen=np.full(32, 7),
            input_ids=selected["token_ids"],
            model=np.asarray(model),
            revision=np.asarray(spec(model)["revision"]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            reference, candidate = (
                Path(tmp) / "reference.npz",
                Path(tmp) / "candidate.npz",
            )
            np.savez(reference, **valid)
            np.savez(candidate, **valid)
            self.assertEqual(
                compare_capture(candidate, reference, selected, model)["status"], "OK"
            )
            corruptions = [
                dict(logits=z[:0]),
                dict(logits=z[:31]),
                dict(logits=np.full_like(z, np.nan)),
                dict(chosen=np.zeros(32, dtype=int)),
                dict(revision=np.asarray("wrong")),
                dict(input_ids=np.zeros(512, dtype=int)),
            ]
            for change in corruptions:
                value = copy.copy(valid)
                value.update(change)
                np.savez(candidate, **value)
                with self.assertRaises(ValueError):
                    compare_capture(candidate, reference, selected, model)


if __name__ == "__main__":
    unittest.main()
