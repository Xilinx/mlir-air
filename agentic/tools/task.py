# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare a deployment task from an HF ID; never compile or access the NPU."""

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATIONS = ((512, 1024), (1024, 2048), (2048, 4096), (4096, 8192))
DEFAULTS = dict(
    mode="auto",
    target="npu2",
    batch_size=1,
    configurations=None,
    arithmetic=dict(
        logical_weights="affine Q4 group32: scale*q + min",
        prefill_gemm="BFP16",
        attention_and_head="model-specific; preserve declared numerical exceptions",
    ),
    placement=dict(
        required="all model blocks and full-vocabulary head on NPU",
        allowed_host_work="embedding, orchestration/sampling, byte-preserving transfers and the declared final single-row prefill norm",
    ),
    gate=dict(
        reference="original HF CPU BF16",
        comparison="mutual_topk",
        outputs=32,
        topk=5,
        full_finite_logits=True,
        ignore_eos=True,
    ),
    timing=dict(
        outputs=128,
        boundary="complete_request",
        warmup_seconds_min=5,
        measurements_min=3,
        measured_seconds_min=10,
    ),
    performance_goal=None,
    budget=None,
)


def merge_requirements(defaults, supplied, prefix=""):
    """Apply explicit task choices, rejecting misspelled/unknown fields."""
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'requirements'} must be an object")
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(
            f"unknown requirements in {prefix or 'task'}: {sorted(unknown)}"
        )
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        name = prefix + key
        if isinstance(defaults[key], dict):
            result[key] = merge_requirements(defaults[key], value, name + ".")
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_requirements(supplied):
    result = merge_requirements(DEFAULTS, supplied)
    if result["mode"] not in ("auto", "deploy", "reproduce", "optimize"):
        raise ValueError("mode must be auto, deploy, reproduce or optimize")
    for label, value in (
        ("batch_size", result["batch_size"]),
        ("gate.outputs", result["gate"]["outputs"]),
        ("gate.topk", result["gate"]["topk"]),
        ("timing.outputs", result["timing"]["outputs"]),
        ("timing.measurements_min", result["timing"]["measurements_min"]),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(label + " must be a positive integer")
    for key in ("warmup_seconds_min", "measured_seconds_min"):
        value = result["timing"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("timing." + key + " must be finite and nonnegative")
    strings = {"target": result["target"], "reference": result["gate"]["reference"]}
    strings.update(result["arithmetic"])
    strings.update(result["placement"])
    for key, value in strings.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(key + " must be a nonempty string")
    for key in ("performance_goal", "budget"):
        if result[key] is not None and (
            not isinstance(result[key], str) or not result[key].strip()
        ):
            raise ValueError(key + " must be null or a nonempty description")
    gate, timing = result["gate"], result["timing"]
    if gate["comparison"] not in ("mutual_topk", "exact_tokens"):
        raise ValueError("gate.comparison must be mutual_topk or exact_tokens")
    if gate["full_finite_logits"] is not True or type(gate["ignore_eos"]) is not bool:
        raise ValueError(
            "complete finite logits are required; ignore_eos must be boolean"
        )
    if timing["boundary"] not in ("complete_request", "first_token"):
        raise ValueError("timing.boundary must be complete_request or first_token")
    if timing["boundary"] == "first_token" and timing["outputs"] != 1:
        raise ValueError("first_token timing requires timing.outputs=1")
    if result["configurations"] is not None:
        rows = result["configurations"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("configurations must be a nonempty list")
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"prefill", "capacity"}:
                raise ValueError("each configuration needs prefill and capacity")
            p, c = row["prefill"], row["capacity"]
            if any(type(n) is not int or n <= 0 for n in (p, c)):
                raise ValueError("prefill and capacity must be positive integers")
            if c < p + max(gate["outputs"], timing["outputs"]) - 1:
                raise ValueError(
                    "capacity cannot hold the prompt and requested outputs"
                )
            if (p, c) in seen:
                raise ValueError("duplicate configuration")
            seen.add((p, c))
    return result


def read_hub(url):
    headers = {"User-Agent": "MLIR-AIR-task-preparation"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=30
    ) as r:
        return json.load(r)


def native_configurations(context):
    if type(context) is not int or context <= 0:
        raise ValueError("native context must be a positive integer")
    return [
        dict(
            prefill=p,
            capacity=c,
            status="REQUIRED" if c <= context else "UNSUPPORTED_NATIVE_CONTEXT",
        )
        for p, c in CONFIGURATIONS
    ]


def registered_models(root):
    """Read the shared CLI registry without importing runtime/build modules."""
    path = root / "programming_examples/llms/shared/deployment.py"
    if not path.is_file():
        return set()
    try:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "MODELS"
                for target in node.targets
            ):
                names = ast.literal_eval(node.value)
                if isinstance(names, (tuple, list)) and all(
                    isinstance(name, str) for name in names
                ):
                    return set(names)
    except (SyntaxError, ValueError, TypeError):
        pass
    return set()


def prepare(
    hf_id,
    *,
    root=ROOT,
    revision=None,
    config=None,
    native_context=None,
    requirements=None,
):
    supplied = {} if requirements is None else requirements
    requested = resolve_requirements(supplied)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", hf_id):
        raise ValueError("expected a Hugging Face owner/model ID")
    matches = []
    for path in sorted((root / "programming_examples/llms").glob("*/model.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("checkpoint") == hf_id:
            matches.append((path, spec))
    if len(matches) > 1:
        raise ValueError("ambiguous existing model specifications")
    existing = matches[0] if matches else None
    if existing:
        path, spec = existing
        if revision and revision != spec["revision"]:
            raise ValueError(
                "revision differs from the existing example; add a new model spec"
            )
        if config is not None:
            raise ValueError(
                "existing example uses its pinned model.json; do not override config"
            )
        revision = spec["revision"]
        context = max(c for _, c in spec["configurations"])
        if native_context is not None:
            native_configurations(native_context)
            context = native_context
        bindings = dict(
            model_module=spec["module"],
            model_spec=path.relative_to(root).as_posix(),
            model_spec_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            config_sha256=spec["config_sha256"],
            bf16_matrices=spec["bf16_matrices"],
            q4_aliases=spec["q4_aliases"],
        )
        inputs = path.with_name("inputs.json")
        bindings.update(
            input_ids_source=inputs.relative_to(root).as_posix(),
            input_ids_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        )
        cohort = [
            dict(
                prefill=p,
                capacity=c,
                status=(
                    "REQUIRED"
                    if [p, c] in spec["configurations"]
                    else "UNSUPPORTED_NATIVE_CONTEXT"
                ),
            )
            for p, c in CONFIGURATIONS
        ]
    else:
        quoted = urllib.parse.quote(hf_id, safe="/")
        if revision is None:
            revision = read_hub("https://huggingface.co/api/models/" + quoted)["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("pin the HF revision to a full 40-character commit")
        if config is None:
            config = read_hub(
                f"https://huggingface.co/{quoted}/resolve/{revision}/config.json"
            )
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")
        context = native_context
        if context is None:
            context = next(
                (
                    config[k]
                    for k in (
                        "max_position_embeddings",
                        "max_sequence_length",
                        "max_context_length",
                    )
                    if type(config.get(k)) is int
                ),
                None,
            )
        if context is None:
            raise ValueError(
                "config has no native context; inspect reference code and supply --native-context"
            )
        cohort = native_configurations(context)
        bindings = dict(
            model_module=None,
            config=config,
            config_content_sha256=hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
            architecture_review="REQUIRED",
            bf16_matrices=None,
            q4_aliases=None,
        )
    if requested["configurations"] is not None:
        cohort = []
        for row in requested["configurations"]:
            p, c = row["prefill"], row["capacity"]
            status = "REQUIRED" if c <= context else "UNSUPPORTED_NATIVE_CONTEXT"
            if existing and native_context is None and c > context:
                # A shipped cohort can be smaller than a model's real context.
                status = "CONTEXT_REVIEW_REQUIRED"
            cohort.append(dict(row, status=status))
    if native_context is not None:
        for row in cohort:
            if row["capacity"] > context:
                row["status"] = "UNSUPPORTED_NATIVE_CONTEXT"
    for row in cohort:
        if row["status"] in ("REQUIRED", "CONTEXT_REVIEW_REQUIRED"):
            outputs = max(requested["gate"]["outputs"], requested["timing"]["outputs"])
            if row["capacity"] < row["prefill"] + outputs - 1:
                raise ValueError(
                    "capacity cannot hold the prompt and requested outputs; select larger configurations"
                )
    if not any(
        row["status"] in ("REQUIRED", "CONTEXT_REVIEW_REQUIRED") for row in cohort
    ):
        raise ValueError(
            "no supported configuration; select a workload within the native context"
        )
    reasons = []
    if not existing:
        reasons.append(
            "new model: inspect architecture and implement/register its adapter"
        )
    else:
        if spec["module"] not in registered_models(root):
            reasons.append(
                "register/review the model in shared/deployment.py before using the shared CLI"
            )
        for row in cohort:
            if row["status"] == "CONTEXT_REVIEW_REQUIRED":
                reasons.append("resolve native context from the pinned reference")
            elif (
                row["status"] == "REQUIRED"
                and [row["prefill"], row["capacity"]] not in spec["configurations"]
            ):
                reasons.append(f"implement P{row['prefill']}/C{row['capacity']}")
    for key in ("target", "batch_size", "arithmetic", "placement", "gate", "timing"):
        if requested[key] != DEFAULTS[key]:
            reasons.append(
                f"adapt the implementation/verification/profiling for requested {key}"
            )
    mode = requested.pop("mode")
    if mode == "auto":
        mode = "reproduce" if existing else "deploy"
    arithmetic = requested["arithmetic"]
    arithmetic["macro"] = (
        "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"
        if arithmetic["prefill_gemm"] == "BFP16"
        else None
    )
    task = dict(
        schema_version=2,
        status="TASK_PREPARED",
        mode=mode,
        model=hf_id,
        revision=revision,
        context_limit_for_this_task=context,
        configuration_basis=(
            "user-supplied native context; reference review required"
            if native_context is not None
            else (
                "pinned example cohort"
                if existing
                else "HF native context; reference review required"
            )
        ),
        bindings=bindings,
        configurations=cohort,
        target=requested["target"],
        batch_size=requested["batch_size"],
        arithmetic=arithmetic,
        placement=requested["placement"],
        gate=requested["gate"],
        timing=requested["timing"],
        timing_boundaries=dict(
            ttft="embedding through first-logit checks and greedy selection",
            kv_handoff="after first token; inside decode/request time, separately recorded for first-token-only timing",
        ),
        performance_goal=requested["performance_goal"],
        budget=requested["budget"],
        explicit_requirements=copy.deepcopy(supplied),
        execution=dict(
            status="ADAPTATION_REQUIRED" if reasons else "SHARED_ADAPTER_AVAILABLE",
            reasons=reasons,
        ),
        outcome=dict(model_gate="NOT_RUN", performance="NOT_RUN"),
    )
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hf_id")
    p.add_argument("--revision")
    p.add_argument(
        "--config",
        type=Path,
        help="local config for a new model; requires a pinned revision",
    )
    p.add_argument("--native-context", type=int)
    p.add_argument(
        "--requirements",
        type=Path,
        help="JSON task overrides; see agentic/references/task-input.md",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("output already exists; use a new task path")
    if args.config and not args.revision:
        p.error("--config requires --revision for offline preparation")
    try:
        task = prepare(
            args.hf_id,
            revision=args.revision,
            config=(
                json.loads(args.config.read_text(encoding="utf-8"))
                if args.config
                else None
            ),
            native_context=args.native_context,
            requirements=(
                json.loads(args.requirements.read_text(encoding="utf-8"))
                if args.requirements
                else None
            ),
        )
    except (ValueError, KeyError, OSError) as exc:
        p.exit(1, f"Task preparation failed: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(task, f, indent=2)
        f.write("\n")
    print(f"Prepared {args.out}. Read the task and architecture before implementation.")
    print(
        f"{task['mode']}: {task['model']} at {task['revision']}; target={task['target']}, batch={task['batch_size']}"
    )
    print(
        f"Weights: {task['arithmetic']['logical_weights']}; prefill GEMM: {task['arithmetic']['prefill_gemm']}"
    )
    print(task["execution"]["status"])
    for reason in task["execution"]["reasons"]:
        print("- " + reason)


if __name__ == "__main__":
    main()
