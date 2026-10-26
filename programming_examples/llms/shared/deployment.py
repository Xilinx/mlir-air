# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Portable bindings for the affine-Q4 model examples.

Model geometry and checkpoint identity live beside each example. Builds and
weights live in user-selected directories; no experiment checkout is required.
"""

from pathlib import Path
import hashlib
import json
import os

LLMS = Path(__file__).resolve().parents[1]
MODELS = (
    "granite33_2b_q4nx",
    "smollm3_3b_q4nx",
    "olmo2_1b_q4nx",
    "openelm_1b_q4nx",
    "phi2_2_7b_q4nx",
    "mistral7b_v03_q4nx",
)


def spec(model):
    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    return json.loads((LLMS / model / "model.json").read_text())


def build_root(model):
    return Path(os.environ.get("AIR_MODEL_BUILD_DIR", LLMS / model / "build")).resolve()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_bfp16(metadata):
    if metadata.get("gemm_bfp16") is not True:
        raise ValueError("rebuild prefill: these examples require BFP16 GEMM artifacts")


def configuration(model, p, ids=None):
    import numpy as np

    matches = [(a, b) for a, b in spec(model)["configurations"] if a == p]
    if len(matches) != 1:
        raise ValueError(f"unsupported native configuration P{p} for {model}")
    if ids is None:
        prompts = json.loads((LLMS / model / "inputs.json").read_text())
        ids = prompts[f"p{p}"]
    a = np.asarray(ids, dtype="<i8")
    if a.shape != (p,) or np.any(a < 0) or np.any(a >= spec(model)["vocab_size"]):
        raise ValueError(f"expected exactly {p} valid token IDs")
    return dict(
        id=f"p{p}",
        tokens=p,
        prefill_build_length=p,
        capacity=matches[0][1],
        token_ids=a.tolist(),
        ids_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
    )


def read_task(model, selected=None):
    s = spec(model)
    q4 = Path(
        os.environ.get("AIR_MODEL_Q4_DIR", build_root(model) / "logical-q4")
    ).resolve()
    manifest = json.loads((q4 / "manifest.json").read_text())
    if (manifest.get("checkpoint"), manifest.get("revision")) != (
        s["checkpoint"],
        s["revision"],
    ):
        raise ValueError("logical Q4 checkpoint/revision differs from model.json")
    # Bind the selected inputs and logical weights to the model session.
    return dict(
        model=dict(
            checkpoint=s["checkpoint"],
            revision=s["revision"],
            vocab_size=s["vocab_size"],
        ),
        assets=dict(
            q4_root=str(q4), q4_manifest_sha256=file_hash(q4 / "manifest.json")
        ),
        prompts=(
            [selected]
            if selected
            else [configuration(model, p) for p, _ in s["configurations"]]
        ),
    )


def runtime(model, selected, work):
    base = build_root(model)
    root = LLMS.parents[1]
    build = base / selected["id"]
    return dict(
        candidate_root=str(root),
        work_dir=str(work),
        configuration=selected,
        build_dir=str(build),
        weights_dir=str(base / "weights"),
        head_dir=str(base / "shared/head"),
        prefill_length=selected["tokens"],
        kv_capacity=selected["capacity"],
        resident_prefill=True,
        build_root=str(base),
    )


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)
