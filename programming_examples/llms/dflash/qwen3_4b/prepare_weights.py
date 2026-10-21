# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare the target and drafter packed caches consumed by the NPU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

from draft_weights import DraftWeights
from model_paths import (
    DRAFT_MODEL,
    DRAFT_REVISION,
    TARGET_MODEL,
    TARGET_REVISION,
    resolve_target_model,
    default_draft_cache,
    default_draft_snapshot,
    default_target_cache,
)
from selected_config import selected_config
from target_config import FUSED, HERE, load_geometry

QWEN = HERE.parents[1] / "qwen3_4b_q4nx"
sys.path.insert(0, str(QWEN))
sys.path.insert(0, str(FUSED))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cache(path: Path, config) -> dict:
    fd = load_geometry(config)
    expected_w = (
        config.layers * fd.W_LAYER + fd.UNI_LM * fd.VOCAB_W_BLOCKS * fd.BLOCK_BF16
    )
    with np.load(path) as values:
        required = {"W", "RMS_in", "RMS_post"}
        if not required.issubset(values.files):
            raise ValueError(
                f"{path} is missing {sorted(required - set(values.files))}"
            )
        if values["W"].size != expected_w:
            raise ValueError((path, "W", values["W"].size, expected_w))
        if (
            values["RMS_in"].shape[0] < config.layers
            or values["RMS_post"].shape[0] < config.layers
        ):
            raise ValueError((path, "RMS layer count"))
        arrays = {
            name: {"shape": list(values[name].shape), "dtype": str(values[name].dtype)}
            for name in values.files
        }
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
        "arrays": arrays,
    }


def build_draft_cache(path: Path, target_model, snapshot: Path) -> None:
    from qwen3_4b_q4nx_requant import _interleave512, _requant_q4k

    config = selected_config("drafter")
    fd = load_geometry(config)
    weights = DraftWeights(target_model=target_model, snapshot=snapshot)
    projections = {
        "q": ("self_attn.q_proj", 4096, 2560),
        "k": ("self_attn.k_proj", 1024, 2560),
        "v": ("self_attn.v_proj", 1024, 2560),
        "o": ("self_attn.o_proj", 2560, 4096),
        "gate": ("mlp.gate_proj", 9728, 2560),
        "up": ("mlp.up_proj", 9728, 2560),
        "down": ("mlp.down_proj", 2560, 9728),
    }

    def matrix(layer: int, name: str) -> np.ndarray:
        stem, rows, cols = projections[name]
        value = weights.tensor(f"layers.{layer}.{stem}.weight").astype(np.float32)
        if value.shape != (rows, cols):
            raise ValueError((layer, name, value.shape, (rows, cols)))
        return value

    packed_layers = []
    rms_in, rms_post = [], []
    for layer in range(config.layers):
        q, k, v = (matrix(layer, name) for name in ("q", "k", "v"))
        phase = [None] * fd.NPH
        phase[0] = _requant_q4k(np.concatenate((q, k, v), axis=0), fd.GROUP)
        phase[fd.OPROJ_PHASE] = _requant_q4k(matrix(layer, "o"), fd.GROUP)
        phase[fd.GLU_PHASE] = _interleave512(
            _requant_q4k(matrix(layer, "up"), fd.GROUP),
            _requant_q4k(matrix(layer, "gate"), fd.GROUP),
            fd.GLU_CHUNK,
        )
        phase[fd.DOWN_PHASE] = _requant_q4k(matrix(layer, "down"), fd.GROUP)
        packed = np.concatenate(
            [
                fd.pack_q4k_cascade(
                    *phase[index],
                    fd.NCX,
                    fd.NCY,
                    iter_major=True,
                    dual_chan=bool(fd.W_DUAL_CHAN),
                )
                for index in range(fd.NPH)
            ]
        )
        if packed.size != fd.W_LAYER:
            raise ValueError((layer, packed.size, fd.W_LAYER))
        packed_layers.append(packed)
        rin, rpost = weights.tensor(
            f"layers.{layer}.input_layernorm.weight"
        ), weights.tensor(f"layers.{layer}.post_attention_layernorm.weight")
        rms_in.append(np.asarray(rin, bfloat16))
        rms_post.append(np.asarray(rpost, bfloat16))
        print(f"packed drafter layer {layer + 1}/{config.layers}", flush=True)

    target_embedding, _, _ = target_model.embed_norm_lmhead()
    padded = np.zeros((fd.VOCAB_SIZE_PADDED_FULL, 2560), np.float32)
    padded[: fd.VOCAB_SIZE] = np.asarray(target_embedding[: fd.VOCAB_SIZE], np.float32)
    q, scale, minimum = _requant_q4k(padded, fd.GROUP)
    vocabulary = [
        fd.pack_q4k_cascade(
            q[index * fd.VOCAB_SIZE_PADDED : (index + 1) * fd.VOCAB_SIZE_PADDED],
            scale[index * fd.VOCAB_SIZE_PADDED : (index + 1) * fd.VOCAB_SIZE_PADDED],
            minimum[index * fd.VOCAB_SIZE_PADDED : (index + 1) * fd.VOCAB_SIZE_PADDED],
            fd.NCX,
            fd.NCY,
            iter_major=True,
            dual_chan=bool(fd.W_DUAL_CHAN),
        )
        for index in range(fd.UNI_LM)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W=np.concatenate(packed_layers + vocabulary).view(np.int16),
        RMS_in=np.stack(rms_in).view(np.int16),
        RMS_post=np.stack(rms_post).view(np.int16),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=TARGET_MODEL)
    parser.add_argument("--target-cache", type=Path, default=default_target_cache())
    parser.add_argument("--draft-cache", type=Path, default=default_draft_cache())
    parser.add_argument("--draft-snapshot", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--manifest", type=Path, default=HERE / "build/weights.json")
    parser.add_argument("--force-draft", action="store_true")
    args = parser.parse_args()
    target_cache = args.target_cache.expanduser().resolve()
    draft_cache = args.draft_cache.expanduser().resolve()

    from qwen3_4b_q4nx_weights import Q4nxModel
    from qwen3_4b_q4nx_requant import build_requant_cache

    source_model = resolve_target_model(args.target_model, offline=args.offline)
    target_model = Q4nxModel(source_model)
    if not target_cache.exists():
        target_cache.parent.mkdir(parents=True, exist_ok=True)
        build_requant_cache(
            source_model,
            load_geometry(selected_config("target")),
            target_cache,
            requant=True,
        )

    snapshot = (args.draft_snapshot or default_draft_snapshot()).expanduser().resolve()
    if (
        not (snapshot / "config.json").is_file()
        or not (snapshot / "model.safetensors").is_file()
    ):
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                DRAFT_MODEL,
                revision=DRAFT_REVISION,
                allow_patterns=("config.json", "model.safetensors"),
                local_files_only=args.offline,
            )
        ).resolve()
    if args.force_draft and draft_cache.exists():
        draft_cache.unlink()
    if not draft_cache.exists():
        build_draft_cache(draft_cache, target_model, snapshot)

    result = {
        "target_model": args.target_model,
        "target_revision": (
            TARGET_REVISION if args.target_model == TARGET_MODEL else None
        ),
        "target_source_sha256": sha(Path(source_model)),
        "draft_model": DRAFT_MODEL,
        "draft_revision": DRAFT_REVISION,
        "draft_snapshot": str(snapshot),
        "target_cache": validate_cache(target_cache, selected_config("target")),
        "draft_cache": validate_cache(draft_cache, selected_config("drafter")),
    }
    manifest = args.manifest.expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
