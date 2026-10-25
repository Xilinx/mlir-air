# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Native Llama single-token decoder, seeded with the same NPU prompt state."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from paths import HERE, FUSED
from common.build_identity import build_identity, check_existing, digest
from model_paths import cache_root, resolve_target_model


def load_baseline(artifacts):
    artifacts = Path(artifacts).resolve()
    check_existing(artifacts, build_identity("baseline", HERE))
    for name in list(os.environ):
        if name.startswith(
            ("DECODE_", "PROJ_", "RMS_", "ATTN_", "Q4K_", "UNI_")
        ) or name in ("W_DUAL_CHAN", "VOCAB_CHUNK_I2"):
            del os.environ[name]
    os.environ["DECODE_ELF"] = "0"
    model = Path(resolve_target_model())
    llms = HERE.parents[1]
    sources = (
        FUSED / "fused_decode.py",
        llms / "llama31_8b_q4nx/q4nx_decode_8b.py",
        llms / "llama32_1b_q4nx/q4nx_requant.py",
        llms / "llama32_1b_q4nx/llama32_1b_q4nx_weights.py",
        llms / "shared/infra/host_weights.py",
    )
    identity = dict(
        model_sha256=digest(model),
        packing_sources={str(p.relative_to(llms.parent)): digest(p) for p in sources},
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache = cache_root() / ("baseline_" + key)
    receipt = cache / "identity.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity or digest(saved["path"]) != saved["sha256"]:
            raise ValueError("Native B1 weight-cache identity mismatch")
    sys.path.insert(0, str(llms / "llama31_8b_q4nx"))
    from q4nx_decode_8b import FusedDecode8B

    decoder = FusedDecode8B(str(model), str(artifacts), max_L=4096, wcache_dir=cache)
    weight_identity = dict(
        identity=identity,
        path=decoder.weight_cache,
        sha256=digest(decoder.weight_cache),
    )
    receipt.write_text(json.dumps(weight_identity, indent=2) + "\n")

    class Baseline:
        batch = 1
        capacity = decoder.ATTN_MAXL

        def seed_kv(self, k, v, position):
            if decoder.seed_kv(k, v) != position:
                raise ValueError("Saved prefill length differs from the KV prefix")

        def verify(self, tokens, position):
            return SimpleNamespace(
                logits=decoder.dispatch(int(tokens[0]), position)[None, :]
            )

    result = Baseline()
    result.weight_identity = weight_identity
    return result
