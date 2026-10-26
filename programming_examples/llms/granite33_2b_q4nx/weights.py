# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Lossless repacking of the fixed affine group-32 Q4 assets.

Device block/cascade layout follows fused_decode/proj_qmm_pack.py (AMD, MIT).
No extrema, code selection, or requantization occurs here.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from .common import file_hash, save_json
from .build import DECODE_ENV, decode_module
from ..shared.affine_q4 import CanonicalWeights
from ..shared.q4_cascade import pack_cascade, check_seed_packer

PROJECTIONS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


def prepare(out, task):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    fd = decode_module()
    reader = CanonicalWeights(task["assets"]["q4_root"])
    total = fd.UNI_DEC * fd.W_LAYER + fd.UNI_LM * fd.VOCAB_W_BLOCKS * 2560
    stream = np.lib.format.open_memmap(
        out / "decode_weights.npy", mode="w+", dtype=np.uint16, shape=(total,)
    )
    offset = 0
    norms = []
    evidence = check_seed_packer()
    for layer in range(40):
        tensors = {
            key: reader.codes(f"model.layers.{layer}.{suffix}.weight")
            for key, suffix in PROJECTIONS.items()
        }
        phases = [
            tuple(
                np.concatenate([tensors[n][a] for n in ("q", "k", "v")])
                for a in range(3)
            ),
            tensors["o"],
            tuple(
                np.stack(
                    [
                        tensors["up"][a].reshape(
                            -1, fd.GLU_CHUNK, *tensors["up"][a].shape[1:]
                        ),
                        tensors["gate"][a].reshape(
                            -1, fd.GLU_CHUNK, *tensors["gate"][a].shape[1:]
                        ),
                    ],
                    axis=1,
                ).reshape(-1, *tensors["up"][a].shape[1:])
                for a in range(3)
            ),
            tensors["down"],
        ]
        for tensor in phases:
            block = pack_cascade(*tensor)
            stream[offset : offset + block.size] = block
            offset += block.size
        assert offset == (layer + 1) * fd.W_LAYER
        # Prefill matrices preserve these very same q/scale/min values, with
        # one BF16 rounding after affine reconstruction, then [out,in]->[in,out].
        for key, suffix in PROJECTIONS.items():
            matrix = reader.dequant(f"model.layers.{layer}.{suffix}.weight").astype(
                bfloat16
            )
            np.save(out / f"layer{layer}_{key}.npy", matrix.T.copy().view(np.uint16))
        norms.append(
            np.stack(
                [
                    reader.bf16(f"model.layers.{layer}.{name}.weight", as_bits=True)
                    for name in ("input_layernorm", "post_attention_layernorm")
                ]
            )
        )
        print(f"packed and checked layer {layer}: {offset * 2} bytes", flush=True)
    for wave in range(fd.UNI_LM):
        lo, hi = wave * fd.VOCAB_SIZE_PADDED, (wave + 1) * fd.VOCAB_SIZE_PADDED
        count = max(0, min(49159, hi) - lo)
        q = np.zeros((fd.VOCAB_SIZE_PADDED, 2048), np.uint8)
        sc = np.ones((fd.VOCAB_SIZE_PADDED, 64), bfloat16).view(np.uint16)
        mn = np.zeros_like(sc)
        if count:
            q[:count], sc[:count], mn[:count] = reader.codes(
                "lm_head.weight", lo, lo + count
            )
        block = pack_cascade(q, sc, mn)
        stream[offset : offset + block.size] = block
        offset += block.size
    assert offset == total
    stream.flush()
    np.save(out / "norms.npy", np.stack(norms))
    for name, tensor in (
        ("embedding", "model.embed_tokens.weight"),
        ("final_norm", "model.norm.weight"),
    ):
        np.save(out / f"{name}.npy", reader.bf16(tensor, as_bits=True))
    record = {
        "logical_manifest_sha256": file_hash(reader.root / "manifest.json"),
        "decode_environment": DECODE_ENV,
        "decode_weight_elements": total,
        "all_packed_codes_and_metadata_checked": True,
        "requantized": False,
        "lm_head_source": "independently quantized lm_head.weight",
        "packer_check": evidence,
        "preparation_seconds": time.perf_counter() - start,
        "files": {
            p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in sorted(out.glob("*.npy"))
        },
    }
    record["preparation_including_hash_seconds"] = time.perf_counter() - start
    save_json(out / "manifest.json", record)
    return record
