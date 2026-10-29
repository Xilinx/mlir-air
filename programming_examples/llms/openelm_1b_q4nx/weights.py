# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Adapted from the declared Granite Q4NX deployment example.
"""Lossless repacking of the fixed affine group-32 Q4 assets.

Device block/cascade layout follows fused_decode/proj_qmm_pack.py (AMD, MIT).
No extrema, code selection, or requantization occurs here.
"""

import time

import numpy as np

from .common import LAYERS, file_hash, save_json
from .build import DECODE_ENV, decode_module
from ..shared.affine_q4 import CanonicalWeights
from ..shared.q4_cascade import (
    pack_cascade,
    check_seed_packer,
    pad,
    slice_rows,
    dequant,
)


def prepare(out, task):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    fd = decode_module()
    reader = CanonicalWeights(task["assets"]["q4_root"])
    total = fd.UNI_DEC * fd.W_LAYER
    stream = np.lib.format.open_memmap(
        out / "decode_weights.npy", mode="w+", dtype=np.uint16, shape=(total,)
    )
    evidence = check_seed_packer()
    offset = 0
    norms = []
    qknorms = []
    for l, (qh, kh, ffn) in enumerate(LAYERS):
        prefix = f"transformer.layers.{l}."
        qkv = reader.codes(prefix + "attn.qkv_proj.weight")
        gu = reader.codes(prefix + "ffn.proj_1.weight")
        tensors = dict(
            q=pad(slice_rows(qkv, 0, qh * 64), 2048, 2048),
            k=pad(slice_rows(qkv, qh * 64, (qh + kh) * 64), 512, 2048),
            v=pad(slice_rows(qkv, (qh + kh) * 64, (qh + 2 * kh) * 64), 512, 2048),
            o=pad(reader.codes(prefix + "attn.out_proj.weight"), 2048, 2048),
            gate=pad(slice_rows(gu, 0, ffn), 8192, 2048),
            up=pad(slice_rows(gu, ffn, 2 * ffn), 8192, 2048),
            down=pad(reader.codes(prefix + "ffn.proj_2.weight"), 2048, 8192),
        )
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
        assert offset == (l + 1) * fd.W_LAYER
        for key, tensor in tensors.items():
            np.save(
                out / f"layer{l}_{key}.npy", dequant(tensor).T.copy().view(np.uint16)
            )
        norms.append(
            np.stack(
                [
                    reader.bf16(prefix + n + ".weight", as_bits=True)
                    for n in ("attn_norm", "ffn_norm")
                ]
            )
        )
        qknorms.append(
            np.stack(
                [
                    reader.bf16(prefix + "attn." + n + ".weight", as_bits=True)
                    for n in ("q_norm", "k_norm")
                ]
            )
        )
        print(
            f"packed layer {l}: real Q={qh} KV={kh} FFN={ffn}; stream {offset*2} bytes",
            flush=True,
        )
    stream.flush()
    assert offset == total
    np.save(out / "norms.npy", np.stack(norms))
    np.save(out / "qknorms.npy", np.stack(qknorms))
    for name, tensor in [
        ("embedding", "transformer.token_embeddings.weight"),
        ("final_norm", "transformer.norm.weight"),
    ]:
        np.save(out / f"{name}.npy", reader.bf16(tensor, as_bits=True))
    record = dict(
        logical_manifest_sha256=file_hash(reader.root / "manifest.json"),
        decode_environment=DECODE_ENV,
        decode_weight_elements=total,
        all_packed_codes_and_metadata_checked=True,
        requantized=False,
        lm_head_source="original BF16 transformer.token_embeddings.weight; shared with embedding",
        layer_geometry=LAYERS,
        packer_check=evidence,
        preparation_seconds=time.perf_counter() - start,
        files={
            p.name: dict(sha256=file_hash(p), bytes=p.stat().st_size)
            for p in sorted(out.glob("*.npy"))
        },
    )
    record["preparation_including_hash_seconds"] = time.perf_counter() - start
    save_json(out / "manifest.json", record)
    return record
