# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Adapted from the declared Granite Q4NX deployment example.
"""Lossless repacking of the fixed affine group-32 Q4 assets.

Device block/cascade layout follows fused_decode/proj_qmm_pack.py (AMD, MIT).
No extrema, code selection, or requantization occurs here.
"""

import argparse
from pathlib import Path
import time

import numpy as np
from ml_dtypes import bfloat16

from .common import ROOT, HERE, file_hash, read_task, save_json
from .build import DECODE_ENV, decode_module
from air_examples.llms.shared.affine_q4 import CanonicalWeights

PROJECTIONS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


def cascade_indices(rows, cols, ncx=4, ncy=4):
    assert rows % (32 * ncx * ncy) == 0 and cols % 256 == 0
    ni, nj = rows // (32 * ncx * ncy), cols // 256
    # [column, shim channel, row iteration, col block, cascade row in half]
    cx, half, i, j, cy = np.indices((ncx, 2, ni, nj, ncy // 2))
    rb = i * ncx * ncy + cx * ncy + half * (ncy // 2) + cy
    return rb.ravel(), j.ravel()


def pack_cascade(q, scale_bits, min_bits, *, verify=True):
    """Vectorized version of iter_major=True, dual_chan=True in the seed."""
    m, k = q.shape
    rb, cb = m // 32, k // 256
    assert scale_bits.shape == min_bits.shape == (m, k // 32)
    blocks = np.empty((rb, cb, 2560), np.uint16)
    blocks[:, :, :256] = (
        scale_bits.reshape(rb, 32, cb, 8).transpose(0, 2, 3, 1).reshape(rb, cb, 256)
    )
    blocks[:, :, 256:512] = (
        min_bits.reshape(rb, 32, cb, 8).transpose(0, 2, 3, 1).reshape(rb, cb, 256)
    )
    qblocks = (
        q.reshape(rb, 32, cb, 256).transpose(0, 2, 1, 3).reshape(rb, cb, 2, 16, 256)
    )
    packed = (
        (qblocks[:, :, :, 0::2, :] | (qblocks[:, :, :, 1::2, :] << 4))
        .transpose(0, 1, 2, 4, 3)
        .copy()
    )
    blocks[:, :, 512:] = packed.reshape(rb, cb, 4096).view(np.uint16)
    ri, ci = cascade_indices(m, k)
    stream = blocks[ri, ci].reshape(-1)
    if verify:
        # Invert the actual stream and all metadata; zero allowed bit mismatches.
        inverse = np.empty_like(blocks)
        inverse[ri, ci] = stream.reshape(-1, 2560)
        sc = (
            inverse[:, :, :256]
            .reshape(rb, cb, 8, 32)
            .transpose(0, 3, 1, 2)
            .reshape(m, k // 32)
        )
        mn = (
            inverse[:, :, 256:512]
            .reshape(rb, cb, 8, 32)
            .transpose(0, 3, 1, 2)
            .reshape(m, k // 32)
        )
        raw = inverse[:, :, 512:].copy().view(np.uint8).reshape(rb, cb, 2, 256, 8)
        qq = np.empty((rb, cb, 2, 16, 256), np.uint8)
        qq[:, :, :, 0::2, :] = (raw & 15).transpose(0, 1, 2, 4, 3)
        qq[:, :, :, 1::2, :] = (raw >> 4).transpose(0, 1, 2, 4, 3)
        qq = qq.reshape(rb, cb, 32, 256).transpose(0, 2, 1, 3).reshape(m, k)
        assert (
            np.array_equal(qq, q)
            and np.array_equal(sc, scale_bits)
            and np.array_equal(mn, min_bits)
        )
    return stream


def check_seed_packer():
    from air_examples.fused_decode.proj_qmm_pack import pack_q4k_cascade

    rng = np.random.default_rng(9401)
    q = rng.integers(0, 16, (1024, 512), dtype=np.uint8)
    sc = rng.uniform(0.001, 0.2, (1024, 16)).astype(bfloat16)
    mn = rng.uniform(-1, 0, (1024, 16)).astype(bfloat16)
    actual = pack_cascade(q, sc.view(np.uint16), mn.view(np.uint16))
    seed = pack_q4k_cascade(
        q,
        sc.astype(np.float32),
        mn.astype(np.float32),
        4,
        4,
        iter_major=True,
        dual_chan=True,
    )
    assert np.array_equal(actual, seed.view(np.uint16))
    return {
        "seed_packer_bitwise_equal": True,
        "test_shape": [1024, 512],
        "uint16_elements_compared": actual.size,
    }


def pad_tensor(tensor, key):
    """Append exact zero affine groups; preserve all original codes/metadata."""
    q, sc, mn = tensor
    if key in ("up", "gate"):
        q = np.pad(q, ((0, 256), (0, 0)))
        sc = np.pad(sc, ((0, 256), (0, 0)))
        mn = np.pad(mn, ((0, 256), (0, 0)))
    elif key == "down":
        q = np.pad(q, ((0, 0), (0, 256)))
        sc = np.pad(sc, ((0, 0), (0, 8)))
        mn = np.pad(mn, ((0, 0), (0, 8)))
    return q, sc, mn


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
    for layer in range(36):
        tensors = {
            key: reader.codes(f"model.layers.{layer}.{suffix}.weight")
            for key, suffix in PROJECTIONS.items()
        }
        tensors = {key: pad_tensor(value, key) for key, value in tensors.items()}
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
            if key in ("up", "gate"):
                matrix = np.pad(matrix, ((0, 256), (0, 0)))
            elif key == "down":
                matrix = np.pad(matrix, ((0, 0), (0, 256)))
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
        count = max(0, min(128256, hi) - lo)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "build/weights")
    args = ap.parse_args()
    prepare(args.out.resolve(), read_task())


if __name__ == "__main__":
    main()
