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

from .common import ROOT, RUN, LAYERS, file_hash, read_task, save_json
from .build import DECODE_ENV, decode_module
from air_examples.llms.shared.affine_q4 import CanonicalWeights


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


def pad(tensor, rows, cols):
    q, sc, mn = tensor
    m, k = q.shape
    assert m <= rows and k <= cols and cols % 32 == 0
    return (
        np.pad(q, ((0, rows - m), (0, cols - k))),
        np.pad(sc, ((0, rows - m), (0, (cols - k) // 32))),
        np.pad(mn, ((0, rows - m), (0, (cols - k) // 32))),
    )


def slice_rows(tensor, lo, hi):
    return tuple(a[lo:hi] for a in tensor)


def dequant(tensor):
    q, s, m = tensor
    return (
        q.astype(np.float32) * s.view(bfloat16).astype(np.float32).repeat(32, axis=1)
        + m.view(bfloat16).astype(np.float32).repeat(32, axis=1)
    ).astype(bfloat16)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / RUN / "weights")
    args = ap.parse_args()
    prepare(args.out.resolve(), read_task())


if __name__ == "__main__":
    main()
