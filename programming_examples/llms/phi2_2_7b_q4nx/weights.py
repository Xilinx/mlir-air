# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Preserve fixed Q4 codes/minima/scales; map logical axes into Phi device pads."""

import argparse
from pathlib import Path
import time
import numpy as np
from ml_dtypes import bfloat16
from air_examples.llms.shared.affine_q4 import CanonicalWeights
from .common import (
    ROOT,
    RUN,
    H,
    HP,
    F,
    FP,
    HEADS,
    VOCAB,
    read_task,
    save_json,
    file_hash,
)
from .packing import pack_cascade, check_seed_packer, dequant
from .build import decode_module, DECODE_ENV


def head_indices():
    h, d = np.indices((32, 80))
    return (h * 128 + (h % 2) * 16 + d).ravel()


def mapped(tensor, rows, cols, rowmap=None, colmap=None, bias=None, bias_col=None):
    q, s, m = tensor
    rm = np.arange(q.shape[0]) if rowmap is None else np.asarray(rowmap)
    cm = np.arange(q.shape[1]) if colmap is None else np.asarray(colmap)
    assert np.all(cm % 32 == np.arange(q.shape[1]) % 32)
    destg, first = np.unique(cm // 32, return_index=True)
    srcg = first // 32
    for d, g in zip(destg, srcg):
        assert np.all(np.flatnonzero(cm // 32 == d) // 32 == g)
    out = (
        np.zeros((rows, cols), np.uint8),
        np.zeros((rows, cols // 32), np.uint16),
        np.zeros((rows, cols // 32), np.uint16),
    )
    out[0][np.ix_(rm, cm)] = q
    out[1][np.ix_(rm, destg)] = s[:, srcg]
    out[2][np.ix_(rm, destg)] = m[:, srcg]
    assert np.array_equal(out[0][np.ix_(rm, cm)], q)
    for a, orig in zip(out[1:], (s, m)):
        assert np.array_equal(
            a[np.ix_(rm, cm // 32)], orig[:, np.arange(q.shape[1]) // 32]
        )
    if bias is not None:
        assert bias_col is not None and bias_col // 32 not in destg
        out[2][rm, bias_col // 32] = np.asarray(bias, dtype=bfloat16).view(np.uint16)
    return out


def bias_only(bias, rows, cols):
    q = np.zeros((rows, cols), np.uint8)
    s = np.zeros((rows, cols // 32), np.uint16)
    m = np.zeros_like(s)
    m[: len(bias), H // 32] = np.asarray(bias, dtype=bfloat16).view(np.uint16)
    return q, s, m


def prepare(out, task):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    fd = decode_module()
    reader = CanonicalWeights(task["assets"]["q4_root"])
    hi = head_indices()
    total = 32 * fd.W_LAYER
    stream = np.lib.format.open_memmap(
        out / "decode_weights.npy", mode="w+", dtype=np.uint16, shape=(total,)
    )
    evidence = check_seed_packer()
    norms = []
    offset = 0
    for layer in range(32):
        prefix = f"model.layers.{layer}."

        def codes(n):
            return reader.codes(prefix + n + ".weight")

        def bias(n):
            return reader.bf16(prefix + n + ".bias")

        qkv = [
            mapped(
                codes("self_attn." + n + "_proj"),
                4096,
                HP,
                hi,
                bias=bias("self_attn." + n + "_proj"),
                bias_col=H,
            )
            for n in ("q", "k", "v")
        ]
        qkv = tuple(np.concatenate([t[a] for t in qkv]) for a in range(3))
        o = mapped(
            codes("self_attn.dense"),
            HP,
            4096,
            colmap=hi,
            bias=bias("self_attn.dense"),
            bias_col=96,
        )
        fc1 = mapped(codes("mlp.fc1"), FP, HP)
        fc1_biased = mapped(codes("mlp.fc1"), FP, HP, bias=bias("mlp.fc1"), bias_col=H)
        up = bias_only(bias("mlp.fc1"), FP, HP)
        gu = tuple(
            np.stack(
                [
                    up[a].reshape(-1, fd.GLU_CHUNK, *up[a].shape[1:]),
                    fc1[a].reshape(-1, fd.GLU_CHUNK, *fc1[a].shape[1:]),
                ],
                axis=1,
            ).reshape(-1, *up[a].shape[1:])
            for a in range(3)
        )
        down = mapped(codes("mlp.fc2"), HP, FP, bias=bias("mlp.fc2"), bias_col=F)
        for t in (qkv, o, gu, down):
            block = pack_cascade(*t)
            stream[offset : offset + len(block)] = block
            offset += len(block)
        assert offset == (layer + 1) * fd.W_LAYER
        for key, t in [("qkv", qkv), ("o", o), ("fc1", fc1_biased), ("down", down)]:
            np.save(
                out / f"layer{layer}_{key}.npy", dequant(t).T.copy().view(np.uint16)
            )
        norms.append(
            np.stack(
                [
                    np.pad(
                        reader.bf16(prefix + "input_layernorm." + n, as_bits=True),
                        (0, HP - H),
                    )
                    for n in ("weight", "bias")
                ]
            )
        )
        print(f"packed layer {layer}; decode bytes {offset*2}", flush=True)
    stream.flush()
    assert offset == total
    np.save(out / "norms.npy", np.stack(norms))
    np.save(
        out / "embedding.npy", reader.bf16("model.embed_tokens.weight", as_bits=True)
    )
    for n in ("weight", "bias"):
        np.save(
            out / f"final_norm_{n}.npy",
            np.pad(
                reader.bf16("model.final_layernorm." + n, as_bits=True), (0, HP - H)
            ),
        )
    hb = reader.bf16("lm_head.bias")
    for part in range(4):
        lo, stop = part * 12800, (part + 1) * 12800
        w = reader.dequant("lm_head.weight", lo, stop).astype(bfloat16)
        arr = np.zeros((12800, HP), bfloat16)
        arr[:, :H] = w
        arr[:, H] = hb[lo:stop]
        np.save(out / f"head{part}.npy", arr.view(np.uint16))
    record = dict(
        logical_manifest_sha256=file_hash(reader.root / "manifest.json"),
        decode_environment=DECODE_ENV,
        decode_weight_elements=total,
        all_packed_codes_and_metadata_checked=True,
        requantized=False,
        padding="alternate padded head offset 16 preserves original K group32 boundaries",
        bias="original BF16 affine coefficients on exact constant-one NPU inputs",
        lm_head="independent canonical Q4 dequantized to BF16; original BF16 bias",
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
