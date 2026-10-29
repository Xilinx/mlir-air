# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare canonical group-32 affine Q4 from a pinned HF checkpoint, rowwise."""

from pathlib import Path
import json
import numpy as np
from ml_dtypes import bfloat16

from .affine_q4 import FORMAT, SafetensorRows, pack_nibbles, bf16_bits, file_hash


def quantize(x):
    """AIR Q4NX min/max codec; choose codes before rounding scale/min to BF16."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] % 32 or not np.isfinite(x).all():
        raise ValueError("expected finite MxK input with K divisible by 32")
    g = x.reshape(x.shape[0], -1, 32)
    lo, hi = g.min(2), g.max(2)
    scale = (hi - lo) / 15.0
    scale = np.where(scale <= 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.round((g - lo[..., None]) / scale[..., None]), 0, 15).astype(
        np.uint8
    )
    sb, mb = bf16_bits(scale), bf16_bits(lo)
    if (
        not np.isfinite(sb.view(bfloat16).astype(np.float32)).all()
        or not np.isfinite(mb.view(bfloat16).astype(np.float32)).all()
        or np.any(sb.view(bfloat16) <= 0)
    ):
        raise ValueError("invalid BF16 quantization scale")
    return q.reshape(x.shape), sb, mb


def prepare(source, out, model, chunk_rows=256):
    source, out = Path(source).resolve(), Path(out).resolve()
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if file_hash(source / "config.json") != model["config_sha256"]:
        raise ValueError("checkpoint config differs from the pinned model")
    raw = SafetensorRows(source)
    expected = {s["file"]: s for s in model["source_shards"]}
    if set(raw.files) != set(expected):
        raise ValueError("checkpoint shard names differ from the pinned model")
    for name, rec in raw.files.items():
        if (
            rec["size"] != expected[name]["bytes"]
            or file_hash(rec["path"]) != expected[name]["sha256"]
        ):
            raise ValueError(f"checkpoint shard differs from the pinned model: {name}")
    out.mkdir(parents=True, exist_ok=False)
    names = {n: n for n in raw.tensors}
    names.update(model["q4_aliases"])
    entries = []
    for name, origin in sorted(names.items()):
        shape = raw.tensors[origin]["shape"]
        is_q4 = len(shape) == 2 and name not in model["bf16_matrices"]
        if is_q4 and shape[1] % 32:
            raise ValueError(f"Q4 tensor K is not divisible by 32: {name}")
        specs = (
            {
                "q": ("q_packed.npy", [shape[0], shape[1] // 2], "u1"),
                "scale": ("scale_bits.npy", [shape[0], shape[1] // 32], "<u2"),
                "min": ("min_bits.npy", [shape[0], shape[1] // 32], "<u2"),
            }
            if is_q4
            else {"value": ("bf16_bits.npy", shape, "<u2")}
        )
        arrays, handles = {}, {}
        for key, (filename, dims, dtype) in specs.items():
            rel = Path("tensors") / name / filename
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            a = np.lib.format.open_memmap(
                out / rel, mode="w+", dtype=dtype, shape=tuple(dims)
            )
            arrays[key] = dict(path=str(rel), shape=dims, dtype=a.dtype.str)
            handles[key] = a
        for start in range(0, shape[0], chunk_rows):
            stop = min(shape[0], start + chunk_rows)
            bits = raw.bits(origin, start, stop)
            if is_q4:
                q, sc, mn = quantize(bits.view(bfloat16).astype(np.float32))
                for key, a in (("q", pack_nibbles(q)), ("scale", sc), ("min", mn)):
                    handles[key][start:stop] = a
            else:
                handles["value"][start:stop] = bits
        for key, a in handles.items():
            a.flush()
            arrays[key]["sha256"] = file_hash(out / arrays[key]["path"])
        entries.append(
            dict(
                name=name,
                source_tensor=origin,
                shape=shape,
                kind="affine_q4_g32" if is_q4 else "bf16",
                arrays=arrays,
            )
        )
        print("Prepared", name, flush=True)
    record = dict(
        format=FORMAT,
        checkpoint=model["checkpoint"],
        revision=model["revision"],
        config_sha256=model["config_sha256"],
        source_shards=model["source_shards"],
        codec=dict(
            nibble_order="even_K_low_odd_K_high",
            group_size=32,
            source_rounding="BF16 nearest-even before quantization",
            parameter_rounding="BF16 after code selection",
        ),
        tensors=entries,
    )
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (out / "COMPLETE.json").write_text(
        json.dumps(dict(manifest_sha256=file_hash(out / "manifest.json"))) + "\n"
    )
