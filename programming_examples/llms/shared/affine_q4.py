# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Offline rowwise reader for canonical affine Q4; no model/device dependencies."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

FORMAT = "affine-q4-g32-v1"


def pack_nibbles(q):
    q = np.asarray(q)
    if q.dtype != np.uint8 or q.ndim != 2 or q.shape[1] % 2:
        raise ValueError("q must be a uint8 matrix with even K")
    if np.any(q > 15):
        raise ValueError("q contains a code outside [0,15]")
    return q[:, 0::2] | (q[:, 1::2] << 4)


def unpack_nibbles(packed):
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError("packed q must be a uint8 matrix")
    q = np.empty((packed.shape[0], packed.shape[1] * 2), np.uint8)
    q[:, 0::2] = packed & 15
    q[:, 1::2] = packed >> 4
    return q


def bf16_bits(x):
    return np.asarray(x, dtype=bfloat16).view(np.uint16).astype("<u2", copy=False)


def bf16_float(bits):
    bits = np.asarray(bits)
    if bits.dtype != np.dtype("<u2"):
        raise ValueError("BF16 bits must be little-endian uint16")
    return bits.view(bfloat16).astype(np.float32)


def reconstruct(q, scale_bits, min_bits):
    q = np.asarray(q)
    if q.ndim != 2 or q.dtype != np.uint8 or q.shape[1] % 32 or np.any(q > 15):
        raise ValueError("invalid affine group32 q")
    m, k = q.shape
    if scale_bits.shape != (m, k // 32) or min_bits.shape != scale_bits.shape:
        raise ValueError("scale/min shape mismatch")
    sc, mn = bf16_float(scale_bits), bf16_float(min_bits)
    if not np.isfinite(sc).all() or not np.isfinite(mn).all() or np.any(sc <= 0):
        raise ValueError("invalid stored BF16 scale/min")
    return (
        q.reshape(m, k // 32, 32).astype(np.float32) * sc[..., None] + mn[..., None]
    ).reshape(m, k)


def file_hash(path, tick=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
            tick()
    return h.hexdigest()


def safe_child(root, rel):
    root = Path(root).resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError("manifest path escapes artifact directory")
    return path


class CanonicalWeights:
    def __init__(self, directory, *, require_complete=True):
        self.root = Path(directory)
        mp = self.root / "manifest.json"
        self.manifest = json.loads(mp.read_text())
        if self.manifest.get("format") not in (FORMAT, "granite-affine-q4-g32-v1"):
            raise ValueError("unsupported format")
        if (
            self.manifest.get("codec", {}).get("nibble_order")
            != "even_K_low_odd_K_high"
        ):
            raise ValueError("unsupported nibble order")
        entries = self.manifest["tensors"]
        self.entries = {e["name"]: e for e in entries}
        if len(entries) != len(self.entries):
            raise ValueError("duplicate tensor names")
        if require_complete:
            complete = json.loads((self.root / "COMPLETE.json").read_text())
            if complete.get("manifest_sha256") != file_hash(mp):
                raise ValueError("complete marker does not bind manifest")
        for entry in entries:
            for arr in entry["arrays"].values():
                safe_child(self.root, arr["path"])

    def _slice(self, rec, start, stop):
        a = np.load(
            safe_child(self.root, rec["path"]), mmap_mode="r", allow_pickle=False
        )
        if list(a.shape) != rec["shape"] or a.dtype.str != rec["dtype"]:
            raise ValueError("array shape/dtype disagrees with manifest")
        return np.array(a[start:stop], copy=True)

    def codes(self, name, start=0, stop=None):
        e = self.entries[name]
        if e["kind"] != "affine_q4_g32":
            raise ValueError("tensor is not Q4")
        stop = e["shape"][0] if stop is None else stop
        if not 0 <= start <= stop <= e["shape"][0]:
            raise ValueError("row slice out of bounds")
        q = unpack_nibbles(self._slice(e["arrays"]["q"], start, stop))
        sc = self._slice(e["arrays"]["scale"], start, stop)
        mn = self._slice(e["arrays"]["min"], start, stop)
        return q, sc, mn

    def dequant(self, name, start=0, stop=None):
        return reconstruct(*self.codes(name, start, stop))

    def bf16(self, name, start=0, stop=None, *, as_bits=False):
        e = self.entries[name]
        if e["kind"] != "bf16":
            raise ValueError("tensor is not BF16")
        stop = e["shape"][0] if stop is None else stop
        if not 0 <= start <= stop <= e["shape"][0]:
            raise ValueError("row slice out of bounds")
        b = self._slice(e["arrays"]["value"], start, stop)
        return b if as_bits else bf16_float(b)


class SafetensorRows:
    """Strict local FP16/BF16 source; bits() returns effective BF16 after explicit nearest-even cast. Source files are unchanged."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            # HF single-shard checkpoints have no index. Derive its exact map
            # from the bounded header, then use the same strict data checks.
            single = self.root / "model.safetensors"
            with single.open("rb") as f:
                raw = f.read(8)
                if len(raw) != 8:
                    raise ValueError("truncated safetensors length")
                size = int.from_bytes(raw, "little")
                if size > 16 * 1024 * 1024 or size + 8 > single.stat().st_size:
                    raise ValueError("invalid single-shard header length")
                header = json.loads(f.read(size))
            self.weight_map = {n: single.name for n in header if n != "__metadata__"}
        self.files = {}
        self.tensors = {}
        for rel in sorted(set(self.weight_map.values())):
            path = safe_child(self.root, rel)
            size = path.stat().st_size
            with path.open("rb") as f:
                raw = f.read(8)
                if len(raw) != 8:
                    raise ValueError("truncated safetensors length")
                hlen = int.from_bytes(raw, "little")
                if hlen > 16 * 1024 * 1024 or hlen + 8 > size:
                    raise ValueError("invalid safetensors header length")
                header = json.loads(f.read(hlen))
            base = 8 + hlen
            ranges = []
            for name, e in header.items():
                if name == "__metadata__":
                    continue
                if name in self.tensors or self.weight_map.get(name) != rel:
                    raise ValueError("duplicate tensor or index/shard mismatch")
                shape = e["shape"]
                if (
                    e["dtype"] not in ("BF16", "F16")
                    or not shape
                    or any(type(n) is not int or n <= 0 for n in shape)
                ):
                    raise ValueError(
                        "expected positive-shape FP16 or BF16 source tensor"
                    )
                lo, hi = e["data_offsets"]
                if lo < 0 or hi - lo != int(np.prod(shape)) * 2 or base + hi > size:
                    raise ValueError("invalid tensor offsets")
                ranges.append((lo, hi))
                self.tensors[name] = {
                    "shape": shape,
                    "file": rel,
                    "offset": base + lo,
                    "bytes": hi - lo,
                    "source_dtype": e["dtype"],
                }
            ranges.sort()
            if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
                raise ValueError("overlapping source tensors")
            self.files[rel] = {
                "path": path,
                "size": size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
        if set(self.tensors) != set(self.weight_map):
            raise ValueError("source index coverage mismatch")

    def bits(self, name, start=0, stop=None):
        e = self.tensors[name]
        shape = e["shape"]
        stop = shape[0] if stop is None else stop
        if not 0 <= start <= stop <= shape[0]:
            raise ValueError("source row slice out of bounds")
        row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        fd = os.open(self.files[e["file"]]["path"], os.O_RDONLY)
        try:
            expected = (stop - start) * row * 2
            raw = os.pread(fd, expected, e["offset"] + start * row * 2)
        finally:
            os.close(fd)
        if len(raw) != expected:
            raise ValueError("short source tensor read")
        bits = np.frombuffer(raw, dtype="<u2").reshape([stop - start] + shape[1:])
        if e["source_dtype"] == "BF16":
            return bits
        # The task defines original-HF BF16 execution from a stored FP16 checkpoint.
        # Exactly expand FP16 to FP32, then round once to nearest-even BF16.
        value = bits.view("<f2").astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite original FP16 source")
        return bf16_bits(value)
