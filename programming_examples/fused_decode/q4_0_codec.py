# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The shared Q4_0 codec for the fused-decode examples: an HF safetensors reader,
# the symmetric signed-int4 quantizer, and the vectorized cascade packer.
#
# Q4_0 vs the Q4NX affine codec both examples' block layout can carry: the
# nibbles are SIGNED and w = q*scale, so the block's `mins` field is unused (and
# zero). Which one a model uses is a per-model #define -- qwen2.5-3b and lfm2-1.2b
# are the Q4_0 ones (fused_decode/models/*.h). The byte layout is identical
# either way (scales 512B | mins 512B | qs 4096B = 2560 bf16), so the
# memtile-cascade stream order does not depend on the codec.
import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from proj_qmm_pack import (
    ROW_BLOCK,
    COL_BLOCK,
    GROUP,
    N_GROUPS,
    PARALLEL,
    BLOCK_BF16,
)


# ---------------------------------------------------------------------------
# HF safetensors reader (mmap, multi-shard, no torch/transformers dependency)
# ---------------------------------------------------------------------------
class HFModel:
    """mmap + parse a (possibly sharded) HF safetensors checkpoint."""

    def __init__(self, model):
        self._shards = {}  # file -> (memmap, header, base)
        self._where = {}  # tensor name -> file
        for f in self._resolve(model):
            mm = np.memmap(f, dtype=np.uint8, mode="r")
            hlen = int(np.frombuffer(mm[:8].tobytes(), dtype="<u8")[0])
            hdr = json.loads(mm[8 : 8 + hlen].tobytes())
            self._shards[f] = (mm, hdr, 8 + hlen)
            for name in hdr:
                if name != "__metadata__":
                    self._where[name] = f

    @staticmethod
    def _resolve(model):
        """`model` = HF repo id, a local dir, or a single .safetensors file."""
        if os.path.isfile(model):
            return [model]
        if os.path.isdir(model):
            d = Path(model)
        else:
            from huggingface_hub import snapshot_download

            d = Path(
                snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])
            )
        idx = d / "model.safetensors.index.json"
        if idx.is_file():
            with open(idx) as fh:
                files = sorted(set(json.load(fh)["weight_map"].values()))
            return [str(d / f) for f in files]
        return [str(p) for p in sorted(d.glob("*.safetensors"))]

    def has(self, name):
        return name in self._where

    def bf16(self, name):
        """A BF16/F32 tensor as float32, in its declared shape."""
        mm, hdr, base = self._shards[self._where[name]]
        e = hdr[name]
        o0, o1 = e["data_offsets"]
        raw = mm[base + o0 : base + o1].tobytes()
        dt = {"BF16": bfloat16, "F32": np.float32, "F16": np.float16}[e["dtype"]]
        return np.frombuffer(raw, dtype=dt).astype(np.float32).reshape(e["shape"])


# ---------------------------------------------------------------------------
# Q4_0 re-quant + vectorized cascade pack
# ---------------------------------------------------------------------------
def requant_q4_0(Wm, group=GROUP):
    """Symmetric signed-4-bit per-group quant of [M, K] -> (nibbles, scale).

    w ~= q * scale with q in [-8, 7]. scale = amax/8 (the llama.cpp Q4_0
    magnitude convention): a positive peak element then rounds to +8 and clips
    to +7, but the finer step everywhere else more than pays for it -- measured
    rel_l2 over the divisor is flat-bottomed around 7.75-8.0 and worse at 7.
    All-zero groups get scale 1 (a 0 scale would make the block undequantizable).
    Returns nibbles as uint8 two's complement (the packer masks with 0xF).
    """
    M, Kc = Wm.shape
    Wg = Wm.reshape(M, Kc // group, group)
    amax = np.abs(Wg).max(2)
    sc = np.where(amax <= 0, 1.0, amax / 8.0).astype(np.float32)
    q = np.clip(np.rint(Wg / sc[..., None]), -8, 7).astype(np.int8)
    return q.reshape(M, Kc).view(np.uint8), sc


def pack_q4k_cascade_fast(q, scale, NCX, NCY, dual_chan=False, mins=None):
    """Vectorized iteration-major cascade pack; == proj_qmm_pack.pack_q4k_cascade
    (iter_major=True), but packs every block at once.

    The reference packer's per-block Python loops cost ~30 min for a 36-layer
    model; this is the same layout computed with numpy reshapes.

    `mins` is the affine Q4NX min term, shaped like `scale`. Omit it for the
    symmetric Q4_0 codec, whose mins are all zero -- that is the default and
    keeps the Q4_0 callers byte-identical.
    """
    M, K = q.shape
    assert M % ROW_BLOCK == 0 and K % COL_BLOCK == 0
    nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
    n_cores = NCX * NCY
    assert nbi % n_cores == 0, (nbi, n_cores)
    nbi_pc = nbi // n_cores
    G = ROW_BLOCK // PARALLEL  # row groups per block (2)

    # scales[block][group(8)][row(32)] as bf16; mins identically, or zero (Q4_0)
    def _per_block(a):
        return (
            a.reshape(nbi, ROW_BLOCK, nbj, N_GROUPS)
            .transpose(0, 2, 3, 1)  # [nbi][nbj][group][row]
            .astype(bfloat16)
            .view(np.int16)
        )

    sc = _per_block(scale)
    mn = np.zeros_like(sc) if mins is None else _per_block(mins)

    # qs[block][rowgrp(2)][col(256)][pair(8)] = (odd<<4) | even
    qv = q.reshape(nbi, ROW_BLOCK, nbj, COL_BLOCK).transpose(
        0, 2, 1, 3
    )  # [nbi][nbj][row][col]
    qv = qv.reshape(nbi, nbj, G, PARALLEL // 2, 2, COL_BLOCK)
    lo = qv[:, :, :, :, 0, :] & 0xF
    hi = qv[:, :, :, :, 1, :] & 0xF
    qs = ((hi << 4) | lo).transpose(0, 1, 2, 4, 3)  # [nbi][nbj][g][col][pair]
    qs = np.ascontiguousarray(qs).reshape(nbi, nbj, -1).view(np.int16)

    blocks = np.concatenate(
        [sc.reshape(nbi, nbj, -1), mn.reshape(nbi, nbj, -1), qs], axis=2
    )
    assert blocks.shape[2] == BLOCK_BF16, blocks.shape

    # stream order [cx][i][j][cy] with gi = i*(NCX*NCY) + cx*NCY + cy
    cx = np.arange(NCX)[:, None, None, None]
    i = np.arange(nbi_pc)[None, :, None, None]
    j = np.arange(nbj)[None, None, :, None]
    cy = np.arange(NCY)[None, None, None, :]
    gi = np.broadcast_to(i * n_cores + cx * NCY + cy, (NCX, nbi_pc, nbj, NCY))
    gj = np.broadcast_to(j, (NCX, nbi_pc, nbj, NCY))
    if dual_chan:
        # Two-MM2S-per-column weight feed: hoist the ROW HALF (= the shim channel)
        # to just inside cx, so each column's slab is [low-row half | high-row half]
        # and each channel reads one contiguous DDR run. Matches FLM's mem_C_1
        # (shim ch0 -> S2MM4 -> rows 2/3, shim ch1 -> S2MM5 -> rows 4/5) and the
        # llama engine's pack_q4k_cascade(dual_chan=True).
        assert NCY % 2 == 0, f"dual_chan needs an even NCY (got {NCY})"
        sh = (NCX, nbi_pc, nbj, 2, NCY // 2)
        gi = gi.reshape(sh).transpose(0, 3, 1, 2, 4)
        gj = gj.reshape(sh).transpose(0, 3, 1, 2, 4)
    return blocks[gi.ravel(), gj.ravel()].reshape(-1)


def _interleave_chunks(a, b, chunk):
    """[a0|b0|a1|b1|...] in `chunk`-row slices (the GLU stream order)."""
    n = a.shape[0] // chunk
    return np.concatenate(
        [
            (a if h == 0 else b)[s * chunk : (s + 1) * chunk]
            for s in range(n)
            for h in (0, 1)
        ]
    )
