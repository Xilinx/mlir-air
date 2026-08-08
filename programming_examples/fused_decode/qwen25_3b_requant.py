# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the Qwen2.5-3B fused-decode weight cache (.npz) straight from the HF
# bf16 checkpoint (Qwen/Qwen2.5-3B-Instruct).
#
# Unlike the Llama path -- which re-quantizes an already-4-bit model.q4nx
# bundle -- no pre-quantized Qwen2.5-3B bundle is needed (or exists locally):
# the decode kernels only require blocks the proj kernel can dequantize, so we
# quantize the full-precision weights once, directly. That is also strictly
# more accurate than requantizing someone else's 4-bit weights.
#
# Quant codec: the Qwen kernels are built with -DQ4_0 (models/qwen2.5-3b.h), so
# q4k_block_t carries SIGNED int4 quants and the `mins` field is unused --
# w = q * scale, symmetric, per 32-column group. The block byte layout is
# otherwise identical to the Q4NX path (scales 512B | mins 512B | qs 4096B =
# 2560 bf16), so the memtile-cascade stream order is the same.
#
# Weight stream per layer = 4 phases, each cascade-packed iteration-major:
#   ph0 QKV   [q(2048) | k(256) | v(256)]  x K=2048   ->  5 rounds
#   ph1 o     [2048]                       x K=2048   ->  4 rounds
#   ph2 gate/up  up|gate interleaved in 512-row chunks x K=2048 -> 44 rounds
#   ph3 down  [2048]                       x K=11264            ->  4 rounds
# INTERMEDIATE is padded 11008 -> 11264 (multiple of 512); the pad rows are
# quantized zeros, so silu(0)*0 = 0 and down's pad columns contribute nothing.
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

HF_REPO = "Qwen/Qwen2.5-3B-Instruct"

_PROJ = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "up": "mlp.up_proj",
    "gate": "mlp.gate_proj",
    "down": "mlp.down_proj",
}
_BIAS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")


# ---------------------------------------------------------------------------
# HF safetensors reader (mmap, multi-shard, no torch/transformers dependency)
# ---------------------------------------------------------------------------
class HFModel:
    """mmap + parse a (possibly sharded) HF safetensors checkpoint."""

    def __init__(self, model=HF_REPO):
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


def pack_q4k_cascade_fast(q, scale, NCX, NCY):
    """Vectorized iteration-major cascade pack; == proj_qmm_pack.pack_q4k_cascade
    (iter_major=True) with all-zero mins, but packs every block at once.

    The reference packer's per-block Python loops cost ~30 min for a 36-layer
    model; this is the same layout computed with numpy reshapes.
    """
    M, K = q.shape
    assert M % ROW_BLOCK == 0 and K % COL_BLOCK == 0
    nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
    n_cores = NCX * NCY
    assert nbi % n_cores == 0, (nbi, n_cores)
    nbi_pc = nbi // n_cores
    G = ROW_BLOCK // PARALLEL  # row groups per block (2)

    # scales[block][group(8)][row(32)] as bf16 (mins stay zero for Q4_0)
    sc = (
        scale.reshape(nbi, ROW_BLOCK, nbj, N_GROUPS)
        .transpose(0, 2, 3, 1)  # [nbi][nbj][group][row]
        .astype(bfloat16)
        .view(np.int16)
    )
    mn = np.zeros_like(sc)

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
    return blocks[gi.ravel(), gj.ravel()].reshape(-1)


def attn_out_perm(fd):
    """Permutation mapping an o-proj input column to the attention output element
    that actually carries it.

    The attention kernel's per-CU output is mmul-TILED, not [head][dv]: device
    element `dvt*64 + hh*8 + dvi` of a CU holds head `hh`, dv column
    `dvt*8 + dvi` (DH = 16 tiles x 8). Measured directly on device -- with a
    uniform-softmax probe whose V column pattern makes each output element report
    the dv index it read, the map is exactly dv(p) = 8*(p//64) + (p%8).

    Rather than de-tiling on chip, permute the o-proj weight COLUMNS to match:
    Wo_device = Wo_true[:, attn_out_perm(fd)]. Free, and it keeps the attention
    output path untouched.
    """
    ncu, dh = fd.N_ATTN_CU, fd.DH
    hpc = fd.Q_HEADS_PER_CU
    dvi = 8
    dvt = dh // dvi
    # [cu][head][dv_tile][dv_inner] = the TRUE (natural) element index ...
    true = np.arange(ncu * hpc * dh).reshape(ncu, hpc, dvt, dvi)
    # ... re-read in device order [cu][dv_tile][head][dv_inner].
    return true.transpose(0, 2, 1, 3).reshape(-1)


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


def _pad_rows(w, m):
    """Zero-pad a [M, K] matrix to [m, K]."""
    if w.shape[0] == m:
        return w
    out = np.zeros((m, w.shape[1]), np.float32)
    out[: w.shape[0]] = w
    return out


def _pad_cols(w, k):
    if w.shape[1] == k:
        return w
    out = np.zeros((w.shape[0], k), np.float32)
    out[:, : w.shape[1]] = w
    return out


# ---------------------------------------------------------------------------
# cache builder
# ---------------------------------------------------------------------------
def build_requant_cache(fd, cache_path, model=HF_REPO, n_layers=None, verbose=True):
    """Re-quantize + cascade-pack the HF Qwen2.5-3B weights into the decode .npz.

    `fd` = the loaded fused_decode_qwen module (supplies the cascade geometry).
    Writes keys:
      W        [n_layers, W_LAYER] int16 -- phase-major [ph][cx][...] weight stream
      RMS_in   [n_layers, K]  bf16 -- input_layernorm
      RMS_post [n_layers, K]  bf16 -- post_attention_layernorm
      BIAS     [n_layers, DQ+DK+DV] bf16 -- q|k|v proj bias (Qwen2.5 has these)
      NORM     [K] bf16 -- final norm
    """
    hf = HFModel(model)
    aperm = attn_out_perm(fd)
    NCX, NCY, NPH, K = fd.NCX, fd.NCY, fd.NPH, fd.K
    RB, NJ = fd._RB, fd._NJ
    DQ, DK, DV = fd.DQ, fd.DK, fd.DV
    INTER = fd.INTERMEDIATE
    GLU_CHUNK = fd.PAYLOAD
    OP, GP, DP = fd.OPROJ_PHASE, fd.GATEUP_PHASE, fd.DOWN_PHASE
    W_LAYER = sum(RB[p] * NJ[p] for p in range(NPH)) * NCX * NCY * BLOCK_BF16
    if n_layers is None:
        n_layers = fd.NLAYERS if hasattr(fd, "NLAYERS") else 1

    W_all, RMS_in, RMS_post, BIAS = [], [], [], []
    for k in range(n_layers):
        R = {nm: hf.bf16(f"model.layers.{k}.{t}.weight") for nm, t in _PROJ.items()}
        ph = [None] * NPH
        ph[0] = requant_q4_0(np.concatenate([R["q"], R["k"], R["v"]], 0))
        # o-proj consumes the attention output, which arrives mmul-tiled: permute
        # its input columns so the device reads each element from where attention
        # actually put it (see attn_out_perm).
        ph[OP] = requant_q4_0(R["o"][:, aperm])
        up = _pad_rows(R["up"], INTER)
        gate = _pad_rows(R["gate"], INTER)
        qu, su = requant_q4_0(up)
        qg, sg = requant_q4_0(gate)
        ph[GP] = (
            _interleave_chunks(qu, qg, GLU_CHUNK),
            _interleave_chunks(su, sg, GLU_CHUNK),
        )
        ph[DP] = requant_q4_0(_pad_cols(R["down"], INTER))
        w = np.concatenate(
            [pack_q4k_cascade_fast(*ph[p], NCX, NCY) for p in range(NPH)]
        )
        assert w.size == W_LAYER, (w.size, W_LAYER)
        W_all.append(w)
        RMS_in.append(hf.bf16(f"model.layers.{k}.input_layernorm.weight"))
        RMS_post.append(hf.bf16(f"model.layers.{k}.post_attention_layernorm.weight"))
        BIAS.append(
            np.concatenate([hf.bf16(f"model.layers.{k}.{t}.bias") for t in _BIAS])
        )
        if verbose:
            print(f"[qwen requant] layer {k} packed ({w.size} bf16)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
    np.savez(
        cache_path,
        W=np.stack(W_all),
        RMS_in=np.stack(RMS_in).astype(bfloat16).view(np.int16),
        RMS_post=np.stack(RMS_post).astype(bfloat16).view(np.int16),
        BIAS=np.stack(BIAS).astype(bfloat16).view(np.int16),
        NORM=hf.bf16("model.norm.weight").astype(bfloat16).view(np.int16),
    )
    if verbose:
        print(f"[qwen requant] wrote {cache_path}", flush=True)
    return cache_path


def to_column_major(w_phase_major, fd):
    """Re-order one layer's weight stream from phase-major [ph][cx][..] to
    column-major [cx][ph][..] (the QWEN_W_PHASE_MAJOR=0 DDR layout)."""
    NCX, NPH = fd.NCX, fd.NPH
    spans = [fd._RB[p] * fd._NJ[p] * fd.NCY * BLOCK_BF16 for p in range(NPH)]
    off, parts = 0, [[] for _ in range(NCX)]
    for p in range(NPH):
        blk = w_phase_major[off : off + NCX * spans[p]].reshape(NCX, spans[p])
        for cx in range(NCX):
            parts[cx].append(blk[cx])
        off += NCX * spans[p]
    return np.concatenate([np.concatenate(c) for c in parts])


def rope_lut(pos, dh, theta=1000000.0):
    """[cos(dh/2) | sin(dh/2)] for absolute position `pos` (NeoX/rotate_half)."""
    inv = 1.0 / (theta ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
    a = pos * inv
    return np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)
