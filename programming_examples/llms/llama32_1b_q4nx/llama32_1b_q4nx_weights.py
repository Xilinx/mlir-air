# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader + dequant for the Llama-3.2-1B Q4NX example.
#
# Reads the per-layer Q4NX weight bins (L{k}_proj_w.bin / L{k}_rms_w.bin) and
# reconstructs the full-precision bf16 weight matrices from the Q4NX blocks
# (reorder with vertical_blocks=2; tensor order q|k|v|o | up/gate interleaved in
# 512-row phases | down). Q4NX = per-block 4-bit affine quant: w = q*scale + min,
# 32x256 blocks, bf16 scale/min per 32-column group.
#
# Layout facts:
#   proj_w = concat[ q(2048x2048) | k(512x2048) | v(512x2048) | o(2048x2048)
#                    | up/gate interleaved per 512-row phase (16384x2048)
#                    | down(2048x8192) ], each Q4NX-reordered (pairs of row-
#                    blocks interleaved column-by-column).
#   rms_w  = [ input_layernorm(2048) | post_attention_layernorm(2048) | 0-pad ]
import numpy as np
from ml_dtypes import bfloat16
from proj_qmm_pack import (
    ROW_BLOCK,
    COL_BLOCK,
    GROUP,
    N_GROUPS,
    ROW_GROUPS,
    PARALLEL,
    BLOCK_BF16,
)

# Llama-3.2-1B dims.
D = 2048  # model dim
DQ = 2048  # q proj out (32 heads * 64)
DK = 512  # k proj out (8 kv heads * 64)
DV = 512
DH = 64  # head dim
N_Q_HEADS = 32
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4
INTER = 8192  # mlp intermediate


def _bf(a):
    return a.astype(bfloat16).astype(np.float32)


def load_bf16(p):
    return np.fromfile(p, dtype=bfloat16).astype(np.float32)


def unpack_block(b):
    """Q4NX block int16[2560] -> (codes[32,256] uint8, scale[32,8], mn[32,8])."""
    raw = b.view(np.int16).tobytes()
    sc = (
        np.frombuffer(raw[0:512], dtype=bfloat16)
        .astype(np.float32)
        .reshape(N_GROUPS, ROW_BLOCK)
    )
    mi = (
        np.frombuffer(raw[512:1024], dtype=bfloat16)
        .astype(np.float32)
        .reshape(N_GROUPS, ROW_BLOCK)
    )
    qs = np.frombuffer(
        raw[1024 : 1024 + ROW_GROUPS * COL_BLOCK * (PARALLEL // 2)], dtype=np.uint8
    ).reshape(ROW_GROUPS, COL_BLOCK, PARALLEL // 2)
    q = np.zeros((ROW_BLOCK, COL_BLOCK), np.uint8)
    for g in range(ROW_GROUPS):
        for h in range(COL_BLOCK):
            for kk in range(PARALLEL // 2):
                pair = qs[g, h, kk]
                q[g * PARALLEL + kk * 2, h] = pair & 0xF
                q[g * PARALLEL + kk * 2 + 1, h] = (pair >> 4) & 0xF
    # scale/mn are stored [group][row]; return as [row, group]
    return q, sc.T.copy(), mi.T.copy()


def unpack_tensor(W_i16, blk0, Mrows, Kcols):
    """De-reorder (vertical_blocks=2) a Q4NX tensor from the proj_w stream.
    Returns (codes[M,K] uint8, scale[M,K/32], mn[M,K/32], next_block_index)."""
    bpr = Kcols // COL_BLOCK
    rb = Mrows // ROW_BLOCK
    q = np.zeros((Mrows, Kcols), np.uint8)
    sc = np.zeros((Mrows, Kcols // GROUP), np.float32)
    mn = np.zeros((Mrows, Kcols // GROUP), np.float32)
    s = blk0
    for rp in range(0, rb, 2):
        for c in range(bpr):
            for i in range(2):
                bq, bs, bm = unpack_block(W_i16[s * BLOCK_BF16 : (s + 1) * BLOCK_BF16])
                r0 = (rp + i) * ROW_BLOCK
                c0 = c * COL_BLOCK
                q[r0 : r0 + ROW_BLOCK, c0 : c0 + COL_BLOCK] = bq
                sc[r0 : r0 + ROW_BLOCK, c0 // GROUP : c0 // GROUP + N_GROUPS] = bs
                mn[r0 : r0 + ROW_BLOCK, c0 // GROUP : c0 // GROUP + N_GROUPS] = bm
                s += 1
    return q, sc, mn, s


def dequant(q, sc, mn):
    """Full-precision matrix from Q4NX codes/scale/mn. w[r,c]=q*scale[c//32]+mn."""
    return q.astype(np.float32) * np.repeat(sc, GROUP, axis=1) + np.repeat(
        mn, GROUP, axis=1
    )


def rmsnorm(x, w, eps=1e-6):
    xf = x.astype(np.float32)
    return _bf(xf / np.sqrt((xf * xf).mean() + eps) * w)


def apply_rope(vec64, cos, sin):
    x1 = vec64[0:32]
    x2 = vec64[32:64]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos])


def load_layer_weights(gd, k):
    """Dequantized bf16 weight matrices [out, K] for layer k, in proj_w order
    (q|k|v|o | up/gate interleaved in 512-row phases | down), from L{k}_proj_w.bin."""
    W = np.fromfile(f"{gd}/L{k}_proj_w.bin", dtype=bfloat16).view(np.int16)
    out = {}
    s = 0

    def take(name, M_, Kc):
        nonlocal s
        q, sc, mn, s = unpack_tensor(W, s, M_, Kc)
        out[name] = dequant(q, sc, mn)

    take("q", DQ, D)
    take("k", DK, D)
    take("v", DV, D)
    take("o", DQ, D)
    up_parts, gate_parts = [], []
    for _ in range(INTER // 512):
        q, sc, mn, s = unpack_tensor(W, s, 512, D)
        up_parts.append((q, sc, mn))
        q, sc, mn, s = unpack_tensor(W, s, 512, D)
        gate_parts.append((q, sc, mn))
    stack = lambda P: tuple(np.concatenate([p[i] for p in P], 0) for i in range(3))
    out["up"] = dequant(*stack(up_parts))
    out["gate"] = dequant(*stack(gate_parts))
    take("down", D, INTER)
    return out


def load_layer_weights_cached(gd, k, cache_dir):
    """load_layer_weights but cached as GEMM input-B [K, out] bf16 .npy (reruns
    skip the slow unpack + the per-call host transpose)."""
    import os
    from pathlib import Path

    cd = Path(cache_dir)
    cd.mkdir(parents=True, exist_ok=True)
    st = os.stat(f"{gd}/L{k}_proj_w.bin")
    tag = f"L{k}_{st.st_size}_{int(st.st_mtime)}"
    names = ("q", "k", "v", "o", "up", "gate", "down")
    if all((cd / f"{tag}_{n}.npy").exists() for n in names):
        return {n: np.load(cd / f"{tag}_{n}.npy").view(bfloat16) for n in names}
    Wt = load_layer_weights(gd, k)
    out = {}
    for n in names:
        bT = np.ascontiguousarray(Wt[n].T, dtype=bfloat16)  # [K, out] GEMM input-B
        np.save(cd / f"{tag}_{n}.npy", bT.view(np.int16))
        out[n] = bT
    return out


def load_layer_q4nx_raw(gd, k):
    """Raw Q4NX (q,sc,mn) per projection for layer k, from L{k}_proj_w.bin."""
    W = np.fromfile(f"{gd}/L{k}_proj_w.bin", dtype=bfloat16).view(np.int16)
    s = 0
    out = {}

    def take(name, M_, K_):
        nonlocal s
        q, sc, mn, s = unpack_tensor(W, s, M_, K_)
        out[name] = (q, sc, mn)

    take("q", DQ, D)
    take("k", DK, D)
    take("v", DV, D)
    take("o", DQ, D)
    up, gate = [], []
    for _ in range(INTER // 512):
        q, sc, mn, s = unpack_tensor(W, s, 512, D)
        up.append((q, sc, mn))
        q, sc, mn, s = unpack_tensor(W, s, 512, D)
        gate.append((q, sc, mn))
    cat = lambda P: tuple(np.concatenate([p[i] for p in P], 0) for i in range(3))
    out["up"], out["gate"] = cat(up), cat(gate)
    take("down", D, INTER)
    return out


# ---------------------------------------------------------------------------
# model.q4nx loader (HF-hosted, self-contained)
#
# The Q4NX model ships as a single safetensors file `model.q4nx` on the Hub
# (repo FastFlowLM/Llama-3.2-1B-NPU2) with standard HF tensor names: per-layer
# {q,k,v,o,gate,up,down}_proj (Q4NX), input/post_attention_layernorm + norm +
# embed_tokens (bf16), and lm_head (Q4NX). The Q4NX storage codec is a per-block
# affine quant `w = scale * (q - min)` (scale/min bf16 per 32-col group), 32x256
# blocks in plain (row-block, col-block) order. This loader mmaps the file,
# parses the safetensors header, and vectorized-dequantizes each tensor.
_G = ROW_BLOCK // PARALLEL  # row groups per block (2)
_EVEN = np.array(
    [g * PARALLEL + 2 * k for g in range(_G) for k in range(PARALLEL // 2)]
)
_ODD = _EVEN + 1


def resolve_q4nx_model(model):
    """Resolve `model` to a local model.q4nx path. `model` may be an HF repo id
    (contains '/'), a directory containing model.q4nx, or a direct file path."""
    import os

    if os.path.isfile(model):
        return model
    if os.path.isdir(model):
        p = os.path.join(model, "model.q4nx")
        if os.path.isfile(p):
            return p
    # treat as an HF repo id
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model, "model.q4nx")


class Q4nxModel:
    """mmap + parse a model.q4nx safetensors file; vectorized Q4NX dequant."""

    def __init__(self, model):
        import json

        path = resolve_q4nx_model(model)
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        hlen = int(np.frombuffer(self._mm[:8].tobytes(), dtype="<u8")[0])
        self._hdr = json.loads(self._mm[8 : 8 + hlen].tobytes())
        self._base = 8 + hlen

    def _raw_i16(self, name):
        o0, o1 = self._hdr[name]["data_offsets"]
        return np.frombuffer(
            self._mm[self._base + o0 : self._base + o1].tobytes(), dtype=np.int16
        )

    def has(self, name):
        return name in self._hdr

    def bf16(self, name):
        """A BF16 tensor as float32, in its declared shape."""
        o0, o1 = self._hdr[name]["data_offsets"]
        v = np.frombuffer(
            self._mm[self._base + o0 : self._base + o1].tobytes(), dtype=bfloat16
        )
        return v.astype(np.float32).reshape(self._hdr[name]["shape"])

    def dequant(self, name):
        """A Q4NX tensor [M, K] dequantized to float32 (w = scale*(q - min))."""
        M, K = self._hdr[name]["shape"]
        nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
        nb = nbi * nbj
        i16 = self._raw_i16(name).reshape(nb, BLOCK_BF16)
        sc = (
            i16[:, 0:256]
            .copy()
            .view(bfloat16)
            .astype(np.float32)
            .reshape(nb, N_GROUPS, ROW_BLOCK)
        )
        mn = (
            i16[:, 256:512]
            .copy()
            .view(bfloat16)
            .astype(np.float32)
            .reshape(nb, N_GROUPS, ROW_BLOCK)
        )
        qb = (
            i16[:, 512:BLOCK_BF16]
            .copy()
            .view(np.uint8)
            .reshape(nb, _G, COL_BLOCK, PARALLEL // 2)
        )
        lo = (qb & 0xF).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        hi = (qb >> 4).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        q = np.zeros((nb, ROW_BLOCK, COL_BLOCK), np.float32)
        q[:, _EVEN, :] = lo
        q[:, _ODD, :] = hi
        w = np.repeat(sc.transpose(0, 2, 1), GROUP, axis=2) * (
            q - np.repeat(mn.transpose(0, 2, 1), GROUP, axis=2)
        )
        return (
            w.reshape(nbi, nbj, ROW_BLOCK, COL_BLOCK)
            .transpose(0, 2, 1, 3)
            .reshape(M, K)
        )

    _PROJ = {
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o": "self_attn.o_proj",
        "up": "mlp.up_proj",
        "gate": "mlp.gate_proj",
        "down": "mlp.down_proj",
    }

    def layer_weights(self, k):
        """Dequantized bf16 GEMM input-B [K, out] per projection for layer k."""
        out = {}
        for nm, t in self._PROJ.items():
            wt = self.dequant(f"model.layers.{k}.{t}.weight")  # [out, K]
            out[nm] = np.ascontiguousarray(wt.T, dtype=bfloat16)  # [K, out]
        return out

    def layer_rms(self, k):
        """(input_layernorm, post_attention_layernorm) float32 for layer k."""
        return (
            self.bf16(f"model.layers.{k}.input_layernorm.weight"),
            self.bf16(f"model.layers.{k}.post_attention_layernorm.weight"),
        )

    def embed_norm_lmhead(self):
        """(embed_tokens [VOCAB,D], final_norm [D], lm_head [VOCAB,D]) float32."""
        embed = self.bf16("model.embed_tokens.weight")
        norm = self.bf16("model.norm.weight")
        lm = (
            self.dequant("lm_head.weight")
            if self._hdr["lm_head.weight"]["dtype"] == "Q4NX"
            else self.bf16("lm_head.weight")
        )
        return embed, norm, lm
