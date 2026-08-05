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
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# The Q4NX block packer/dequant reference lives in the standalone fused_decode
# example (this dir already depends on it for the decode path).
_FUSED_DECODE = str(Path(__file__).resolve().parents[2] / "fused_decode")
if _FUSED_DECODE not in sys.path:
    sys.path.insert(0, _FUSED_DECODE)
from proj_qmm_pack import (  # noqa: E402
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


# Pinned model.q4nx revisions for the known FastFlowLM repos. The Hub bundles are
# periodically re-packed to newer Q4NX block layouts (e.g. a 4096-i16/block form)
# that this loader's codec (2560 i16/block; see proj_qmm_pack) does not parse, so
# a bare `hf_hub_download` of the latest revision breaks with a reshape error.
# Pin the last revision that matches the codec above. Custom repo ids / local
# paths are always used as-is (unpinned).
_PINNED_Q4NX_REVISION = {
    "FastFlowLM/Llama-3.2-1B-NPU2": "d0c7f84ac9c5cf796db0fc8255afac42592d9db3",
    "FastFlowLM/Llama-3.2-3B-NPU2": "790271a87c7bb8158e52e9684a586a496d1fb1c9",
}


def resolve_q4nx_model(model):
    """Resolve `model` to a local model.q4nx path. `model` may be an HF repo id
    (contains '/'), a directory containing model.q4nx, or a direct file path.

    For a known FastFlowLM repo id, a compatible revision is pinned (see
    `_PINNED_Q4NX_REVISION`); other repo ids download their latest revision."""
    import os

    if os.path.isfile(model):
        return model
    if os.path.isdir(model):
        p = os.path.join(model, "model.q4nx")
        if os.path.isfile(p):
            return p
    # treat as an HF repo id
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        model, "model.q4nx", revision=_PINNED_Q4NX_REVISION.get(model)
    )


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
        """(embed_tokens [VOCAB,D], final_norm [D], lm_head [VOCAB,D]) float32.

        Llama-3.2-1B ties the LM head to the embedding (config
        tie_word_embeddings=true), so the LM head IS the full-precision embed
        matrix. The bundle also carries a separate Q4NX-quantized lm_head tensor;
        the tied fp embed is used instead (lossless, matches HF)."""
        embed = self.bf16("model.embed_tokens.weight")
        norm = self.bf16("model.norm.weight")
        return embed, norm, embed
