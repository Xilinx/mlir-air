# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader + dequant for the Gemma3-4B (text) Q4NX example.
#
# Mirrors llama32_1b_q4nx_weights.py (same Q4NX codec, reuses proj_qmm_pack) and
# adds the Gemma-specific host weight-prep deltas confirmed against FLM's built
# reference (FLM_Xclbin/Gemma3/gemma_npu_bin):
#   - FOUR RMSNorms per layer: input / post_attention / pre_feedforward /
#     post_feedforward (kernel does plain norm*w). The Gemma (1+weight) convention
#     is ALREADY folded in the q4nx bundle (it is a passthrough of
#     gemma-3-4b-it-Q4_1.gguf, and llama.cpp's GGUF Gemma conversion folds +1 into
#     the *_norm weights), so they are fed AS-IS -- NOT re-folded here. eps = 1e-6.
#   - qk-norm: per-head RMSNorm weights q_norm/k_norm (length DH, GGUF 1+w folded),
#     packed after the cos/sin LUT into rope_w = [cos/sin(DH), q_norm(DH), k_norm(DH)].
#   - dual RoPE: local layers theta=10000; global layers (every 6th) theta=1e6 with
#     linear position scaling factor 8. Sliding-window pattern 5 local : 1 global.
#   - embedding scale x sqrt(hidden_size): applied at gather in the driver (the LM
#     head ties to embed_tokens but is NOT scaled), so this loader exposes
#     EMBED_SCALE and returns the unscaled tied matrix.
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# The Q4NX block packer/dequant reference lives in the standalone fused_decode
# example (this dir depends on it for the decode path), same as llama32_1b_q4nx.
_FUSED_DECODE = str(Path(__file__).resolve().parents[2] / "fused_decode")
if _FUSED_DECODE not in sys.path:
    sys.path.insert(0, _FUSED_DECODE)
from proj_qmm_pack import (  # noqa: E402
    ROW_BLOCK,
    COL_BLOCK,
    GROUP,
    N_GROUPS,
    PARALLEL,
    BLOCK_BF16,
)

# Gemma3-4B (text) dims (models/Gemma3-4B-NPU2/config.json).
D = 2560  # hidden_size
DH = 256  # head_dim
N_Q_HEADS = 8
N_KV_HEADS = 4
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 2 (GQA)
DQ = N_Q_HEADS * DH  # 2048  q proj out
DK = N_KV_HEADS * DH  # 1024  k proj out
DV = DK  # 1024
INTER = 10240  # mlp intermediate
NUM_LAYERS = 34
VOCAB = 262208
RMS_EPS = 1e-6

# Gemma normalizer: input embeddings are scaled by sqrt(hidden_size). Applied at
# the host embedding gather (NOT folded into the tied embed/lm_head matrix).
EMBED_SCALE = float(np.sqrt(D))

# Dual RoPE (Gemma3): local (sliding-window) layers vs global layers.
ROPE_LOCAL_THETA = 10000.0
ROPE_GLOBAL_THETA = 1000000.0
ROPE_GLOBAL_LINEAR_FACTOR = 8.0  # linear position scaling on the global layers
SLIDING_WINDOW = 1024
SLIDING_PATTERN = 6  # every 6th layer is global; the other 5 are local


def is_global_layer(layer_idx):
    """Gemma3 sliding-window pattern: 5 local : 1 global, global every 6th layer."""
    return (layer_idx + 1) % SLIDING_PATTERN == 0


def _bf(a):
    return a.astype(bfloat16).astype(np.float32)


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
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model, "model.q4nx")


# Row-group reorder (identical Q4NX codec to llama; w = scale*(q - min), 32x256
# blocks, parallel=16 -> 2 row-groups/block interleaved even/odd).
_G = ROW_BLOCK // PARALLEL  # 2
_EVEN = np.array(
    [g * PARALLEL + 2 * k for g in range(_G) for k in range(PARALLEL // 2)]
)
_ODD = _EVEN + 1


class Q4nxModel:
    """mmap + parse a model.q4nx safetensors file; vectorized Q4NX dequant.

    Identical codec/loader to the llama example; Gemma-specific accessors add the
    4-norm (1+w) fold, qk-norm weights, dual-theta rope LUT, and embed scale."""

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

    def dequant(self, name, M, K):
        """A Q4NX tensor dequantized to float32 [M, K] (w = scale*(q - min)).

        The Gemma bundle stores each Q4NX tensor block-major as [nb, block_bytes]
        (nb = (M/32)*(K/256), block = 256 scales + 256 mins + 32*256 nibbles =
        2560 int16), so the logical [M,K] must be supplied by the caller (unlike
        the llama bundle whose header shape is [M,K])."""
        nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
        nb = nbi * nbj
        assert (
            self._hdr[name]["shape"][0] == nb
        ), f"{name}: header nb={self._hdr[name]['shape'][0]} != (M/32)*(K/256)={nb}"
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
        # Q4NX is additive (GGUF Q4_1): w = scale*q + min, min = the negative
        # additive offset (min/scale ~ -7.4). This matches the AIR kernel q4_k.h
        # (c += (q*b)*scale; c += min*sum(b) => w = scale*q + min). Verified by an
        # exact round-trip against FLM's converter _pack_q4nx.
        w = np.repeat(sc.transpose(0, 2, 1), GROUP, axis=2) * q + np.repeat(
            mn.transpose(0, 2, 1), GROUP, axis=2
        )
        return (
            w.reshape(nbi, nbj, ROW_BLOCK, COL_BLOCK)
            .transpose(0, 2, 1, 3)
            .reshape(M, K)
        )

    # name -> (tensor suffix, out, in) logical dims (Gemma bundle is block-major).
    _PROJ = {
        "q": ("self_attn.q_proj", DQ, D),
        "k": ("self_attn.k_proj", DK, D),
        "v": ("self_attn.v_proj", DV, D),
        "o": ("self_attn.o_proj", D, DQ),
        "up": ("mlp.up_proj", INTER, D),
        "gate": ("mlp.gate_proj", INTER, D),
        "down": ("mlp.down_proj", D, INTER),
    }

    def layer_weights(self, k):
        """Dequantized bf16 GEMM input-B [K, out] per projection for layer k."""
        out = {}
        for nm, (t, M, K) in self._PROJ.items():
            wt = self.dequant(f"model.layers.{k}.{t}.weight", M, K)  # [out, K]
            out[nm] = np.ascontiguousarray(wt.T, dtype=bfloat16)  # [K, out]
        return out

    # NOTE on the Gemma (1+weight) convention: the FLM kernel does plain norm*w,
    # and the q4nx bundle is a passthrough of gemma-3-4b-it-Q4_1.gguf. llama.cpp's
    # GGUF conversion for Gemma ALREADY folds +1 into every *_norm weight, so the
    # stored q4nx norm weights are ALREADY (1+w). Feed them AS-IS -- adding +1 here
    # would double-fold. (Confirmed by the stored norm mean ~8, not ~0.)
    def layer_rms(self, k):
        """The FOUR Gemma norm weights for layer k (input, post_attention,
        pre_feedforward, post_feedforward), fed as-is (GGUF already folded 1+w)."""
        names = [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        return tuple(
            self.bf16(f"model.layers.{k}.{n}.weight").astype(np.float32) for n in names
        )

    def layer_qk_norm(self, k):
        """Per-head qk-norm weights (q_norm, k_norm), length DH, as-is (GGUF 1+w)."""
        qn = self.bf16(f"model.layers.{k}.self_attn.q_norm.weight").astype(np.float32)
        kn = self.bf16(f"model.layers.{k}.self_attn.k_norm.weight").astype(np.float32)
        return qn, kn

    def embed_norm_lmhead(self):
        """(embed_in [VOCAB,D], final_norm [D], lm_head [VOCAB,D]) float32.

        The Gemma bundle stores the two tied matrices SEPARATELY, already resolving
        the Gemma embed-scale convention (verified against FLM's gemma3.py converter):
          - model.embed_tokens.weight (bf16) is the input embedding ALREADY scaled by
            sqrt(hidden_size) -> gather it AS-IS (do NOT re-apply EMBED_SCALE).
          - lm_head.weight (Q4NX) is the RAW (unscaled) tied matrix -> use for logits.
        final_norm carries the (1+w) fold (fed as-is)."""
        embed_in = self.bf16("model.embed_tokens.weight")  # already * sqrt(hidden)
        norm = self.bf16("model.norm.weight").astype(np.float32)  # GGUF already 1+w
        lm_head = self.dequant("lm_head.weight", VOCAB, D)  # raw embed (Q4NX additive)
        return embed_in, norm, lm_head


def generate_rope_lut(position, theta, linear_factor=1.0):
    """Half-split (NEOX) RoPE cos/sin LUT for a single decode position, matching
    fused_decode/kernels/rope.cc apply_rope: rope_w[:DH] = [cos(DH/2) ++ sin(DH/2)].

    Gemma3 dual-theta: pass ROPE_LOCAL_THETA (local layers) or ROPE_GLOBAL_THETA
    with linear_factor=ROPE_GLOBAL_LINEAR_FACTOR (global layers)."""
    half = DH // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    p = position / linear_factor
    ang = p * inv_freq
    cos = np.cos(ang).astype(bfloat16).astype(np.float32)
    sin = np.sin(ang).astype(bfloat16).astype(np.float32)
    return np.concatenate([cos, sin]).astype(np.float32)  # length DH


def rope_w_layer(position, layer_idx, qn, kn):
    """Assemble the 3*DH rope weight buffer for a (position, layer): the dual-theta
    cos/sin LUT (local vs global by the sliding-window pattern) followed by the
    (1+w)-folded q_norm and k_norm. Matches rope.cc rope_w = [cos/sin, q_norm, k_norm].
    """
    if is_global_layer(layer_idx):
        lut = generate_rope_lut(position, ROPE_GLOBAL_THETA, ROPE_GLOBAL_LINEAR_FACTOR)
    else:
        lut = generate_rope_lut(position, ROPE_LOCAL_THETA)
    return np.concatenate([lut, qn, kn]).astype(bfloat16)


# ---------------------------------------------------------------------------
# NumPy reference forward (the decode gate's golden + KV seed).
#
# This is the Gemma3 analogue of the numpy layer reference the Llama-1B decode
# was brought up against (llama32_1b_q4nx_weights.forward_layer). It runs the
# prompt through the 34 layers to produce (a) per-layer roped-K / raw-V for the
# AIR decode KV seed and (b) the per-position logits for the HF-cross-check
# golden. Intermediates are bf16-rounded at each GEMM output to approximate the
# device datapath (the seeded KV must be close to what the device would compute).
# Math is matched to the kernels: rms_residual.cc (norm*w, eps 1e-6, plain -- the
# 1+w fold is already in the weights), rope.cc (qk_norm RMSNorm(DH) before
# half-split NEOX apply_rope), gelu.cc (gelu_tanh), attn scale 1/sqrt(256).
ATTN_SCALE = 0.0625  # query_pre_attn_scalar=256 -> 1/sqrt(256)
_DH_2 = DH // 2


def _rmsnorm(x, w, eps=RMS_EPS):
    """RMSNorm over the last axis: x/sqrt(mean(x^2)+eps) * w (w fed as-is; the
    Gemma 1+w fold is already in the stored weights)."""
    x = x.astype(np.float32)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return _bf(x * (1.0 / np.sqrt(ms + eps)) * w.astype(np.float32))


def _apply_rope_halfsplit(x, cos, sin):
    """Half-split NEOX rope on a [..., DH] vector, matching rope.cc apply_rope:
    y[:DH/2] = x1*cos - x2*sin ; y[DH/2:] = x1*sin + x2*cos (cos/sin length DH/2)."""
    x1 = x[..., :_DH_2]
    x2 = x[..., _DH_2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _gelu_tanh(x):
    """gelu_pytorch_tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3)))."""
    x = x.astype(np.float32)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def forward_prompt(model, prompt_ids):
    """Run the prompt through all NUM_LAYERS Gemma3 layers in numpy.

    Returns (Kc, Vc, logits):
      Kc, Vc : float32 [NUM_LAYERS, P, DK] -- per-layer ROPED-K / RAW-V (the AIR
               decode KV seed; heads concatenated in CU order [h0|h1|h2|h3]).
      logits : float32 [P, VOCAB] -- tied-lm-head logits per position (final-norm
               applied); logits[-1] argmax is the greedy first token (HF golden)."""
    embed, final_norm, lm_head = model.embed_norm_lmhead()
    ids = np.asarray(prompt_ids, dtype=np.int64)
    P = ids.shape[0]
    # embed is ALREADY scaled by sqrt(hidden) in the bundle -> gather as-is.
    x = _bf(embed[ids].astype(np.float32))  # [P, D]

    Kc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    Vc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    pos = np.arange(P)

    for L in range(NUM_LAYERS):
        w = model.layer_weights(L)  # {q,k,v,o,gate,up,down} [K, out] bf16
        n_in, n_pa, n_pf, n_pf2 = model.layer_rms(L)  # 4 norms [D]
        qn, kn = model.layer_qk_norm(L)  # per-head norms [DH]

        # ---- attention sublayer (pre-norm in, post-norm out, residual) ----
        residual = x
        h = _rmsnorm(x, n_in)  # input_layernorm
        q = _bf(h @ w["q"]).reshape(P, N_Q_HEADS, DH).astype(np.float32)
        k = _bf(h @ w["k"]).reshape(P, N_KV_HEADS, DH).astype(np.float32)
        v = _bf(h @ w["v"]).reshape(P, N_KV_HEADS, DH).astype(np.float32)

        # qk-norm (RMSNorm over DH per head) then dual-theta half-split rope.
        theta = ROPE_GLOBAL_THETA if is_global_layer(L) else ROPE_LOCAL_THETA
        lf = ROPE_GLOBAL_LINEAR_FACTOR if is_global_layer(L) else 1.0
        p_scaled = pos / lf
        inv = 1.0 / (theta ** (np.arange(_DH_2) / _DH_2))
        ang = p_scaled[:, None] * inv[None, :]  # [P, DH/2]
        cos = np.cos(ang).astype(bfloat16).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(bfloat16).astype(np.float32)[:, None, :]
        q = _apply_rope_halfsplit(_rmsnorm(q, qn), cos, sin)
        k = _apply_rope_halfsplit(_rmsnorm(k, kn), cos, sin)
        Kc[L] = _bf(k).reshape(P, DK)
        Vc[L] = _bf(v).reshape(P, DK)

        # GQA attention: causal, plus sliding-window=1024 on local layers.
        win = None if is_global_layer(L) else SLIDING_WINDOW
        o = np.zeros((P, N_Q_HEADS, DH), np.float32)
        for hq in range(N_Q_HEADS):
            hk = hq // Q_PER_KV
            scores = (q[:, hq, :] @ k[:, hk, :].T) * ATTN_SCALE  # [P, P]
            mask = pos[None, :] > pos[:, None]  # causal (future)
            if win is not None:
                mask |= pos[None, :] <= (pos[:, None] - win)
            scores = np.where(mask, -1e30, scores)
            scores -= scores.max(axis=-1, keepdims=True)
            e = np.exp(scores)
            attn = e / e.sum(axis=-1, keepdims=True)
            o[:, hq, :] = attn @ v[:, hk, :]
        attn_out = _bf(o.reshape(P, DQ) @ w["o"])  # o_proj [P, D]
        attn_out = _rmsnorm(attn_out, n_pa)  # post_attention_layernorm
        x = _bf(residual + attn_out)

        # ---- MLP sublayer (pre-norm in, post-norm out, residual) ----
        residual = x
        h2 = _rmsnorm(x, n_pf)  # pre_feedforward_layernorm
        act = _bf(_gelu_tanh(_bf(h2 @ w["gate"])) * _bf(h2 @ w["up"]))
        down = _bf(act @ w["down"])  # down_proj [P, D]
        down = _rmsnorm(down, n_pf2)  # post_feedforward_layernorm
        x = _bf(residual + down)

    xf = _rmsnorm(x, final_norm)  # model.norm
    logits = xf @ lm_head.T.astype(np.float32)  # tied lm-head [P, VOCAB]
    return Kc, Vc, logits.astype(np.float32)
