# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader + dequant for the Qwen3-8B Q4NX example.
#
# Mirrors llama32_1b_q4nx_weights.py (same Q4NX codec, reuses proj_qmm_pack).
# Qwen3-8B is the LLAMA-shaped q4nx topology (ATTN_IMPL_2x4x1, DH=128,
# PAIR_ROWS=2, standard 2-norm pre-norm, SiLU GLU, single-theta RoPE) plus one
# Qwen3 delta:
#   - qk-norm: per-head RMSNorm weights q_norm/k_norm (length DH), packed after
#     the cos/sin LUT into rope_w = [cos/sin(DH), q_norm(DH), k_norm(DH)].
#
# NOT Gemma: no (1+w) norm fold, no embedding scale, no dual-theta / sliding
# window, and the lm_head is NOT tied. The q4nx norm weights were verified
# BYTE-IDENTICAL to the HF bf16 checkpoint (input_layernorm/post_attention/
# q_norm/model.norm all match), so they are plain `w` and are fed as-is.
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

# Qwen3-8B dims (FastFlowLM/Qwen3-8B-NPU2 config.json).
D = 4096  # hidden_size
DH = 128  # head_dim
N_Q_HEADS = 32
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4 (GQA)
DQ = N_Q_HEADS * DH  # 4096  q proj out (== D, so o_proj is square)
DK = N_KV_HEADS * DH  # 1024  k proj out
DV = DK  # 1024
INTER = 12288  # mlp intermediate
NUM_LAYERS = 36
VOCAB = 151936
RMS_EPS = 1e-6

# Single-theta RoPE (no Gemma dual-theta / linear scaling / sliding window).
ROPE_THETA = 1000000.0


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

    This bundle has only ever been I8-packed, so `shape` is
    [n_blocks, bytes_per_block] and `dequant` takes the logical (M, K) from the
    caller. llama32_1b_q4nx_weights.Q4nxModel now handles that header too, plus
    the older Q4NX one; the codec half of this class duplicates it and could
    fold in. Kept separate here because the accessors below are Qwen3-specific:
    the 4-norm (1+w) fold, qk-norm weights, dual-theta rope LUT, embed scale."""

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
        """A Q4NX tensor dequantized to float32 [M, K] (w = scale*q + min).

        The FLM bundle stores each Q4NX tensor block-major as [nb, block_bytes]
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

    # name -> (tensor suffix, out, in) logical dims (bundle is block-major).
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

    def layer_rms(self, k):
        """The two Qwen3 pre-norm weights for layer k (input, post_attention).
        Plain `w` -- verified byte-identical to the HF bf16 checkpoint, so there
        is no Gemma-style (1+w) fold to undo."""
        return tuple(
            self.bf16(f"model.layers.{k}.{n}.weight").astype(np.float32)
            for n in ("input_layernorm", "post_attention_layernorm")
        )

    def layer_qk_norm(self, k):
        """Per-head qk-norm weights (q_norm, k_norm), length DH, plain w."""
        qn = self.bf16(f"model.layers.{k}.self_attn.q_norm.weight").astype(np.float32)
        kn = self.bf16(f"model.layers.{k}.self_attn.k_norm.weight").astype(np.float32)
        return qn, kn

    def embed_norm_lmhead(self):
        """(embed_in [VOCAB,D] bf16, final_norm [D] f32, lm_head [VOCAB,D] f32).

        Qwen3-8B does NOT tie: the bundle carries a bf16 `model.embed_tokens`
        for the input gather and a SEPARATE Q4NX-packed `lm_head` (75968x5120
        int8 = 151936x4096 at 0.625 B/param) for the logits. No embedding scale
        (that is a Gemma convention)."""
        embed_in = self.bf16("model.embed_tokens.weight")
        norm = self.bf16("model.norm.weight").astype(np.float32)
        lm_head = self.dequant("lm_head.weight", VOCAB, D)
        return embed_in, norm, lm_head


def _proj_dims(c):
    """Logical (out, K) per projection, for I8-packed Q4NX headers."""
    dq = c.n_heads * c.head_dim
    dkv = c.n_kv_heads * c.head_dim
    return {
        "q": (dq, c.emb_dim),
        "k": (dkv, c.emb_dim),
        "v": (dkv, c.emb_dim),
        "o": (c.emb_dim, dq),
        "gate": (c.hidden_dim, c.emb_dim),
        "up": (c.hidden_dim, c.emb_dim),
        "down": (c.emb_dim, c.hidden_dim),
    }


def load_q4nx_weights(model_source, config=None):
    """Load Qwen3-8B weights from a `model.q4nx` bundle into a LlamaWeights.

    Assembles the dequantized bf16 matrices into the `qwen3_4b` containers so the
    config-parameterized qwen3_4b prefill driver runs unchanged -- Q4NX only
    changes the weight source. Unlike llama, Qwen3 carries per-head qk-norm
    weights, and 8B does NOT tie embeddings (lm_head is its own Q4NX tensor).
    """
    _QWEN3_4B = str(Path(__file__).resolve().parents[1] / "qwen3_4b")
    if _QWEN3_4B not in sys.path:
        sys.path.insert(0, _QWEN3_4B)
    from qwen3_4b_weights import LayerWeights, LlamaWeights

    if config is None:
        config = qwen3_8b_config()

    qm = Q4nxModel(model_source)
    dims = _proj_dims(config)
    assert dims == {
        k: (m, kk) for k, (_, m, kk) in Q4nxModel._PROJ.items()
    }, "config dims disagree with the bundle's projection table"
    embed, norm, lm_head = qm.embed_norm_lmhead()

    layers = []
    for k in range(config.n_layers):
        w = qm.layer_weights(k)  # each [K, out] bf16
        rms_in, rms_post = qm.layer_rms(k)
        qn, kn = qm.layer_qk_norm(k)
        layers.append(
            LayerWeights(
                attn_norm=np.asarray(rms_in, bfloat16),
                wq=np.asarray(w["q"], bfloat16),
                wk=np.asarray(w["k"], bfloat16),
                wv=np.asarray(w["v"], bfloat16),
                wo=np.asarray(w["o"], bfloat16),
                ffn_norm=np.asarray(rms_post, bfloat16),
                w_gate=np.asarray(w["gate"], bfloat16),
                w_up=np.asarray(w["up"], bfloat16),
                w_down=np.asarray(w["down"], bfloat16),
                q_norm=np.asarray(qn, bfloat16),
                k_norm=np.asarray(kn, bfloat16),
            )
        )

    return LlamaWeights(
        embed_table=np.asarray(embed, bfloat16),
        layers=layers,
        final_norm=np.asarray(norm, bfloat16),
        lm_head=np.asarray(lm_head, bfloat16),
    )


def qwen3_8b_config(n_layers=NUM_LAYERS):
    """The qwen3_4b LlamaConfig re-parameterized to Qwen3-8B."""
    _QWEN3_4B = str(Path(__file__).resolve().parents[1] / "qwen3_4b")
    if _QWEN3_4B not in sys.path:
        sys.path.insert(0, _QWEN3_4B)
    from qwen3_4b_weights import LlamaConfig

    return LlamaConfig(
        n_layers=n_layers,
        emb_dim=D,
        n_heads=N_Q_HEADS,
        head_dim=DH,
        n_kv_heads=N_KV_HEADS,
        hidden_dim=INTER,
        vocab_size=VOCAB,
        rope_base=ROPE_THETA,
        tie_word_embeddings=False,  # 8B ships a separate Q4NX lm_head
    )


def generate_rope_lut(position, theta=ROPE_THETA):
    """Half-split (NEOX) RoPE cos/sin LUT for a single decode position, matching
    fused_decode/kernels/rope.cc apply_rope: rope_w[:DH] = [cos(DH/2) ++ sin(DH/2)]."""
    half = DH // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    ang = position * inv_freq
    cos = np.cos(ang).astype(bfloat16).astype(np.float32)
    sin = np.sin(ang).astype(bfloat16).astype(np.float32)
    return np.concatenate([cos, sin]).astype(np.float32)  # length DH


def rope_w_layer(position, layer_idx, qn, kn):
    """The 3*DH rope weight buffer for a position: cos/sin LUT followed by the
    per-head q_norm and k_norm. Matches rope.cc rope_w = [cos/sin, q_norm, k_norm].
    `layer_idx` is unused (single-theta) but kept for driver call-site parity."""
    del layer_idx
    return np.concatenate([generate_rope_lut(position), qn, kn]).astype(bfloat16)


# ---------------------------------------------------------------------------
# NumPy reference forward (the decode gate's golden + KV seed).
#
# Runs the prompt through the 36 layers to produce (a) per-layer roped-K / raw-V
# for the AIR decode KV seed and (b) the per-position logits for the HF-cross-check
# golden. Intermediates are bf16-rounded at each GEMM output to approximate the
# device datapath (the seeded KV must be close to what the device would compute).
# Math is matched to the kernels: rms_residual.cc (norm*w, eps 1e-6, plain w),
# rope.cc (qk_norm RMSNorm(DH) before half-split NEOX apply_rope), glu.cc (SiLU),
# attn scale 1/sqrt(128).
ATTN_SCALE = 0.08838834764831843  # 1/sqrt(128)
_DH_2 = DH // 2


def _rmsnorm(x, w, eps=RMS_EPS):
    """RMSNorm over the last axis: x/sqrt(mean(x^2)+eps) * w (plain w)."""
    x = x.astype(np.float32)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return _bf(x * (1.0 / np.sqrt(ms + eps)) * w.astype(np.float32))


def _apply_rope_halfsplit(x, cos, sin):
    """Half-split NEOX rope on a [..., DH] vector, matching rope.cc apply_rope:
    y[:DH/2] = x1*cos - x2*sin ; y[DH/2:] = x1*sin + x2*cos (cos/sin length DH/2)."""
    x1 = x[..., :_DH_2]
    x2 = x[..., _DH_2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _silu(x):
    """SiLU / swish: x * sigmoid(x)."""
    x = x.astype(np.float32)
    return x / (1.0 + np.exp(-x))


def forward_prompt(model, prompt_ids):
    """Run the prompt through all NUM_LAYERS Qwen3-8B layers in numpy.

    Returns (Kc, Vc, logits):
      Kc, Vc : float32 [NUM_LAYERS, P, DK] -- per-layer ROPED-K / RAW-V (the AIR
               decode KV seed; heads concatenated in CU order [h0|h1|h2|h3]).
      logits : float32 [P, VOCAB] -- lm-head logits per position (final-norm
               applied); logits[-1] argmax is the greedy first token (HF golden)."""
    embed, final_norm, lm_head = model.embed_norm_lmhead()
    ids = np.asarray(prompt_ids, dtype=np.int64)
    P = ids.shape[0]
    x = _bf(embed[ids].astype(np.float32))  # [P, D]  (no Gemma embed scale)

    Kc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    Vc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    pos = np.arange(P)

    for L in range(NUM_LAYERS):
        w = model.layer_weights(L)  # {q,k,v,o,gate,up,down} [K, out] bf16
        n_in, n_pa = model.layer_rms(L)  # 2 norms [D]
        qn, kn = model.layer_qk_norm(L)  # per-head norms [DH]

        # ---- attention sublayer (pre-norm in, post-norm out, residual) ----
        residual = x
        h = _rmsnorm(x, n_in)  # input_layernorm
        q = _bf(h @ w["q"]).reshape(P, N_Q_HEADS, DH).astype(np.float32)
        k = _bf(h @ w["k"]).reshape(P, N_KV_HEADS, DH).astype(np.float32)
        v = _bf(h @ w["v"]).reshape(P, N_KV_HEADS, DH).astype(np.float32)

        # qk-norm (RMSNorm over DH per head) then half-split rope.
        inv = 1.0 / (ROPE_THETA ** (np.arange(_DH_2) / _DH_2))
        ang = pos[:, None] * inv[None, :]  # [P, DH/2]
        cos = np.cos(ang).astype(bfloat16).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(bfloat16).astype(np.float32)[:, None, :]
        q = _apply_rope_halfsplit(_rmsnorm(q, qn), cos, sin)
        k = _apply_rope_halfsplit(_rmsnorm(k, kn), cos, sin)
        Kc[L] = _bf(k).reshape(P, DK)
        Vc[L] = _bf(v).reshape(P, DK)

        # GQA attention: causal (Qwen3 has no sliding window).
        o = np.zeros((P, N_Q_HEADS, DH), np.float32)
        for hq in range(N_Q_HEADS):
            hk = hq // Q_PER_KV
            scores = (q[:, hq, :] @ k[:, hk, :].T) * ATTN_SCALE  # [P, P]
            mask = pos[None, :] > pos[:, None]  # causal (future)
            scores = np.where(mask, -1e30, scores)
            scores -= scores.max(axis=-1, keepdims=True)
            e = np.exp(scores)
            attn = e / e.sum(axis=-1, keepdims=True)
            o[:, hq, :] = attn @ v[:, hk, :]
        attn_out = _bf(o.reshape(P, DQ) @ w["o"])  # o_proj [P, D]
        x = _bf(residual + attn_out)

        # ---- MLP sublayer (standard pre-norm, no post-norm) ----
        residual = x
        h2 = _rmsnorm(x, n_pa)  # post_attention_layernorm == FFN pre-norm
        act = _bf(_silu(_bf(h2 @ w["gate"])) * _bf(h2 @ w["up"]))
        x = _bf(residual + _bf(act @ w["down"]))

    xf = _rmsnorm(x, final_norm)  # model.norm
    logits = xf @ lm_head.T.astype(np.float32)  # untied lm-head [P, VOCAB]
    return Kc, Vc, logits.astype(np.float32)
