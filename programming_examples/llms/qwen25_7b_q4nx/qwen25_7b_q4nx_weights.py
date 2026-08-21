# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader + dequant for the Qwen2.5-7B Q4NX example.
#
# Same Q4NX codec as the llama / Qwen3-8B q4nx examples (reuses proj_qmm_pack),
# and the same LLAMA-shaped topology (DH=128, standard 2-norm pre-norm, SiLU GLU,
# single-theta half-split RoPE). Two Qwen2.5 deltas:
#   - q/k/v projection BIAS (bf16, unquantized), packed after the cos/sin LUT
#     into rope_w = [cos/sin(DH), q_bias(DQ), k_bias(DK), v_bias(DV)]; the kernel
#     adds it in place before RoPE (fused_decode/kernels/rope.cc add_q_k_v_bias).
#   - no qk-norm (that is the Qwen3 convention).
#
# The lm_head is NOT tied (unlike Qwen2.5-3B), so the bundle ships a separate
# Q4NX-packed lm_head at 152064x3584.
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# The Q4NX block packer/dequant reference lives in the standalone fused_decode
# example (this dir depends on it for the decode path), same as qwen3_8b_q4nx.
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

# Qwen2.5-7B-Instruct dims (HF config.json; matches FLM's
# generic_decoding_layer/models/qwen2.5-7b.h).
D = 3584  # hidden_size
DH = 128  # head_dim
N_Q_HEADS = 28
N_KV_HEADS = 4
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 7 (GQA)
DQ = N_Q_HEADS * DH  # 3584  q proj out (== D, so o_proj is square)
DK = N_KV_HEADS * DH  # 512   k proj out
DV = DK  # 512
INTER = 18944  # mlp intermediate
NUM_LAYERS = 28
VOCAB = 152064
RMS_EPS = 1e-6

# Single-theta RoPE. NOTE 1e6, not the 1e7 of the -1M long-context variant.
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


# Row-group reorder (identical Q4NX codec to llama; w = scale*q + min, 32x256
# blocks, parallel=16 -> 2 row-groups/block interleaved even/odd).
_G = ROW_BLOCK // PARALLEL  # 2
_EVEN = np.array(
    [g * PARALLEL + 2 * k for g in range(_G) for k in range(PARALLEL // 2)]
)
_ODD = _EVEN + 1


class Q4nxModel:
    """mmap + parse a model.q4nx safetensors file; vectorized Q4NX dequant.

    The codec half mirrors qwen3_8b_q4nx_weights.Q4nxModel (I8-packed bundle:
    `shape` is [n_blocks, bytes_per_block], so the logical (M, K) comes from the
    caller). The accessors are Qwen2.5-specific: a 2-norm pre-norm layer with
    plain weights, q/k/v projection biases, no qk-norm, and an untied Q4NX
    lm_head."""

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
        2560 int16), so the logical [M,K] must be supplied by the caller."""
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

    def proj_fp(self, k, nm):
        """Full-precision [out, K] for one projection (the requant cache's input)."""
        t, M, K = self._PROJ[nm]
        return self.dequant(f"model.layers.{k}.{t}.weight", M, K)

    def lm_head_fp(self):
        """Full-precision lm_head [VOCAB, D]."""
        return self.dequant("lm_head.weight", VOCAB, D)

    def layer_weights(self, k):
        """Dequantized bf16 GEMM input-B [K, out] per projection for layer k."""
        out = {}
        for nm in self._PROJ:
            wt = self.proj_fp(k, nm)  # [out, K]
            out[nm] = np.ascontiguousarray(wt.T, dtype=bfloat16)  # [K, out]
        return out

    def layer_rms(self, k):
        """The two Qwen2.5 pre-norm weights for layer k (input, post_attention),
        plain `w` -- no Gemma-style (1+w) fold."""
        return tuple(
            self.bf16(f"model.layers.{k}.{n}.weight").astype(np.float32)
            for n in ("input_layernorm", "post_attention_layernorm")
        )

    def layer_qkv_bias(self, k):
        """The q/k/v projection biases (bf16, unquantized) for layer k, as
        (bq[DQ], bk[DK], bv[DV]) float32."""
        return tuple(
            self.bf16(f"model.layers.{k}.self_attn.{n}_proj.bias").astype(np.float32)
            for n in ("q", "k", "v")
        )

    def embed_norm_lmhead(self):
        """(embed_in [VOCAB,D] bf16, final_norm [D] f32, lm_head [VOCAB,D] f32).

        Qwen2.5-7B does NOT tie (unlike the 3B): the bundle carries a bf16
        `model.embed_tokens` for the input gather and a SEPARATE Q4NX-packed
        `lm_head` (66528x5120 int8 = 152064x3584 at 0.625 B/param)."""
        embed_in = self.bf16("model.embed_tokens.weight")
        norm = self.bf16("model.norm.weight").astype(np.float32)
        lm_head = self.dequant("lm_head.weight", VOCAB, D)
        return embed_in, norm, lm_head


class HFQ4nxModel:
    """A full-precision HF checkpoint, rounded onto the Q4NX grid on load.

    The DEFAULT source. FastFlowLM publishes no Qwen2.5-7B NPU2 bundle (their
    Qwen2.5 line stops at 3B), so there is nothing to download in the Q4NX
    packing the AIR decode expects -- and their converter's Qwen2.5 output uses a
    custom nibble interleave (q4nx/gguf_tensor.py transform_nibble_layout) that
    the Llama/Qwen3 bundles do not, so it is not interchangeable either.

    Quantizing here instead makes the example self-contained and reproducible
    from an ungated upstream checkpoint. Same shape of decision as the
    Qwen2.5-3B Q4_0 sibling, which quantizes from HF for the analogous reason
    (its device codec disagrees with the bundle's).

    Every projection goes through `quantize_dequantize_q4nx` -- the same
    quantizer the decode cascade cache uses -- so prefill and decode see
    BIT-IDENTICAL weights. Norms and the q/k/v biases stay bf16 (the reference
    design does not quantize them either)."""

    _PROJ = Q4nxModel._PROJ

    def __init__(self, model):
        _FD = str(Path(__file__).resolve().parents[2] / "fused_decode")
        if _FD not in sys.path:
            sys.path.insert(0, _FD)
        from qwen25_3b_requant import HFModel

        self._hf = HFModel(model)

    def has(self, name):
        return self._hf.has(name)

    def bf16(self, name):
        return self._hf.bf16(name)

    def proj_fp(self, k, nm):
        """Q4NX-rounded [out, K] for one projection of layer k."""
        from qwen25_7b_q4nx_requant import quantize_dequantize_q4nx

        t, _M, _K = self._PROJ[nm]
        return quantize_dequantize_q4nx(self.bf16(f"model.layers.{k}.{t}.weight"))

    def lm_head_fp(self):
        from qwen25_7b_q4nx_requant import quantize_dequantize_q4nx

        return quantize_dequantize_q4nx(self.bf16("lm_head.weight"))

    def layer_weights(self, k):
        """Q4NX-rounded bf16 GEMM input-B [K, out] per projection for layer k."""
        return {
            nm: np.ascontiguousarray(self.proj_fp(k, nm).T, dtype=bfloat16)
            for nm in self._PROJ
        }

    def layer_rms(self, k):
        return tuple(
            self.bf16(f"model.layers.{k}.{n}.weight").astype(np.float32)
            for n in ("input_layernorm", "post_attention_layernorm")
        )

    def layer_qkv_bias(self, k):
        return tuple(
            self.bf16(f"model.layers.{k}.self_attn.{n}_proj.bias").astype(np.float32)
            for n in ("q", "k", "v")
        )

    def embed_norm_lmhead(self):
        return (
            self.bf16("model.embed_tokens.weight"),
            self.bf16("model.norm.weight").astype(np.float32),
            self.lm_head_fp(),
        )


def open_weight_source(model):
    """Q4nxModel for a `model.q4nx` bundle, HFQ4nxModel for anything else.

    Only a path that actually resolves to a model.q4nx takes the bundle path;
    an HF repo id or a checkpoint directory gets quantized on load."""
    import os

    if os.path.isfile(model) and model.endswith(".q4nx"):
        return Q4nxModel(model)
    if os.path.isdir(model) and os.path.isfile(os.path.join(model, "model.q4nx")):
        return Q4nxModel(model)
    return HFQ4nxModel(model)


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
    """Load Qwen2.5-7B weights into a LlamaWeights, Q4NX-rounded.

    `model_source` is an HF repo id / checkpoint dir (quantized on load, the
    default) or a `model.q4nx` bundle -- see open_weight_source. Either way the
    matrices land in the `qwen25_3b` containers, so the config-parameterized
    qwen25_3b prefill builders run unchanged; those containers already carry the
    q/k/v bias fields (bq/bk/bv) the rms_qkv_bias_rope kernel needs."""
    _QWEN25_3B = str(Path(__file__).resolve().parents[1] / "qwen25_3b")
    if _QWEN25_3B not in sys.path:
        sys.path.insert(0, _QWEN25_3B)
    from qwen25_3b_weights import LayerWeights, LlamaWeights

    if config is None:
        config = qwen25_7b_config()

    qm = open_weight_source(model_source)
    dims = _proj_dims(config)
    assert dims == {
        k: (m, kk) for k, (_, m, kk) in Q4nxModel._PROJ.items()
    }, "config dims disagree with the bundle's projection table"
    embed, norm, lm_head = qm.embed_norm_lmhead()

    layers = []
    for k in range(config.n_layers):
        w = qm.layer_weights(k)  # each [K, out] bf16
        rms_in, rms_post = qm.layer_rms(k)
        bq, bk, bv = qm.layer_qkv_bias(k)
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
                bq=np.asarray(bq, bfloat16),
                bk=np.asarray(bk, bfloat16),
                bv=np.asarray(bv, bfloat16),
            )
        )

    return LlamaWeights(
        embed_table=np.asarray(embed, bfloat16),
        layers=layers,
        final_norm=np.asarray(norm, bfloat16),
        lm_head=np.asarray(lm_head, bfloat16),
    )


def qwen25_7b_config(n_layers=NUM_LAYERS):
    """The qwen25_3b LlamaConfig re-parameterized to Qwen2.5-7B."""
    _QWEN25_3B = str(Path(__file__).resolve().parents[1] / "qwen25_3b")
    if _QWEN25_3B not in sys.path:
        sys.path.insert(0, _QWEN25_3B)
    from qwen25_3b_weights import LlamaConfig

    return LlamaConfig(
        n_layers=n_layers,
        emb_dim=D,
        n_heads=N_Q_HEADS,
        head_dim=DH,
        n_kv_heads=N_KV_HEADS,
        hidden_dim=INTER,
        vocab_size=VOCAB,
        rope_base=ROPE_THETA,
        qkv_bias=True,
        qk_norm=False,
        tie_word_embeddings=False,  # 7B ships a separate Q4NX lm_head
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


def rope_w_layer(position, layer_idx, bq, bk, bv):
    """The DH+DQ+DK+DV rope weight buffer for a position: the cos/sin LUT followed
    by the q/k/v bias slab. Matches rope.cc, which reads the bias at rope_w+DH and
    adds it to the concatenated qkv before applying RoPE.
    `layer_idx` is unused (single-theta) but kept for driver call-site parity."""
    del layer_idx
    return np.concatenate([generate_rope_lut(position), bq, bk, bv]).astype(bfloat16)


# ---------------------------------------------------------------------------
# NumPy reference forward (the decode gate's golden + KV seed).
#
# Runs the prompt through the 28 layers to produce (a) per-layer roped-K / raw-V
# for the AIR decode KV seed and (b) the per-position logits for the HF-cross-check
# golden. Intermediates are bf16-rounded at each GEMM output to approximate the
# device datapath. Math is matched to the kernels: rms_residual.cc (norm*w, eps
# 1e-6, plain w), rope.cc (q/k/v bias add, then half-split NEOX apply_rope, no
# qk-norm), glu.cc (SiLU), attn scale 1/sqrt(128).
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
    """Run the prompt through all NUM_LAYERS Qwen2.5-7B layers in numpy.

    Returns (Kc, Vc, logits):
      Kc, Vc : float32 [NUM_LAYERS, P, DK] -- per-layer ROPED-K / RAW-V (the AIR
               decode KV seed; heads concatenated in CU order [h0|h1|h2|h3]).
      logits : float32 [P, VOCAB] -- lm-head logits per position (final-norm
               applied); logits[-1] argmax is the greedy first token (HF golden)."""
    embed, final_norm, lm_head = model.embed_norm_lmhead()
    ids = np.asarray(prompt_ids, dtype=np.int64)
    P = ids.shape[0]
    x = _bf(embed[ids].astype(np.float32))  # [P, D]

    Kc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    Vc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    pos = np.arange(P)

    inv = 1.0 / (ROPE_THETA ** (np.arange(_DH_2) / _DH_2))
    ang = pos[:, None] * inv[None, :]  # [P, DH/2]
    cos = np.cos(ang).astype(bfloat16).astype(np.float32)[:, None, :]
    sin = np.sin(ang).astype(bfloat16).astype(np.float32)[:, None, :]

    for L in range(NUM_LAYERS):
        w = model.layer_weights(L)  # {q,k,v,o,gate,up,down} [K, out] bf16
        n_in, n_pa = model.layer_rms(L)  # 2 norms [D]
        bq, bk, bv = model.layer_qkv_bias(L)  # q/k/v bias

        # ---- attention sublayer (pre-norm in, residual) ----
        residual = x
        h = _rmsnorm(x, n_in)  # input_layernorm
        # Bias is added to the raw projection output, before RoPE (rope.cc order).
        q = _bf(_bf(h @ w["q"]) + bq).reshape(P, N_Q_HEADS, DH).astype(np.float32)
        k = _bf(_bf(h @ w["k"]) + bk).reshape(P, N_KV_HEADS, DH).astype(np.float32)
        v = _bf(_bf(h @ w["v"]) + bv).reshape(P, N_KV_HEADS, DH).astype(np.float32)

        # Half-split rope (no qk-norm -- that is Qwen3).
        q = _apply_rope_halfsplit(q, cos, sin)
        k = _apply_rope_halfsplit(k, cos, sin)
        Kc[L] = _bf(k).reshape(P, DK)
        Vc[L] = _bf(v).reshape(P, DK)

        # GQA attention: causal (Qwen2.5-7B has no sliding window).
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
