# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Weight loader for the Qwen2.5-3B Q4 example: FastFlowLM's shipped model.q4nx.
#
# Same .q4nx CONTAINER as the llama / Qwen2.5-7B examples (32x256 blocks of
# 256 scales | 256 mins | 4096 nibbles, reuses proj_qmm_pack) but the Qwen2.5-3B
# bundle stores the Q4_0 variant of it, which is why the kernels are built
# -DQ4_0 (fused_decode/models/qwen2.5-3b.h):
#   - the nibbles are SIGNED two's-complement int4 and w = q*scale, symmetric;
#     the `mins` field is unused and is exactly zero in EVERY tensor of the
#     bundle (checked, all 36 layers + lm_head). Reading it as the affine Q4NX
#     codec gives rel_l2 ~2.8 against the reference, i.e. noise.
#   - scales may be negative (the llama.cpp d = max/-8 convention).
#
# The bundle also folds an AWQ-style per-input-channel smoothing into the
# weights: RMSNorm w' = w*s with q/k/v divided by s, and the GLU's `up` rows
# scaled by t with `down` columns divided by t. Both are exact rewrites -- the
# bundle is self-consistent, and a tensor-by-tensor comparison against the HF
# checkpoint disagrees by design. Nothing here needs to know s or t.
#
# Qwen2.5 topology deltas vs llama, as in the 7B sibling:
#   - q/k/v projection BIAS (bf16, unquantized), packed after the cos/sin LUT
#     into rope_w = [cos/sin(DH), q_bias(DQ), k_bias(DK), v_bias(DV)]; the kernel
#     adds it in place before RoPE (fused_decode/kernels/rope.cc add_q_k_v_bias).
#   - no qk-norm (that is the Qwen3 convention).
#
# The config ties the LM head to the embedding, but the bundle's bf16 embedding
# has itself been through a 4-bit round trip, so the tied-fp shortcut
# llama32_1b_q4nx takes buys nothing here: the shipped `lm_head.weight` is used.
#
# INTERMEDIATE is padded 11008 -> 11264 (a multiple of 512) on the device, and the
# shipped bundle is already padded that way with exact zeros -- silu(0)*0 = 0 and
# down's pad columns contribute nothing. Accessors return the PADDED shape (what
# the decode cascade packs); the host containers get the logical 11008 slice.
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# The Q4NX block packer/dequant reference lives in the standalone fused_decode
# example (this dir depends on it for the decode path), same as the 7B sibling.
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

# Qwen2.5-3B-Instruct dims (HF config.json).
D = 2048  # hidden_size
DH = 128  # head_dim
N_Q_HEADS = 16
N_KV_HEADS = 2
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 8 (GQA)
DQ = N_Q_HEADS * DH  # 2048  q proj out (== D, so o_proj is square)
DK = N_KV_HEADS * DH  # 256   k proj out
DV = DK  # 256
INTER = 11008  # mlp intermediate (logical)
INTER_PAD = 11264  # ... padded to a multiple of 512 for the device
NUM_LAYERS = 36
VOCAB = 151936
RMS_EPS = 1e-6
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


# Row-group reorder (same block geometry as the llama bundles: 32x256 blocks,
# parallel=16 -> 2 row-groups/block interleaved even/odd).
_G = ROW_BLOCK // PARALLEL  # 2
_EVEN = np.array(
    [g * PARALLEL + 2 * k for g in range(_G) for k in range(PARALLEL // 2)]
)
_ODD = _EVEN + 1


class Q4nxModel:
    """mmap + parse a model.q4nx safetensors file; vectorized Q4_0 dequant.

    The container half mirrors qwen25_7b_q4nx_weights.Q4nxModel (I8-packed
    bundle: `shape` is [n_blocks, bytes_per_block], so the logical (M, K) comes
    from the caller); the codec is Q4_0, see the module header. The accessors are
    Qwen2.5-specific: a 2-norm pre-norm layer with plain weights, q/k/v
    projection biases, no qk-norm, and a separately quantized lm_head."""

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

    def raw_q4_0(self, name, M, K):
        """One quantized tensor in ROW-MAJOR Q4_0 form, unpacked from the bundle's
        block layout: (q[M,K] uint8 holding ONE two's-complement signed nibble per
        element, scale[M, K/GROUP] float32). w = int8(q)*scale.

        The FLM bundle stores each tensor block-major as [nb, block_bytes]
        (nb = (M/32)*(K/256), block = 256 scales + 256 unused mins + 32*256
        nibbles = 2560 int16), so the logical [M,K] must be supplied by the
        caller. Handing the codes back untouched is what lets the decode cascade
        pack the SHIPPED weights rather than a re-quantization of them."""
        nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
        nb = nbi * nbj
        assert (
            self._hdr[name]["shape"][0] == nb
        ), f"{name}: header nb={self._hdr[name]['shape'][0]} != (M/32)*(K/256)={nb}"
        i16 = self._raw_i16(name).reshape(nb, BLOCK_BF16)
        assert not i16[:, 256:512].any(), f"{name}: Q4_0 expects zero mins"
        sc = (
            i16[:, 0:256]
            .copy()
            .view(bfloat16)
            .astype(np.float32)
            .reshape(nb, N_GROUPS, ROW_BLOCK)
            .transpose(0, 2, 1)  # [block][row][group]
        )
        qb = (
            i16[:, 512:BLOCK_BF16]
            .copy()
            .view(np.uint8)
            .reshape(nb, _G, COL_BLOCK, PARALLEL // 2)
        )
        lo = (qb & 0xF).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        hi = (qb >> 4).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        q = np.zeros((nb, ROW_BLOCK, COL_BLOCK), np.uint8)
        q[:, _EVEN, :] = lo
        q[:, _ODD, :] = hi
        # sign-extend the nibble: 0..15 -> -8..7, kept as the raw uint8 byte.
        q = (((q ^ 8).astype(np.int8)) - np.int8(8)).view(np.uint8)
        unblock = lambda a, w: (  # [nb, ROW_BLOCK, w] -> [M, nbj*w]
            a.reshape(nbi, nbj, ROW_BLOCK, w).transpose(0, 2, 1, 3).reshape(M, nbj * w)
        )
        return unblock(q, COL_BLOCK), unblock(sc, N_GROUPS)

    def dequant(self, name, M, K):
        """A Q4_0 tensor dequantized to float32 [M, K]: w = int8(q) * scale."""
        q, sc = self.raw_q4_0(name, M, K)
        return q.view(np.int8).astype(np.float32) * np.repeat(sc, GROUP, axis=1)

    # name -> (tensor suffix, out, in) PADDED dims (the bundle is block-major, and
    # ships the GLU axis already zero-padded to INTER_PAD).
    _PROJ = {
        "q": ("self_attn.q_proj", DQ, D),
        "k": ("self_attn.k_proj", DK, D),
        "v": ("self_attn.v_proj", DV, D),
        "o": ("self_attn.o_proj", D, DQ),
        "up": ("mlp.up_proj", INTER_PAD, D),
        "gate": ("mlp.gate_proj", INTER_PAD, D),
        "down": ("mlp.down_proj", D, INTER_PAD),
    }

    def proj_fp(self, k, nm):
        """Full-precision PADDED [out, K] for one projection (what the decode
        cascade packs; the pad rows/columns are exact zeros)."""
        t, M, K = self._PROJ[nm]
        return self.dequant(f"model.layers.{k}.{t}.weight", M, K)

    def lm_head_fp(self):
        """Full-precision lm_head [VOCAB, D] -- the bundle's own quantized head."""
        return self.dequant("lm_head.weight", VOCAB, D)

    def lm_head_raw(self):
        """The lm-head's Q4_0 codes + scales, for a lossless cascade pack."""
        return self.raw_q4_0("lm_head.weight", VOCAB, D)

    def proj_raw(self, k, nm):
        """PADDED Q4_0 codes + scales for one projection of layer k."""
        t, M, K = self._PROJ[nm]
        return self.raw_q4_0(f"model.layers.{k}.{t}.weight", M, K)

    def layer_weights(self, k):
        """Dequantized bf16 GEMM input-B [K, out] per projection for layer k, at
        the LOGICAL (unpadded) GLU dim -- the host containers' shape."""
        return {
            nm: np.ascontiguousarray(_unpad(nm, self.proj_fp(k, nm)).T, dtype=bfloat16)
            for nm in self._PROJ
        }

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

        The config ties the head to the embedding, but the bundle's bf16
        embedding has itself been through a 4-bit round trip, so the tied-fp
        shortcut llama32_1b_q4nx takes gains nothing: the shipped Q4_0 lm_head
        is dequantized instead."""
        return (
            self.bf16("model.embed_tokens.weight"),
            self.bf16("model.norm.weight").astype(np.float32),
            self.lm_head_fp(),
        )


def _unpad(nm, w):
    """Drop the GLU zero padding from a PADDED [out, K] projection."""
    if nm in ("up", "gate"):
        return w[:INTER]
    if nm == "down":
        return w[:, :INTER]
    return w


def _pad(nm, w):
    """Zero-pad a LOGICAL [out, K] projection onto the device's GLU dim."""
    if nm in ("up", "gate"):
        out = np.zeros((INTER_PAD, w.shape[1]), np.float32)
        out[: w.shape[0]] = w
        return out
    if nm == "down":
        out = np.zeros((w.shape[0], INTER_PAD), np.float32)
        out[:, : w.shape[1]] = w
        return out
    return w


class HFQ40Model:
    """A full-precision HF checkpoint, rounded onto the Q4_0 grid on load.

    The fallback source, for reproducing the example from the ungated upstream
    checkpoint when the FastFlowLM bundle is not on hand. It carries none of the
    bundle's AWQ smoothing, so it is a strictly worse starting point -- but every
    projection goes through the SAME `requant_q4_0` the decode cascade packs
    with, so prefill and decode still see bit-identical weights. Norms and the
    q/k/v biases stay bf16 (the reference design does not quantize them either).
    """

    _PROJ = Q4nxModel._PROJ

    def __init__(self, model):
        from q4_0_codec import HFModel

        self._hf = HFModel(model)

    def has(self, name):
        return self._hf.has(name)

    def bf16(self, name):
        return self._hf.bf16(name)

    def proj_raw(self, k, nm):
        """PADDED Q4_0 codes + scales for one projection of layer k."""
        from q4_0_codec import requant_q4_0

        t, _M, _K = self._PROJ[nm]
        return requant_q4_0(_pad(nm, self.bf16(f"model.layers.{k}.{t}.weight")))

    def proj_fp(self, k, nm):
        """Q4_0-rounded PADDED [out, K] for one projection of layer k."""
        q, sc = self.proj_raw(k, nm)
        return q.view(np.int8).astype(np.float32) * np.repeat(sc, GROUP, axis=1)

    def lm_head_raw(self):
        """Tied: the fp embedding matrix, Q4_0-rounded."""
        from q4_0_codec import requant_q4_0

        return requant_q4_0(self.bf16("model.embed_tokens.weight"))

    def lm_head_fp(self):
        q, sc = self.lm_head_raw()
        return q.view(np.int8).astype(np.float32) * np.repeat(sc, GROUP, axis=1)

    def layer_weights(self, k):
        return {
            nm: np.ascontiguousarray(_unpad(nm, self.proj_fp(k, nm)).T, dtype=bfloat16)
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
    """Q4nxModel for a `model.q4nx` bundle, HFQ40Model for a checkpoint.

    A local path is classified by what is on disk. A REPO ID has to be probed:
    the default source is FastFlowLM's bundle-only repo, which ships a
    model.q4nx and no safetensors, so guessing HFQ40Model there sends
    snapshot_download looking for files that do not exist."""
    import os

    if os.path.isfile(model):
        return Q4nxModel(model) if model.endswith(".q4nx") else HFQ40Model(model)
    if os.path.isdir(model):
        has_bundle = os.path.isfile(os.path.join(model, "model.q4nx"))
        return Q4nxModel(model) if has_bundle else HFQ40Model(model)
    # Repo id: try the bundle, fall back to the checkpoint. Only a missing-file
    # error is a fallback; anything else (auth, network) is the caller's problem
    # and must not be reported as "this repo has no bundle".
    try:
        return Q4nxModel(model)
    except Exception as e:
        from huggingface_hub.errors import EntryNotFoundError

        if not isinstance(e, EntryNotFoundError):
            raise
    return HFQ40Model(model)


def qwen25_3b_config(n_layers=NUM_LAYERS):
    """The qwen25_3b LlamaConfig at this example's layer count."""
    _QWEN25_3B = str(Path(__file__).resolve().parents[1] / "qwen25_3b")
    if _QWEN25_3B not in sys.path:
        sys.path.insert(0, _QWEN25_3B)
    from qwen25_3b_weights import LlamaConfig

    return LlamaConfig(n_layers=n_layers)


def load_q4nx_weights(model_source, config=None, n_layers=None, verbose=True):
    """Load Qwen2.5-3B weights into the qwen25_3b `LlamaWeights`, Q4_0-rounded.

    `model_source` is a `model.q4nx` bundle (the default) or an HF repo id /
    checkpoint dir quantized on load -- see open_weight_source. Shape-for-shape
    identical to qwen25_3b_weights.load_weights(), so every downstream builder /
    preloader / block runner works unchanged."""
    _QWEN25_3B = str(Path(__file__).resolve().parents[1] / "qwen25_3b")
    if _QWEN25_3B not in sys.path:
        sys.path.insert(0, _QWEN25_3B)
    from qwen25_3b_weights import LayerWeights, LlamaWeights

    config = config or qwen25_3b_config()
    n_layers = config.n_layers if n_layers is None else n_layers
    qm = open_weight_source(model_source)
    if verbose:
        print(f"[qwen25_3b_q4] Q4_0 load from {model_source} ({n_layers} layers)...")

    layers = []
    for k in range(n_layers):
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
        if verbose and (k + 1) % 6 == 0:
            print(f"  ... {k + 1}/{n_layers} layers", flush=True)

    embed, norm, lm_head = qm.embed_norm_lmhead()
    return LlamaWeights(
        embed_table=np.asarray(embed, bfloat16),
        layers=layers,
        final_norm=np.asarray(norm, bfloat16),
        lm_head=np.asarray(lm_head, bfloat16),
    )


# The name the prefill imports.
load_q4_weights = load_q4nx_weights


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
# Runs the prompt through the 36 layers to produce (a) per-layer roped-K / raw-V
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
    """Run the prompt through all NUM_LAYERS Qwen2.5-3B layers in numpy.

    Returns (Kc, Vc, logits):
      Kc, Vc : float32 [NUM_LAYERS, P, DK] -- per-layer ROPED-K / RAW-V (the AIR
               decode KV seed; heads concatenated in CU order [h0|h1]).
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

        # GQA attention: causal (Qwen2.5-3B does not use its sliding window).
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
    logits = xf @ lm_head.T.astype(np.float32)  # tied lm-head [P, VOCAB]
    return Kc, Vc, logits.astype(np.float32)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Qwen2.5-3B Q4_0 weight loader")
    ap.add_argument("--model", default="FastFlowLM/Qwen2.5-3B-Instruct-NPU2")
    ap.add_argument("--n-layers", type=int, default=1)
    args = ap.parse_args()

    w = load_q4nx_weights(args.model, n_layers=args.n_layers)
    L0 = w.layers[0]
    print(f"embed {w.embed_table.shape} norm {w.final_norm.shape}")
    print(f"L0 wq {L0.wq.shape} wk {L0.wk.shape} wo {L0.wo.shape} bq {L0.bq.shape}")
