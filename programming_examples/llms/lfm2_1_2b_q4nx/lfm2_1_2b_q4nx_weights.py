# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Q4_0 weight loader for LFM2-1.2B.

Quantizes the full-precision HuggingFace checkpoint (`LiquidAI/LFM2-1.2B`)
directly to Q4_0, rather than re-quantizing a pre-quantized 4-bit bundle.
That mirrors `llms/qwen25_3b_q4`, the in-tree precedent for a Q4_0 model.

## Why not read the pre-quantized bundle

This is the same split the Qwen2.5-3B port hit, for the same reason. The
bundle's codec and the device's codec disagree:

  * The device kernels are built `-DQ4_0` -- SYMMETRIC signed int4,
    `w = q * scale`, `q` in [-8, 7], `scale = amax/8`, per 32-element group
    along the reduction dim. There is no offset term.
  * The Llama/Gemma/Phi4 bundles are block-AFFINE (`w = scale*q + min`), which
    is the same shape of codec the device wants for *those* models, so their
    dequant/requant round-trip lands back on the same grid and reading the
    bundle is lossless.

Symmetric cannot represent an offset range, so pushing affine 4-bit weights
into it would quantize twice and lose more than starting from full precision.
Quantizing the fp checkpoint once is both simpler and strictly more accurate.

An earlier attempt tried to invert the bundle's packing instead and did not
converge; `docs/Q4NX_DECODE_STATUS.md` records that dead end so it is not
re-explored. Nothing here depends on the bundle.

## Consequences of sourcing from HF rather than the bundle

  * **Tensor names are HF's**, not the bundle's llama-style renaming. HF LFM2
    uses `operator_norm` / `ffn_norm`, `feed_forward.w1|w2|w3`,
    `self_attn.out_proj`, `self_attn.q_layernorm`, and `conv.*`.
  * **`lm_head` is TIED.** HF LFM2 sets `tie_word_embeddings: true` and ships
    no `lm_head.weight` at all (148 tensors, none of them a head). The head is
    the embedding matrix, kept full precision -- the same treatment the Llama
    and Qwen Q4 examples give their tied heads.
  * **The final norm is `model.embedding_norm`**, not `model.norm`, and is
    applied after the layer loop.

## What is and is not quantized

Quantized (Q4_0, group 32 along the reduction dim): all seven projections per
attention/FFN layer, plus the ShortConv `in_proj` and `out_proj`.

Left BF16: every norm, the token embedding (= the tied head), and the
depthwise taps `conv.conv.weight`. The taps are only 6144 values and the
reference design keeps them full precision, so the Conv1D kernel needs no
quantized path.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LLMS_DIR = os.path.dirname(_THIS_DIR)
_PROG_DIR = os.path.dirname(_LLMS_DIR)
# The Q4_0 codec + the dependency-free safetensors reader live with the
# fused_decode example; reuse them rather than restating the codec.
for _p in (_LLMS_DIR, os.path.join(_PROG_DIR, "fused_decode")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen25_3b_requant import HFModel, requant_q4_0  # noqa: E402
from proj_qmm_pack import GROUP  # noqa: E402

DEFAULT_HF_REPO = "LiquidAI/LFM2-1.2B"


def dequant_q4_0(q_u8, scale, group=GROUP):
    """Inverse of `requant_q4_0`: ([M, K] uint8 holding ONE two's-complement
    signed-int4 code per element, [M, K/group] scale) -> float32 [M, K].

    `w = int8(q) * scale`, symmetric, per `group` columns. The input is one
    byte per weight, NOT packed nibbles -- `requant_q4_0` returns unpacked
    codes in [-8, 7] and the two-per-byte packing happens later, in the packer.
    """
    return q_u8.view(np.int8).astype(np.float32) * np.repeat(scale, group, axis=1)


@dataclass
class Lfm2Q4nxConfig:
    """LFM2-1.2B hyperparameters (identical to the bf16 config; the codec
    changes, the architecture does not)."""

    n_layers: int = 16
    emb_dim: int = 2048
    n_heads: int = 32
    head_dim: int = 64
    n_kv_heads: int = 8
    hidden_dim: int = 8192  # NOT config.intermediate_size (which reports 12288)
    vocab_size: int = 65536
    rope_base: float = 1000000.0
    norm_eps: float = 1e-5
    full_attn_idxs: Tuple[int, ...] = (2, 5, 8, 10, 12, 14)
    conv_dim: int = 2048
    conv_L_cache: int = 3
    dtype: np.dtype = bfloat16

    def is_attn_layer(self, i):
        return i in self.full_attn_idxs

    @property
    def q_dim(self):
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_dim

    @property
    def n_attn_layers(self):
        return len(self.full_attn_idxs)

    @property
    def n_conv_layers(self):
        return self.n_layers - len(self.full_attn_idxs)


@dataclass
class AttnWeights:
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray


@dataclass
class ConvWeights:
    w_in: np.ndarray  # (emb, 3*conv_dim), columns ordered B | C | v
    w_conv: np.ndarray  # (conv_dim, 3) bf16, oldest-tap-first
    w_out: np.ndarray  # (conv_dim, emb)


@dataclass
class LayerWeights:
    operator_norm: np.ndarray
    ffn_norm: np.ndarray
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray
    attn: Optional[AttnWeights] = None
    conv: Optional[ConvWeights] = None

    @property
    def is_attn(self):
        return self.attn is not None


@dataclass
class Lfm2Weights:
    embed_table: np.ndarray
    layers: List[LayerWeights] = field(default_factory=list)
    final_norm: np.ndarray = None
    lm_head: np.ndarray = None


class Lfm2Q4Model:
    """HF bf16 checkpoint -> Q4_0 -> bf16, tensor by tensor.

    Deliberately quantize-then-dequantize rather than using the bf16 weights
    directly: the prefill must run on the SAME 4-bit values the NPU decode
    consumes, so that the prefill's KV cache and conv state are valid decode
    inputs.
    """

    def __init__(self, model=DEFAULT_HF_REPO, dtype=bfloat16):
        self.hf = HFModel(model)
        self.source = model
        self.dtype = dtype

    def q4_T(self, name):
        """One HF projection [out, in] -> Q4_0 round-trip -> bf16 [in, out].

        Grouping runs along the last axis, which for HF's (out, in) layout is
        the reduction dim -- exactly where the device kernel expects its groups.
        """
        w = self.hf.bf16(name)  # [out, in] float32
        q, sc = requant_q4_0(w)
        return np.ascontiguousarray(dequant_q4_0(q, sc).T, dtype=self.dtype)

    def bf(self, name):
        return np.asarray(self.hf.bf16(name), self.dtype)


def load_weights(model=DEFAULT_HF_REPO, config=None, dtype=bfloat16, verbose=False):
    """Load + Q4_0-round-trip the LFM2 weights into numpy arrays.

    All projections are returned in the `y = x @ W` convention, i.e.
    (in_features, out_features) -- HF stores (out, in), so each tensor is
    transposed once here.
    """
    cfg = config or Lfm2Q4nxConfig()
    m = Lfm2Q4Model(model, dtype=dtype)
    if verbose:
        print(f"[lfm2_1_2b_q4nx] Q4_0 requant from {model} ({cfg.n_layers} layers)...")

    emb, hid, cdim = cfg.emb_dim, cfg.hidden_dim, cfg.conv_dim

    embed_table = m.bf("model.embed_tokens.weight")
    assert embed_table.shape == (cfg.vocab_size, emb), embed_table.shape

    layers = []
    for li in range(cfg.n_layers):
        p = f"model.layers.{li}"
        common = dict(
            operator_norm=m.bf(f"{p}.operator_norm.weight"),
            ffn_norm=m.bf(f"{p}.ffn_norm.weight"),
            # LFM2's SwiGLU names its projections w1/w3/w2 = gate/up/down.
            w_gate=m.q4_T(f"{p}.feed_forward.w1.weight"),
            w_up=m.q4_T(f"{p}.feed_forward.w3.weight"),
            w_down=m.q4_T(f"{p}.feed_forward.w2.weight"),
        )
        if cfg.is_attn_layer(li):
            attn = AttnWeights(
                wq=m.q4_T(f"{p}.self_attn.q_proj.weight"),
                wk=m.q4_T(f"{p}.self_attn.k_proj.weight"),
                wv=m.q4_T(f"{p}.self_attn.v_proj.weight"),
                wo=m.q4_T(f"{p}.self_attn.out_proj.weight"),
                q_norm=m.bf(f"{p}.self_attn.q_layernorm.weight"),
                k_norm=m.bf(f"{p}.self_attn.k_layernorm.weight"),
            )
            conv = None
        else:
            attn = None
            conv = ConvWeights(
                w_in=m.q4_T(f"{p}.conv.in_proj.weight"),
                # BF16, and stored [conv_dim, 1, 3] -- the singleton channel
                # axis is HF's Conv1d groups dim; squeeze it to [conv_dim, 3].
                w_conv=np.ascontiguousarray(
                    m.bf(f"{p}.conv.conv.weight").reshape(cdim, cfg.conv_L_cache)
                ),
                w_out=m.q4_T(f"{p}.conv.out_proj.weight"),
            )
        lw = LayerWeights(**common, attn=attn, conv=conv)
        _assert_layer(lw, cfg, li)
        layers.append(lw)
        if verbose and (li + 1) % 4 == 0:
            print(f"  ... {li + 1}/{cfg.n_layers} layers", flush=True)

    final_norm = m.bf("model.embedding_norm.weight")
    # TIED: HF LFM2 sets tie_word_embeddings=true and ships no lm_head tensor.
    # The head stays full precision, as in the Llama/Qwen Q4 examples.
    lm_head = embed_table

    return Lfm2Weights(
        embed_table=embed_table,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


def _assert_layer(lw, cfg, li):
    emb, hid, cdim = cfg.emb_dim, cfg.hidden_dim, cfg.conv_dim
    assert lw.operator_norm.shape == (emb,), (li, lw.operator_norm.shape)
    assert lw.ffn_norm.shape == (emb,), (li, lw.ffn_norm.shape)
    assert lw.w_gate.shape == (emb, hid), (li, lw.w_gate.shape)
    assert lw.w_up.shape == (emb, hid), (li, lw.w_up.shape)
    assert lw.w_down.shape == (hid, emb), (li, lw.w_down.shape)
    if lw.is_attn:
        a = lw.attn
        assert a.wq.shape == (emb, cfg.q_dim), (li, a.wq.shape)
        assert a.wk.shape == (emb, cfg.kv_dim), (li, a.wk.shape)
        assert a.wv.shape == (emb, cfg.kv_dim), (li, a.wv.shape)
        assert a.wo.shape == (cfg.q_dim, emb), (li, a.wo.shape)
        assert a.q_norm.shape == (cfg.head_dim,), (li, a.q_norm.shape)
        assert a.k_norm.shape == (cfg.head_dim,), (li, a.k_norm.shape)
    else:
        c = lw.conv
        assert c.w_in.shape == (emb, 3 * cdim), (li, c.w_in.shape)
        assert c.w_conv.shape == (cdim, cfg.conv_L_cache), (li, c.w_conv.shape)
        assert c.w_out.shape == (cdim, emb), (li, c.w_out.shape)


def generate_rope_lut(config=None, seq_len=2048, dtype=bfloat16):
    """Half-split RoPE LUT: [cos_0..cos_{h-1}, sin_0..sin_{h-1}] per position."""
    cfg = config or Lfm2Q4nxConfig()
    hd, half, theta = cfg.head_dim, cfg.head_dim // 2, cfg.rope_base
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.outer(np.arange(seq_len, dtype=np.float64), inv)
    lut = np.empty((seq_len, hd), dtype=np.float64)
    lut[:, :half], lut[:, half:] = np.cos(ang), np.sin(ang)
    return lut.astype(dtype)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LFM2-1.2B Q4_0 weight loader")
    ap.add_argument("--model", default=DEFAULT_HF_REPO)
    ap.add_argument("--gate", action="store_true", help="run the quant-error gate only")
    args = ap.parse_args()

    m = Lfm2Q4Model(args.model)

    # Gate: Q4_0 round-trip error on the layer-2 q_proj, compared UN-transposed
    # against the HF tensor. The bar the Q4 bundles hit is cosine ~0.997.
    ref = m.hf.bf16("model.layers.2.self_attn.q_proj.weight")  # [2048, 2048]
    q, sc = requant_q4_0(ref)
    got = dequant_q4_0(q, sc)
    cos = float(
        (got.ravel() @ ref.ravel())
        / (np.linalg.norm(got.ravel()) * np.linalg.norm(ref.ravel()))
    )
    print(
        f"gate  L2 q_proj {ref.shape}  cosine={cos:.6f}  "
        f"std_got={got.std():.4f} std_ref={ref.std():.4f}  "
        f"rel_l2={np.linalg.norm(got - ref) / np.linalg.norm(ref):.4f}"
    )
    assert cos > 0.99, f"Q4_0 round-trip cosine {cos:.6f} -- wrong codec"
    print("gate  PASS")

    if args.gate:
        raise SystemExit(0)

    cfg = Lfm2Q4nxConfig()
    w = load_weights(args.model, config=cfg, verbose=True)
    print(f"  embed_table {w.embed_table.shape}  final_norm {w.final_norm.shape}")
    print(f"  lm_head     {w.lm_head.shape}  (tied={w.lm_head is w.embed_table})")
    for li in (0, 2):
        lw = w.layers[li]
        kind = "ATTN" if lw.is_attn else "CONV"
        print(f"  L{li} [{kind}] gate{lw.w_gate.shape} down{lw.w_down.shape}", end=" ")
        if lw.is_attn:
            print(f"wq{lw.attn.wq.shape} q_norm{lw.attn.q_norm.shape}")
        else:
            print(f"w_in{lw.conv.w_in.shape} w_conv{lw.conv.w_conv.shape}")
    print("OK")
