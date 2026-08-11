# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4_0 weight loader for the Qwen2.5-3B Q4 example.
#
# Mirrors llama32_1b_q4nx_weights.Q4nxModel, with two deltas that follow from
# the model rather than from a choice:
#
#   1. CODEC. What differs is the ON-DEVICE format, not the bundle. FastFlowLM
#      ships Qwen2.5-3B as a `model.q4nx` bundle in the same per-block AFFINE
#      encoding as Llama and Gemma (w = scale*q + min, unsigned nibbles), but
#      their Qwen decode design sets `#define Q4_0` (FastFlowLM's
#      Qwen2_5/decoding_3b/models/qwen2_3b.h), which switches the kernel to a
#      SYMMETRIC signed-int4 form (w = q*scale, q in
#      [-8,7], scale = amax/8, per 32-element group along the reduction dim).
#      The AIR port mirrors that and builds its kernels -DQ4_0.
#
#      So the bundle's codec and the device's disagree, and this loader
#      quantizes the full-precision HF checkpoint directly instead. For Llama
#      and Gemma the two agree (affine to affine, same groups, same 4 bits), so
#      their dequant/requant round-trip lands back on the same grid; symmetric
#      cannot represent an offset range, so re-quantizing affine 4-bit weights
#      into it would quantize twice and lose more than starting from fp.
#
#      The quantizer is `qwen25_3b_requant.requant_q4_0`, the same function the
#      fused_decode Qwen weight cache uses, so the prefill and the fused decode
#      see BIT-IDENTICAL weight values.
#
#   2. QKV BIAS. Qwen2.5 has q/k/v projection bias. Bias is NOT quantized (it is
#      not quantized in the reference design either) -- it is carried as bf16.
#
# Everything else -- dequant to bf16, transpose HF (out,in) -> GEMM (in,out),
# tied lm_head, bf16 norms -- matches the Llama Q4NX loader, so the resulting
# LlamaWeights container drops straight into the qwen25_3b prefill builders.
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# The Q4_0 quantizer + HF safetensors reader live with the fused_decode example
# (single source of truth for the Qwen quant codec).
_FUSED_DECODE = str(Path(__file__).resolve().parents[2] / "fused_decode")
_QWEN25_3B = str(Path(__file__).resolve().parent.parent / "qwen25_3b")
for _p in (_FUSED_DECODE, _QWEN25_3B):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen25_3b_requant import HFModel, HF_REPO, requant_q4_0  # noqa: E402
from proj_qmm_pack import GROUP  # noqa: E402
from qwen25_3b_weights import (  # noqa: E402
    LlamaConfig,
    LayerWeights,
    LlamaWeights,
    generate_rope_lut,
)

# Qwen2.5-3B dims (mirrors the Llama loader's module-level dim block).
D = 2048  # model dim
DQ = 2048  # q proj out (16 heads * 128)
DK = 256  # k proj out (2 kv heads * 128)
DV = 256
DH = 128  # head dim
N_Q_HEADS = 16
N_KV_HEADS = 2
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 8
INTER = 11008  # mlp intermediate
VOCAB = 151936

_PROJ = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "up": "mlp.up_proj",
    "gate": "mlp.gate_proj",
    "down": "mlp.down_proj",
}
_BIAS = {"bq": "self_attn.q_proj", "bk": "self_attn.k_proj", "bv": "self_attn.v_proj"}


def _bf(a):
    return a.astype(bfloat16).astype(np.float32)


def dequant_q4_0(q_u8, scale, group=GROUP):
    """Inverse of requant_q4_0: ([M,K] uint8 holding ONE two's-complement
    signed-int4 value per element, [M,K/g] scale) -> float32 [M, K].
    w = int8(q) * scale, symmetric, per `group` columns.

    Note the input is one byte per weight, NOT packed nibbles: requant_q4_0
    returns unpacked codes in [-8, 7] and the two-per-byte packing happens
    later, in the cascade packer."""
    return q_u8.view(np.int8).astype(np.float32) * np.repeat(scale, group, axis=1)


class Qwen25Q4Model:
    """HF bf16 checkpoint -> Q4_0 -> bf16, tensor by tensor.

    Deliberately quantize-then-dequantize (rather than using the bf16 weights
    directly): the point of this example is to run the prefill on the SAME
    4-bit values the NPU decode consumes, so prefill and decode agree and the
    prefill's KV cache is a valid decode input.
    """

    def __init__(self, model=HF_REPO):
        self.hf = HFModel(model)
        self.source = model

    def _q4(self, name):
        """One HF projection [out, in] -> Q4_0 round-trip -> bf16 [in, out]."""
        w = self.hf.bf16(name)  # [out, in] float32
        q, sc = requant_q4_0(w)
        return np.ascontiguousarray(dequant_q4_0(q, sc).T, dtype=bfloat16)

    def layer_weights(self, k):
        """Q4_0-round-tripped bf16 GEMM input-B [in, out] per projection."""
        return {nm: self._q4(f"model.layers.{k}.{t}.weight") for nm, t in _PROJ.items()}

    def layer_bias(self, k):
        """(bq, bk, bv) bf16 -- NOT quantized (matches the reference design)."""
        return tuple(
            np.asarray(self.hf.bf16(f"model.layers.{k}.{t}.bias"), bfloat16)
            for t in (_BIAS["bq"], _BIAS["bk"], _BIAS["bv"])
        )

    def layer_rms(self, k):
        """(input_layernorm, post_attention_layernorm) float32 for layer k."""
        return (
            self.hf.bf16(f"model.layers.{k}.input_layernorm.weight"),
            self.hf.bf16(f"model.layers.{k}.post_attention_layernorm.weight"),
        )

    def embed_norm_lmhead(self):
        """(embed [VOCAB,D], final_norm [D], lm_head [VOCAB,D]) float32.

        Qwen2.5-3B sets tie_word_embeddings=true, so the LM head IS the
        full-precision embedding matrix (kept fp, exactly as the Llama Q4NX
        example keeps its tied head fp)."""
        embed = self.hf.bf16("model.embed_tokens.weight")
        norm = self.hf.bf16("model.norm.weight")
        return embed, norm, embed


def load_q4_weights(model=HF_REPO, config=None, n_layers=None, verbose=True):
    """Build the qwen25_3b `LlamaWeights` container from Q4_0 weights.

    Shape-for-shape identical to qwen25_3b_weights.load_weights(), so every
    downstream builder / preloader / block runner works unchanged."""
    config = config or LlamaConfig()
    n_layers = config.n_layers if n_layers is None else n_layers
    qm = Qwen25Q4Model(model)
    if verbose:
        print(f"[qwen25_3b_q4] Q4_0 requant from {model} ({n_layers} layers)...")

    layers = []
    for k in range(n_layers):
        W = qm.layer_weights(k)
        rms_in, rms_post = qm.layer_rms(k)
        bq, bk, bv = qm.layer_bias(k)
        layers.append(
            LayerWeights(
                attn_norm=np.asarray(rms_in, bfloat16),
                wq=W["q"],
                wk=W["k"],
                wv=W["v"],
                wo=W["o"],
                ffn_norm=np.asarray(rms_post, bfloat16),
                w_gate=W["gate"],
                w_up=W["up"],
                w_down=W["down"],
                bq=bq,
                bk=bk,
                bv=bv,
            )
        )
        if verbose and (k + 1) % 6 == 0:
            print(f"  ... {k + 1}/{n_layers} layers", flush=True)

    embed, final_norm, lm_head = qm.embed_norm_lmhead()
    return LlamaWeights(
        embed_table=np.asarray(embed, bfloat16),
        layers=layers,
        final_norm=np.asarray(final_norm, bfloat16),
        lm_head=np.asarray(lm_head, bfloat16),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Qwen2.5-3B Q4_0 weight loader")
    ap.add_argument("--model", default=HF_REPO)
    ap.add_argument("--n-layers", type=int, default=1)
    args = ap.parse_args()

    w = load_q4_weights(args.model, n_layers=args.n_layers)
    L0 = w.layers[0]
    print(f"embed {w.embed_table.shape} norm {w.final_norm.shape}")
    print(f"L0 wq {L0.wq.shape} wk {L0.wk.shape} wo {L0.wo.shape} bq {L0.bq.shape}")

    # Quant error vs the fp reference, per projection.
    qm = Qwen25Q4Model(args.model)
    for nm, t in _PROJ.items():
        ref = qm.hf.bf16(f"model.layers.0.{t}.weight").T
        got = np.asarray(
            getattr(
                L0,
                {
                    "q": "wq",
                    "k": "wk",
                    "v": "wv",
                    "o": "wo",
                    "gate": "w_gate",
                    "up": "w_up",
                    "down": "w_down",
                }[nm],
            ),
            np.float32,
        )
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        print(f"  {nm:5s} {str(ref.shape):16s} rel_l2={rel:.4f}")
