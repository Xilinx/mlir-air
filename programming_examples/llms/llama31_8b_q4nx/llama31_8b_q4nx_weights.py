# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader for the Llama-3.1-8B Q4NX example.
#
# Reuses the shape-agnostic Q4NX safetensors reader/dequantizer (`Q4nxModel`)
# from the 1B example; only the dimensions and two model facts differ from the
# 3B loader:
#
#   1. UNTIED LM HEAD. Llama-3.2-1B/3B set tie_word_embeddings=true, so their
#      LM head IS the bf16 embedding matrix. Llama-3.1-8B sets it FALSE and the
#      bundle carries a real quantized `lm_head.weight` (I8, 64128x5120 packed
#      = logical 128256x4096). Falling back to the embedding here would be a
#      silently wrong LM head, so it is dequantized like any other projection.
#
#   2. ROPE SCALING FACTOR 8.0. Llama-3.1-8B applies llama3 rope scaling with
#      factor=8.0 (config rope_scaling), not the 32.0 that Llama-3.2 uses.
#      FastFlowLM's own `llama_3b_8b_rope` table is this exact curve (checked
#      numerically: factor 8.0 reproduces it to 4e-05, the header's print
#      precision; 32.0 is off by 75%), so FLM-faithful and HF-correct coincide.
#
# Everything else -- per-block 4-bit dequant to bf16, transpose HF (out,in) ->
# GEMM (in,out), bf16 norms -- matches the Llama Q4NX loaders.
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_LLMS = _HERE.parent
_LLAMA1B_Q4NX = _LLMS / "llama32_1b_q4nx"
_LLAMA3B = _LLMS / "llama32_3b"
for _p in (str(_LLMS), str(_LLAMA1B_Q4NX), str(_LLAMA3B), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Dimension-agnostic, but NOT header-agnostic: the loader needs the logical
# [out, K] for an I8-packed bundle, which is what this model ships.
from llama32_1b_q4nx_weights import Q4nxModel  # noqa: E402

# The container dataclasses are plain dimension-carrying records, shared with
# the 3B example rather than re-declared here.
from llama32_3b_weights import (  # noqa: E402
    LlamaConfig,
    LayerWeights,
    LlamaWeights,
)

# HF repo id of the bf16 reference `make verify` compares against. FastFlowLM's
# bundle declares base_model: meta-llama/Llama-3.1-8B-Instruct, which is gated;
# this is a faithful ungated re-upload whose config matches the FLM bundle's
# exactly, including the 3-way eos_token_id list. Same pattern as the Gemma3
# example's unsloth mirror.
HF_REPO = "NousResearch/Meta-Llama-3.1-8B-Instruct"

# Q4NX bundle (FastFlowLM's NPU2 packaging) the NPU weights come from.
Q4NX_REPO = "FastFlowLM/Llama-3.1-8B-NPU2"


def llama31_8b_config():
    """LlamaConfig carrying the Llama-3.1-8B dimensions."""
    return LlamaConfig(
        n_layers=32,
        emb_dim=4096,
        n_heads=32,
        head_dim=128,
        n_kv_heads=8,  # GQA: 4 Q heads per KV head (32 / 8)
        hidden_dim=14336,
        vocab_size=128256,
        rope_base=500000.0,
    )


# llama3 rope scaling (config rope_scaling). Factor 8.0 is Llama-3.1; the
# remaining three are shared with Llama-3.2.
ROPE_FACTOR = 8.0
ROPE_LOW_FREQ_FACTOR = 1.0
ROPE_HIGH_FREQ_FACTOR = 4.0
ROPE_OLD_CTX = 8192.0


def llama3_inv_freq(
    dim=128,
    theta=500000.0,
    factor=ROPE_FACTOR,
    low_freq_factor=ROPE_LOW_FREQ_FACTOR,
    high_freq_factor=ROPE_HIGH_FREQ_FACTOR,
    old_ctx=ROPE_OLD_CTX,
):
    """llama3-scaled inverse frequencies [dim/2].

    High-frequency dims (wavelength < old_ctx/high_freq_factor) pass through,
    low-frequency dims (wavelength > old_ctx/low_freq_factor) are divided by
    `factor`, and the band between is linearly interpolated -- the HF
    `rope_type: llama3` rule.
    """
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    wl = 2 * np.pi / inv
    low_wl = old_ctx / low_freq_factor
    high_wl = old_ctx / high_freq_factor
    smooth = (old_ctx / wl - low_freq_factor) / (high_freq_factor - low_freq_factor)
    med = (1 - smooth) * (inv / factor) + smooth * inv
    out = np.where(wl > low_wl, inv / factor, inv)
    return np.where((wl <= low_wl) & (wl >= high_wl), med, out)


def generate_rope_lut(config=None, seq_len=2048, dtype=bfloat16):
    """RoPE LUT [seq_len, head_dim] as concatenated [cos... , sin...].

    Matches the half-split RoPE kernel + HF Llama convention, and applies the
    llama3 frequency scaling that Llama-3.1-8B specifies (the un-scaled table
    diverges by up to 7x on the low-frequency dims).
    """
    if config is None:
        config = llama31_8b_config()
    head_dim = config.head_dim
    half = head_dim // 2
    inv = llama3_inv_freq(dim=head_dim, theta=config.rope_base)
    angles = np.outer(np.arange(seq_len, dtype=np.float64), inv)
    lut = np.empty((seq_len, head_dim), dtype=np.float64)
    lut[:, :half] = np.cos(angles)
    lut[:, half:] = np.sin(angles)
    return lut.astype(dtype)


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


def load_q4nx_weights(model_source=None, config=None):
    """Load Llama-3.1-8B weights from a `model.q4nx` bundle into a LlamaWeights.

    `model_source` may be an HF repo id (contains '/'), a directory containing
    model.q4nx, or a direct model.q4nx file path; it defaults to Q4NX_REPO.
    Q4NX projections are dequant'd to bf16 on the host; norms/embeddings are
    read as bf16.
    """
    if config is None:
        config = llama31_8b_config()
    if model_source is None:
        model_source = Q4NX_REPO

    # The 8B bundle carries I8-packed headers, which do not encode the logical
    # [out, K] -- supply it (Q4nxModel docstring).
    qm = Q4nxModel(model_source)
    embed = np.asarray(qm.bf16("model.embed_tokens.weight"), bfloat16)
    norm = np.asarray(qm.bf16("model.norm.weight"), bfloat16)
    # UNTIED: a real quantized LM head, dequantized like any other projection.
    lm_head = np.asarray(
        qm.dequant("lm_head.weight", config.vocab_size, config.emb_dim), bfloat16
    )

    layers = []
    for k in range(config.n_layers):
        w = qm.layer_weights(k, _proj_dims(config))  # each [K, out] bf16
        rms_in, rms_post = qm.layer_rms(k)
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
            )
        )

    return LlamaWeights(
        embed_table=embed,
        layers=layers,
        final_norm=norm,
        lm_head=lm_head,
    )


def _cosine(a, b):
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    """Dequant-fidelity check: cosine of every layer-0 projection (plus the
    untied LM head) against the bf16 reference. Q4NX round-trips to ~0.997."""
    import argparse

    ap = argparse.ArgumentParser(description="Llama-3.1-8B Q4NX weight loader")
    ap.add_argument("--model", default=Q4NX_REPO, help="Q4NX bundle source")
    ap.add_argument("--hf", default=HF_REPO, help="bf16 reference repo")
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    cfg = llama31_8b_config()
    qm = Q4nxModel(args.model)

    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    ref_dir = snapshot_download(args.hf, allow_patterns=["*.safetensors", "*.json"])
    index = {}
    for f in sorted(Path(ref_dir).glob("*.safetensors")):
        with safe_open(f, framework="np") as h:
            for k in h.keys():
                index[k] = f

    def ref(name):
        with safe_open(index[name], framework="np") as h:
            return h.get_tensor(name)

    dims = _proj_dims(cfg)
    print(f"layer {args.layer} projections (Q4NX vs {args.hf}):")
    worst = 1.0
    for key, t in Q4nxModel._PROJ.items():
        M, K = dims[key]
        got = qm.dequant(f"model.layers.{args.layer}.{t}.weight", M, K)
        c = _cosine(got, ref(f"model.layers.{args.layer}.{t}.weight"))
        worst = min(worst, c)
        print(f"  {key:5s} {str((M, K)):16s} cosine {c:.6f}")
    c = _cosine(
        qm.dequant("lm_head.weight", cfg.vocab_size, cfg.emb_dim), ref("lm_head.weight")
    )
    worst = min(worst, c)
    print(
        f"  {'lm':5s} {str((cfg.vocab_size, cfg.emb_dim)):16s} cosine {c:.6f} (untied)"
    )
    print(f"\nworst cosine {worst:.6f} -> {'PASS' if worst > 0.99 else 'FAIL'}")
    return 0 if worst > 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
