# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader for the Phi-4-mini-instruct Q4NX example.
#
# Reuses the shape-agnostic Q4NX safetensors reader/dequantizer (`Q4nxModel`)
# from the 1B example. Three Phi-4 facts shape this file:
#
#   1. THE BUNDLE IS ALREADY SPLIT. HF's Phi3ForCausalLM stores FUSED
#      `self_attn.qkv_proj` [5120,3072] and `mlp.gate_up_proj` [16384,3072],
#      and FastFlowLM's GGUF converter reads the equally-fused GGUF
#      `blk.N.attn_qkv` / `blk.N.ffn_up`. But it WRITES the q4nx bundle already
#      split into q/k/v and gate/up (verified in the header), so the loader
#      needs no unfusing -- only the cosine check below has to split the HF
#      side to compare like with like.
#
#   2. TIED LM HEAD. config tie_word_embeddings=true and the HF checkpoint has
#      no `lm_head.weight` at all, so the LM head IS the bf16 embedding (the
#      bundle's separate quantized lm_head is redundant and lossier). This is
#      the same convention as Llama-3.2-1B/3B.
#
#   3. PARTIAL ROTARY + LongRoPE. partial_rotary_factor=0.75 -> RoPE covers only
#      the leading 96 of 128 head dims, so there are 48 frequencies, and the
#      bundle carries `rope.short.weight[48]` / `rope.long.weight[48]` LongRoPE
#      factor tables. short_factor is all 1.0, so within
#      original_max_position_embeddings=4096 the frequencies reduce to plain
#      inv_freq(theta=1e4, dim=96) -- exactly FastFlowLM's `phi4_rope` short
#      half. The long table (up to 47.75) only engages past 4096.
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

from llama32_1b_q4nx_weights import Q4nxModel  # noqa: E402
from llama32_3b_weights import (  # noqa: E402
    LlamaConfig,
    LayerWeights,
    LlamaWeights,
)

# bf16 reference `make verify` compares against -- ungated, and the same repo
# FastFlowLM's own reference generator uses (models_simple/verify_phi4.py).
HF_REPO = "microsoft/Phi-4-mini-instruct"
Q4NX_REPO = "FastFlowLM/Phi4-mini-Instruct-NPU2"

# Rotary geometry (config partial_rotary_factor=0.75 * head_dim 128).
ROPE_DIM = 96
ROPE_BASE = 10000.0
ROPE_ORIG_MAX_POS = 4096


def phi4_mini_config():
    """LlamaConfig carrying the Phi-4-mini dimensions."""
    return LlamaConfig(
        n_layers=32,
        emb_dim=3072,
        n_heads=24,
        head_dim=128,
        n_kv_heads=8,  # GQA: 3 Q heads per KV head (24 / 8)
        hidden_dim=8192,
        vocab_size=200064,
        rope_base=ROPE_BASE,
    )


def longrope_attention_scaling(max_pos=131072, orig_max=ROPE_ORIG_MAX_POS):
    """LongRoPE cos/sin multiplier, sqrt(1 + ln(max/orig)/ln(orig)).

    HF Phi3 scales BOTH cos and sin by this whenever
    original_max_position_embeddings < max_position_embeddings; for Phi-4-mini it
    is 1.1902380714238083, and it applies at every context length, not only past
    orig_max. Leaving it out shifts every rotated q/k by ~19% and reorders the
    first-token logits enough to fail the top-k gate (measured: 4/8 prompts).

    NOTE: FastFlowLM does NOT apply it -- their get_rope_scaling(PHI4_4B) returns
    1.0f with this exact constant commented out beside it. We follow HF here,
    because HF bf16 is what `make verify` gates against and what the model card
    specifies; this is a deliberate, measured divergence from FLM.
    """
    return float(np.sqrt(1.0 + np.log(max_pos / orig_max) / np.log(orig_max)))


def longrope_inv_freq(factors, dim=ROPE_DIM, theta=ROPE_BASE):
    """LongRoPE inverse frequencies [dim/2] for a given factor table.

    HF Phi3: inv_freq = 1 / (ext_factors * theta**(arange(0,dim,2)/dim)).
    `factors` is the short table at contexts <= original_max_position_embeddings
    and the long table beyond it.
    """
    f = np.asarray(factors, np.float64).reshape(-1)
    base = theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    if f.size != base.size:
        raise ValueError(f"factor table has {f.size} entries, expected {base.size}")
    return 1.0 / (f * base)


def load_rope_factors(model_source=None):
    """(short, long) LongRoPE factor tables [48] from the bundle."""
    qm = Q4nxModel(model_source or Q4NX_REPO)
    return (
        np.asarray(qm.bf16("rope.short.weight"), np.float64),
        np.asarray(qm.bf16("rope.long.weight"), np.float64),
    )


def generate_rope_lut(
    config=None, seq_len=2048, dtype=bfloat16, model_source=None, width=None
):
    """RoPE LUT [seq_len, ROPE_DIM] as concatenated [cos..., sin...].

    ROPE_DIM (96) wide by default, not head_dim: the trailing 32 dims are not
    rotated (the kernel copies them through), so there are no cos/sin entries for
    them. Picks the short or long LongRoPE factor table by seq_len, as HF does.

    `width` zero-pads each row out to that many columns. The DECODE wants the bare
    96 (its rope_w buffer is ROPE_W_LEN). The PREFILL wants head_dim=128, because
    its rope launch DMAs the LUT row at the same offset and width as the data row
    -- the pad columns are never read by `rope_partial`.
    """
    if config is None:
        config = phi4_mini_config()
    short, long_ = load_rope_factors(model_source)
    factors = short if seq_len <= ROPE_ORIG_MAX_POS else long_
    inv = longrope_inv_freq(factors, dim=ROPE_DIM, theta=config.rope_base)
    half = ROPE_DIM // 2
    scale = longrope_attention_scaling()
    angles = np.outer(np.arange(seq_len, dtype=np.float64), inv)
    lut = np.empty((seq_len, ROPE_DIM), dtype=np.float64)
    lut[:, :half] = np.cos(angles) * scale
    lut[:, half:] = np.sin(angles) * scale
    if width is not None and width != ROPE_DIM:
        if width < ROPE_DIM:
            raise ValueError(f"width {width} < ROPE_DIM {ROPE_DIM}")
        padded = np.zeros((seq_len, width), dtype=np.float64)
        padded[:, :ROPE_DIM] = lut
        lut = padded
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
    """Load Phi-4-mini weights from a `model.q4nx` bundle into a LlamaWeights."""
    if config is None:
        config = phi4_mini_config()
    if model_source is None:
        model_source = Q4NX_REPO

    qm = Q4nxModel(model_source)
    embed = np.asarray(qm.bf16("model.embed_tokens.weight"), bfloat16)
    norm = np.asarray(qm.bf16("model.norm.weight"), bfloat16)

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
        lm_head=embed,  # tied (config tie_word_embeddings=true)
    )


def _cosine(a, b):
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    """Dequant-fidelity check vs the bf16 reference. The HF side is FUSED, so q/k/v
    are sliced out of qkv_proj and gate/up out of gate_up_proj to compare."""
    import argparse

    ap = argparse.ArgumentParser(description="Phi-4-mini Q4NX weight loader")
    ap.add_argument("--model", default=Q4NX_REPO)
    ap.add_argument("--hf", default=HF_REPO)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    cfg = phi4_mini_config()
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

    L = args.layer
    dq = cfg.n_heads * cfg.head_dim
    dkv = cfg.n_kv_heads * cfg.head_dim
    qkv = ref(f"model.layers.{L}.self_attn.qkv_proj.weight")
    gu = ref(f"model.layers.{L}.mlp.gate_up_proj.weight")
    hf = {
        "q": qkv[:dq],
        "k": qkv[dq : dq + dkv],
        "v": qkv[dq + dkv :],
        "o": ref(f"model.layers.{L}.self_attn.o_proj.weight"),
        "gate": gu[: cfg.hidden_dim],
        "up": gu[cfg.hidden_dim :],
        "down": ref(f"model.layers.{L}.mlp.down_proj.weight"),
    }

    dims = _proj_dims(cfg)
    print(f"layer {L} projections (Q4NX vs {args.hf}; HF qkv/gate_up split):")
    worst = 1.0
    for key, t in Q4nxModel._PROJ.items():
        M, K = dims[key]
        got = qm.dequant(f"model.layers.{L}.{t}.weight", M, K)
        c = _cosine(got, hf[key])
        worst = min(worst, c)
        print(f"  {key:5s} {str((M, K)):16s} cosine {c:.6f}")

    # Tie check: the bundle also ships a quantized lm_head; confirm it really is
    # the embedding (so using the lossless tied bf16 embed is right).
    emb = np.asarray(qm.bf16("model.embed_tokens.weight"), np.float32)
    lmq = qm.dequant("lm_head.weight", cfg.vocab_size, cfg.emb_dim)
    print(f"  tie   cosine(bundle lm_head, embed) {_cosine(lmq, emb):.6f} -> tied")

    print(f"\nworst cosine {worst:.6f} -> {'PASS' if worst > 0.99 else 'FAIL'}")
    return 0 if worst > 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
