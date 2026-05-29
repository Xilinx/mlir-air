"""Regenerate decode golden fixtures by running Cell D for layer 0 at current_pos=7.

Uses deterministic synthetic inputs (numpy seed=42).
Outputs:
  golden/golden_rms_gemv_rope_decode.npz
  golden/golden_o_gemv_ffn_decode.npz
  golden/golden_meta.json

Usage:
  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 golden/regen_golden.py
"""

import hashlib
import json
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

# sys.path setup — make decode/, ablation/, llama32_1b/, programming_examples/ importable
_THIS = os.path.dirname(os.path.abspath(__file__))
_DECODE = os.path.dirname(_THIS)
_ABLATION = os.path.dirname(_DECODE)
_LLAMA = os.path.dirname(_ABLATION)
_PE = os.path.dirname(_LLAMA)
for p in (_PE, _LLAMA, _ABLATION, os.path.join(_ABLATION, "prefill"), _DECODE):
    if p not in sys.path:
        sys.path.insert(0, p)

from kernel_builder.cache import KernelCache
from cells.cell_d_merged import (
    compile_cell_d,
    preload_cell_d,
    run_rms_gemv_rope_d,
    run_o_gemv_ffn_d,
)

CONFIG = {
    "seq_len": 1,  # decode is single-token; seq_len present for shape-helper compatibility
    "emb_dim": 2048,
    "kv_dim": 512,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 64,
    "hidden_dim": 8192,
    "n_layers": 16,
    "max_seq": 2048,
    "vocab_size": 128256,
}

PROMPT_LEN = 7
CURRENT_POS = 7  # decode generates the token at position 7 (after a 7-token prefill)
SEED = 42


def synthetic_layer_weights(layer_idx, config, seed):
    """Per-layer weights — already in production-decode transposed shape.

    GEMV convention: W at slot 0 with shape (out_dim, in_dim). HuggingFace
    storage uses (out, in) too, but production pre-transposes; for synthetic
    inputs we just generate at the production shape directly.
    """
    rng = np.random.default_rng(seed + layer_idx)
    emb = config["emb_dim"]
    kv = config["kv_dim"]
    hid = config["hidden_dim"]
    return {
        "norm_w": rng.standard_normal(emb).astype(bfloat16),
        "wq": (rng.standard_normal((emb, emb)) * 0.02).astype(bfloat16),
        "wk": (rng.standard_normal((kv, emb)) * 0.02).astype(bfloat16),
        "wv": (rng.standard_normal((kv, emb)) * 0.02).astype(bfloat16),
        "wo": (rng.standard_normal((emb, emb)) * 0.02).astype(bfloat16),
        "ffn_norm_w": rng.standard_normal(emb).astype(bfloat16),
        "w_gate": (rng.standard_normal((hid, emb)) * 0.02).astype(bfloat16),
        "w_up": (rng.standard_normal((hid, emb)) * 0.02).astype(bfloat16),
        "w_down": (rng.standard_normal((emb, hid)) * 0.02).astype(bfloat16),
    }


def synthetic_x_in(config, seed):
    """The token's embedding entering layer 0."""
    rng = np.random.default_rng(seed + 9999)
    return rng.standard_normal(config["emb_dim"]).astype(bfloat16)


def synthetic_lut(config, seed):
    """Synthetic RoPE LUT slice at the timed current_pos (constant across trials)."""
    rng = np.random.default_rng(seed + 8888)
    emb = config["emb_dim"]
    kv = config["kv_dim"]
    return {
        "lut_q": rng.standard_normal(emb).astype(bfloat16),
        "lut_k": rng.standard_normal(kv).astype(bfloat16),
    }


def synthetic_attn_out(config, seed):
    """Synthetic post-attention activation entering o_gemv_ffn.

    For golden generation we don't actually run CPU attention — we just need
    a deterministic byte-stable input for the o_gemv_ffn golden. The validation
    gate compares Cell D against this golden in isolation; what feeds o_gemv_ffn
    in actual inference is decode_attention_cpu(q_roped, k/v cache, ...) but that
    data flow is exercised by the per-token loop test, not by this golden.
    """
    rng = np.random.default_rng(seed + 7777)
    return rng.standard_normal(config["emb_dim"]).astype(bfloat16)


def main():
    print("=" * 60)
    print("Plan 2 (full decode) golden regeneration")
    print(f"  current_pos={CURRENT_POS}, prompt_len={PROMPT_LEN}, seed={SEED}")
    print("=" * 60)

    cache_dir = os.path.join(_DECODE, "build")
    os.makedirs(cache_dir, exist_ok=True)
    cache = KernelCache(cache_dir=cache_dir, verbose=True)
    cache.load_manifest()

    # 1. Compile both ELFs
    print("\n[1/5] Compiling Cell D ELFs (rms_gemv_rope + o_gemv_ffn)...")
    compile_cell_d(cache, CONFIG)

    # 2. Generate synthetic per-layer weights (just layer 0 for goldens)
    print("\n[2/5] Generating synthetic weights for layer 0 (seed=42)...")
    weights_layer0 = synthetic_layer_weights(layer_idx=0, config=CONFIG, seed=SEED)
    lut = synthetic_lut(CONFIG, SEED)
    x_in = synthetic_x_in(CONFIG, SEED)
    attn_out_synth = synthetic_attn_out(CONFIG, SEED)

    # 3. Pre-load layer 0 weights into Cell D's BOs
    print("\n[3/5] Pre-loading layer 0 weights into Cell D BOs...")
    preload_cell_d(cache, [weights_layer0], lut["lut_q"], lut["lut_k"], CONFIG)

    # 4. Run rms_gemv_rope Cell D, capture outputs as golden
    print("\n[4/5] Running rms_gemv_rope (Cell D) → golden_rms_gemv_rope_decode.npz")
    rg_inputs = {
        "x_in": x_in,
        "norm_w": weights_layer0["norm_w"],
        "wq": weights_layer0["wq"],
        "wk": weights_layer0["wk"],
        "wv": weights_layer0["wv"],
        "lut_q": lut["lut_q"],
        "lut_k": lut["lut_k"],
    }
    rg_out = run_rms_gemv_rope_d(cache, rg_inputs, layer_idx=0)
    rg_path = os.path.join(_THIS, "golden_rms_gemv_rope_decode.npz")
    np.savez(
        rg_path,
        normed=rg_out["normed"],
        q=rg_out["q"],
        k=rg_out["k"],
        v=rg_out["v"],
        q_roped=rg_out["q_roped"],
        k_roped=rg_out["k_roped"],
    )
    print(f"  → wrote {rg_path}  ({os.path.getsize(rg_path)} bytes)")

    # 5. Run o_gemv_ffn Cell D with synthetic attn_out, capture output as golden
    print("\n[5/5] Running o_gemv_ffn (Cell D) → golden_o_gemv_ffn_decode.npz")
    of_inputs = {
        "wo": weights_layer0["wo"],
        "attn_out": attn_out_synth,
        "x_residual": x_in,
        "ffn_norm_w": weights_layer0["ffn_norm_w"],
        "w_gate": weights_layer0["w_gate"],
        "w_up": weights_layer0["w_up"],
        "w_down": weights_layer0["w_down"],
    }
    of_out = run_o_gemv_ffn_d(cache, of_inputs, layer_idx=0)
    of_path = os.path.join(_THIS, "golden_o_gemv_ffn_decode.npz")
    np.savez(of_path, output=of_out["output"])
    print(f"  → wrote {of_path}  ({os.path.getsize(of_path)} bytes)")

    # Meta JSON
    def _h(arr):
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    meta = {
        "config": CONFIG,
        "prompt_len": PROMPT_LEN,
        "current_pos": CURRENT_POS,
        "seed": SEED,
        "layer_idx": 0,
        "rms_gemv_rope_outputs": {
            k: _h(v) for k, v in rg_out.items() if not k.startswith("_")
        },
        "o_gemv_ffn_outputs": {"output": _h(of_out["output"])},
    }
    meta_path = os.path.join(_THIS, "golden_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  → wrote {meta_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
