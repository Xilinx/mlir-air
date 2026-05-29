"""Regenerate prefill golden fixtures by running Cell D once for each kernel-group.

Uses deterministic synthetic inputs (numpy seed=42 for layer 0).
Outputs:
  golden/golden_rms_gemms_rope_prefill.npz
  golden/golden_o_ffn_prefill.npz
  golden/golden_meta.json
"""

import hashlib
import json
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernel_builder.cache import KernelCache
from cells.cell_d_merged import (
    CONFIG,
    compile_cell_d_rms_gemms_rope,
    compile_cell_d_o_ffn,
    run_cell_d_rms_gemms_rope,
    run_cell_d_o_ffn,
)


def _synthetic_layer_inputs(layer_idx, config):
    """Deterministic synthetic inputs for one prefill layer (seq=2048).

    Same seeding scheme as Plan 1: seed = 42 + layer_idx.
    """
    rng = np.random.default_rng(42 + layer_idx)
    seq = config["seq_len"]
    emb = config["emb_dim"]
    kv = config["kv_dim"]
    hid = config["hidden_dim"]
    return {
        "x_in": rng.standard_normal((seq, emb)).astype(bfloat16),
        "norm_w": rng.standard_normal(emb).astype(bfloat16),
        "wq": rng.standard_normal((emb, emb)).astype(bfloat16),
        "wk": rng.standard_normal((emb, kv)).astype(bfloat16),
        "wv": rng.standard_normal((emb, kv)).astype(bfloat16),
        "lut_q": rng.standard_normal(seq * emb).astype(bfloat16),
        "lut_k": rng.standard_normal(seq * kv).astype(bfloat16),
        "wo": rng.standard_normal((emb, emb)).astype(bfloat16),
        "ffn_norm_w": rng.standard_normal(emb).astype(bfloat16),
        "w_gate": rng.standard_normal((emb, hid)).astype(bfloat16),
        "w_up": rng.standard_normal((emb, hid)).astype(bfloat16),
        "w_down": rng.standard_normal((hid, emb)).astype(bfloat16),
    }


def main():
    cache = KernelCache(cache_dir="standalone_cache", verbose=True)
    cache.load_manifest()
    compile_cell_d_rms_gemms_rope(cache)
    compile_cell_d_o_ffn(cache)

    inputs = _synthetic_layer_inputs(0, CONFIG)

    # rms_gemms_rope golden
    rg_inputs = {
        k: inputs[k] for k in ["x_in", "norm_w", "wq", "wk", "wv", "lut_q", "lut_k"]
    }
    rg_out = run_cell_d_rms_gemms_rope(cache, rg_inputs, layer_idx=0)
    rg_path = os.path.join(
        os.path.dirname(__file__), "golden_rms_gemms_rope_prefill.npz"
    )
    np.savez(rg_path, **{k: v for k, v in rg_out.items() if not k.startswith("_")})

    # For o_ffn golden, attn_out comes from FA in production. For the golden
    # we use a CPU FA reference computed from rg_out's q_roped/k_roped/v —
    # since FA is invariant across cells, all cells will see the same attn_out.
    # Simplest: synthesize attn_out from the same RNG (it is what flows into
    # o_ffn's slot 0 in every cell; the bytes are determined upstream).
    attn_out = (
        np.random.default_rng(42 + 0 + 1000)
        .standard_normal((CONFIG["seq_len"], CONFIG["emb_dim"]))
        .astype(bfloat16)
    )
    of_inputs = {
        "attn_out": attn_out,
        "wo": inputs["wo"],
        "x_residual": inputs["x_in"],  # the residual is the layer input
        "ffn_norm_w": inputs["ffn_norm_w"],
        "w_gate": inputs["w_gate"],
        "w_up": inputs["w_up"],
        "w_down": inputs["w_down"],
    }
    of_out = run_cell_d_o_ffn(cache, of_inputs, layer_idx=0)
    of_path = os.path.join(os.path.dirname(__file__), "golden_o_ffn_prefill.npz")
    np.savez(of_path, **{k: v for k, v in of_out.items() if not k.startswith("_")})

    meta = {
        "config": CONFIG,
        "rms_gemms_rope": {
            "input_hashes": {
                k: hashlib.sha256(v.tobytes()).hexdigest()[:16]
                for k, v in rg_inputs.items()
            },
            "output_hashes": {
                k: hashlib.sha256(v.tobytes()).hexdigest()[:16]
                for k, v in rg_out.items()
                if not k.startswith("_")
            },
        },
        "o_ffn": {
            "input_hashes": {
                k: hashlib.sha256(v.tobytes()).hexdigest()[:16]
                for k, v in of_inputs.items()
            },
            "output_hashes": {
                k: hashlib.sha256(v.tobytes()).hexdigest()[:16]
                for k, v in of_out.items()
                if not k.startswith("_")
            },
        },
    }
    with open(os.path.join(os.path.dirname(__file__), "golden_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {rg_path}, {of_path}, golden_meta.json")


if __name__ == "__main__":
    main()
