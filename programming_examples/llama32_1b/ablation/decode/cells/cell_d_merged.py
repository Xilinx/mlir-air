"""Cell D — production-merged decode ELFs.

Compiles and invokes:
- rms_gemv_rope.elf (6 stitched launches in one xrt.run)
- o_gemv_ffn.elf   (8 stitched launches in one xrt.run)

Mirrors production llama32_1b_inference.py decode dispatch (static_input_indices
+ bo_key per layer). The lm_head_gemv ELF is compiled here too but invoked via
cells.lm_head_const (held INVARIANT across cells).

Three public functions:
- compile_cell_d(cache, config): compile rms_gemv_rope + o_gemv_ffn ELFs.
- preload_cell_d(cache, weights_per_layer, rope_lut_pos_q, rope_lut_pos_k, config):
    one-time per-layer BO + weight preload.
- run_rms_gemv_rope_d(cache, layer_inputs, layer_idx) → dict.
- run_o_gemv_ffn_d(cache, layer_inputs, layer_idx) → dict.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import RGR_BACKEND, OGF_BACKEND

# Production decode static_input_indices (mirrors llama32_1b_inference.py preload):
#   rms_gemv_rope: {1, 3, 5, 7} = norm_w, wq, wk, wv  (LUTs at 9, 10 NOT static)
#   o_gemv_ffn:    {0, 5, 7, 9, 12} = wo, ffn_norm_w, w_gate, w_up, w_down
_RGR_STATIC = {1, 3, 5, 7}
_RGR_INTERMEDIATE = {2, 4, 6, 8, 11, 12}
_OGF_STATIC = {0, 5, 7, 9, 12}
_OGF_INTERMEDIATE = {2, 4, 6, 8, 10, 11, 13, 14}


def compile_cell_d(cache: KernelCache, config):
    """Compile production rms_gemv_rope and o_gemv_ffn ELFs (one-time)."""
    if "rms_gemv_rope" not in cache.artifacts:
        from multi_launch_builder.rms_gemv_rope_multi import build_rms_gemv_rope_module

        mod = build_rms_gemv_rope_module(
            emb_dim=config["emb_dim"],
            kv_dim=config["kv_dim"],
            n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"],
            head_dim=config["head_dim"],
        )
        cache.compile_and_cache(
            "rms_gemv_rope",
            mod,
            {**RGR_BACKEND, "verbose": getattr(cache, "verbose", False)},
        )
        cache._save_manifest()

    if "o_gemv_ffn" not in cache.artifacts:
        from multi_launch_builder.o_gemv_ffn_multi import build_o_gemv_ffn_module

        mod = build_o_gemv_ffn_module(
            emb_dim=config["emb_dim"],
            hidden_dim=config["hidden_dim"],
        )
        cache.compile_and_cache(
            "o_gemv_ffn",
            mod,
            {**OGF_BACKEND, "verbose": getattr(cache, "verbose", False)},
        )
        cache._save_manifest()


def preload_cell_d(cache, weights_per_layer, lut_q, lut_k, config):
    """Pre-load per-layer weights into per-layer BOs.

    Mirrors production llama32_1b_inference.py preload pattern. After this,
    each layer's BO set holds its weights resident on the NPU; subsequent
    run_*_d calls only upload activations (slot 0/1) and LUTs (9, 10).
    """
    emb = config["emb_dim"]
    kv = config["kv_dim"]
    hid = config["hidden_dim"]

    for layer_idx, w in enumerate(weights_per_layer):
        # rms_gemv_rope: 13 args
        cache.load_and_run(
            "rms_gemv_rope",
            RGR_BACKEND,
            np.zeros(emb, dtype=bfloat16),  # 0 x_in (placeholder)
            w["norm_w"],  # 1 (static)
            np.zeros(emb, dtype=bfloat16),  # 2 normed
            w["wq"],  # 3 (static)
            np.zeros(emb, dtype=bfloat16),  # 4 q
            w["wk"],  # 5 (static)
            np.zeros(kv, dtype=bfloat16),  # 6 k
            w["wv"],  # 7 (static)
            np.zeros(kv, dtype=bfloat16),  # 8 v
            lut_q,  # 9 (NOT static)
            lut_k,  # 10 (NOT static)
            np.zeros(emb, dtype=bfloat16),  # 11 q_roped
            np.zeros(kv, dtype=bfloat16),  # 12 k_roped
            output_indices=[8, 11, 12],
            static_input_indices=_RGR_STATIC,
            intermediate_indices=_RGR_INTERMEDIATE,
            bo_key=f"D_rms_gemv_rope_L{layer_idx}",
        )

        # o_gemv_ffn: 15 args
        cache.load_and_run(
            "o_gemv_ffn",
            OGF_BACKEND,
            w["wo"],  # 0 (static)
            np.zeros(emb, dtype=bfloat16),  # 1 attn_out (placeholder)
            np.zeros(emb, dtype=bfloat16),  # 2 proj
            np.zeros(emb, dtype=bfloat16),  # 3 x_residual (placeholder)
            np.zeros(emb, dtype=bfloat16),  # 4 res1
            w["ffn_norm_w"],  # 5 (static)
            np.zeros(emb, dtype=bfloat16),  # 6 normed2
            w["w_gate"],  # 7 (static)
            np.zeros(hid, dtype=bfloat16),  # 8 gate
            w["w_up"],  # 9 (static)
            np.zeros(hid, dtype=bfloat16),  # 10 up
            np.zeros(hid, dtype=bfloat16),  # 11 swiglu
            w["w_down"],  # 12 (static)
            np.zeros(emb, dtype=bfloat16),  # 13 down
            np.zeros(emb, dtype=bfloat16),  # 14 output
            output_indices=[14],
            static_input_indices=_OGF_STATIC,
            intermediate_indices=_OGF_INTERMEDIATE,
            bo_key=f"D_o_gemv_ffn_L{layer_idx}",
        )


def run_rms_gemv_rope_d(cache, layer_inputs, layer_idx=0):
    """Production merged dispatch — 6 stitched launches in 1 xrt.run.

    layer_inputs keys: x_in, norm_w, wq, wk, wv, lut_q, lut_k.
    Returns dict with normed, q, k, v, q_roped, k_roped, _wall_s.
    """
    emb = layer_inputs["x_in"].shape[-1]
    # Determine kv_dim from wk shape (W is at slot 0 of GEMV, shape [kv, emb])
    kv = layer_inputs["wk"].shape[0]

    args = [
        layer_inputs["x_in"].astype(bfloat16).flatten(),  # 0
        layer_inputs["norm_w"].astype(bfloat16),  # 1 (static)
        np.zeros(emb, dtype=bfloat16),  # 2 normed
        layer_inputs["wq"],  # 3 (static)
        np.zeros(emb, dtype=bfloat16),  # 4 q
        layer_inputs["wk"],  # 5 (static)
        np.zeros(kv, dtype=bfloat16),  # 6 k
        layer_inputs["wv"],  # 7 (static)
        np.zeros(kv, dtype=bfloat16),  # 8 v
        layer_inputs["lut_q"].astype(bfloat16),  # 9
        layer_inputs["lut_k"].astype(bfloat16),  # 10
        np.zeros(emb, dtype=bfloat16),  # 11 q_roped
        np.zeros(kv, dtype=bfloat16),  # 12 k_roped
    ]
    t0 = time.perf_counter()
    out = cache.load_and_run(
        "rms_gemv_rope",
        RGR_BACKEND,
        *args,
        output_indices=[2, 4, 6, 8, 11, 12],
        static_input_indices=_RGR_STATIC,
        intermediate_indices=_RGR_INTERMEDIATE,
        bo_key=f"D_rms_gemv_rope_L{layer_idx}",
    )
    elapsed = time.perf_counter() - t0
    return {
        "normed": out[2],
        "q": out[4],
        "k": out[6],
        "v": out[8],
        "q_roped": out[11],
        "k_roped": out[12],
        "_wall_s": elapsed,
    }


def run_o_gemv_ffn_d(cache, layer_inputs, layer_idx=0):
    """Production merged dispatch — 8 stitched launches in 1 xrt.run.

    layer_inputs keys: wo, attn_out, x_residual, ffn_norm_w, w_gate, w_up, w_down.
    Returns dict with output, _wall_s.
    """
    emb = layer_inputs["attn_out"].shape[-1]
    hid = layer_inputs["w_gate"].shape[0]

    args = [
        layer_inputs["wo"],  # 0 (static)
        layer_inputs["attn_out"].astype(bfloat16).flatten(),  # 1
        np.zeros(emb, dtype=bfloat16),  # 2 proj
        layer_inputs["x_residual"].astype(bfloat16).flatten(),  # 3
        np.zeros(emb, dtype=bfloat16),  # 4 res1
        layer_inputs["ffn_norm_w"].astype(bfloat16),  # 5 (static)
        np.zeros(emb, dtype=bfloat16),  # 6 normed2
        layer_inputs["w_gate"],  # 7 (static)
        np.zeros(hid, dtype=bfloat16),  # 8 gate
        layer_inputs["w_up"],  # 9 (static)
        np.zeros(hid, dtype=bfloat16),  # 10 up
        np.zeros(hid, dtype=bfloat16),  # 11 swiglu
        layer_inputs["w_down"],  # 12 (static)
        np.zeros(emb, dtype=bfloat16),  # 13 down
        np.zeros(emb, dtype=bfloat16),  # 14 output
    ]
    t0 = time.perf_counter()
    out = cache.load_and_run(
        "o_gemv_ffn",
        OGF_BACKEND,
        *args,
        output_indices=[14],
        static_input_indices=_OGF_STATIC,
        intermediate_indices=_OGF_INTERMEDIATE,
        bo_key=f"D_o_gemv_ffn_L{layer_idx}",
    )
    elapsed = time.perf_counter() - t0
    return {"output": out[14], "_wall_s": elapsed}
