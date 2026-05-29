# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cell D — production: invoke the merged ELFs (rms_gemms_rope.elf with 6
launches; o_ffn.elf with 8 launches) using the production KernelCache +
backend presets.
"""

import os
import sys

# Ensure llama32_1b/ is on sys.path so kernel_builder and multi_launch_builder
# are importable whether this file is run directly or imported from the
# prefill/ package root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LLAMA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _LLAMA_DIR not in sys.path:
    sys.path.insert(0, _LLAMA_DIR)

import time

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import RMS_GEMMS_ROPE_BACKEND, O_FFN_BACKEND
from multi_launch_builder.rms_gemms_rope_multi import build_rms_gemms_rope_module
from multi_launch_builder.o_ffn_multi import build_o_ffn_module

CONFIG = {
    "seq_len": 2048,
    "emb_dim": 2048,
    "kv_dim": 512,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 64,
    "hidden_dim": 8192,
}


def compile_cell_d_rms_gemms_rope(cache: KernelCache):
    if "rms_gemms_rope" in cache.artifacts:
        return
    mod = build_rms_gemms_rope_module(
        seq_len=CONFIG["seq_len"],
        emb_dim=CONFIG["emb_dim"],
        kv_dim=CONFIG["kv_dim"],
        n_heads=CONFIG["n_heads"],
        n_kv_heads=CONFIG["n_kv_heads"],
        head_dim=CONFIG["head_dim"],
    )
    cache.compile_and_cache(
        "rms_gemms_rope", mod, {"verbose": cache.verbose, **RMS_GEMMS_ROPE_BACKEND}
    )
    cache._save_manifest()


def compile_cell_d_o_ffn(cache: KernelCache):
    if "o_ffn" in cache.artifacts:
        return
    mod = build_o_ffn_module(
        seq_len=CONFIG["seq_len"],
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
    )
    cache.compile_and_cache("o_ffn", mod, {"verbose": cache.verbose, **O_FFN_BACKEND})
    cache._save_manifest()


def run_cell_d_rms_gemms_rope(cache, layer_inputs, layer_idx=0):
    """One rms_gemms_rope call (6 launches in one xrt.run).
    layer_inputs has keys: x_in, norm_w, wq, wk, wv, lut_q, lut_k.
    Returns dict with normed, q, k, v, q_roped, k_roped, _wall_s.
    """
    seq = CONFIG["seq_len"]
    emb = CONFIG["emb_dim"]
    kv = CONFIG["kv_dim"]
    args = [
        layer_inputs["x_in"],
        layer_inputs["norm_w"],
        np.zeros((seq, emb), dtype=bfloat16),  # normed
        layer_inputs["wq"],
        np.zeros((seq, emb), dtype=bfloat16),  # q
        layer_inputs["wk"],
        np.zeros((seq, kv), dtype=bfloat16),  # k
        layer_inputs["wv"],
        np.zeros((seq, kv), dtype=bfloat16),  # v
        layer_inputs["lut_q"],
        layer_inputs["lut_k"],
        np.zeros((seq, emb), dtype=bfloat16),  # q_roped
        np.zeros((seq, kv), dtype=bfloat16),  # k_roped
    ]
    t0 = time.perf_counter()
    out = cache.load_and_run(
        "rms_gemms_rope",
        RMS_GEMMS_ROPE_BACKEND,
        *args,
        output_indices=[2, 4, 6, 8, 11, 12],
        static_input_indices={1, 3, 5, 7, 9, 10},
        intermediate_indices={2, 4, 6, 8, 11, 12},
        bo_key=f"D_rms_gemms_rope_L{layer_idx}",
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


def run_cell_d_o_ffn(cache, layer_inputs, layer_idx=0):
    """One o_ffn call (8 launches in one xrt.run).
    layer_inputs has: attn_out, wo, x_residual, ffn_norm_w, w_gate, w_up, w_down.
    Returns dict with output, _wall_s.
    """
    seq = CONFIG["seq_len"]
    emb = CONFIG["emb_dim"]
    hid = CONFIG["hidden_dim"]
    n_total = seq * emb
    args = [
        layer_inputs["attn_out"],  # 0
        layer_inputs["wo"],  # 1
        np.zeros((seq, emb), dtype=bfloat16),  # 2 proj
        layer_inputs["x_residual"],  # 3
        np.zeros((seq, emb), dtype=bfloat16),  # 4 res1
        layer_inputs["ffn_norm_w"],  # 5
        np.zeros((seq, emb), dtype=bfloat16),  # 6 normed2
        layer_inputs["w_gate"],  # 7
        np.zeros((seq, hid), dtype=bfloat16),  # 8 gate
        layer_inputs["w_up"],  # 9
        np.zeros((seq, hid), dtype=bfloat16),  # 10 up
        np.zeros((seq, hid), dtype=bfloat16),  # 11 swiglu
        layer_inputs["w_down"],  # 12
        np.zeros((seq, emb), dtype=bfloat16),  # 13 down
        np.zeros(n_total, dtype=bfloat16),  # 14 output (1D)
    ]
    t0 = time.perf_counter()
    out = cache.load_and_run(
        "o_ffn",
        O_FFN_BACKEND,
        *args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
        bo_key=f"D_o_ffn_L{layer_idx}",
    )
    return {"output": out[14], "_wall_s": time.perf_counter() - t0}
