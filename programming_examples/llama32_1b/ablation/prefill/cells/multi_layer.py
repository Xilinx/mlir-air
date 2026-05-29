# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""16-layer prefill wrapper.

Threads:  rms_gemms_rope[L] -> FA[L] -> o_ffn[L] -> rms_gemms_rope[L+1]

The cell-A/B/C/D dispatch strategy is independent of this wrapper; we
take the cell's per-kernel-group runner as a parameter.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from cells.flash_attn_const import run_flash_attn


def run_16_layer_prefill(
    cache,
    config,
    run_rms_gemms_rope,
    run_o_ffn,
    layer_inputs_per_layer,
):
    """Run a 16-layer prefill via the supplied per-kernel-group runners.

    Args:
        cache: shared KernelCache (FA + both groups + standalones all reside here)
        config: dict from cell_d_merged.CONFIG
        run_rms_gemms_rope(cache, layer_inputs, layer_idx) -> {normed,q,k,v,q_roped,k_roped, _wall_s}
        run_o_ffn(cache, layer_inputs, layer_idx) -> {output, _wall_s}
        layer_inputs_per_layer: list of N dicts, each with all per-layer weights+LUTs+x_in[layer 0 only]

    Returns dict with:
        per_layer_wall: list of N floats (wall time per layer including FA)
        total_wall: float
        final_output: numpy array (last layer's o_ffn output, reshaped to (seq, emb))
    """
    n_layers = len(layer_inputs_per_layer)
    per_layer_wall = []
    x_in = layer_inputs_per_layer[0]["x_in"]
    final_output = None

    t_total_start = time.perf_counter()
    for L in range(n_layers):
        layer_in = dict(layer_inputs_per_layer[L])
        layer_in["x_in"] = x_in  # threaded from previous layer

        t_layer_start = time.perf_counter()

        # 1. rms_gemms_rope
        rg_out = run_rms_gemms_rope(cache, layer_in, layer_idx=L)
        # 2. FA (invariant)
        # rms_gemms_rope returns 1D flat arrays; FA expects 2D (seq, dim)
        seq = config["seq_len"]
        emb = config["emb_dim"]
        kv = config["kv_dim"]
        q_roped_2d = rg_out["q_roped"].reshape(seq, emb)
        k_roped_2d = rg_out["k_roped"].reshape(seq, kv)
        v_2d = rg_out["v"].reshape(seq, kv)
        fa_out = run_flash_attn(cache, q_roped_2d, k_roped_2d, v_2d, layer_idx=L)
        # 3. o_ffn — assemble inputs
        of_in = {
            "attn_out": fa_out["attn_out"],
            "wo": layer_in["wo"],
            "x_residual": x_in,
            "ffn_norm_w": layer_in["ffn_norm_w"],
            "w_gate": layer_in["w_gate"],
            "w_up": layer_in["w_up"],
            "w_down": layer_in["w_down"],
        }
        of_out = run_o_ffn(cache, of_in, layer_idx=L)
        # The o_ffn output (slot 14) is 1D (n_total = seq*emb); reshape for next layer
        x_in = of_out["output"].reshape(config["seq_len"], config["emb_dim"])
        final_output = x_in

        per_layer_wall.append(time.perf_counter() - t_layer_start)

    total_wall = time.perf_counter() - t_total_start
    return {
        "per_layer_wall": per_layer_wall,
        "total_wall": total_wall,
        "final_output": final_output,
    }
