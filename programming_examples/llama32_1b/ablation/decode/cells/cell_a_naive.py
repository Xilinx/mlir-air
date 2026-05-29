# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cell A -- Naive no-merge for a generic KernelGroupSpec.

Walks spec.sub_launches in order. For each sub-launch:
  1. Build the 3-element args list per the spec's slot semantics.
  2. Invoke cache.load_and_run with naive=True (writes everything,
     reads everything every call).
  3. Store output in results dict keyed by sub.name.

Cross-sub-launch data flows via the host (extracted to numpy in a results
dict, then passed to the next call as input).

naive=True forces load_and_run to:
  - set output_indices = list(range(len(inputs)))  (read back all slots)
  - skip static_input_indices and intermediate_indices optimizations

The returned result[slot] is always a 1D flat numpy array. Baton-link values
are passed directly as inputs to downstream sub-launches; the BO write uses
raw bytes so 1D vs 2D shape does not matter as long as byte counts match.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from cells.common import compile_standalone_kernels


def _output_shape_for(spec_name, sub_name, config):
    """Return numpy shape of the output buffer for (spec_name, sub_name).

    The output buffer is allocated as zeros with this shape and passed at
    sub.output_slot_in_standalone. The kernel writes into it; load_and_run
    returns a 1D flat view (byte-compatible with the 2D shape).
    """
    seq = config["seq_len"]
    emb = config["emb_dim"]
    kv = config["kv_dim"]
    hid = config["hidden_dim"]
    n_total = seq * emb

    if spec_name == "rms_gemms_rope":
        return {
            "rmsnorm": (seq, emb),
            "q_gemm": (seq, emb),
            "k_gemm": (seq, kv),
            "v_gemm": (seq, kv),
            "rope_q": (seq, emb),
            "rope_k": (seq, kv),
        }[sub_name]

    if spec_name == "o_ffn":
        return {
            "o_gemm": (seq, emb),
            "res_add": (seq, emb),
            "ffn_rmsnorm": (seq, emb),
            "gate_gemm": (seq, hid),
            "up_gemm": (seq, hid),
            "swiglu": (seq, hid),
            "down_gemm": (seq, emb),
            "ffn_add": (n_total,),  # 1D output (standalone emits 1D; see o_ffn.py)
        }[sub_name]

    # ---- Decode (single-token, 1D outputs) ----
    if spec_name == "rms_gemv_rope":
        return {
            "rmsnorm": (emb,),
            "q_gemv": (emb,),
            "k_gemv": (kv,),
            "v_gemv": (kv,),
            "rope_q": (emb,),  # n_heads * head_dim = 32*64 = emb
            "rope_k": (kv,),  # n_kv_heads * head_dim = 8*64 = kv
        }[sub_name]

    if spec_name == "o_gemv_ffn":
        return {
            "o_gemv": (emb,),
            "add_attn_residual": (emb,),
            "ffn_rmsnorm": (emb,),
            "gate_gemv": (hid,),
            "up_gemv": (hid,),
            "swiglu": (hid,),
            "down_gemv_k8192": (emb,),
            "add_ffn_residual": (emb,),
        }[sub_name]

    raise ValueError(f"unknown spec {spec_name!r}")


def _static_input_for(spec_name, sub_name, slot, layer_inputs):
    """Return the static (weight/LUT/layer-level) array for this slot, or None.

    Returns None when the slot should come from a baton link (upstream
    sub-launch output) or from the output buffer.
    """
    if spec_name == "rms_gemms_rope":
        # Slot conventions (from rms_gemms_rope.py docstring):
        #   rmsnorm:  (x_in[slot0], norm_w[slot1], out[slot2])
        #   gemm:     (A[slot0],    B_weight[slot1], C[slot2])
        #   rope_2d:  (in[slot0],   lut[slot1],      out[slot2])
        if sub_name == "rmsnorm":
            if slot == 0:
                return layer_inputs["x_in"]
            if slot == 1:
                return layer_inputs["norm_w"]
        elif sub_name == "q_gemm":
            if slot == 1:
                return layer_inputs["wq"]
            # slot 0 comes from rmsnorm baton
        elif sub_name == "k_gemm":
            if slot == 1:
                return layer_inputs["wk"]
            # slot 0 comes from rmsnorm baton
        elif sub_name == "v_gemm":
            if slot == 1:
                return layer_inputs["wv"]
            # slot 0 comes from rmsnorm baton
        elif sub_name == "rope_q":
            if slot == 1:
                return layer_inputs["lut_q"]
            # slot 0 comes from q_gemm baton
        elif sub_name == "rope_k":
            if slot == 1:
                return layer_inputs["lut_k"]
            # slot 0 comes from k_gemm baton
        return None

    if spec_name == "o_ffn":
        # Slot conventions (from o_ffn.py docstring):
        #   gemm:         (A[slot0], B_weight[slot1], C[slot2])
        #   add_2d_to_2d: (A[slot0], B[slot1],        C[slot2])   no weight
        #   rmsnorm:      (x[slot0], w[slot1],         out[slot2])
        #   swiglu_2d:    (gate[slot0], up[slot1],     out[slot2]) no weight
        #   ffn_add:      (A[slot0], B[slot1],          out[slot2]) no weight
        if sub_name == "o_gemm":
            if slot == 0:
                return layer_inputs["attn_out"]
            if slot == 1:
                return layer_inputs["wo"]
        elif sub_name == "res_add":
            # slot0 = proj (from o_gemm baton); slot1 = x_residual (static)
            if slot == 1:
                return layer_inputs["x_residual"]
        elif sub_name == "ffn_rmsnorm":
            if slot == 1:
                return layer_inputs["ffn_norm_w"]
            # slot 0 comes from res_add baton
        elif sub_name == "gate_gemm":
            if slot == 1:
                return layer_inputs["w_gate"]
            # slot 0 comes from ffn_rmsnorm baton
        elif sub_name == "up_gemm":
            if slot == 1:
                return layer_inputs["w_up"]
            # slot 0 comes from ffn_rmsnorm baton
        elif sub_name == "swiglu":
            # both slot0 (gate) and slot1 (up) come from batons
            pass
        elif sub_name == "down_gemm":
            if slot == 1:
                return layer_inputs["w_down"]
            # slot 0 comes from swiglu baton
        elif sub_name == "ffn_add":
            # slot0 = down (from down_gemm baton); slot1 = res1 (from res_add baton)
            pass
        return None

    # ---- Decode kernel-groups ----
    # CRITICAL: GEMV slot convention differs from prefill GEMM!
    #   gemv: (W_weight[slot0], x[slot1], y[slot2])  ← W is at slot 0, NOT slot 1
    if spec_name == "rms_gemv_rope":
        # Slot conventions for decode rms_gemv_rope sub-launches:
        #   rmsnorm: (x_in[slot0], norm_w[slot1], out[slot2])
        #   gemv:    (W[slot0],    x[slot1],      y[slot2])
        #   rope:    (in[slot0],   lut[slot1],    out[slot2])
        if sub_name == "rmsnorm":
            if slot == 0:
                return layer_inputs["x_in"]
            if slot == 1:
                return layer_inputs["norm_w"]
        elif sub_name == "q_gemv":
            if slot == 0:
                return layer_inputs["wq"]
            # slot 1 (x = normed) comes from rmsnorm baton
        elif sub_name == "k_gemv":
            if slot == 0:
                return layer_inputs["wk"]
        elif sub_name == "v_gemv":
            if slot == 0:
                return layer_inputs["wv"]
        elif sub_name == "rope_q":
            if slot == 1:
                return layer_inputs["lut_q"]
            # slot 0 (in = q) comes from q_gemv baton
        elif sub_name == "rope_k":
            if slot == 1:
                return layer_inputs["lut_k"]
        return None

    if spec_name == "o_gemv_ffn":
        # Slot conventions for decode o_gemv_ffn sub-launches:
        #   gemv:    (W[slot0],    x[slot1],     y[slot2])
        #   add:     (A[slot0],    B[slot1],     out[slot2])  no weight
        #   rmsnorm: (x[slot0],    w[slot1],     out[slot2])
        #   swiglu:  (gate[slot0], up[slot1],    out[slot2])  no weight
        if sub_name == "o_gemv":
            if slot == 0:
                return layer_inputs["wo"]
            if slot == 1:
                return layer_inputs["attn_out"]
        elif sub_name == "add_attn_residual":
            # slot 0 = proj (from o_gemv baton); slot 1 = x_residual
            if slot == 1:
                return layer_inputs["x_residual"]
        elif sub_name == "ffn_rmsnorm":
            if slot == 1:
                return layer_inputs["ffn_norm_w"]
            # slot 0 (x = res1) comes from add_attn_residual baton
        elif sub_name == "gate_gemv":
            if slot == 0:
                return layer_inputs["w_gate"]
            # slot 1 (x = normed2) comes from ffn_rmsnorm baton
        elif sub_name == "up_gemv":
            if slot == 0:
                return layer_inputs["w_up"]
        elif sub_name == "swiglu":
            # both slot 0 (gate) and slot 1 (up) come from batons
            pass
        elif sub_name == "down_gemv_k8192":
            if slot == 0:
                return layer_inputs["w_down"]
            # slot 1 (x = swiglu) comes from swiglu baton
        elif sub_name == "add_ffn_residual":
            # slot 0 = down (from down_gemv baton); slot 1 = res1 (from add_attn baton)
            pass
        return None

    raise ValueError(f"unknown spec {spec_name!r}")


def compile_cell_a(cache, spec, backend_preset):
    """Compile the standalone ELFs for this kernel-group into cache."""
    registry = [(s.name, s.builder_ref, s.build_kwargs) for s in spec.sub_launches]
    compile_standalone_kernels(cache, spec.name, registry, backend_preset)


def run_cell_a(cache, spec, layer_inputs, config, backend_preset, layer_idx=0):
    """Run all spec.sub_launches sequentially with naive=True.

    Each sub-launch is a separate xrt.run() call. All host<->device transfers
    are done unconditionally (naive=True means no skipping of static or
    intermediate buffers).

    Args:
        cache: KernelCache with manifested artifacts.
        spec: KernelGroupSpec (rms_gemms_rope or o_ffn).
        layer_inputs: dict of numpy arrays keyed by semantic name
            (e.g. "x_in", "norm_w", "wq", "attn_out", etc.).
        config: dict with seq_len, emb_dim, kv_dim, hidden_dim.
        backend_preset: backend kwargs dict (instance_name will be removed).
        layer_idx: layer index (unused in Cell A, present for API consistency).

    Returns:
        dict keyed by sub.name -> 1D flat numpy array of that sub-launch's
        output, plus "_wall_s" for total wall time.
    """
    # Strip instance_name; compile_cell_a sets it per-kernel.
    backend = {**backend_preset}
    backend.pop("instance_name", None)

    results = {}
    t0 = time.perf_counter()

    for idx, sub in enumerate(spec.sub_launches):
        out_shape = _output_shape_for(spec.name, sub.name, config)
        out_buf = np.zeros(out_shape, dtype=bfloat16)

        # Build the 3-arg list (all standalones have exactly 3 args).
        args = [None, None, None]

        for slot in range(3):
            if slot == sub.output_slot_in_standalone:
                args[slot] = out_buf
                continue

            # Try static (weight/layer-level) lookup first.
            v = _static_input_for(spec.name, sub.name, slot, layer_inputs)
            if v is not None:
                args[slot] = v
                continue

            # Otherwise this slot is fed by an upstream baton link.
            for link in spec.baton_links:
                if link.consumer_idx == idx and link.consumer_in_slot == slot:
                    producer_name = spec.sub_launches[link.producer_idx].name
                    args[slot] = results[producer_name]
                    break

            assert args[slot] is not None, (
                f"[cell_a] no source found for {spec.name}/{sub.name} slot={slot}. "
                f"Check baton_links and _static_input_for."
            )

        kernel_name = f"{spec.name}__{sub.name}"
        result = cache.load_and_run(
            kernel_name,
            backend,
            *args,
            naive=True,
        )
        # naive=True sets output_indices = list(range(3)), so result is a 3-tuple.
        # The output is at sub.output_slot_in_standalone.
        results[sub.name] = result[sub.output_slot_in_standalone]

    elapsed = time.perf_counter() - t0
    results["_wall_s"] = elapsed
    return results
