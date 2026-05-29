# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cell B -- Cell A + per-layer weight BOs + static_input_indices.

Same dataflow as Cell A (walks spec.sub_launches, threads via baton links),
but weights are pre-loaded once into per-layer BOs during preload phase.
The timed run phase skips the weight host->device sync via static_input_indices.

Two public phases:

  preload_cell_b(cache, spec, weights_per_layer, config, backend_preset)
      Called once before timing. For each (layer_idx, sub_launch):
        - Builds a 3-arg list with the actual weight at weight_slot_in_standalone
          and dummy zeros at all other slots.
        - Calls load_and_run with output_indices=[output_slot],
          static_input_indices={weight_slot}, and
          bo_key=f"B_{spec.name}_{sub.name}_L{layer_idx}".
      Sub-launches with weight_slot_in_standalone=None are skipped (no weight
      to preload; those sub-launches just use default bo_key in the timed run).

  run_cell_b(cache, spec, layer_inputs, config, backend_preset, layer_idx=0)
      Same loop as Cell A but:
        - No naive=True.
        - Passes static_input_indices={sub.weight_slot_in_standalone} (or empty
          set if None) and output_indices=[sub.output_slot_in_standalone].
        - Passes bo_key=f"B_{spec.name}_{sub.name}_L{layer_idx}" -- must
          byte-match the preload bo_key.

Helpers _output_shape_for and _static_input_for are imported from cell_a_naive
to avoid duplication.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from cells.cell_a_naive import _output_shape_for, _static_input_for
from cells.common import compile_standalone_kernels


def _activation_shape_for(spec_name, sub_name, config):
    """Return the numpy shape of the activation (non-weight, non-output) input slot.

    This is needed during preload to allocate a correctly-sized dummy BO for the
    activation slot. All current standalones have exactly 3 args:
    (activation, weight, output). The activation is always at slot 0.

    Shapes must match what _static_input_for / baton links would supply at
    run time, because the BO is allocated on the first call (preload) and
    reused on subsequent calls (run). A size mismatch raises a ValueError
    inside KernelCache.load_and_run when it tries to copy src into the BO.
    """
    seq = config["seq_len"]
    emb = config["emb_dim"]
    kv = config["kv_dim"]
    hid = config["hidden_dim"]

    if spec_name == "rms_gemms_rope":
        # All sub-launches: activation at slot 0 is either x_in (seq,emb) or
        # the normed/q/k output fed via baton -- all are (seq, emb) or (seq, kv).
        return {
            # rmsnorm: x_in is (seq, emb)
            "rmsnorm": (seq, emb),
            # gemms: A input is (seq, emb) -- the normed activation
            "q_gemm": (seq, emb),
            "k_gemm": (seq, emb),
            "v_gemm": (seq, emb),
            # ropes: activation slot is the q/k output
            "rope_q": (seq, emb),
            "rope_k": (seq, kv),
        }[sub_name]

    if spec_name == "o_ffn":
        return {
            # o_gemm: activation = attn_out (seq, emb)
            "o_gemm": (seq, emb),
            # ffn_rmsnorm: activation = res1 (seq, emb)
            "ffn_rmsnorm": (seq, emb),
            # gate/up gemms: activation = normed2 (seq, emb)
            "gate_gemm": (seq, emb),
            "up_gemm": (seq, emb),
            # down_gemm: activation = swiglu (seq, hid)
            "down_gemm": (seq, hid),
        }[sub_name]

    # ---- Decode (single-token, 1D activations) ----
    if spec_name == "rms_gemv_rope":
        # All activations are 1D. The activation slot is whichever non-weight,
        # non-output slot exists; preload sets a dummy of this size in any
        # missing slot.
        return {
            "rmsnorm": (emb,),  # x_in at slot 0
            "q_gemv": (emb,),  # x at slot 1 (input dim K=emb)
            "k_gemv": (emb,),  # x at slot 1
            "v_gemv": (emb,),  # x at slot 1
            "rope_q": (emb,),  # in at slot 0 (n_heads * head_dim = emb)
            "rope_k": (kv,),  # in at slot 0 (n_kv_heads * head_dim = kv)
        }[sub_name]

    if spec_name == "o_gemv_ffn":
        return {
            "o_gemv": (emb,),  # attn_out at slot 1
            "add_attn_residual": (emb,),  # A & B at slots 0,1 both (emb,)
            "ffn_rmsnorm": (emb,),  # res1 at slot 0
            "gate_gemv": (emb,),  # normed2 at slot 1 (input dim K=emb)
            "up_gemv": (emb,),  # normed2 at slot 1
            "swiglu": (hid,),  # gate, up both (hid,)
            "down_gemv_k8192": (hid,),  # swiglu at slot 1 (input dim K=hid)
            "add_ffn_residual": (emb,),  # A & B at slots 0,1
        }[sub_name]

    raise ValueError(f"unknown spec {spec_name!r} or sub {sub_name!r}")


def compile_cell_b(cache, spec, backend_preset):
    """Compile the standalone ELFs for this kernel-group into cache."""
    registry = [(s.name, s.builder_ref, s.build_kwargs) for s in spec.sub_launches]
    compile_standalone_kernels(cache, spec.name, registry, backend_preset)


def preload_cell_b(cache, spec, weights_per_layer, config, backend_preset):
    """Pre-load per-layer weights into dedicated BOs.

    For each (layer_idx, weights) pair and each sub-launch with a weight slot,
    run a one-shot load_and_run that writes the weight into the BO. Subsequent
    timed runs reuse the same BO (identified by bo_key) and skip the write.

    Args:
        cache: KernelCache with manifested artifacts.
        spec: KernelGroupSpec (rms_gemms_rope or o_ffn).
        weights_per_layer: list of dicts (one per layer), each keyed by semantic
            weight name (same keys accepted by _static_input_for / Cell A).
        config: dict with seq_len, emb_dim, kv_dim, hidden_dim.
        backend_preset: backend kwargs dict (instance_name will be removed).
    """
    backend = {**backend_preset}
    backend.pop("instance_name", None)

    for layer_idx, layer_weights in enumerate(weights_per_layer):
        for sub in spec.sub_launches:
            if sub.weight_slot_in_standalone is None:
                # No weight slot -- nothing to preload for this sub-launch.
                continue

            out_shape = _output_shape_for(spec.name, sub.name, config)
            out_buf = np.zeros(out_shape, dtype=bfloat16)

            # Build the 3-arg list: weight at weight_slot, output at output_slot,
            # dummy zeros at remaining slot(s).
            args = [None, None, None]
            weight_slot = sub.weight_slot_in_standalone
            output_slot = sub.output_slot_in_standalone
            args[output_slot] = out_buf

            # Retrieve the weight array using the same lookup as Cell A.
            weight_arr = _static_input_for(
                spec.name, sub.name, weight_slot, layer_weights
            )
            assert weight_arr is not None, (
                f"[cell_b preload] _static_input_for returned None for "
                f"{spec.name}/{sub.name} slot={weight_slot}. "
                f"Check weight keys in weights_per_layer."
            )
            args[weight_slot] = weight_arr

            # Fill any remaining slot with a correctly-sized dummy zero array.
            # The BO is allocated on this first call and reused in run_cell_b;
            # the size must match what the real activation will supply.
            for slot in range(3):
                if args[slot] is None:
                    act_shape = _activation_shape_for(spec.name, sub.name, config)
                    args[slot] = np.zeros(act_shape, dtype=bfloat16)

            bo_key = f"B_{spec.name}_{sub.name}_L{layer_idx}"
            kernel_name = f"{spec.name}__{sub.name}"

            cache.load_and_run(
                kernel_name,
                backend,
                *args,
                output_indices=[output_slot],
                static_input_indices={weight_slot},
                bo_key=bo_key,
            )


def run_cell_b(cache, spec, layer_inputs, config, backend_preset, layer_idx=0):
    """Run all spec.sub_launches sequentially with pre-loaded weight BOs.

    Same dataflow as Cell A (batons via results dict) but:
      - Uses static_input_indices={weight_slot} to skip weight write on this call.
      - Uses output_indices=[output_slot] instead of naive read-all.
      - Uses bo_key matching the preload phase so the same BO set is reused.

    Sub-launches with weight_slot_in_standalone=None (e.g. swiglu, ffn_add)
    have no static weight -- they use an empty static_input_indices set and
    the same bo_key pattern for BO identity.

    Args:
        cache: KernelCache with manifested artifacts.
        spec: KernelGroupSpec (rms_gemms_rope or o_ffn).
        layer_inputs: dict of numpy arrays keyed by semantic name.
        config: dict with seq_len, emb_dim, kv_dim, hidden_dim.
        backend_preset: backend kwargs dict (instance_name will be removed).
        layer_idx: layer index used to select the right pre-loaded BO set.

    Returns:
        dict keyed by sub.name -> 1D flat numpy array of that sub-launch's
        output, plus "_wall_s" for total wall time.
    """
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
                f"[cell_b] no source found for {spec.name}/{sub.name} slot={slot}. "
                f"Check baton_links and _static_input_for."
            )

        # Determine static_input_indices for this sub-launch.
        if sub.weight_slot_in_standalone is not None:
            static_indices = {sub.weight_slot_in_standalone}
        else:
            static_indices = set()

        kernel_name = f"{spec.name}__{sub.name}"
        bo_key = f"B_{spec.name}_{sub.name}_L{layer_idx}"

        result = cache.load_and_run(
            kernel_name,
            backend,
            *args,
            output_indices=[sub.output_slot_in_standalone],
            static_input_indices=static_indices,
            bo_key=bo_key,
        )
        results[sub.name] = result[sub.output_slot_in_standalone]

    elapsed = time.perf_counter() - t0
    results["_wall_s"] = elapsed
    return results
