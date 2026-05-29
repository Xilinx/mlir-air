# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cell C -- Cell B + shared intermediate BOs across separate xrt.run() calls,
parameterized over a KernelGroupSpec. Walks spec.baton_links to alias BOs.

Two public phases:

  preload_cell_c(cache, spec, weights_per_layer, config, backend_preset)
      Called once before timing. For each (layer_idx, layer_weights) pair:
        1. Run each sub-launch once (allocates BOs and writes weights via
           static_input_indices). Uses bo_key=f"C_{spec.name}_{sub.name}_L{li}".
        2. Walk spec.baton_links and alias each producer's output BO into
           the consumer's input BO slot via _share_bo.

  run_cell_c(cache, spec, layer_inputs, config, backend_preset, layer_idx=0)
      Same dataflow as Cell B but with:
        - bo_key=f"C_{spec.name}_{sub.name}_L{layer_idx}" (matches preload).
        - intermediate_indices: producer output slots and consumer input slots
          that are baton-managed (host skips writing those BOs).

For a baton-aliased slot, a np.zeros placeholder is passed to load_and_run;
the bytes are NOT written to device because the slot is in intermediate_indices.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from cells.cell_a_naive import _output_shape_for, _static_input_for
from cells.common import compile_standalone_kernels, _share_bo

# ---------------------------------------------------------------------------
# Compile (same registry walk as Cell A / Cell B)
# ---------------------------------------------------------------------------


def compile_cell_c(cache, spec, backend_preset):
    """Compile the standalone ELFs for this kernel-group into cache."""
    registry = [(s.name, s.builder_ref, s.build_kwargs) for s in spec.sub_launches]
    compile_standalone_kernels(cache, spec.name, registry, backend_preset)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _slot_shape_for(spec_name, sub_name, slot, config):
    """Return the numpy shape for an arbitrary (sub_name, slot) pair.

    Covers both weight slots and activation/baton slots so that the preload
    loop can allocate correctly-sized BOs for all sub-launches, including
    those with no weight slot (res_add, swiglu, ffn_add).

    For weight slots this returns the weight shape (2-D for GEMMs, 1-D for
    norms/LUTs).  For activation/baton slots it returns the activation shape.
    """
    seq = config["seq_len"]
    emb = config["emb_dim"]
    kv = config["kv_dim"]
    hid = config["hidden_dim"]

    if spec_name == "rms_gemms_rope":
        # slot 2 = output for every sub-launch; handled by _output_shape_for.
        table = {
            #           slot0           slot1
            "rmsnorm": [(seq, emb), (emb,)],
            "q_gemm": [(seq, emb), (emb, emb)],
            "k_gemm": [(seq, emb), (emb, kv)],
            "v_gemm": [(seq, emb), (emb, kv)],
            "rope_q": [(seq, emb), (seq * emb,)],
            "rope_k": [(seq, kv), (seq * kv,)],
        }
        return table[sub_name][slot]

    if spec_name == "o_ffn":
        table = {
            #                slot0          slot1
            "o_gemm": [(seq, emb), (emb, emb)],
            "res_add": [(seq, emb), (seq, emb)],
            "ffn_rmsnorm": [(seq, emb), (emb,)],
            "gate_gemm": [(seq, emb), (emb, hid)],
            "up_gemm": [(seq, emb), (emb, hid)],
            "swiglu": [(seq, hid), (seq, hid)],
            "down_gemm": [(seq, hid), (hid, emb)],
            "ffn_add": [(seq, emb), (seq, emb)],
        }
        return table[sub_name][slot]

    raise ValueError(f"unknown spec {spec_name!r} or sub {sub_name!r}")


# ---------------------------------------------------------------------------
# Baton-link helpers
# ---------------------------------------------------------------------------


def _intermediate_slots_for_sub(spec, sub_idx):
    """For a given sub-launch index, return the set of slots that are
    baton-managed (either produced or consumed via a baton link).

    These slots are passed as intermediate_indices to load_and_run so the
    host skips writing them:
    - Producer output slot: the kernel writes here; downstream reads from the
      same BO via the alias.
    - Consumer input slot: upstream already wrote to it via the shared BO;
      host must not overwrite with zeros.
    """
    slots = set()
    for link in spec.baton_links:
        if link.producer_idx == sub_idx:
            slots.add(link.producer_out_slot)
        if link.consumer_idx == sub_idx:
            slots.add(link.consumer_in_slot)
    return slots


# ---------------------------------------------------------------------------
# Preload phase
# ---------------------------------------------------------------------------


def preload_cell_c(cache, spec, weights_per_layer, config, backend_preset):
    """One-shot allocation: run each sub-launch once to materialise BOs, then
    alias intermediate BOs across sub-launches per spec.baton_links.

    Phase 1 (inner loop over sub_launches): Each sub-launch is invoked once
    with its actual weight in place and dummy zeros for all other inputs.
    This causes KernelCache to allocate the BO set for that bo_key.

    Phase 2 (inner loop over baton_links): _share_bo aliases the producer's
    output BO into the consumer's input BO slot so that both operations refer
    to the same xrt.bo object.
    """
    backend = {**backend_preset}
    backend.pop("instance_name", None)

    for li, layer_weights in enumerate(weights_per_layer):
        # --- Phase 1: allocate BOs for every sub-launch ---
        for sub in spec.sub_launches:
            out_shape = _output_shape_for(spec.name, sub.name, config)
            args = [None, None, None]

            for slot in range(3):
                if slot == sub.output_slot_in_standalone:
                    args[slot] = np.zeros(out_shape, dtype=bfloat16)
                    continue
                if (
                    sub.weight_slot_in_standalone is not None
                    and slot == sub.weight_slot_in_standalone
                ):
                    # Use the actual weight so the BO is populated from the start.
                    w = _static_input_for(spec.name, sub.name, slot, layer_weights)
                    assert w is not None, (
                        f"[cell_c preload] _static_input_for returned None for "
                        f"{spec.name}/{sub.name} slot={slot}"
                    )
                    args[slot] = w
                    continue
                # Activation or baton-fed slot: correctly-sized dummy zeros.
                args[slot] = np.zeros(
                    _slot_shape_for(spec.name, sub.name, slot, config), dtype=bfloat16
                )

            static_idx = (
                {sub.weight_slot_in_standalone}
                if sub.weight_slot_in_standalone is not None
                else set()
            )
            kernel_name = f"{spec.name}__{sub.name}"
            bo_key = f"C_{spec.name}_{sub.name}_L{li}"

            cache.load_and_run(
                kernel_name,
                backend,
                *args,
                output_indices=[sub.output_slot_in_standalone],
                static_input_indices=static_idx,
                bo_key=bo_key,
            )

        # --- Phase 2: alias BOs per baton_links ---
        for link in spec.baton_links:
            producer = spec.sub_launches[link.producer_idx]
            consumer = spec.sub_launches[link.consumer_idx]
            _share_bo(
                cache,
                f"C_{spec.name}_{producer.name}_L{li}",
                link.producer_out_slot,
                f"C_{spec.name}_{consumer.name}_L{li}",
                link.consumer_in_slot,
            )


# ---------------------------------------------------------------------------
# Timed run phase
# ---------------------------------------------------------------------------


def run_cell_c(cache, spec, layer_inputs, config, backend_preset, layer_idx=0):
    """Run all spec.sub_launches sequentially with pre-loaded weight BOs and
    shared intermediate BOs (baton-pass).

    Differences from Cell B:
    - bo_key uses "C_" prefix (matches preload).
    - intermediate_indices is set for each sub-launch based on baton_links:
        * producer's output slot  -> kernel overwrites it; don't host-write
        * consumer's input slot   -> aliased to upstream BO; don't host-write

    For baton-fed input slots the numpy arg is np.zeros (placeholder); bytes
    are skipped because the slot is in intermediate_indices.

    Args:
        cache: KernelCache with manifested artifacts (preload must have run).
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

        # Build the 3-arg list.
        args = [None, None, None]

        for slot in range(3):
            if slot == sub.output_slot_in_standalone:
                args[slot] = np.zeros(out_shape, dtype=bfloat16)
                continue

            # Try static (weight/LUT/layer-level) lookup first.
            v = _static_input_for(spec.name, sub.name, slot, layer_inputs)
            if v is not None:
                args[slot] = v
                continue

            # Baton-fed slot: host won't write it (intermediate_indices); use
            # a correctly-sized zero placeholder so the array shape is valid.
            args[slot] = np.zeros(
                _slot_shape_for(spec.name, sub.name, slot, config), dtype=bfloat16
            )

        intermediate_idx = _intermediate_slots_for_sub(spec, idx)
        static_idx = (
            {sub.weight_slot_in_standalone}
            if sub.weight_slot_in_standalone is not None
            else set()
        )

        kernel_name = f"{spec.name}__{sub.name}"
        bo_key = f"C_{spec.name}_{sub.name}_L{layer_idx}"

        result = cache.load_and_run(
            kernel_name,
            backend,
            *args,
            output_indices=[sub.output_slot_in_standalone],
            static_input_indices=static_idx,
            intermediate_indices=intermediate_idx,
            bo_key=bo_key,
        )
        results[sub.name] = result[sub.output_slot_in_standalone]

    elapsed = time.perf_counter() - t0
    results["_wall_s"] = elapsed
    return results
