# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the prefill 4-cell ablation.

Modes:
  --scope=single-layer    5 trials × 1-layer cell call (per kernel-group)
  --scope=16-layer        5 trials × 16-layer triple (rms->FA->o_ffn) loop
  --scope=both (default)  both above

Run from programming_examples/llama32_1b/ablation/prefill/build/
(where standalone_cache/ lives and xclbins are found).
"""

import argparse
import json
import os
import sys
import time

# Path setup: this script lives in prefill/; CWD is build/ (where standalone_cache/ lives)
# prefill/ -> ablation/ -> llama32_1b/ -> programming_examples/
_PREFILL = os.path.dirname(os.path.abspath(__file__))
_ABLATION = os.path.dirname(_PREFILL)
_LLAMA = os.path.dirname(_ABLATION)
_PROG_EXAMPLES = os.path.dirname(_LLAMA)

# Insert in ascending priority: _PROG_EXAMPLES appended, _PREFILL at front.
# Use append for lower-priority dirs so they don't shadow prefill's 'cells' package.
for p in (_PROG_EXAMPLES, _LLAMA, _ABLATION):
    if p not in sys.path:
        sys.path.append(p)
# _PREFILL must be at index 0 so prefill/cells/ wins over ablation/cells/.
if _PREFILL in sys.path:
    sys.path.remove(_PREFILL)
sys.path.insert(0, _PREFILL)

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import RMS_GEMMS_ROPE_BACKEND, O_FFN_BACKEND

from validate import validate_against_golden, GoldenMismatch
from cells import cell_a_naive, cell_b_static, cell_c_charitable, cell_d_merged
from cells.flash_attn_const import compile_flash_attn
from cells.multi_layer import run_16_layer_prefill
from specs.rms_gemms_rope import SPEC as RG_SPEC
from specs.o_ffn import SPEC as OF_SPEC
from golden.regen_golden import _synthetic_layer_inputs

GOLDEN_DIR = os.path.join(_PREFILL, "golden")


# ---------------------------------------------------------------------------
# Output key adapters: convert cell A/B/C sub-launch dicts to golden-comparable
# ---------------------------------------------------------------------------


def _rg_cell_outputs(out, cell):
    """Map run_cell_* output dict to golden keys for rms_gemms_rope."""
    if cell == "D":
        # Cell D already returns {normed, q, k, v, q_roped, k_roped, _wall_s}
        return {k: v for k, v in out.items() if not k.startswith("_")}
    # Cell A/B/C: sub-launch names as keys
    return {
        "normed": out["rmsnorm"],
        "q": out["q_gemm"],
        "k": out["k_gemm"],
        "v": out["v_gemm"],
        "q_roped": out["rope_q"],
        "k_roped": out["rope_k"],
    }


def _of_cell_outputs(out, cell):
    """Map run_cell_* output dict to golden keys for o_ffn."""
    if cell == "D":
        # Cell D returns {output, _wall_s}
        return {"output": out["output"]}
    # Cell A/B/C: last sub-launch is "ffn_add"; golden only checks "output"
    return {"output": out["ffn_add"].reshape(-1)}


# ---------------------------------------------------------------------------
# Cell runners (single-layer) — unified interface
# ---------------------------------------------------------------------------


def _run_rg(cell, cache, layer_inputs):
    """Run rms_gemms_rope for the given cell. Returns raw output dict."""
    if cell == "A":
        return cell_a_naive.run_cell_a(
            cache, RG_SPEC, layer_inputs, cell_d_merged.CONFIG, RMS_GEMMS_ROPE_BACKEND
        )
    if cell == "B":
        return cell_b_static.run_cell_b(
            cache, RG_SPEC, layer_inputs, cell_d_merged.CONFIG, RMS_GEMMS_ROPE_BACKEND
        )
    if cell == "C":
        return cell_c_charitable.run_cell_c(
            cache, RG_SPEC, layer_inputs, cell_d_merged.CONFIG, RMS_GEMMS_ROPE_BACKEND
        )
    if cell == "D":
        rg_in = {
            k: layer_inputs[k]
            for k in ["x_in", "norm_w", "wq", "wk", "wv", "lut_q", "lut_k"]
        }
        return cell_d_merged.run_cell_d_rms_gemms_rope(cache, rg_in)
    raise ValueError(f"unknown cell {cell!r}")


def _run_of(cell, cache, layer_inputs):
    """Run o_ffn for the given cell. Returns raw output dict.

    layer_inputs must contain: attn_out, wo, x_residual, ffn_norm_w,
    w_gate, w_up, w_down (plus any extra keys ignored by A/B/C).
    """
    if cell == "A":
        return cell_a_naive.run_cell_a(
            cache, OF_SPEC, layer_inputs, cell_d_merged.CONFIG, O_FFN_BACKEND
        )
    if cell == "B":
        return cell_b_static.run_cell_b(
            cache, OF_SPEC, layer_inputs, cell_d_merged.CONFIG, O_FFN_BACKEND
        )
    if cell == "C":
        return cell_c_charitable.run_cell_c(
            cache, OF_SPEC, layer_inputs, cell_d_merged.CONFIG, O_FFN_BACKEND
        )
    if cell == "D":
        of_in = {
            k: layer_inputs[k]
            for k in [
                "attn_out",
                "wo",
                "x_residual",
                "ffn_norm_w",
                "w_gate",
                "w_up",
                "w_down",
            ]
        }
        return cell_d_merged.run_cell_d_o_ffn(cache, of_in)
    raise ValueError(f"unknown cell {cell!r}")


# ---------------------------------------------------------------------------
# 16-layer adapter: convert cell A/B/C output to multi_layer-expected shape
# ---------------------------------------------------------------------------


def _make_rg_runner_16layer(cell, cache):
    """Return a run_rms_gemms_rope(cache, layer_in, layer_idx) adapter for multi_layer.

    multi_layer.py expects the function to return a dict with keys:
        q_roped, k_roped, v  (and others, unused by multi_layer)
    all as 1D flat arrays (it reshapes them internally before calling FA).
    """

    def run(c, layer_in, layer_idx=0):
        if cell in ("A", "B", "C"):
            out = _run_rg(cell, c, layer_in)
            # Convert sub-launch names to canonical names for multi_layer
            out["q_roped"] = out["rope_q"]
            out["k_roped"] = out["rope_k"]
            out["q"] = out["q_gemm"]
            out["k"] = out["k_gemm"]
            out["v"] = out["v_gemm"]
            out["normed"] = out["rmsnorm"]
        else:
            out = _run_rg(cell, c, layer_in)
        return out

    return run


def _make_of_runner_16layer(cell, cache):
    """Return a run_o_ffn(cache, of_in, layer_idx) adapter for multi_layer.

    multi_layer.py assembles of_in with all needed keys (attn_out, wo,
    x_residual, ffn_norm_w, w_gate, w_up, w_down) and calls this.
    We need to return a dict with key 'output' as a 1D array that multi_layer
    reshapes for the next layer's x_in.
    """

    def run(c, of_in, layer_idx=0):
        out = _run_of(cell, c, of_in)
        if cell in ("A", "B", "C"):
            # Rename ffn_add -> output for multi_layer compatibility
            out["output"] = out["ffn_add"].reshape(-1)
        return out

    return run


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------


def _unload_all_contexts(cache):
    """Unload all XRT HW contexts and drop all cached BOs.

    The NPU has a limited number of HW context slots (~16).  When switching
    between single-layer (14+ standalone contexts) and 16-layer (up to 15
    contexts for Cell A/B/C), we must release all contexts first to avoid
    hitting the limit.

    BOs are allocated against a specific XRT device handle; after unloading
    the backend that handle is nulled, so the old BO objects are unusable.
    We must also clear _cached_bos so the next load_and_run allocates fresh
    BOs against the new device.  This means preloaded Cell B/C weights are
    lost and will be re-written on the next call (acceptable since the
    16-layer loop only runs one cell at a time anyway).
    """
    for name, (backend, _) in list(cache._loaded.items()):
        try:
            backend.unload()
        except Exception:
            pass
    cache._loaded.clear()
    cache._cached_bos.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument(
        "--scope",
        choices=["single-layer", "16-layer", "both"],
        default="both",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache = KernelCache(cache_dir="standalone_cache", verbose=False)
    cache.load_manifest()

    # ---- Compile all cells + FA (idempotent — skips if already cached) ----
    print("=== Compiling kernels (idempotent) ===")
    cell_a_naive.compile_cell_a(cache, RG_SPEC, RMS_GEMMS_ROPE_BACKEND)
    cell_a_naive.compile_cell_a(cache, OF_SPEC, O_FFN_BACKEND)
    cell_b_static.compile_cell_b(cache, RG_SPEC, RMS_GEMMS_ROPE_BACKEND)
    cell_b_static.compile_cell_b(cache, OF_SPEC, O_FFN_BACKEND)
    cell_c_charitable.compile_cell_c(cache, RG_SPEC, RMS_GEMMS_ROPE_BACKEND)
    cell_c_charitable.compile_cell_c(cache, OF_SPEC, O_FFN_BACKEND)
    cell_d_merged.compile_cell_d_rms_gemms_rope(cache)
    cell_d_merged.compile_cell_d_o_ffn(cache)
    compile_flash_attn(cache, cell_d_merged.CONFIG)
    print("All kernels compiled/cached.\n")

    # ---- Generate per-layer synthetic inputs (all 16 layers) ----
    layer_inputs_per_layer = [
        _synthetic_layer_inputs(L, cell_d_merged.CONFIG) for L in range(16)
    ]

    # ---- Pre-load weights for Cell B and Cell C (both kernel-groups, all 16 layers) ----
    print("=== Pre-loading weights for Cell B and Cell C ===")
    rg_weights = [
        {k: li[k] for k in ["norm_w", "wq", "wk", "wv", "lut_q", "lut_k"]}
        for li in layer_inputs_per_layer
    ]
    of_weights = [
        {k: li[k] for k in ["wo", "ffn_norm_w", "w_gate", "w_up", "w_down"]}
        for li in layer_inputs_per_layer
    ]

    cell_b_static.preload_cell_b(
        cache, RG_SPEC, rg_weights, cell_d_merged.CONFIG, RMS_GEMMS_ROPE_BACKEND
    )
    cell_b_static.preload_cell_b(
        cache, OF_SPEC, of_weights, cell_d_merged.CONFIG, O_FFN_BACKEND
    )
    cell_c_charitable.preload_cell_c(
        cache, RG_SPEC, rg_weights, cell_d_merged.CONFIG, RMS_GEMMS_ROPE_BACKEND
    )
    cell_c_charitable.preload_cell_c(
        cache, OF_SPEC, of_weights, cell_d_merged.CONFIG, O_FFN_BACKEND
    )
    print("Preload done.\n")

    results = {
        "config": cell_d_merged.CONFIG,
        "trials": args.trials,
        "scope": args.scope,
        "cells": {},
    }

    # ---- Build layer-0 inputs for single-layer validation and timing ----
    layer0 = layer_inputs_per_layer[0]
    # o_ffn needs attn_out (from FA in production; synthesized here to match regen_golden)
    attn_out_layer0 = (
        np.random.default_rng(42 + 0 + 1000)
        .standard_normal(
            (cell_d_merged.CONFIG["seq_len"], cell_d_merged.CONFIG["emb_dim"])
        )
        .astype(bfloat16)
    )
    of_layer0 = dict(layer0)
    of_layer0["attn_out"] = attn_out_layer0
    of_layer0["x_residual"] = layer0["x_in"]

    # ---- Validation: single-layer Cell A/B/C/D vs both goldens ----
    print("=== Validation (layer 0, single-layer) ===")
    for cell in ("A", "B", "C", "D"):
        cell_results = {}

        # rms_gemms_rope validation
        try:
            rg_out = _run_rg(cell, cache, layer0)
            rg_cell_out = _rg_cell_outputs(rg_out, cell)
            validate_against_golden(
                rg_cell_out, GOLDEN_DIR, "golden_rms_gemms_rope_prefill.npz"
            )
            cell_results["rms_gemms_rope"] = {"validation": "PASS"}
            print(f"  Cell {cell} rms_gemms_rope: PASS")
        except GoldenMismatch as e:
            cell_results["rms_gemms_rope"] = {"validation": "FAIL", "error": str(e)}
            print(f"  Cell {cell} rms_gemms_rope: FAIL - {e}")
        except Exception as e:
            cell_results["rms_gemms_rope"] = {"validation": "ERROR", "error": str(e)}
            print(f"  Cell {cell} rms_gemms_rope: ERROR - {e}")

        # o_ffn validation
        try:
            of_out = _run_of(cell, cache, of_layer0)
            of_cell_out = _of_cell_outputs(of_out, cell)
            validate_against_golden(of_cell_out, GOLDEN_DIR, "golden_o_ffn_prefill.npz")
            cell_results["o_ffn"] = {"validation": "PASS"}
            print(f"  Cell {cell} o_ffn: PASS")
        except GoldenMismatch as e:
            cell_results["o_ffn"] = {"validation": "FAIL", "error": str(e)}
            print(f"  Cell {cell} o_ffn: FAIL - {e}")
        except Exception as e:
            cell_results["o_ffn"] = {"validation": "ERROR", "error": str(e)}
            print(f"  Cell {cell} o_ffn: ERROR - {e}")

        results["cells"][cell] = cell_results

    print()

    # ---- Timing: single-layer scope ----
    if args.scope in ("single-layer", "both"):
        print("=== Timing: single-layer scope ===")
        for cell in ("A", "B", "C", "D"):
            cr = results["cells"][cell]

            # rms_gemms_rope timing
            if cr.get("rms_gemms_rope", {}).get("validation") == "PASS":
                times_rg = []
                for _ in range(args.trials):
                    o = _run_rg(cell, cache, layer0)
                    times_rg.append(o["_wall_s"])
                keep = sorted(times_rg[1:])
                med_rg = keep[len(keep) // 2]
                cr["rms_gemms_rope"]["single_layer"] = {
                    "all_trials_s": times_rg,
                    "median_s": med_rg,
                    "min_s": min(keep),
                    "max_s": max(keep),
                }
                print(
                    f"  Cell {cell} rg single-layer: "
                    f"med={med_rg * 1000:.2f}ms  "
                    f"[{min(keep)*1000:.2f}-{max(keep)*1000:.2f}ms] "
                    f"(warmup={times_rg[0]*1000:.2f}ms)"
                )

            # o_ffn timing
            if cr.get("o_ffn", {}).get("validation") == "PASS":
                times_of = []
                for _ in range(args.trials):
                    o = _run_of(cell, cache, of_layer0)
                    times_of.append(o["_wall_s"])
                keep = sorted(times_of[1:])
                med_of = keep[len(keep) // 2]
                cr["o_ffn"]["single_layer"] = {
                    "all_trials_s": times_of,
                    "median_s": med_of,
                    "min_s": min(keep),
                    "max_s": max(keep),
                }
                print(
                    f"  Cell {cell} of single-layer: "
                    f"med={med_of * 1000:.2f}ms  "
                    f"[{min(keep)*1000:.2f}-{max(keep)*1000:.2f}ms] "
                    f"(warmup={times_of[0]*1000:.2f}ms)"
                )
        print()

    # ---- Timing: 16-layer scope ----
    if args.scope in ("16-layer", "both"):
        print("=== Timing: 16-layer scope ===")
        for cell in ("A", "B", "C", "D"):
            cr = results["cells"][cell]
            rg_ok = cr.get("rms_gemms_rope", {}).get("validation") == "PASS"
            of_ok = cr.get("o_ffn", {}).get("validation") == "PASS"
            if not (rg_ok and of_ok):
                print(
                    f"  Cell {cell}: skipping 16-layer (validation failed for "
                    f"{'rms_gemms_rope' if not rg_ok else 'o_ffn'})"
                )
                continue

            # Unload all previously opened XRT contexts and BOs before each
            # cell's 16-layer run.  The NPU has ~16 HW context slots; Cell A/B/C
            # each need 14 standalone contexts + FA = 15 total.  Starting fresh
            # per cell avoids hitting the limit.
            # Cell B/C weights are lost with the BOs — re-preload them below.
            _unload_all_contexts(cache)

            # Re-preload weights for B and C after the context reset.
            if cell == "B":
                cell_b_static.preload_cell_b(
                    cache,
                    RG_SPEC,
                    rg_weights,
                    cell_d_merged.CONFIG,
                    RMS_GEMMS_ROPE_BACKEND,
                )
                cell_b_static.preload_cell_b(
                    cache, OF_SPEC, of_weights, cell_d_merged.CONFIG, O_FFN_BACKEND
                )
            elif cell == "C":
                cell_c_charitable.preload_cell_c(
                    cache,
                    RG_SPEC,
                    rg_weights,
                    cell_d_merged.CONFIG,
                    RMS_GEMMS_ROPE_BACKEND,
                )
                cell_c_charitable.preload_cell_c(
                    cache, OF_SPEC, of_weights, cell_d_merged.CONFIG, O_FFN_BACKEND
                )

            run_rg_16 = _make_rg_runner_16layer(cell, cache)
            run_of_16 = _make_of_runner_16layer(cell, cache)

            times_total = []
            for trial in range(args.trials):
                r = run_16_layer_prefill(
                    cache,
                    cell_d_merged.CONFIG,
                    run_rg_16,
                    run_of_16,
                    layer_inputs_per_layer,
                )
                times_total.append(r["total_wall"])

            keep = sorted(times_total[1:])
            med = keep[len(keep) // 2]
            cr["16_layer"] = {
                "all_trials_s": times_total,
                "median_s": med,
                "min_s": min(keep),
                "max_s": max(keep),
            }
            print(
                f"  Cell {cell} 16-layer total: "
                f"med={med:.3f}s  "
                f"[{min(keep):.3f}-{max(keep):.3f}s] "
                f"(warmup={times_total[0]:.3f}s)"
            )
        print()

    # ---- Dump JSON ----
    out_path = args.out or f"results_prefill_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
