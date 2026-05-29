"""Run the Plan 2 (full decode) 4-cell ablation.

Per cell:
  - Compile (idempotent, skipped if cached)
  - Preload weights into per-layer BOs (Cells B/C/D; Cell A skips this)
  - Validate: run rms_gemv_rope and o_gemv_ffn ONCE for layer 0 with synthetic
    inputs, compare bytes to committed goldens
  - 5 timed trials of per_token_loop generating ONE decode token at fixed
    current_pos, drop trial 1 as warmup
  - Median + (min, max) of trials 2-5

Per-kernel-group medians for rms_gemv_rope and o_gemv_ffn are extracted
from per_token_loop's per-layer wall arrays (medians across the 16 layers
within trial 2-5).

Usage:
  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 run_ablation.py --trials 5
"""

import argparse
import json
import os
import sys
import time

# sys.path setup (mirrors conftest.py)
_THIS = os.path.dirname(os.path.abspath(__file__))
_ABLATION = os.path.dirname(_THIS)
_LLAMA = os.path.dirname(_ABLATION)
_PE = os.path.dirname(_LLAMA)
for p in (_PE, _LLAMA, _ABLATION, os.path.join(_ABLATION, "prefill")):
    if p not in sys.path:
        sys.path.append(p)
# Decode dir at sys.path[0] so decode/cells/ wins over prefill/cells/
if _THIS in sys.path:
    sys.path.remove(_THIS)
sys.path.insert(0, _THIS)
# Drop any stale `cells`/`specs`/`standalone_builders` modules from prior imports
for _stale in [
    m
    for m in list(sys.modules)
    if m.startswith(("cells", "specs", "standalone_builders"))
]:
    del sys.modules[_stale]

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import RGR_BACKEND, OGF_BACKEND

from validate import GoldenMismatch, validate_against_golden
from cells import cell_a_naive, cell_b_static, cell_c_charitable, cell_d_merged
from cells.kv_cache import build_initial_kv_cache, reset_position
from cells.lm_head_const import (
    compile_lm_head,
    preload_lm_head,
    _LM_N_PART,
    _LM_N_PARTITIONS,
)
from cells.per_token_loop import run_one_decode_token
from specs.rms_gemv_rope import SPEC as RGR_SPEC
from specs.o_gemv_ffn import SPEC as OGF_SPEC
from golden.regen_golden import (
    CONFIG,
    PROMPT_LEN,
    CURRENT_POS,
    SEED,
    synthetic_layer_weights,
    synthetic_lut,
    synthetic_x_in,
    synthetic_attn_out,
)

GOLDEN_DIR = os.path.join(_THIS, "golden")


# ---------------------------------------------------------------------------
# Cell-specific dispatch adapters for the per-token loop
# ---------------------------------------------------------------------------


def _wrap_rg_runner(cell, spec):
    """Return a (cache, layer_inputs, layer_idx) -> dict adapter.

    Output dict normalizes sub-launch names to {normed, q, k, v, q_roped,
    k_roped} for downstream consumers (per_token_loop, validation).
    """
    if cell == "D":

        def _run(cache, layer_inputs, layer_idx=0):
            return cell_d_merged.run_rms_gemv_rope_d(cache, layer_inputs, layer_idx)

        return _run

    if cell == "A":
        runner = cell_a_naive.run_cell_a
    elif cell == "B":
        runner = cell_b_static.run_cell_b
    elif cell == "C":
        runner = cell_c_charitable.run_cell_c
    else:
        raise ValueError(f"unknown cell {cell!r}")

    def _run(cache, layer_inputs, layer_idx=0):
        out = runner(
            cache, spec, layer_inputs, CONFIG, RGR_BACKEND, layer_idx=layer_idx
        )
        # Normalize keys for downstream consumers
        return {
            "normed": out["rmsnorm"],
            "q": out["q_gemv"],
            "k": out["k_gemv"],
            "v": out["v_gemv"],
            "q_roped": out["rope_q"],
            "k_roped": out["rope_k"],
            "_wall_s": out["_wall_s"],
        }

    return _run


def _wrap_of_runner(cell, spec):
    if cell == "D":

        def _run(cache, layer_inputs, layer_idx=0):
            return cell_d_merged.run_o_gemv_ffn_d(cache, layer_inputs, layer_idx)

        return _run

    if cell == "A":
        runner = cell_a_naive.run_cell_a
    elif cell == "B":
        runner = cell_b_static.run_cell_b
    elif cell == "C":
        runner = cell_c_charitable.run_cell_c
    else:
        raise ValueError(f"unknown cell {cell!r}")

    def _run(cache, layer_inputs, layer_idx=0):
        out = runner(
            cache, spec, layer_inputs, CONFIG, OGF_BACKEND, layer_idx=layer_idx
        )
        # Cells A/B/C return all 8 sub-launch outputs; the per_token_loop
        # only needs the final residual add as 'output'.
        return {"output": out["add_ffn_residual"], "_wall_s": out["_wall_s"]}

    return _run


# ---------------------------------------------------------------------------
# Validation: run layer 0 once, compare to goldens
# ---------------------------------------------------------------------------


def _validate_cell(cell, cache, layer0_weights, lut, x_in, attn_out_synth):
    """Run rms_gemv_rope and o_gemv_ffn for layer 0 (synthetic inputs) and
    bit-exact compare to committed goldens. Raises GoldenMismatch on diff."""
    rg_runner = _wrap_rg_runner(cell, RGR_SPEC)
    of_runner = _wrap_of_runner(cell, OGF_SPEC)

    rg_in = {
        "x_in": x_in,
        "norm_w": layer0_weights["norm_w"],
        "wq": layer0_weights["wq"],
        "wk": layer0_weights["wk"],
        "wv": layer0_weights["wv"],
        "lut_q": lut["lut_q"],
        "lut_k": lut["lut_k"],
    }
    rg_out = rg_runner(cache, rg_in, layer_idx=0)
    rg_compare = {k: rg_out[k] for k in ("normed", "q", "k", "v", "q_roped", "k_roped")}
    validate_against_golden(rg_compare, GOLDEN_DIR, "golden_rms_gemv_rope_decode.npz")

    of_in = {
        "wo": layer0_weights["wo"],
        "attn_out": attn_out_synth,
        "x_residual": x_in,
        "ffn_norm_w": layer0_weights["ffn_norm_w"],
        "w_gate": layer0_weights["w_gate"],
        "w_up": layer0_weights["w_up"],
        "w_down": layer0_weights["w_down"],
    }
    of_out = of_runner(cache, of_in, layer_idx=0)
    of_compare = {"output": of_out["output"]}
    validate_against_golden(of_compare, GOLDEN_DIR, "golden_o_gemv_ffn_decode.npz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache_dir = os.path.join(_THIS, "build")
    os.makedirs(cache_dir, exist_ok=True)
    cache = KernelCache(cache_dir=cache_dir, verbose=False)
    cache.load_manifest()

    # ------ 1. Compile all cells (idempotent) ------
    print("=== Compiling cells (idempotent) ===")
    cell_a_naive.compile_cell_a(cache, RGR_SPEC, RGR_BACKEND)
    cell_a_naive.compile_cell_a(cache, OGF_SPEC, OGF_BACKEND)
    cell_b_static.compile_cell_b(cache, RGR_SPEC, RGR_BACKEND)
    cell_b_static.compile_cell_b(cache, OGF_SPEC, OGF_BACKEND)
    cell_c_charitable.compile_cell_c(cache, RGR_SPEC, RGR_BACKEND)
    cell_c_charitable.compile_cell_c(cache, OGF_SPEC, OGF_BACKEND)
    cell_d_merged.compile_cell_d(cache, CONFIG)
    compile_lm_head(cache, CONFIG)
    print("All compiled.\n")

    # ------ 2. Generate synthetic inputs ------
    n_layers = CONFIG["n_layers"]
    weights_per_layer = [
        synthetic_layer_weights(L, CONFIG, SEED) for L in range(n_layers)
    ]
    lut = synthetic_lut(CONFIG, SEED)
    x_in = synthetic_x_in(CONFIG, SEED)  # token embedding entering layer 0
    attn_out_synth = synthetic_attn_out(CONFIG, SEED)  # for golden validation only

    # Synthetic LM head partitions
    rng = np.random.default_rng(SEED + 6666)
    lm_weight_parts = [
        (rng.standard_normal((_LM_N_PART, CONFIG["emb_dim"])) * 0.02).astype(bfloat16)
        for _ in range(_LM_N_PARTITIONS)
    ]
    final_norm_w = rng.standard_normal(CONFIG["emb_dim"]).astype(bfloat16)

    # ------ 3. Per-cell weight prep helpers (called inside per-cell loop) ------

    rg_weights_per_layer = [
        {k: w[k] for k in ("norm_w", "wq", "wk", "wv")} for w in weights_per_layer
    ]
    for d in rg_weights_per_layer:
        d["lut_q"] = lut["lut_q"]
        d["lut_k"] = lut["lut_k"]

    of_weights_per_layer = [
        {k: w[k] for k in ("wo", "ffn_norm_w", "w_gate", "w_up", "w_down")}
        for w in weights_per_layer
    ]

    def _preload_for_cell(cell):
        """Preload BOs for the given cell. Cell A doesn't preload (naive=True)."""
        if cell == "B":
            cell_b_static.preload_cell_b(
                cache, RGR_SPEC, rg_weights_per_layer, CONFIG, RGR_BACKEND
            )
            cell_b_static.preload_cell_b(
                cache, OGF_SPEC, of_weights_per_layer, CONFIG, OGF_BACKEND
            )
        elif cell == "C":
            cell_c_charitable.preload_cell_c(
                cache, RGR_SPEC, rg_weights_per_layer, CONFIG, RGR_BACKEND
            )
            cell_c_charitable.preload_cell_c(
                cache, OGF_SPEC, of_weights_per_layer, CONFIG, OGF_BACKEND
            )
        elif cell == "D":
            cell_d_merged.preload_cell_d(
                cache, weights_per_layer, lut["lut_q"], lut["lut_k"], CONFIG
            )
        # LM head invariant — preload for every cell (held INVARIANT in ablation)
        preload_lm_head(cache, lm_weight_parts, CONFIG)

    def _unload_all_contexts():
        """Free up all NPU HW context slots and drop cached BOs.

        The NPU HW context limit is ~16. Cells A/B/C each load 14 standalone
        ELFs + 1 LM head = 15 contexts; switching cells without unloading
        would exceed the limit. We unload after each cell finishes its trials
        so the next cell starts with a clean slot table.
        """
        for name, (backend, _) in list(cache._loaded.items()):
            try:
                backend.unload()
            except Exception:
                pass
        cache._loaded.clear()
        cache._cached_bos.clear()

    # ------ 4. Run each cell: preload + validate + 5 trials + unload ------
    results = {
        "config": CONFIG,
        "current_pos": CURRENT_POS,
        "prompt_len": PROMPT_LEN,
        "trials": args.trials,
        "cells": {},
    }

    for cell in ["A", "B", "C", "D"]:
        print(f"=== Cell {cell}: preload + validate + {args.trials} trials ===")
        _preload_for_cell(cell)
        # Validate against goldens (single layer 0 run)
        try:
            _validate_cell(
                cell,
                cache,
                weights_per_layer[0],
                lut,
                x_in,
                attn_out_synth,
            )
            validation = "PASS"
            print(f"  Cell {cell}: VALIDATION PASS")
        except GoldenMismatch as e:
            validation = f"FAIL: {e}"
            print(f"  Cell {cell}: VALIDATION FAIL — {e}")
            results["cells"][cell] = {"validation": validation}
            continue

        # Build per-layer inputs for the per_token_loop
        layer_inputs_per_layer = []
        for L in range(n_layers):
            li = {
                "norm_w": weights_per_layer[L]["norm_w"],
                "wq": weights_per_layer[L]["wq"],
                "wk": weights_per_layer[L]["wk"],
                "wv": weights_per_layer[L]["wv"],
                "wo": weights_per_layer[L]["wo"],
                "ffn_norm_w": weights_per_layer[L]["ffn_norm_w"],
                "w_gate": weights_per_layer[L]["w_gate"],
                "w_up": weights_per_layer[L]["w_up"],
                "w_down": weights_per_layer[L]["w_down"],
                "lut_q": lut["lut_q"],
                "lut_k": lut["lut_k"],
            }
            layer_inputs_per_layer.append(li)

        # Build the cell-specific runners
        rg_runner = _wrap_rg_runner(cell, RGR_SPEC)
        of_runner = _wrap_of_runner(cell, OGF_SPEC)

        # 5 timed trials
        trial_results = []
        for trial in range(args.trials):
            # Reset KV cache to a fresh pre-filled state
            kv_cache = build_initial_kv_cache(CONFIG, prompt_len=PROMPT_LEN, seed=SEED)
            # Reset position CURRENT_POS so subsequent trials don't carry over the
            # previously-generated k/v at slot CURRENT_POS
            reset_position(kv_cache, CURRENT_POS)

            out = run_one_decode_token(
                cache=cache,
                config=CONFIG,
                kv_cache=kv_cache,
                layer_inputs_per_layer=layer_inputs_per_layer,
                final_norm_w=final_norm_w,
                lm_weight_parts=lm_weight_parts,
                initial_x_decode=x_in,
                current_pos=CURRENT_POS,
                run_rms_gemv_rope=rg_runner,
                run_o_gemv_ffn=of_runner,
            )
            trial_results.append(out)
            print(
                f"  trial {trial+1}: total={out['total_wall_s']*1000:.2f}ms"
                f"  cpu_attn={out['cpu_attn_wall_s']*1000:.2f}ms"
                f"  lm_head={out['lm_head_wall_s']*1000:.2f}ms"
            )

        # Drop trial 1 (warmup), median + (min,max) of remaining
        kept = trial_results[1:]
        kept_total = sorted([t["total_wall_s"] for t in kept])
        median_total = kept_total[len(kept_total) // 2]

        # Per-kernel-group medians: median over (16 layers × 4 kept trials) of per-layer wall
        rg_walls = [w for t in kept for w in t["per_layer_rms_gemv_rope_wall_s"]]
        of_walls = [w for t in kept for w in t["per_layer_o_gemv_ffn_wall_s"]]
        rg_walls_sorted = sorted(rg_walls)
        of_walls_sorted = sorted(of_walls)
        rg_median_per_call = rg_walls_sorted[len(rg_walls_sorted) // 2]
        of_median_per_call = of_walls_sorted[len(of_walls_sorted) // 2]

        # CPU attention floor (median across kept trials)
        cpu_walls = sorted([t["cpu_attn_wall_s"] for t in kept])
        lm_walls = sorted([t["lm_head_wall_s"] for t in kept])

        cell_summary = {
            "validation": validation,
            "all_trials_total_s": [t["total_wall_s"] for t in trial_results],
            "median_total_s": median_total,
            "min_total_s": min([t["total_wall_s"] for t in kept]),
            "max_total_s": max([t["total_wall_s"] for t in kept]),
            "rms_gemv_rope_per_call_median_s": rg_median_per_call,
            "o_gemv_ffn_per_call_median_s": of_median_per_call,
            "cpu_attn_total_median_s": cpu_walls[len(cpu_walls) // 2],
            "lm_head_median_s": lm_walls[len(lm_walls) // 2],
            "next_token": trial_results[-1]["next_token"],
        }
        results["cells"][cell] = cell_summary
        print(
            f"  Cell {cell} median total: {median_total*1000:.2f}ms  "
            f"rg/call: {rg_median_per_call*1000:.2f}ms  "
            f"of/call: {of_median_per_call*1000:.2f}ms"
        )

        # Free up NPU HW context slots before next cell loads its ELFs
        _unload_all_contexts()
        print(f"  (unloaded contexts)\n")

    # ------ 5. Write results JSON ------
    out_path = args.out or os.path.join(_THIS, f"results_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
