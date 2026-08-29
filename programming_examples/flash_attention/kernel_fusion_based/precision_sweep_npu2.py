#!/usr/bin/env python3
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flash Attention numerical stability sweep — correlation vs sequence length.

Measures Pearson correlation, max_err, mean_err, and fail_pct against a CPU F32
reference for the paper Table 1 model configurations across sequence lengths.

Usage:
    cd programming_examples/flash_attention/kernel_fusion_based
    python3 precision_sweep_npu2.py                           # Full sweep
    python3 precision_sweep_npu2.py --configs gpt2            # Single config
    python3 precision_sweep_npu2.py --configs gpt2 --seq-lengths 512  # Smoke test
    python3 precision_sweep_npu2.py --per-head                # Include per-head breakdown

Prerequisites:
    - MLIR-AIR environment (source utils/env_setup.sh)
    - NPU2 hardware
    - Compiled kernel .o files in build_peano/:
        make compile-kernel DK=64 DV=64 LKP=64 LQP=256
        make compile-kernel DK=128 DV=128 LKP=64 LQP=256
"""

import argparse
import csv
import os
import sys
import time
from math import sqrt

import numpy as np
from ml_dtypes import bfloat16
import filelock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_npu2 import build_module
from air.backend.xrt import XRTBackend

# ---------------------------------------------------------------------------
# Configuration registry (paper Table 1)
# ---------------------------------------------------------------------------
CONFIGS = {
    "gpt2": dict(
        label="GPT-2 Small",
        dk=64,
        dv=64,
        nh=12,
        nkv=12,
        causal=False,
        seqs=[256, 512, 1024, 2048, 4096, 8192, 16384],
    ),
    "llama2": dict(
        label="LLaMA-2 7B",
        dk=128,
        dv=128,
        nh=32,
        nkv=32,
        causal=False,
        seqs=[256, 512, 1024, 2048, 4096, 8192],
    ),
    "llama3": dict(
        label="LLaMA-3.1 8B",
        dk=128,
        dv=128,
        nh=32,
        nkv=8,
        causal=False,
        seqs=[256, 512, 1024, 2048, 4096, 8192],
    ),
    "qwen25": dict(
        label="Qwen-2.5 7B",
        dk=128,
        dv=128,
        nh=28,
        nkv=4,
        causal=False,
        seqs=[256, 512, 1024, 2048, 4096, 8192],
    ),
    "gpt2_causal": dict(
        label="GPT-2 Small (causal)",
        dk=64,
        dv=64,
        nh=12,
        nkv=12,
        causal=True,
        seqs=[256, 512, 1024, 2048, 4096, 8192, 16384],
    ),
}

LKP = 64
LQP = 256
NUM_Q_TILES = 4
NUM_CASCADE_STAGES = 4


# ---------------------------------------------------------------------------
# Reference and metrics (from correlation_test.py)
# ---------------------------------------------------------------------------
def cpu_reference(Q, K, V, num_heads, num_kv_heads, causal):
    """Standard scaled dot-product attention in F32."""
    LQ, DK = Q.shape[1], Q.shape[2]
    DV = V.shape[2]
    scale = 1.0 / sqrt(DK)
    gqa_group = num_heads // num_kv_heads

    ref = np.zeros((num_heads, LQ, DV), dtype=np.float32)
    for h in range(num_heads):
        kv_h = h // gqa_group
        scores = (Q[h].astype(np.float32) @ K[kv_h].astype(np.float32).T) * scale
        if causal:
            mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
        mx = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - mx)
        P = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        ref[h] = (P @ V[kv_h].astype(np.float32)).astype(bfloat16).astype(np.float32)
    return ref


def precision_metrics(npu, ref):
    """Compute precision metrics.

    Returns both aggregate (all heads flattened) and mean per-head correlation.
    Per-head correlation is the more meaningful metric for multi-head attention
    since aggregate correlation is dominated by between-head variance.
    """
    num_heads = npu.shape[0]
    diff = np.abs(npu - ref)
    corr = np.corrcoef(npu.flatten(), ref.flatten())[0, 1]
    tol = np.maximum(0.15, 0.04 * (np.abs(npu) + np.abs(ref)))
    fail_pct = np.sum(diff > tol) / diff.size * 100

    # Per-head correlation (mean across heads)
    head_corrs = []
    for h in range(num_heads):
        c = np.corrcoef(npu[h].flatten(), ref[h].flatten())[0, 1]
        head_corrs.append(c)
    mean_head_corr = np.mean(head_corrs)
    min_head_corr = np.min(head_corrs)

    return {
        "corr": corr,
        "head_corr": mean_head_corr,
        "head_corr_min": min_head_corr,
        "max_err": float(diff.max()),
        "mean_err": float(diff.mean()),
        "fail_pct": fail_pct,
        "out_std": float(ref.std()),
    }


def per_head_correlation(npu, ref, num_heads):
    """Compute Pearson correlation per head."""
    corrs = []
    for h in range(num_heads):
        c = np.corrcoef(npu[h].flatten(), ref[h].flatten())[0, 1]
        corrs.append(c)
    return corrs


# ---------------------------------------------------------------------------
# V-transpose helpers (matching attn_npu2.py host logic)
# ---------------------------------------------------------------------------
def transpose_v(v_orig, num_kv_heads, lk, dv, lkp):
    """Transpose V from [nkv, lk, dv] to [nkv*dv_chunks, lk, dv_tile]."""
    dv_chunks = dv // lkp
    return (
        v_orig.reshape(num_kv_heads, lk, dv_chunks, lkp)
        .transpose(0, 2, 1, 3)
        .reshape(num_kv_heads * dv_chunks, lk, lkp)
        .copy()
    )


def untranspose_output(out, num_heads, lq, dv, lkp):
    """Un-transpose output from [nh*dv_chunks, lq, dv_tile] to [nh, lq, dv]."""
    dv_chunks = dv // lkp
    return (
        out.reshape(num_heads, dv_chunks, lq, lkp)
        .transpose(0, 2, 1, 3)
        .reshape(num_heads, lq, dv)
    )


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------
def run_config_sweep(cfg_name, cfg, seq_lengths, seeds, val_range, per_head, build_dir):
    """Run precision sweep for one configuration across sequence lengths."""
    dk, dv, nh, nkv = cfg["dk"], cfg["dv"], cfg["nh"], cfg["nkv"]
    causal = cfg["causal"]
    dv_chunks = dv // LKP

    results = []
    per_head_results = {}

    for lk in seq_lengths:
        lq = lk if causal else 2048

        print(f"  LK={lk}, LQ={lq} ... ", end="", flush=True)
        t0 = time.time()

        # Build MLIR module and compile (once per seq length)
        os.chdir(build_dir)
        mlir_module = build_module(
            lk=lk,
            lkp=LKP,
            lq=lq,
            lqp=LQP,
            dk=dk,
            dv=dv,
            num_q_tiles=NUM_Q_TILES,
            num_cascade_stages=NUM_CASCADE_STAGES,
            num_heads=nh,
            num_kv_heads=nkv,
            causal=causal,
        )

        backend = XRTBackend(
            omit_while_true_loop=False,
            omit_pingpong="all",
            output_format="elf",
            instance_name="attention_bf16",
            target_device="npu2",
        )
        artifact = backend.compile(mlir_module)

        for seed in seeds:
            rng = np.random.default_rng(seed)
            Q = rng.uniform(0, val_range, (nh, lq, dk)).astype(bfloat16)
            K = rng.uniform(0, val_range, (nkv, lk, dk)).astype(bfloat16)
            V_orig = rng.uniform(0, val_range, (nkv, lk, dv)).astype(bfloat16)
            V_transposed = transpose_v(V_orig, nkv, lk, dv, LKP)
            O = np.zeros((nh * dv_chunks, lq, LKP), dtype=bfloat16)

            # Load, run, unload for each seed — XRT ELF runtime requires
            # a fresh load between invocations to properly reset NPU state.
            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(artifact)
                npu_raw = invoker(Q, K, V_transposed, O)
                backend.unload()

            npu_out = npu_raw[3].reshape(nh * dv_chunks, lq, LKP)
            npu_out = untranspose_output(npu_out, nh, lq, dv, LKP).astype(np.float32)

            ref = cpu_reference(Q, K, V_orig, nh, nkv, causal)
            m = precision_metrics(npu_out, ref)
            m["config"] = cfg_name
            m["lk"] = lk
            m["seed"] = seed
            results.append(m)

            # Per-head breakdown at lk=2048
            if per_head and lk == 2048 and seed == seeds[0]:
                per_head_results[cfg_name] = per_head_correlation(npu_out, ref, nh)
        elapsed = time.time() - t0
        # Print summary for this seq length (mean across seeds)
        seed_hcorrs = [r["head_corr"] for r in results if r["lk"] == lk]
        mean_hcorr = np.mean(seed_hcorrs)
        print(f"head_corr={mean_hcorr:.6f} ({elapsed:.1f}s)")

    return results, per_head_results


# ---------------------------------------------------------------------------
# Adversarial cascade merge stress test
# ---------------------------------------------------------------------------
ADVERSARIAL_CONFIGS = {
    "gpt2": CONFIGS["gpt2"],
}

SIGMA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def run_adversarial_sweep(build_dir):
    """Run precision sweep over input standard deviation.

    Draws Q, K, V from N(0, sigma^2) and sweeps sigma. This directly
    controls the attention score magnitude: E[q_i * k_j] = sigma^2,
    so the raw QK dot product scales as dk * sigma^2 before the
    1/sqrt(dk) normalization. Higher sigma produces more peaked softmax
    distributions and larger bf16 truncation errors in the score buffer.
    """
    lk = 2048
    seed = 42

    all_results = []

    for cfg_name, cfg in ADVERSARIAL_CONFIGS.items():
        dk, dv, nh, nkv = cfg["dk"], cfg["dv"], cfg["nh"], cfg["nkv"]
        causal = cfg["causal"]
        dv_chunks = dv // LKP
        lq = 2048

        print(f"\n--- {cfg['label']} (dk={dk}, NH={nh}, NKV={nkv}) ---")

        # Compile once per config (fixed lk=2048)
        os.chdir(build_dir)
        mlir_module = build_module(
            lk=lk,
            lkp=LKP,
            lq=lq,
            lqp=LQP,
            dk=dk,
            dv=dv,
            num_q_tiles=NUM_Q_TILES,
            num_cascade_stages=NUM_CASCADE_STAGES,
            num_heads=nh,
            num_kv_heads=nkv,
            causal=causal,
        )
        backend = XRTBackend(
            omit_while_true_loop=False,
            omit_pingpong="all",
            output_format="elf",
            instance_name="attention_bf16",
            target_device="npu2",
        )
        artifact = backend.compile(mlir_module)

        for sigma in SIGMA_VALUES:
            print(f"  sigma={sigma:.2f} ... ", end="", flush=True)
            t0 = time.time()

            # Generate inputs from N(0, sigma^2)
            rng = np.random.default_rng(seed)
            Q = rng.normal(0, sigma, (nh, lq, dk)).astype(bfloat16)
            K = rng.normal(0, sigma, (nkv, lk, dk)).astype(bfloat16)
            V_orig = rng.normal(0, sigma, (nkv, lk, dv)).astype(bfloat16)
            V_transposed = transpose_v(V_orig, nkv, lk, dv, LKP)
            O = np.zeros((nh * dv_chunks, lq, LKP), dtype=bfloat16)

            # Run on NPU
            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(artifact)
                npu_raw = invoker(Q, K, V_transposed, O)
                backend.unload()

            npu_out = npu_raw[3].reshape(nh * dv_chunks, lq, LKP)
            npu_out = untranspose_output(npu_out, nh, lq, dv, LKP).astype(np.float32)

            # F32 reference with same inputs
            ref = cpu_reference(Q, K, V_orig, nh, nkv, causal)
            m = precision_metrics(npu_out, ref)
            m["config"] = cfg_name
            m["sigma"] = sigma

            all_results.append(m)
            elapsed = time.time() - t0
            print(
                f"head_corr={m['head_corr']:.6f}  "
                f"mean_err={m['mean_err']:.4f}  "
                f"fail_pct={m['fail_pct']:.3f}%  ({elapsed:.1f}s)"
            )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["all"],
        choices=list(CONFIGS.keys()) + ["all"],
        help="Configurations to sweep (default: all)",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Override sequence lengths (default: per-config defaults)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 7777],
        help="Random seeds (default: 42 123 7777)",
    )
    parser.add_argument(
        "--val-range",
        type=float,
        default=2.5,
        help="Input uniform range [0, val_range) (default: 2.5)",
    )
    parser.add_argument(
        "--per-head",
        action="store_true",
        help="Report per-head correlation at lk=2048",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="precision_sweep_results.csv",
        help="Output CSV file (default: precision_sweep_results.csv)",
    )
    parser.add_argument(
        "--adversarial-sweep",
        action="store_true",
        help="Run cascade merge stress test with escalating K magnitude per stage",
    )
    args = parser.parse_args()

    # Build directory
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_peano")
    os.makedirs(build_dir, exist_ok=True)
    orig_dir = os.getcwd()

    # -----------------------------------------------------------------------
    # Adversarial sweep mode
    # -----------------------------------------------------------------------
    if args.adversarial_sweep:
        print("=" * 72)
        print("Numerical Stability — Input Variance Sweep")
        print("  Q, K, V ~ N(0, sigma^2)")
        print(f"  sigma sweep: {SIGMA_VALUES}")
        print("  LK=2048, LQ=2048, seed=42")
        print("=" * 72)

        results = run_adversarial_sweep(build_dir)
        os.chdir(orig_dir)

        # Summary table
        print()
        print("=" * 72)
        print("INPUT VARIANCE SWEEP SUMMARY")
        print("=" * 72)
        for cfg_name in ADVERSARIAL_CONFIGS:
            cfg = ADVERSARIAL_CONFIGS[cfg_name]
            print(f"\n{cfg['label']}:")
            print(
                f"  {'sigma':>10s}  {'head_corr':>10s}  {'mean_err':>10s}  "
                f"{'max_err':>10s}  {'fail_pct':>10s}"
            )
            print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
            for r in results:
                if r["config"] != cfg_name:
                    continue
                print(
                    f"  {r['sigma']:>10.2f}  "
                    f"{r['head_corr']:>10.6f}  "
                    f"{r['mean_err']:>10.4f}  "
                    f"{r['max_err']:>10.4f}  "
                    f"{r['fail_pct']:>9.3f}%"
                )

        # Write CSV
        csv_path = os.path.join(orig_dir, args.output_csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "config",
                    "sigma",
                    "corr",
                    "head_corr",
                    "head_corr_min",
                    "max_err",
                    "mean_err",
                    "fail_pct",
                    "out_std",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in writer.fieldnames})
        print(f"\nResults saved to {csv_path}")
        return

    # Resolve config list
    if "all" in args.configs:
        config_names = list(CONFIGS.keys())
    else:
        config_names = args.configs

    all_results = []
    all_per_head = {}

    print("=" * 72)
    print("Flash Attention Numerical Stability Sweep")
    print(f"  val_range={args.val_range}, seeds={args.seeds}")
    print("=" * 72)

    for cfg_name in config_names:
        cfg = CONFIGS[cfg_name]
        seq_lengths = args.seq_lengths if args.seq_lengths else cfg["seqs"]

        print()
        print(
            f"--- {cfg['label']} (dk={cfg['dk']}, NH={cfg['nh']}, "
            f"NKV={cfg['nkv']}, causal={cfg['causal']}) ---"
        )
        print(f"    Sequence lengths: {seq_lengths}")

        os.chdir(orig_dir)
        results, ph = run_config_sweep(
            cfg_name,
            cfg,
            seq_lengths,
            args.seeds,
            args.val_range,
            args.per_head,
            build_dir,
        )
        all_results.extend(results)
        all_per_head.update(ph)
        os.chdir(orig_dir)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY: Mean Correlation (std) across seeds")
    print("=" * 72)

    # Collect unique configs and seq lengths
    for cfg_name in config_names:
        cfg = CONFIGS[cfg_name]
        seq_lengths = args.seq_lengths if args.seq_lengths else cfg["seqs"]
        print(f"\n{cfg['label']}:")
        print(
            f"  {'LK':>6s}  {'head_corr':>14s}  {'agg_corr':>14s}  "
            f"{'max_err':>10s}  {'mean_err':>10s}  {'fail_pct':>10s}  {'out_std':>8s}"
        )
        print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

        for lk in seq_lengths:
            rows = [r for r in all_results if r["config"] == cfg_name and r["lk"] == lk]
            if not rows:
                continue
            head_corrs = [r["head_corr"] for r in rows]
            corrs = [r["corr"] for r in rows]
            max_errs = [r["max_err"] for r in rows]
            mean_errs = [r["mean_err"] for r in rows]
            fail_pcts = [r["fail_pct"] for r in rows]
            out_stds = [r["out_std"] for r in rows]
            print(
                f"  {lk:>6d}  "
                f"{np.mean(head_corrs):.6f}+/-{np.std(head_corrs):.6f}  "
                f"{np.mean(corrs):.6f}+/-{np.std(corrs):.6f}  "
                f"{np.mean(max_errs):>10.4f}  "
                f"{np.mean(mean_errs):>10.4f}  "
                f"{np.mean(fail_pcts):>9.3f}%  "
                f"{np.mean(out_stds):>8.4f}"
            )

    # -----------------------------------------------------------------------
    # Per-head table
    # -----------------------------------------------------------------------
    if args.per_head and all_per_head:
        print()
        print("=" * 72)
        print("PER-HEAD CORRELATION at LK=2048 (seed=", args.seeds[0], ")")
        print("=" * 72)
        for cfg_name, corrs in all_per_head.items():
            cfg = CONFIGS[cfg_name]
            print(f"\n{cfg['label']} ({len(corrs)} heads):")
            print(
                f"  min={min(corrs):.6f}  mean={np.mean(corrs):.6f}  "
                f"max={max(corrs):.6f}"
            )
            # Print individual heads in groups of 8
            for i in range(0, len(corrs), 8):
                chunk = corrs[i : i + 8]
                vals = "  ".join(f"{c:.5f}" for c in chunk)
                print(f"  heads {i:>2d}-{i+len(chunk)-1:>2d}: {vals}")

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(orig_dir, args.output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "lk",
                "seed",
                "corr",
                "head_corr",
                "head_corr_min",
                "max_err",
                "mean_err",
                "fail_pct",
                "out_std",
            ],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\nResults saved to {csv_path}")

    # Check pass/fail (use per-head min correlation)
    min_head_corr = min(r["head_corr_min"] for r in all_results)
    min_agg_corr = min(r["corr"] for r in all_results)
    print(f"\nMin per-head correlation: {min_head_corr:.6f}")
    print(f"Min aggregate correlation: {min_agg_corr:.6f}")
    if min_head_corr < 0.99:
        print(f"WARNING: Min per-head correlation {min_head_corr:.6f} < 0.99")
        sys.exit(1)
    else:
        print("All configurations PASS (all per-head correlations >= 0.99)")


if __name__ == "__main__":
    main()
