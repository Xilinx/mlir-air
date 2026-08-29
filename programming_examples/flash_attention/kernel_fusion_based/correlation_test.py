#!/usr/bin/env python3
"""Flash Attention precision analysis — self-contained reproduction script.

Measures correlation, max_err, mean_err, and 4%-fail rate against
CPU F32 standard attention reference. Designed for developers to
verify kernel correctness.

Usage:
    cd programming_examples/flash_attention/kernel_fusion_based
    python3 test_precision.py                           # Test default configs
    python3 test_precision.py --num-heads 8             # Single config
    python3 test_precision.py --num-heads 32 --causal   # LLAMA causal config

Prerequisites:
    - MLIR-AIR environment (source utils/env_setup.sh)
    - NPU2 hardware (ryzen_ai_npu2)
"""

import argparse, os, sys
from math import sqrt

import numpy as np
from ml_dtypes import bfloat16
import filelock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn import build_module
from air.backend.xrt import XRTBackend


def cpu_reference(Q, K, V, num_heads, num_kv_heads, causal):
    """Standard scaled dot-product attention in F32.

    This is the mathematically correct reference — NOT the tiled flash
    attention reference used by the standalone test.
    """
    LQ, DK = Q.shape[1], Q.shape[2]
    LK, DV = V.shape[1], V.shape[2]
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
    """Compute precision metrics matching IRON's reporting."""
    diff = np.abs(npu - ref)
    corr = np.corrcoef(npu.flatten(), ref.flatten())[0, 1]
    tol = np.maximum(0.15, 0.04 * (np.abs(npu) + np.abs(ref)))
    fail_pct = np.sum(diff > tol) / diff.size * 100
    return {
        "corr": corr,
        "max_err": float(diff.max()),
        "mean_err": float(diff.mean()),
        "fail_pct": fail_pct,
        "out_range": (float(ref.min()), float(ref.max())),
        "out_mean": float(ref.mean()),
        "out_std": float(ref.std()),
    }


def run_one(lq, lk, lqp, lkp, dk, dv, num_heads, num_kv_heads, causal, val_range=3.0):
    """Compile, run on NPU, and measure precision for one config."""
    os.makedirs("build_peano", exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir("build_peano")

    try:
        mlir_module = build_module(
            lk=lk,
            lkp=lkp,
            lq=lq,
            lqp=lqp,
            dk=dk,
            dv=dv,
            num_q_tiles=4,
            num_cascade_stages=4,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            causal=causal,
        )

        enable_shared = lkp == dk
        omit_loop = False if causal else not enable_shared
        backend = XRTBackend(
            omit_while_true_loop=omit_loop,
            omit_pingpong="all",
            output_format="elf",
            instance_name="attention_bf16",
        )
        artifact = backend.compile(mlir_module)

        np.random.seed(42)
        Q = np.random.uniform(0, val_range, (num_heads, lq, dk)).astype(bfloat16)
        K = np.random.uniform(0, val_range, (num_kv_heads, lk, dk)).astype(bfloat16)
        V = np.random.uniform(0, val_range, (num_kv_heads, lk, dv)).astype(bfloat16)
        M = np.zeros((num_heads, lq, lk), dtype=bfloat16)
        O = np.zeros((num_heads, lq, dv), dtype=bfloat16)

        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)
            results = invoker(Q, K, V, M, O)

        npu = results[len([Q, K, V, M]) :][0]
        npu = npu.reshape(num_heads, lq, dv).astype(np.float32)
        backend.unload()

    finally:
        os.chdir(orig_dir)

    ref = cpu_reference(Q, K, V, num_heads, num_kv_heads, causal)
    return precision_metrics(npu, ref)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lq", type=int, default=2048)
    parser.add_argument("--lk", type=int, default=2048)
    parser.add_argument("--lqp", type=int, default=256)
    parser.add_argument("--lkp", type=int, default=64)
    parser.add_argument("--dk", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="If not set, runs sweep of multiple configs",
    )
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--val-range", type=float, default=3.0)
    args = parser.parse_args()

    if args.num_heads is not None:
        # Single config
        nkv = args.num_kv_heads or args.num_heads
        print(
            f"FlashAttention: LQ={args.lq} LK={args.lk} NH={args.num_heads} "
            f"NKV={nkv} causal={args.causal}"
        )
        m = run_one(
            args.lq,
            args.lk,
            args.lqp,
            args.lkp,
            args.dk,
            args.dv,
            args.num_heads,
            nkv,
            args.causal,
            args.val_range,
        )
        print(
            f"Output range: [{m['out_range'][0]:.3f}, {m['out_range'][1]:.3f}], "
            f"mean={m['out_mean']:.3f}, std={m['out_std']:.4f}"
        )
        print(
            f"Precision -- corr: {m['corr']:.6f}, max_err: {m['max_err']:.4f}, "
            f"mean_err: {m['mean_err']:.4f}, 4%-fail: {m['fail_pct']:.2f}%"
        )
    else:
        # Sweep multiple configs
        configs = [
            ("4h MHA", 2048, 2048, 256, 64, 64, 64, 4, 4, False),
            ("8h MHA", 2048, 2048, 256, 64, 64, 64, 8, 8, False),
            ("8h GQA", 2048, 2048, 256, 64, 64, 64, 8, 4, False),
            ("LLAMA", 2048, 2048, 256, 64, 64, 64, 32, 8, False),
            ("LLAMA causal", 2048, 2048, 256, 64, 64, 64, 32, 8, True),
        ]

        print("Flash Attention Precision Sweep (LQ=LK=2048, DK=DV=64, val_range=3)")
        print(
            f"{'Config':18s} {'corr':>10s} {'max_err':>8s} {'mean_err':>8s} {'4%-fail':>8s} {'out_std':>8s}"
        )
        print("-" * 65)

        for name, lq, lk, lqp, lkp, dk, dv, nh, nkv, causal in configs:
            try:
                m = run_one(lq, lk, lqp, lkp, dk, dv, nh, nkv, causal, args.val_range)
                print(
                    f"{name:18s} {m['corr']:>10.6f} {m['max_err']:>8.4f} "
                    f"{m['mean_err']:>8.4f} {m['fail_pct']:>7.2f}% {m['out_std']:>8.4f}"
                )
            except Exception as e:
                print(f"{name:18s} ERROR: {str(e)[:50]}")

        print()
        print("Expected: corr > 0.99 for correct kernel (IRON MHA achieves 0.998)")
        print("          corr < 0.35 indicates kernel is computing wrong attention")


if __name__ == "__main__":
    main()
