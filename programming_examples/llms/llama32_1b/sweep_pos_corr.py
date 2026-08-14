"""Sweep (layer, pos) and report per-head correlation vs CPU reference.

Built to add comprehensive debugging data to Xilinx/mlir-air#1600
(the 0.65 outlier head correlation in attn_decode_npu2 with real
Llama-3.2-1B weights).

Sweeps:
  - layer indices: configurable, default {0, 1, 8, 15}
  - pos values:    configurable, default {10, 100, 500, 1000, 1500, 2000, 2046}

For each (layer, pos), reports:
  - overall correlation
  - per-Q-head correlation (32 heads = 8 kv groups × 4 Q heads/group)
  - which Q-head index is the worst-corr outlier
  - max abs error
  - max abs of CPU output (to see if the error is large in absolute terms)

Useful for answering:
  - Is the same Q-head always the outlier?  (if yes, kernel-side bug in
    that specific head/group)
  - Does correlation degrade with pos?       (if yes, attn-2 accumulator
    saturation suspect)
  - Does it vary across layers?              (if yes, weight-magnitude
    suspect)

Usage:
    python3 sweep_pos_corr.py --layers 0 1 8 15 --pos 10 100 500 1000 1500 2046

Reuses fused_pathD_cache from chat_fused_d.py runs to skip recompiles.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ATTN_DECODE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "attention_decode")
)
sys.path.insert(0, _ATTN_DECODE_DIR)

from llama32_1b_weights import LlamaConfig, load_weights
from llama_kernel_builder.cache import KernelCache
from attn_decode_npu2 import build_module as build_fused_attn_module

N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
EMB_DIM = N_HEADS * HEAD_DIM  # 2048
GROUP_SIZE = N_HEADS // N_KV_HEADS  # 4
GEMV_COUNT = GROUP_SIZE + 2  # 6
TILE_K = 128
TILE_N = HEAD_DIM
ROPE_BASE = 500000.0

FUSED_ELF_BACKEND = dict(
    verbose=False,
    omit_while_true_loop=False,
    omit_pingpong=True,
    output_format="elf",
    instance_name="mha_bf16",
    target_device="npu2",
    stack_size=0xC00,
)


# ---------- CPU reference (matches kernel: half-split RoPE, rope_base=500000) ----------

def rms_norm_cpu(x_bf16, w_bf16, eps=1e-5):
    x = x_bf16.astype(np.float32)
    w = w_bf16.astype(np.float32)
    rstd = 1.0 / np.sqrt(np.mean(x * x) + eps)
    return (x * rstd * w).astype(bfloat16)


def apply_rope_halfsplit_cpu(vec_bf16, pos):
    vec = vec_bf16.astype(np.float32).copy()
    half = HEAD_DIM // 2
    freqs = pos / np.power(ROPE_BASE, 2 * np.arange(half) / HEAD_DIM)
    cos_v = np.cos(freqs)
    sin_v = np.sin(freqs)
    x1 = vec[:half].copy()
    x2 = vec[half:].copy()
    vec[:half] = x1 * cos_v - x2 * sin_v
    vec[half:] = x1 * sin_v + x2 * cos_v
    return vec.astype(bfloat16)


def cpu_reference(x_in_bf16, lw, k_cache, v_cache, pos):
    x_norm = rms_norm_cpu(x_in_bf16, lw.attn_norm.reshape(EMB_DIM))
    x_norm_f = x_norm.astype(np.float32)
    q_full = (x_norm_f @ lw.wq.astype(np.float32)).astype(bfloat16)
    k_full = (x_norm_f @ lw.wk.astype(np.float32)).astype(bfloat16)
    v_full = (x_norm_f @ lw.wv.astype(np.float32)).astype(bfloat16)

    q_heads = q_full.reshape(N_HEADS, HEAD_DIM)
    k_heads = k_full.reshape(N_KV_HEADS, HEAD_DIM)
    v_heads = v_full.reshape(N_KV_HEADS, HEAD_DIM)

    q_roped = np.zeros_like(q_heads)
    for h in range(N_HEADS):
        q_roped[h] = apply_rope_halfsplit_cpu(q_heads[h], pos)
    k_roped = np.zeros_like(k_heads)
    for h in range(N_KV_HEADS):
        k_roped[h] = apply_rope_halfsplit_cpu(k_heads[h], pos)

    for kv in range(N_KV_HEADS):
        k_cache[kv, pos, :] = k_roped[kv]
        v_cache[kv, pos, :] = v_heads[kv]

    inv_sqrt_n = 1.0 / sqrt(HEAD_DIM)
    xb = np.zeros((N_KV_HEADS, GROUP_SIZE, HEAD_DIM), dtype=bfloat16)
    for kv in range(N_KV_HEADS):
        Kc = k_cache[kv, : pos + 1, :].astype(np.float32)
        Vc = v_cache[kv, : pos + 1, :].astype(np.float32)
        for g in range(GROUP_SIZE):
            q_head_idx = kv * GROUP_SIZE + g
            q = q_roped[q_head_idx].astype(np.float32)
            scores = Kc @ q * inv_sqrt_n
            scores -= scores.max()
            p = np.exp(scores)
            p /= p.sum()
            xb[kv, g, :] = (p @ Vc).astype(bfloat16)
    return xb


# ---------- Weight + xrms packing ----------

def pack_weights_to_B(lw):
    B = np.zeros((N_KV_HEADS, GEMV_COUNT, EMB_DIM, HEAD_DIM), dtype=bfloat16)
    for kv in range(N_KV_HEADS):
        for g in range(GROUP_SIZE):
            q_head = kv * GROUP_SIZE + g
            B[kv, g] = lw.wq[:, q_head * HEAD_DIM : (q_head + 1) * HEAD_DIM]
        B[kv, GROUP_SIZE] = lw.wk[:, kv * HEAD_DIM : (kv + 1) * HEAD_DIM]
        B[kv, GROUP_SIZE + 1] = lw.wv[:, kv * HEAD_DIM : (kv + 1) * HEAD_DIM]
    return B


def pack_xrms(x_in_bf16, attn_norm_bf16):
    xrms = np.zeros((TILE_K, TILE_N), dtype=bfloat16)
    flat = xrms.reshape(-1)
    flat[:EMB_DIM] = x_in_bf16
    flat[EMB_DIM : 2 * EMB_DIM] = attn_norm_bf16
    return xrms


# ---------- External .o (one-time per seq_len) ----------

def compile_attn_decode_o(seq_len):
    cc_src = os.path.join(_ATTN_DECODE_DIR, "attn_decode_npu2.cc")
    o_path = os.path.join(os.getcwd(), "attn_decode_npu2.o")
    peano = os.environ["PEANO_INSTALL_DIR"]
    aieopt = os.environ.get("AIEOPT_DIR") or os.environ["MLIR_AIE_INSTALL_DIR"]
    cmd = [
        f"{peano}/bin/clang++",
        "-Os", "-std=c++20", "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-DNDEBUG", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-I", f"{aieopt}/include",
        f"-DSEQ_LEN={seq_len}", f"-DGROUP_SIZE={GROUP_SIZE}",
        f"-DDIM_K={EMB_DIM}", f"-DTILE_K={TILE_K}",
        f"-DDIM_N={HEAD_DIM}", f"-DHEAD_SIZE={HEAD_DIM}",
        "-c", cc_src, "-o", o_path,
    ]
    subprocess.run(cmd, check=True)


def round_seq_len(p):
    return ((p + 1 + 15) // 16) * 16


# ---------- Compile + run one (layer, pos) point ----------

def run_one(cache, lw, layer_idx, pos, seq_len, rng_seed):
    """Compile fused-attn ELF for this pos (cache reuses), run on NPU,
    run CPU reference, return per-head corr table.
    """
    name = f"sweep_pos{pos}_seq{seq_len}"
    if name not in cache.artifacts:
        mod = build_fused_attn_module(
            EMB_DIM, HEAD_DIM, TILE_K, TILE_N, seq_len,
            bfloat16, bfloat16, bfloat16, pos,
            group_size=GROUP_SIZE, nkv=N_KV_HEADS,
        )
        cache.compile_and_cache(name, mod, FUSED_ELF_BACKEND)
        cache._save_manifest()

    rng = np.random.default_rng(rng_seed)
    x_in = rng.standard_normal(EMB_DIM).astype(bfloat16)
    # KV magnitude knob: real Llama-3.2-1B prefill K/V is at magnitude ~5;
    # the sweep originally used 0.1 which understates the kernel's bf16 sensitivity.
    kv_scale = float(os.environ.get("SWEEP_KV_SCALE", "0.1"))
    k_cache_init = (rng.standard_normal((N_KV_HEADS, seq_len, HEAD_DIM)) * kv_scale).astype(bfloat16)
    v_cache_init = (rng.standard_normal((N_KV_HEADS, seq_len, HEAD_DIM)) * kv_scale).astype(bfloat16)
    k_cache_init[:, pos:, :] = 0
    v_cache_init[:, pos:, :] = 0

    B = pack_weights_to_B(lw)
    xrms = pack_xrms(x_in, lw.attn_norm.reshape(EMB_DIM).astype(bfloat16))
    xb_buf = np.zeros((N_KV_HEADS, GROUP_SIZE, HEAD_DIM), dtype=bfloat16)

    # CPU reference (mutates k/v caches in place)
    k_ref = k_cache_init.copy()
    v_ref = v_cache_init.copy()
    xb_ref = cpu_reference(x_in, lw, k_ref, v_ref, pos)

    # NPU run
    res = cache.load_and_run(
        name, FUSED_ELF_BACKEND,
        xrms, B, k_cache_init.copy(), v_cache_init.copy(), xb_buf,
        output_indices=[4],
        static_input_indices={1, 2, 3},
        intermediate_indices={4},
        bo_key=f"{name}_L{layer_idx}",
    )
    xb_npu = np.asarray(res[4]).reshape(N_KV_HEADS, GROUP_SIZE, HEAD_DIM)

    # Per-head correlation table
    corr_table = np.zeros((N_KV_HEADS, GROUP_SIZE), dtype=np.float64)
    abs_err_table = np.zeros((N_KV_HEADS, GROUP_SIZE), dtype=np.float64)
    cpu_mag_table = np.zeros((N_KV_HEADS, GROUP_SIZE), dtype=np.float64)
    xb_ref_f = xb_ref.astype(np.float32)
    xb_npu_f = xb_npu.astype(np.float32)
    for kv in range(N_KV_HEADS):
        for g in range(GROUP_SIZE):
            r = xb_ref_f[kv, g].flatten()
            n = xb_npu_f[kv, g].flatten()
            denom = np.linalg.norm(r) * np.linalg.norm(n)
            corr = float(np.dot(r, n) / denom) if denom > 1e-12 else 1.0
            corr_table[kv, g] = corr
            abs_err_table[kv, g] = float(np.max(np.abs(r - n)))
            cpu_mag_table[kv, g] = float(np.max(np.abs(r)))

    overall_corr = float(
        np.dot(xb_ref_f.flatten(), xb_npu_f.flatten())
        / (np.linalg.norm(xb_ref_f) * np.linalg.norm(xb_npu_f) + 1e-12)
    )
    return {
        "overall_corr": overall_corr,
        "corr_table": corr_table,
        "abs_err_table": abs_err_table,
        "cpu_mag_table": cpu_mag_table,
    }


def fmt_qhead_idx(kv, g):
    return f"Q{kv * GROUP_SIZE + g} (kv={kv}, g={g})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 8, 15])
    parser.add_argument("--pos", type=int, nargs="+",
                        default=[10, 100, 500, 1000, 1500, 2046])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--cache-dir", default="./sweep_cache")
    args = parser.parse_args()

    print("=" * 78)
    print(f"Sweep: layers={args.layers}  pos={args.pos}  seed={args.seed}")
    print(f"Model: {args.model}")
    print("=" * 78)

    config = LlamaConfig()
    print(f"\nLoading weights ({args.model})...")
    weights = load_weights(args.model, dtype=bfloat16, config=config)

    # Compile .o once at the largest seq_len needed
    max_pos = max(args.pos)
    seq_len = round_seq_len(max_pos)
    print(f"Compiling attn_decode_npu2.o (SEQ_LEN={seq_len})...")
    compile_attn_decode_o(seq_len)

    cache = KernelCache(cache_dir=args.cache_dir, verbose=False)
    cache.load_manifest()

    # Per-(layer, pos) results
    results = {}
    print()
    for layer_idx in args.layers:
        lw = weights.layers[layer_idx]
        for pos in args.pos:
            print(f"  L{layer_idx:2d} pos={pos:4d}: ", end="", flush=True)
            t0 = time.time()
            r = run_one(cache, lw, layer_idx, pos, seq_len, args.seed)
            dt = time.time() - t0
            results[(layer_idx, pos)] = r
            cmin = float(r["corr_table"].min())
            cmean = float(r["corr_table"].mean())
            outlier_idx = np.unravel_index(
                np.argmin(r["corr_table"]), r["corr_table"].shape
            )
            outlier_head = outlier_idx[0] * GROUP_SIZE + outlier_idx[1]
            print(
                f"corr_overall={r['overall_corr']:.4f}  "
                f"per-head min={cmin:.4f} mean={cmean:.4f}  "
                f"outlier=Q{outlier_head:2d} (kv={outlier_idx[0]}, g={outlier_idx[1]})  "
                f"max_abs_err={float(r['abs_err_table'].max()):.4f}  ({dt:.1f}s)"
            )

    # ---- Summary 1: outlier-head consistency ----
    print()
    print("=" * 78)
    print("Summary 1: outlier Q-head index per (layer, pos)")
    print("=" * 78)
    header = "layer\\pos  " + "".join(f"  pos={p:5d}" for p in args.pos)
    print(header)
    for layer_idx in args.layers:
        cells = []
        for pos in args.pos:
            ct = results[(layer_idx, pos)]["corr_table"]
            outlier = np.unravel_index(np.argmin(ct), ct.shape)
            cells.append(f"  Q{outlier[0]*GROUP_SIZE + outlier[1]:2d}({ct.min():.2f})")
        print(f"L{layer_idx:2d}      " + "".join(c.rjust(11) for c in cells))

    # ---- Summary 2: per-Q-head min correlation across all (layer, pos) ----
    print()
    print("=" * 78)
    print("Summary 2: per-Q-head min correlation across all swept (layer, pos)")
    print("=" * 78)
    head_min = {h: 1.0 for h in range(N_HEADS)}
    head_n_below_threshold = {h: 0 for h in range(N_HEADS)}
    for r in results.values():
        for kv in range(N_KV_HEADS):
            for g in range(GROUP_SIZE):
                h = kv * GROUP_SIZE + g
                c = float(r["corr_table"][kv, g])
                head_min[h] = min(head_min[h], c)
                if c < 0.95:
                    head_n_below_threshold[h] += 1
    bad_heads = sorted([h for h in range(N_HEADS) if head_min[h] < 0.95],
                       key=lambda h: head_min[h])
    print(f"Q-heads with corr < 0.95 in at least one config "
          f"(out of {len(args.layers)*len(args.pos)} configs):")
    for h in bad_heads:
        kv, g = divmod(h, GROUP_SIZE)
        print(f"  Q{h:2d} (kv={kv}, g={g}): worst corr = {head_min[h]:.4f}, "
              f"#configs where corr<0.95: {head_n_below_threshold[h]}")
    if not bad_heads:
        print("  None (all heads stay >=0.95 across all configs)")

    # ---- Summary 3: corr-vs-pos for layer 0 (single-layer trend) ----
    if 0 in args.layers and len(args.pos) > 1:
        print()
        print("=" * 78)
        print("Summary 3: layer 0, per-Q-head correlation vs pos")
        print("=" * 78)
        # Header
        print(f"  Q-head     " + "".join(f"  pos={p:5d}" for p in args.pos))
        for kv in range(N_KV_HEADS):
            for g in range(GROUP_SIZE):
                h = kv * GROUP_SIZE + g
                row = f"  Q{h:2d} (kv={kv},g={g}) "
                for pos in args.pos:
                    c = float(results[(0, pos)]["corr_table"][kv, g])
                    row += f"   {c:.4f} "
                print(row)

    # ---- Summary 4: outlier-head's metrics per pos ----
    if 0 in args.layers:
        # Use the most-frequent outlier across pos as the head of interest
        from collections import Counter
        outliers = []
        for pos in args.pos:
            ct = results[(0, pos)]["corr_table"]
            o = np.unravel_index(np.argmin(ct), ct.shape)
            outliers.append(o[0] * GROUP_SIZE + o[1])
        most_common_outlier, _ = Counter(outliers).most_common(1)[0]
        kv, g = divmod(most_common_outlier, GROUP_SIZE)
        print()
        print("=" * 78)
        print(f"Summary 4: layer 0, head Q{most_common_outlier} (kv={kv}, g={g}) "
              f"— most-frequent outlier across pos")
        print("=" * 78)
        print(f"  pos      corr     max_abs_err   max(|cpu|)   err/mag")
        for pos in args.pos:
            r = results[(0, pos)]
            c = float(r["corr_table"][kv, g])
            ae = float(r["abs_err_table"][kv, g])
            mg = float(r["cpu_mag_table"][kv, g])
            ratio = ae / max(mg, 1e-9)
            print(f"  {pos:5d}    {c:.4f}    {ae:.4f}        {mg:.4f}      {ratio:.3f}")

    print()


if __name__ == "__main__":
    main()
