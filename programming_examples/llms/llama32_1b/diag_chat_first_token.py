"""Diagnostic: at decode token 1 layer 0 with REAL prefill K/V cache,
compare three xb computations:

  A. Pure CPU reference   (sweep_pos_corr.py's cpu_reference function)
  B. Fused NPU kernel     (the one chat_fused_c.py uses, via FusedAttnSwapRunner)
  C. Baseline production  (rms_gemv_rope NPU + decode_attention_cpu)

If A ≈ C: production chain matches my CPU reference (sanity)
If A ≈ B: the fused NPU kernel matches my CPU reference (consistent with sweep's 0.999)
If C ≈ B: the fused NPU kernel matches the production chain (chat would be correct)

If A ≈ B but C ≠ B, my CPU reference and the kernel agree but neither matches
production → wrap/format issue. If A ≠ B on real inputs, kernel-level bug
that the random-data sweep didn't catch.
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ATTN_DECODE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "attention_decode")
)
sys.path.insert(0, _ATTN_DECODE_DIR)

from llama32_1b_inference import prepare_runtime, run_npu_prefill, _LM_N_PARTITIONS, _LM_N_PART
from llama32_1b_prefill import compile_all_kernels as compile_prefill_kernels
from llama32_1b_decode import compile_decode_kernels, decode_attention_cpu
from llama32_1b_weights import LlamaConfig, load_weights, generate_rope_lut
from llama_kernel_builder.cache import KernelCache
from llama_kernel_builder.backend_presets import RGR_BACKEND
from chat_fused_c import (
    FusedAttnSwapRunner, pack_weights_to_B, pack_xrms,
    EMB_DIM, HEAD_DIM, N_HEADS, N_KV_HEADS, GROUP_SIZE, GEMV_COUNT,
    TILE_K, TILE_N, FUSED_ATTN_BACKEND, compile_pos_artifacts,
)
from sweep_pos_corr import cpu_reference

KV_DIM = N_KV_HEADS * HEAD_DIM


def per_head_corr(xb_a, xb_b):
    """Per-Q-head correlation (8x4=32) + overall."""
    af = xb_a.astype(np.float32)
    bf = xb_b.astype(np.float32)
    table = np.zeros((N_KV_HEADS, GROUP_SIZE), dtype=np.float64)
    err_table = np.zeros_like(table)
    for kv in range(N_KV_HEADS):
        for g in range(GROUP_SIZE):
            r = af[kv, g].flatten()
            n = bf[kv, g].flatten()
            denom = np.linalg.norm(r) * np.linalg.norm(n)
            table[kv, g] = float(np.dot(r, n) / denom) if denom > 1e-12 else 1.0
            err_table[kv, g] = float(np.max(np.abs(r - n)))
    overall = float(
        np.dot(af.flatten(), bf.flatten())
        / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)
    )
    return overall, table, err_table


def fmt_corr_table(name, overall, ct, et):
    print(f"\n=== {name} ===")
    print(f"  overall corr: {overall:.6f}")
    print(f"  per-head min/mean/max: {ct.min():.4f} / {ct.mean():.4f} / {ct.max():.4f}")
    print(f"  per-head max_abs_err min/mean/max: "
          f"{et.min():.4f} / {et.mean():.4f} / {et.max():.4f}")
    worst = np.unravel_index(np.argmin(ct), ct.shape)
    print(f"  worst head: Q{worst[0]*GROUP_SIZE + worst[1]} "
          f"(kv={worst[0]}, g={worst[1]}) corr={ct[worst]:.4f} "
          f"err={et[worst]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading weights ({args.model})...")
    config = LlamaConfig()
    weights = load_weights(args.model, dtype=bfloat16, config=config)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if "Instruct" in args.model:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt_text = args.prompt
    raw_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    SEQ_PAD = 2048
    if len(raw_tokens) < SEQ_PAD:
        raw_tokens = raw_tokens + [tokenizer.eos_token_id] * (SEQ_PAD - len(raw_tokens))
    prompt_len = len(raw_tokens)
    print(f"  padded prompt_len={prompt_len}")

    rope_lut_bf16 = generate_rope_lut(config=config, seq_len=prompt_len + 4, dtype=bfloat16)

    prefill_cache = KernelCache(cache_dir="./prefill_kernel_cache")
    decode_cache = KernelCache(cache_dir="./decode_kernel_cache")
    prefill_cache.load_manifest()
    decode_cache.load_manifest()
    if not prefill_cache.artifacts:
        compile_prefill_kernels(prefill_cache, config, prompt_len, cpu_attn=False)
    if not decode_cache.artifacts:
        compile_decode_kernels(decode_cache, config)

    fused_seq_len = ((prompt_len + 1 + 15) // 16) * 16
    xclbin_path, insts_per_pos = compile_pos_artifacts(
        prompt_len, 1, fused_seq_len, cache_dir="./fused_pathC_cache"
    )

    prepare_runtime(prefill_cache, decode_cache, weights, config,
                    seq_len=prompt_len, rope_lut_bf16=rope_lut_bf16)

    print("\nRunning NPU prefill to populate REAL KV cache...")
    t0 = time.time()
    prefill_token, k_cache, v_cache, _ = run_npu_prefill(
        raw_tokens, weights, config, prefill_cache, decode_cache,
        rope_lut_bf16, max_seq=fused_seq_len,
        tokenizer=tokenizer, cpu_attn=False, profile=False, verify=False,
        quiet=True,
    )
    print(f"  Prefill: {time.time() - t0:.2f}s, first token={prefill_token}")

    layer_idx = args.layer
    lw = weights.layers[layer_idx]
    current_pos = prompt_len
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    print(f"\n--- Diagnostic at layer={layer_idx}, pos={current_pos} ---")
    print(f"  x_decode magnitude: max(|x|)={float(np.max(np.abs(x_decode.astype(np.float32)))):.4f}, "
          f"mean(|x|)={float(np.mean(np.abs(x_decode.astype(np.float32)))):.4f}")
    print(f"  k_cache[L{layer_idx}] (prefill) at slot 0: "
          f"max(|k|)={float(np.max(np.abs(k_cache[layer_idx, :, 0, :].astype(np.float32)))):.4f}")
    print(f"  k_cache[L{layer_idx}] (prefill) at slot prompt_len-1: "
          f"max(|k|)={float(np.max(np.abs(k_cache[layer_idx, :, prompt_len-1, :].astype(np.float32)))):.4f}")
    print(f"  k_cache[L{layer_idx}] (prefill) at slot prompt_len (uninit): "
          f"max(|k|)={float(np.max(np.abs(k_cache[layer_idx, :, prompt_len, :].astype(np.float32)))):.4f}")

    # ---------- A. Pure CPU reference ----------
    print(f"\n[A] Pure CPU reference (sweep_pos_corr.cpu_reference)...")
    k_a = k_cache[layer_idx].copy()
    v_a = v_cache[layer_idx].copy()
    t0 = time.time()
    xb_A = cpu_reference(x_decode, lw, k_a, v_a, current_pos)
    print(f"    {time.time() - t0:.2f}s, xb shape={xb_A.shape}")

    # ---------- B. Fused NPU kernel ----------
    print(f"\n[B] Fused NPU kernel (FusedAttnSwapRunner.run)...")
    fused_runner = FusedAttnSwapRunner(xclbin_path, insts_per_pos, config.n_layers)
    B = pack_weights_to_B(lw)
    fused_runner.add_layer(layer_idx, B, k_cache[layer_idx], v_cache[layer_idx])
    xrms = pack_xrms(
        x_decode.flatten().astype(bfloat16),
        lw.attn_norm.reshape(EMB_DIM).astype(bfloat16),
    )
    t0 = time.time()
    xb_B = fused_runner.run(layer_idx, current_pos, xrms)
    print(f"    {time.time() - t0:.3f}s, xb shape={xb_B.shape}")

    # ---------- C. Baseline production: rms_gemv_rope NPU + decode_attention_cpu ----------
    print(f"\n[C] Baseline production (rms_gemv_rope NPU + decode_attention_cpu)...")
    # Reproduce what run_decode_block does for layer 0 of decode token 1 (per llama32_1b_decode.py:run_decode_block)
    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, 64)
    lut_q = np.tile(rope_lut_pos, (N_HEADS, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (N_KV_HEADS, 1)).flatten().astype(bfloat16)
    x_in = x_decode.flatten().astype(bfloat16)
    w_norm = lw.attn_norm.reshape(EMB_DIM).astype(bfloat16)
    normed_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    q_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    k_buf = np.zeros(KV_DIM, dtype=bfloat16)
    v_buf = np.zeros(KV_DIM, dtype=bfloat16)
    q_roped_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    k_roped_buf = np.zeros(KV_DIM, dtype=bfloat16)

    t0 = time.time()
    res = decode_cache.load_and_run(
        "rms_gemv_rope", RGR_BACKEND,
        x_in, w_norm, normed_buf,
        lw._wq_t, q_buf, lw._wk_t, k_buf, lw._wv_t, v_buf,
        lut_q, lut_k, q_roped_buf, k_roped_buf,
        output_indices=[8, 11, 12],
        static_input_indices={1, 3, 5, 7},
        intermediate_indices={2, 4, 6, 8, 11, 12},
        bo_key=f"rms_gemv_rope_L{layer_idx}",
    )
    v_C = np.asarray(res[8]).astype(bfloat16)
    q_roped_C = np.asarray(res[11]).reshape(N_HEADS, HEAD_DIM).astype(bfloat16)
    k_roped_C = np.asarray(res[12]).reshape(N_KV_HEADS, HEAD_DIM).astype(bfloat16)

    k_c = k_cache[layer_idx].copy()
    v_c = v_cache[layer_idx].copy()
    k_c[:, current_pos, :] = k_roped_C
    v_c[:, current_pos, :] = v_C.reshape(N_KV_HEADS, HEAD_DIM)

    attn_out_C = decode_attention_cpu(
        q_roped_C.flatten(), k_c, v_c, current_pos,
        N_HEADS, N_KV_HEADS, HEAD_DIM,
    )
    print(f"    {time.time() - t0:.3f}s, attn_out shape={attn_out_C.shape}")
    # Reshape attn_out_C to per-head (NKV, GROUP_SIZE, HEAD_DIM) for comparison
    xb_C = attn_out_C.reshape(N_HEADS, HEAD_DIM)
    xb_C = xb_C.reshape(N_KV_HEADS, GROUP_SIZE, HEAD_DIM)

    # ---------- Compare ----------
    print("\n" + "=" * 78)
    print("Pairwise correlation summary (real prefill K/V, layer 0, pos=prompt_len)")
    print("=" * 78)

    # A vs B (sweep equivalent: pure CPU vs NPU)
    o, ct, et = per_head_corr(xb_A, xb_B)
    fmt_corr_table("A (pure CPU) vs B (NPU fused)", o, ct, et)

    # A vs C (my CPU ref vs production CPU ref)
    o, ct, et = per_head_corr(xb_A, xb_C)
    fmt_corr_table("A (pure CPU) vs C (production rms_gemv_rope + cpu_attn)", o, ct, et)

    # B vs C (NPU fused vs production)
    o, ct, et = per_head_corr(xb_B, xb_C)
    fmt_corr_table("B (NPU fused) vs C (production)", o, ct, et)

    # ---------- Print sample values from each ----------
    print("\n" + "=" * 78)
    print("Sample xb values (kv=0, g=0, first 8 features)")
    print("=" * 78)
    print(f"  A (pure CPU): {xb_A[0, 0, :8].astype(np.float32)}")
    print(f"  B (NPU fused): {xb_B[0, 0, :8].astype(np.float32)}")
    print(f"  C (production): {xb_C[0, 0, :8].astype(np.float32)}")

    print("\nSample xb values (kv=0, g=0) magnitude statistics:")
    print(f"  A max(|xb|)={float(np.max(np.abs(xb_A.astype(np.float32)))):.4f}")
    print(f"  B max(|xb|)={float(np.max(np.abs(xb_B.astype(np.float32)))):.4f}")
    print(f"  C max(|xb|)={float(np.max(np.abs(xb_C.astype(np.float32)))):.4f}")

    fused_runner.shutdown()


if __name__ == "__main__":
    main()
