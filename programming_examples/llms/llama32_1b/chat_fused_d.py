"""Path D: chat with fused NPU attention as ELF (one ELF per pos).

Now that PR #1604 is applied, the standalone ELF path works. Path D
replaces (rms_gemv_rope NPU + decode_attention_cpu) with the fused
attn_decode_npu2 kernel compiled as ELF for each decode pos.

Architectural advantage over path C (xclbin + insts swap):
  - All decode kernels stay in ELF format → no xclbin↔ELF context switch
    (we measured 820 us/call penalty in path C; ELF↔ELF was 0 us).

Architectural cost vs path C:
  - Each pos is a separate XRTBackend instance → BOs can't be shared
    across pos. To avoid per-call B writes (12 MB × 16 layers = 192 MB),
    we PRE-WARM all (pos, layer) combinations at startup. This allocates
    N × 16 separate BO sets in NPU-mappable host memory.

Memory budget at startup (N decode tokens):
  - N × 16 × (12 MB B + 2 MB kc + 2 MB vc + small xrms/xb)
  - For N=20: ~5 GB (BOs are host_only; lives in system RAM)

Pre-warm cost: N × 16 × ~10 ms ≈ a few seconds.

Steady-state per-call (post-warm): just xrms write + kernel + xb read,
target ~730 us (matches standalone benchmark).

Usage:
    python3 chat_fused_d.py --prompt "What is 2+2?" --n-tokens 5
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ATTN_DECODE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "attention_decode")
)
sys.path.insert(0, _ATTN_DECODE_DIR)

from llama32_1b_inference import prepare_runtime, run_npu_prefill, _LM_N_PARTITIONS, _LM_N_PART
from llama32_1b_prefill import compile_all_kernels as compile_prefill_kernels
from llama32_1b_decode import compile_decode_kernels
from llama32_1b_weights import LlamaConfig, load_weights, generate_rope_lut
from llama32_1b_reference import rms_norm
from llama_kernel_builder.cache import KernelCache
from llama_kernel_builder.backend_presets import LM_GEMV_BACKEND, OGF_BACKEND
from attn_decode_npu2 import build_module as build_fused_attn_module

N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
EMB_DIM = N_HEADS * HEAD_DIM  # 2048
HIDDEN_DIM = 8192
GROUP_SIZE = N_HEADS // N_KV_HEADS
GEMV_COUNT = GROUP_SIZE + 2
TILE_K = 128
TILE_N = HEAD_DIM

# ELF-flavored backend kwargs for fused attn (matches standalone profile-decode-elf flags).
FUSED_ELF_BACKEND = dict(
    verbose=False,
    omit_while_true_loop=False,
    omit_pingpong=True,
    output_format="elf",
    instance_name="mha_bf16",
    target_device="npu2",
    stack_size=0xC00,
)


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
    print(f"  Compiling attn_decode_npu2.cc -> attn_decode_npu2.o")
    subprocess.run(cmd, check=True)


def compile_fused_elfs(cache, prompt_len, n_tokens, seq_len):
    """Compile N fused-attn ELFs (one per decode pos)."""
    print(f"\n[fused-pathD] Compiling {n_tokens} ELF kernels for pos in "
          f"[{prompt_len}, {prompt_len + n_tokens - 1}], seq_len={seq_len}")
    compile_attn_decode_o(seq_len)
    t_total = time.time()
    n_compiled = 0
    for i, pos in enumerate(range(prompt_len, prompt_len + n_tokens)):
        name = f"fused_attn_pos{pos}_seq{seq_len}"
        if name in cache.artifacts:
            print(f"  [{i + 1}/{n_tokens}] pos={pos}: cached")
            continue
        mod = build_fused_attn_module(
            EMB_DIM, HEAD_DIM, TILE_K, TILE_N, seq_len,
            bfloat16, bfloat16, bfloat16, pos,
            group_size=GROUP_SIZE, nkv=N_KV_HEADS,
        )
        t0 = time.time()
        cache.compile_and_cache(name, mod, FUSED_ELF_BACKEND)
        n_compiled += 1
        print(f"  [{i + 1}/{n_tokens}] pos={pos}: {time.time() - t0:.1f}s")
    cache._save_manifest()
    if n_compiled:
        print(f"[fused-pathD] Total compile: {time.time() - t_total:.1f}s")


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


def fused_call(fused_cache, layer_idx, pos, seq_len, xrms, B, kc, vc, xb_buf,
               kc_host=None, vc_host=None):
    """One fused-attn ELF invocation. KernelCache reuses BOs across calls
    when bo_key matches, so subsequent calls (same (pos, layer)) skip
    static input writes.

    If kc_host/vc_host are provided, after the call we read K_new/V_new at
    slot=pos back from the device bo_kc/bo_vc and store into kc_host[:,pos,:]
    / vc_host[:,pos,:]. This is required for multi-token decode because each
    (layer, pos) bo_key allocates its own bo_kc — K_new written by a prior
    pos's call lives only in that pos's BO, so we have to round-trip via the
    host array to make it visible to the next pos's first call.
    """
    name = f"fused_attn_pos{pos}_seq{seq_len}"
    bo_key = f"fused_L{layer_idx}_pos{pos}_seq{seq_len}"
    res = fused_cache.load_and_run(
        name, FUSED_ELF_BACKEND,
        xrms, B, kc, vc, xb_buf,
        output_indices=[4],
        # Only B (12 MB weights) is shared across all decode steps for this
        # (layer, pos). kc/vc must be re-uploaded each call so that K_new
        # written by previous pos calls (and persisted to host k_cache via
        # kc_host write-back below) is visible. Marking kc/vc static would
        # leave each (L, pos) BO frozen with the prefill K cache.
        static_input_indices={1},
        intermediate_indices={4},
        bo_key=bo_key,
    )
    if kc_host is not None:
        import pyxrt as xrt
        bos = fused_cache._cached_bos[bo_key]
        bos[2].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        bos[3].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        kc_view = np.frombuffer(bos[2].map(), dtype=bfloat16, count=kc.size).reshape(kc.shape)
        vc_view = np.frombuffer(bos[3].map(), dtype=bfloat16, count=vc.size).reshape(vc.shape)
        kc_host[:, pos, :] = kc_view[:, pos, :]
        vc_host[:, pos, :] = vc_view[:, pos, :]
    return np.asarray(res[4]).reshape(N_KV_HEADS, GROUP_SIZE, HEAD_DIM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="What is 2+2?")
    parser.add_argument("--n-tokens", type=int, default=5)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--no-prewarm", action="store_true",
                        help="Skip pre-warming all (pos, layer) BOs (faster startup, slower per-call)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Path D: chat with fused NPU attn (one ELF per pos, N XRTBackends)")
    print(f"  Prompt: {args.prompt!r}    N tokens: {args.n_tokens}    "
          f"prewarm: {not args.no_prewarm}")
    print("=" * 70)

    print("\n[1/7] Loading weights + tokenizer...")
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
    actual_prompt_len = len(raw_tokens)
    SEQ_PAD = 2048  # prefill kernel is fixed-size; pad input but track actual
    if len(raw_tokens) < SEQ_PAD:
        raw_tokens = raw_tokens + [tokenizer.eos_token_id] * (SEQ_PAD - len(raw_tokens))
    prefill_seq = len(raw_tokens)
    print(f"  actual_prompt_len={actual_prompt_len}, prefill seq={prefill_seq}")

    rope_lut_bf16 = generate_rope_lut(config=config,
                                      seq_len=prefill_seq + args.n_tokens + 1,
                                      dtype=bfloat16)

    prefill_cache = KernelCache(cache_dir="./prefill_kernel_cache", verbose=args.verbose)
    decode_cache = KernelCache(cache_dir="./decode_kernel_cache", verbose=args.verbose)
    fused_cache = KernelCache(cache_dir="./fused_pathD_cache", verbose=args.verbose)
    prefill_cache.load_manifest()
    decode_cache.load_manifest()
    fused_cache.load_manifest()

    if not prefill_cache.artifacts:
        print("\n[2a/7] Compiling prefill kernels (one-time)...")
        compile_prefill_kernels(prefill_cache, config, prefill_seq, cpu_attn=False)
    else:
        print("\n[2a/7] Reusing cached prefill kernels.")
    if not decode_cache.artifacts:
        print("[2b/7] Compiling decode kernels (one-time)...")
        compile_decode_kernels(decode_cache, config)
    else:
        print("[2b/7] Reusing cached decode kernels.")

    # Fused decode kernel attends only [0..current_pos]; compile for actual
    # prompt length to keep K cache range small and avoid attending to the
    # EOS-padded prefill K/V slots (would bias softmax toward special tokens).
    raw_seq = actual_prompt_len + args.n_tokens + 1
    fused_seq_len = ((raw_seq + 15) // 16) * 16
    print(f"\n[3/7] Compiling fused-attn ELFs (pos in [{actual_prompt_len}, "
          f"{actual_prompt_len + args.n_tokens - 1}], fused_seq_len={fused_seq_len})...")
    compile_fused_elfs(fused_cache, actual_prompt_len, args.n_tokens, fused_seq_len)

    print(f"\n[4/7] Preparing runtime (existing prefill+decode preload)...")
    prepare_runtime(prefill_cache, decode_cache, weights, config,
                    seq_len=prefill_seq, rope_lut_bf16=rope_lut_bf16)

    print(f"\n[5/7] NPU prefill...")
    t0 = time.time()
    prefill_token, k_cache, v_cache, _ = run_npu_prefill(
        raw_tokens, weights, config, prefill_cache, decode_cache,
        rope_lut_bf16, max_seq=prefill_seq,
        tokenizer=tokenizer, cpu_attn=False, profile=False, verify=False,
        quiet=True,
    )
    print(f"  Prefill: {time.time() - t0:.2f}s, first token={prefill_token}")

    # Slice K/V to fused_seq_len and zero the EOS-padded tail; the fused kernel
    # attends over fused_seq_len slots and slot t > current_pos must contribute zero.
    k_cache_sliced = np.zeros(
        (config.n_layers, N_KV_HEADS, fused_seq_len, HEAD_DIM), dtype=bfloat16,
    )
    v_cache_sliced = np.zeros_like(k_cache_sliced)
    k_cache_sliced[:, :, :actual_prompt_len, :] = k_cache[:, :, :actual_prompt_len, :]
    v_cache_sliced[:, :, :actual_prompt_len, :] = v_cache[:, :, :actual_prompt_len, :]
    k_cache, v_cache = k_cache_sliced, v_cache_sliced

    # Pre-pack B once per layer (same B used at every pos for that layer).
    print(f"\n[6/7] Pre-packing B per layer...")
    t0 = time.time()
    B_per_layer = [pack_weights_to_B(weights.layers[L]) for L in range(config.n_layers)]
    print(f"  done: {time.time() - t0:.2f}s ({config.n_layers} × {B_per_layer[0].nbytes // (1024*1024)} MB)")

    # Pre-warm: fire each (pos, layer) once with dummy xrms so KernelCache
    # allocates+writes static B/kc/vc BOs, leaving steady-state per-call cost
    # to xrms write + kernel + xb read only.
    if not args.no_prewarm:
        print(f"\n[7a/7] Pre-warming all (pos × layer) BOs...")
        t0 = time.time()
        dummy_xrms = pack_xrms(
            np.zeros(EMB_DIM, dtype=bfloat16),
            np.zeros(EMB_DIM, dtype=bfloat16),
        )
        xb_dummy = np.zeros((N_KV_HEADS, GROUP_SIZE, HEAD_DIM), dtype=bfloat16)
        n_pos = args.n_tokens
        for pos in range(actual_prompt_len, actual_prompt_len + n_pos):
            for layer_idx in range(config.n_layers):
                fused_call(
                    fused_cache, layer_idx, pos, fused_seq_len,
                    dummy_xrms,
                    B_per_layer[layer_idx],
                    k_cache[layer_idx],
                    v_cache[layer_idx],
                    xb_dummy,
                )
        print(f"  done: {time.time() - t0:.2f}s "
              f"({n_pos * config.n_layers} (pos, layer) BO sets allocated)")

    print(f"\n[7b/7] Decoding (fused NPU attn, path D) for up to {args.n_tokens} tokens...")
    generated = [prefill_token]
    current_pos = actual_prompt_len  # decode at unpadded prompt length
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)
    decode_times = []
    fused_call_times = []
    ogf_call_times = []
    pack_xrms_times = []
    lm_head_times = []
    final_rms_times = []

    xb_buf = np.zeros((N_KV_HEADS, GROUP_SIZE, HEAD_DIM), dtype=bfloat16)

    for token_idx in range(args.n_tokens):
        t_token = time.perf_counter()
        x = x_decode.copy()
        for layer_idx in range(config.n_layers):
            lw = weights.layers[layer_idx]

            t_pk = time.perf_counter()
            xrms = pack_xrms(
                x.flatten().astype(bfloat16),
                lw.attn_norm.reshape(EMB_DIM).astype(bfloat16),
            )
            pack_xrms_times.append(time.perf_counter() - t_pk)
            t_fused = time.perf_counter()
            xb = fused_call(
                fused_cache, layer_idx, current_pos, fused_seq_len,
                xrms, B_per_layer[layer_idx],
                k_cache[layer_idx], v_cache[layer_idx],
                xb_buf,
                kc_host=k_cache[layer_idx], vc_host=v_cache[layer_idx],
            )
            fused_call_times.append(time.perf_counter() - t_fused)

            attn_out = xb.reshape(EMB_DIM).astype(bfloat16)

            t_ogf = time.perf_counter()
            results = decode_cache.load_and_run(
                "o_gemv_ffn", OGF_BACKEND,
                lw._wo_t, attn_out,
                np.zeros(EMB_DIM, dtype=bfloat16),
                x.flatten().astype(bfloat16),
                np.zeros(EMB_DIM, dtype=bfloat16),
                lw.ffn_norm.reshape(EMB_DIM).astype(bfloat16),
                np.zeros(EMB_DIM, dtype=bfloat16),
                lw._wgate_t, np.zeros(HIDDEN_DIM, dtype=bfloat16),
                lw._wup_t, np.zeros(HIDDEN_DIM, dtype=bfloat16),
                np.zeros(HIDDEN_DIM, dtype=bfloat16),
                lw._wdown_t, np.zeros(EMB_DIM, dtype=bfloat16),
                np.zeros(EMB_DIM, dtype=bfloat16),
                output_indices=[14],
                static_input_indices={0, 5, 7, 9, 12},
                intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
                bo_key=f"o_gemv_ffn_L{layer_idx}",
            )
            x = np.asarray(results[14]).astype(bfloat16)
            ogf_call_times.append(time.perf_counter() - t_ogf)

        t_fr = time.perf_counter()
        x_normed = rms_norm(
            x.astype(np.float32).reshape(1, EMB_DIM),
            weights.final_norm.astype(np.float32),
        )
        x_lm = x_normed.flatten().astype(bfloat16)
        final_rms_times.append(time.perf_counter() - t_fr)
        t_lm = time.perf_counter()
        lm_inputs = [x_lm]
        lm_output_indices = []
        for p in range(_LM_N_PARTITIONS):
            lm_inputs.append(weights._lm_weight_parts_gemv[p])
            lm_inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))
            lm_output_indices.append(2 + 2 * p)
        lm_results = decode_cache.load_and_run(
            "lm_head_gemv", LM_GEMV_BACKEND,
            *lm_inputs,
            output_indices=lm_output_indices,
            static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
            intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
        )
        lm_head_times.append(time.perf_counter() - t_lm)
        vocab_size = weights.lm_head.shape[0]
        logits = np.zeros((1, vocab_size), dtype=np.float32)
        for p in range(_LM_N_PARTITIONS):
            n_start = p * _LM_N_PART
            n_end = min(n_start + _LM_N_PART, vocab_size)
            logits[0, n_start:n_end] = lm_results[2 + 2 * p][: n_end - n_start].astype(
                np.float32
            )
        next_token = int(np.argmax(logits[0]))

        decode_times.append(time.perf_counter() - t_token)
        print(f"  token {token_idx + 1}: id={next_token}, "
              f"time={decode_times[-1]*1000:.0f}ms")

        generated.append(next_token)
        current_pos += 1
        x_decode = weights.embed_table[next_token].astype(bfloat16)

        if next_token in {tokenizer.eos_token_id, 128009, 128001}:
            print(f"  (EOS at token {token_idx + 1})")
            break

    answer = tokenizer.decode(generated, skip_special_tokens=True)
    print()
    print("=" * 70)
    print(f"Q: {args.prompt}")
    print(f"A: {answer}")
    print("=" * 70)
    if decode_times:
        avg_token_ms = np.mean(decode_times) * 1000
        n_layers = config.n_layers
        # Per-call averages
        f_ms = np.mean(fused_call_times) * 1000
        o_ms = np.mean(ogf_call_times) * 1000
        x_ms = np.mean(pack_xrms_times) * 1000
        l_ms = np.mean(lm_head_times) * 1000
        r_ms = np.mean(final_rms_times) * 1000
        # Per-token (× n_layers for the per-layer ones)
        f_tot = f_ms * n_layers
        o_tot = o_ms * n_layers
        x_tot = x_ms * n_layers
        accounted = f_tot + o_tot + x_tot + l_ms + r_ms
        unaccounted = avg_token_ms - accounted
        print(f"\nPer-token latency: {avg_token_ms:.1f} ms ({1000/avg_token_ms:.2f} tok/s)")
        print(f"  fused_attn (NPU): {f_ms:>5.2f} ms/call * {n_layers} = {f_tot:>5.1f} ms ({100*f_tot/avg_token_ms:>4.1f}%)")
        print(f"  o_gemv_ffn (NPU): {o_ms:>5.2f} ms/call * {n_layers} = {o_tot:>5.1f} ms ({100*o_tot/avg_token_ms:>4.1f}%)")
        print(f"  pack_xrms (host): {x_ms:>5.2f} ms/call * {n_layers} = {x_tot:>5.1f} ms ({100*x_tot/avg_token_ms:>4.1f}%)")
        print(f"  lm_head_gemv (NPU+host stitch): {l_ms:>5.1f} ms ({100*l_ms/avg_token_ms:>4.1f}%)")
        print(f"  final RMS-norm (CPU, f32): {r_ms:>5.2f} ms ({100*r_ms/avg_token_ms:>4.1f}%)")
        print(f"  unaccounted (other host): {unaccounted:>5.1f} ms ({100*unaccounted/avg_token_ms:>4.1f}%)")
    print()


if __name__ == "__main__":
    main()
