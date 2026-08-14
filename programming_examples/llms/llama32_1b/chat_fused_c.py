"""Path-C integration: chat with fused NPU attention. ONE xclbin loaded
once, swap .insts.bin per decode step.

Replaces (rms_gemv_rope NPU + decode_attention_cpu) with the fused
attn_decode_npu2 kernel. Other components (NPU prefill, o_gemv_ffn ELF,
LM head ELF) are unchanged.

Architecture (per the diagnostic work in the prior session):
  - The fused kernel's xclbin/PDI is pos-invariant (verified by diffing
    artifacts from two pos values: only ~64 bytes of insts.bin BD config
    change; the xclbin diff is metadata-only).
  - We compile N kernels at startup (one per decode pos), keep ONE xclbin,
    and swap the per-pos insts.bin into bo_instr per call.
  - Per-layer B/kc/vc BOs are allocated and written ONCE at startup, so
    the only host->device traffic per call is 13 KB insts + 8 KB xrms.

Known limitations:
  - Output is gibberish until issue Xilinx/mlir-air#1600 is resolved
    (one Q-head correlates at ~0.65 against the CPU reference, error
    compounds across the autoregressive loop).
  - Per-call cost is bounded at ~1.55 ms vs the standalone 0.73 ms
    because every call alternates between fused-attn (xclbin) and
    o_gemv_ffn (ELF), incurring a +820us context-switch penalty.
    Eliminating that requires moving fused-attn to ELF format
    (option D), currently blocked by a Python ELF-loader bug for
    this specific kernel pattern (ERT_CMD_STATE_TIMEOUT — file separately).

Usage:
    python3 chat_fused_c.py --prompt "What is 2+2?" --n-tokens 5
"""

import argparse
import os
import shutil
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
from air.backend.xrt import XRTBackend, XRTCompileArtifact

# Llama-3.2-1B shape constants
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
EMB_DIM = N_HEADS * HEAD_DIM  # 2048
HIDDEN_DIM = 8192
GROUP_SIZE = N_HEADS // N_KV_HEADS  # 4
GEMV_COUNT = GROUP_SIZE + 2  # 6
TILE_K = 128
TILE_N = HEAD_DIM

FUSED_ATTN_BACKEND = dict(
    verbose=False,
    omit_while_true_loop=False,
    omit_pingpong=True,
    output_format="xclbin",
    instance_name="mha_bf16",
    target_device="npu2",
    stack_size=0xC00,
)


# ---------- One-time external .o build ----------

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


# ---------- Compile N kernels, keep one xclbin + N insts.bin ----------

def compile_pos_artifacts(prompt_len, n_tokens, seq_len, cache_dir="./fused_pathC_cache"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fused-pathC] Compiling {n_tokens} kernels for pos in "
          f"[{prompt_len}, {prompt_len + n_tokens - 1}], seq_len={seq_len}")
    compile_attn_decode_o(seq_len)

    kc = KernelCache(cache_dir=str(cache_dir / "scratch"), verbose=False)

    xclbin_path = None
    insts_per_pos = {}

    for i, pos in enumerate(range(prompt_len, prompt_len + n_tokens)):
        name = f"fused_attn_pos{pos}_seq{seq_len}"
        cached_insts = cache_dir / f"{name}.insts.bin"
        cached_xclbin = cache_dir / "fused_attn.xclbin"

        if cached_insts.exists() and cached_xclbin.exists():
            insts_per_pos[pos] = str(cached_insts)
            if xclbin_path is None:
                xclbin_path = str(cached_xclbin)
            print(f"  [{i + 1}/{n_tokens}] pos={pos}: cached")
            continue

        mod = build_fused_attn_module(
            EMB_DIM, HEAD_DIM, TILE_K, TILE_N, seq_len,
            bfloat16, bfloat16, bfloat16, pos,
            group_size=GROUP_SIZE, nkv=N_KV_HEADS,
        )
        t0 = time.time()
        kc.compile_and_cache(name, mod, FUSED_ATTN_BACKEND)
        art = kc.artifacts[name]

        if xclbin_path is None:
            shutil.copy2(art.output_binary, cached_xclbin)
            xclbin_path = str(cached_xclbin)

        shutil.copy2(art.insts, cached_insts)
        insts_per_pos[pos] = str(cached_insts)
        print(f"  [{i + 1}/{n_tokens}] pos={pos}: {time.time() - t0:.1f}s")

    return xclbin_path, insts_per_pos


# ---------- Path-C runner: ONE xclbin, swap insts per call ----------

class FusedAttnSwapRunner:
    """One xclbin loaded into one XRTBackend; one bo_instr we overwrite per
    decode step; persistent per-layer B/kc/vc BOs allocated and written
    ONCE at startup; shared xrms/xb BOs.
    """

    KERNEL_NAME_HINT = "MLIR_AIE"

    def __init__(self, xclbin_path, insts_per_pos, n_layers):
        import pyxrt as xrt

        self.xrt = xrt
        self.n_layers = n_layers

        sample_pos, sample_insts = next(iter(insts_per_pos.items()))
        artifact = XRTCompileArtifact(xclbin_path, self.KERNEL_NAME_HINT, sample_insts)

        self.backend = XRTBackend(**FUSED_ATTN_BACKEND)
        self._invoker_unused = self.backend.load(artifact)

        self.insts_per_pos = {}
        for pos, path in insts_per_pos.items():
            with open(path, "rb") as f:
                data = f.read()
            self.insts_per_pos[pos] = np.frombuffer(data, dtype=np.uint32)

        sizes = {len(v) for v in self.insts_per_pos.values()}
        assert len(sizes) == 1, f"insts.bin sizes differ: {sizes}"
        self.insts_n_words = sizes.pop()

        self.bo_xrms = xrt.bo(
            self.backend.device,
            TILE_K * TILE_N * 2,
            xrt.bo.host_only,
            self.backend.kernel.group_id(0 + 3),
        )
        self.bo_xb = xrt.bo(
            self.backend.device,
            N_KV_HEADS * GROUP_SIZE * HEAD_DIM * 2,
            xrt.bo.host_only,
            self.backend.kernel.group_id(4 + 3),
        )

        self.bo_B = {}
        self.bo_kc = {}
        self.bo_vc = {}

    def add_layer(self, layer_idx, B_bf16, k_cache_init_bf16, v_cache_init_bf16):
        import pyxrt as xrt

        bo_B = xrt.bo(self.backend.device, B_bf16.size * 2, xrt.bo.host_only,
                      self.backend.kernel.group_id(1 + 3))
        bo_kc = xrt.bo(self.backend.device, k_cache_init_bf16.size * 2,
                       xrt.bo.host_only, self.backend.kernel.group_id(2 + 3))
        bo_vc = xrt.bo(self.backend.device, v_cache_init_bf16.size * 2,
                       xrt.bo.host_only, self.backend.kernel.group_id(3 + 3))

        bo_B.write(B_bf16.view(np.int16), 0)
        bo_B.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_kc.write(k_cache_init_bf16.view(np.int16), 0)
        bo_kc.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_vc.write(v_cache_init_bf16.view(np.int16), 0)
        bo_vc.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        self.bo_B[layer_idx] = bo_B
        self.bo_kc[layer_idx] = bo_kc
        self.bo_vc[layer_idx] = bo_vc

    def run(self, layer_idx, pos, xrms_bf16):
        import pyxrt as xrt

        insts_arr = self.insts_per_pos[pos]
        self.backend.bo_instr.write(insts_arr, 0)
        self.backend.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        self.bo_xrms.write(xrms_bf16.view(np.int16), 0)
        self.bo_xrms.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        h = self.backend.kernel(
            3, self.backend.bo_instr, self.insts_n_words,
            self.bo_xrms,
            self.bo_B[layer_idx],
            self.bo_kc[layer_idx],
            self.bo_vc[layer_idx],
            self.bo_xb,
        )
        h.wait()

        self.bo_xb.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        xb_view = np.frombuffer(self.bo_xb.map(), dtype=bfloat16,
                                count=N_KV_HEADS * GROUP_SIZE * HEAD_DIM)
        return xb_view.reshape(N_KV_HEADS, GROUP_SIZE, HEAD_DIM).copy()

    def shutdown(self):
        self.backend.unload()


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


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="What is 2+2?")
    parser.add_argument("--n-tokens", type=int, default=5)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Path C: chat with fused NPU attn (one xclbin loaded once, swap insts)")
    print(f"  Prompt: {args.prompt!r}    N tokens: {args.n_tokens}")
    print("=" * 70)

    print("\n[1/6] Loading weights + tokenizer...")
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
    SEQ_PAD = 2048  # prefill kernel is compiled for fixed seq_len=2048; pad the input
    if len(raw_tokens) < SEQ_PAD:
        raw_tokens = raw_tokens + [tokenizer.eos_token_id] * (SEQ_PAD - len(raw_tokens))
    prefill_seq = len(raw_tokens)
    print(f"  actual_prompt_len={actual_prompt_len}, prefill seq={prefill_seq}")

    rope_lut_bf16 = generate_rope_lut(
        config=config,
        seq_len=prefill_seq + args.n_tokens + 1,
        dtype=bfloat16,
    )

    prefill_cache = KernelCache(cache_dir="./prefill_kernel_cache", verbose=args.verbose)
    decode_cache = KernelCache(cache_dir="./decode_kernel_cache", verbose=args.verbose)
    prefill_cache.load_manifest()
    decode_cache.load_manifest()

    if not prefill_cache.artifacts:
        print("\n[2a/6] Compiling prefill kernels (one-time)...")
        compile_prefill_kernels(prefill_cache, config, prefill_seq, cpu_attn=False)
    else:
        print("\n[2a/6] Reusing cached prefill kernels.")
    if not decode_cache.artifacts:
        print("[2b/6] Compiling decode kernels (one-time)...")
        compile_decode_kernels(decode_cache, config)
    else:
        print("[2b/6] Reusing cached decode kernels.")

    # Fused decode kernel attends only over [0..current_pos]; its compiled
    # seq_len needs to cover actual_prompt_len + n_tokens, NOT the full padded
    # 2048 (otherwise EOS-padded prefill slots are attended to, producing
    # gibberish biased toward special tokens).
    raw_seq = actual_prompt_len + args.n_tokens + 1
    fused_seq_len = ((raw_seq + 15) // 16) * 16
    print(f"\n[3/6] Compiling fused-attn artifacts (one xclbin + {args.n_tokens} insts.bin), "
          f"fused_seq_len={fused_seq_len}, pos in [{actual_prompt_len}, {actual_prompt_len + args.n_tokens - 1}]...")
    xclbin_path, insts_per_pos = compile_pos_artifacts(
        actual_prompt_len, args.n_tokens, fused_seq_len, cache_dir="./fused_pathC_cache"
    )

    print(f"\n[4/6] Preparing runtime (existing prefill+decode preload)...")
    prepare_runtime(
        prefill_cache, decode_cache, weights, config,
        seq_len=prefill_seq, rope_lut_bf16=rope_lut_bf16,
    )

    print(f"\n[5/6] NPU prefill...")
    t0 = time.time()
    # Use prefill_seq for max_seq (prefill writes K cache for the full padded
    # seq); we slice to fused_seq_len before passing K cache to the fused runner.
    prefill_token, k_cache, v_cache, _ = run_npu_prefill(
        raw_tokens, weights, config, prefill_cache, decode_cache,
        rope_lut_bf16, max_seq=prefill_seq,
        tokenizer=tokenizer, cpu_attn=False, profile=False, verify=False,
        quiet=True,
    )
    print(f"  Prefill: {time.time() - t0:.2f}s, first token={prefill_token}")

    print(f"\n[6/6] Initializing FusedAttnSwapRunner...")
    t0 = time.time()
    fused_runner = FusedAttnSwapRunner(xclbin_path, insts_per_pos, config.n_layers)
    print(f"  xclbin loaded + bo_instr allocated: {time.time() - t0:.2f}s")

    print(f"  Pre-packing B per layer + writing per-layer BOs (one-time)...")
    t0 = time.time()
    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]
        B = pack_weights_to_B(lw)
        # Slice K/V cache to fused_seq_len, AND zero the slots beyond
        # actual_prompt_len. Prefill wrote K/V for the full padded seq
        # (positions actual_prompt_len..prefill_seq-1 are EOS-padding garbage);
        # the fused kernel attends over all fused_seq_len slots, so let those
        # tail slots contribute zero (overwritten as decode advances current_pos).
        kc_slice = np.ascontiguousarray(k_cache[layer_idx, :, :fused_seq_len, :])
        vc_slice = np.ascontiguousarray(v_cache[layer_idx, :, :fused_seq_len, :])
        kc_slice[:, actual_prompt_len:, :] = 0
        vc_slice[:, actual_prompt_len:, :] = 0
        fused_runner.add_layer(layer_idx, B, kc_slice, vc_slice)
    print(f"  All {config.n_layers} layers preloaded: {time.time() - t0:.2f}s")
    print(f"  (B BOs: {config.n_layers} * {B.nbytes // (1024*1024)} MB = "
          f"{config.n_layers * B.nbytes // (1024*1024)} MB)")

    print(f"\nDecoding (fused NPU attn, path C) for up to {args.n_tokens} tokens...")
    generated = [prefill_token]
    current_pos = actual_prompt_len  # decode at the unpadded prompt length, not 2048
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)
    decode_times = []
    fused_call_times = []

    for token_idx in range(args.n_tokens):
        t_token = time.perf_counter()
        x = x_decode.copy()
        for layer_idx in range(config.n_layers):
            lw = weights.layers[layer_idx]

            xrms = pack_xrms(
                x.flatten().astype(bfloat16),
                lw.attn_norm.reshape(EMB_DIM).astype(bfloat16),
            )
            t_fused = time.perf_counter()
            xb = fused_runner.run(layer_idx, current_pos, xrms)
            fused_call_times.append(time.perf_counter() - t_fused)

            attn_out = xb.reshape(EMB_DIM).astype(bfloat16)

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

        x_normed = rms_norm(
            x.astype(np.float32).reshape(1, EMB_DIM),
            weights.final_norm.astype(np.float32),
        )
        x_lm = x_normed.flatten().astype(bfloat16)
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

    fused_runner.shutdown()

    answer = tokenizer.decode(generated, skip_special_tokens=True)
    print()
    print("=" * 70)
    print(f"Q: {args.prompt}")
    print(f"A: {answer}")
    print("=" * 70)
    if decode_times:
        avg_token_ms = np.mean(decode_times) * 1000
        avg_fused_ms = np.mean(fused_call_times) * 1000
        n_layers = config.n_layers
        print(f"Per-token latency:   {avg_token_ms:.1f} ms  ({1000/avg_token_ms:.2f} tok/s)")
        print(f"Per-fused-attn-call: {avg_fused_ms:.2f} ms  "
              f"(sum across {n_layers} layers = "
              f"{avg_fused_ms * n_layers:.1f} ms/token)")
    print()


if __name__ == "__main__":
    main()
