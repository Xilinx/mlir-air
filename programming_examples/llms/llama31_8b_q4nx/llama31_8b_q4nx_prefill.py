# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Llama-3.1-8B Q4NX prefill in mlir-air.
#
# Runs the prefill MECHANISM on the AMD NPU2: Q4NX weights, host dequant
# Q4NX->bf16 at load, then op-by-op bf16 GEMM / RMSNorm / RoPE / SwiGLU /
# causal-GQA-flash-attention all ON THE NPU with RESIDENT weight BOs, and the
# final LM head as an on-device GEMV. The whole transformer block runs through
# the two llama32_3b multi-launch stitchers (rms_gemms_rope + o_ffn) plus the
# seq-first temporal-causal flash attention (head_dim=128), driven by
# KernelCache.load_and_run
# with per-layer resident BOs (weights written once).
#
# This is the 3B analog of llama32_1b_q4nx_prefill.py: same structure, 3B config.
# It captures per-layer roped-K + raw-V so the fused decode can be seeded with a
# warm KV cache (see q4nx_decode_3b.FusedDecode3B.seed_kv) -- prompt tokens then
# cost one batched prefill instead of one full decode dispatch each.
#
# Gate: first prompt token argmax 12366 (" Paris") for "The capital city of France
# is called" (see PROMPT below for why not the siblings' bare phrasing).
#
# Weight source (env-overridable):
#   Q4NX_MODEL_SOURCE : the model.q4nx bundle -- HF repo id or a local dir/file.
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_PROG = str(_HERE.parent.parent)  # programming_examples
_LLMS = str(_HERE.parent)  # llms
# The stitcher/FA builders are dimension-driven (compile_all_kernels takes the
# config), and 8B shares 3B's head_dim=128 head-first FA path, so they are
# reused as-is rather than re-authored for these dims.
_LLAMA3B = str(_HERE.parent / "llama32_3b")  # fused-stitcher prefill driver
_LLAMA1B = str(_HERE.parent / "llama32_1b")  # shared stitcher internals
for _p in (_PROG, _LLMS, _LLAMA3B, _LLAMA1B, str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.infra.cache import KernelCache  # noqa: E402

# Default weight source: the self-contained model.q4nx bundle on the Hub. May be
# overridden with --model / Q4NX_MODEL_SOURCE (an HF repo id, or a local dir/file).
MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Llama-3.1-8B-NPU2")

VOCAB = 128256
# The bare "The capital of France is" that the 1B/3B siblings gate on does NOT
# work for this model: Llama-3.1-8B-Instruct ranks " a" (264, logit 16.000)
# fractionally above " Paris" (12366, 15.938) there, and the HF bf16 reference
# agrees -- so that gate would fail on a perfectly correct build. "...is called"
# makes it decisive: " Paris" leads the runner-up by 3.4 logits.
PROMPT = [
    128000,
    791,
    6864,
    3363,
    315,
    9822,
    374,
    2663,
]  # "The capital city of France is called"
EXPECT_FIRST = 12366  # " Paris"

# On-device LM head GEMV: 8 partitions of M=16384 (8*16384=131072 >= VOCAB), K=emb_dim.
_LM_N_PART = 16384
_LM_N_PARTITIONS = 8
# K=4096 breaks both default GEMV budgets, exactly as it does for the Qwen3-8B
# sibling: the L2 A stage needs herd_m*tile_m*(K+1)*2 <= 512 KiB (so
# herd_m*tile_m <= 63, vs the default 64 -- it overflows by 128 B), and m_input=4
# puts the ping-pong L1 A tiles at 2*32 KiB = all of L1. herd_m must stay 8 (4
# builds but hangs the dispatch), so tile_m halves instead, which means mv.o has
# to be rebuilt at DIM_M_OUTPUT=tile_m.
_LM_HERD_M = 8
_LM_TILE_M = 4
_LM_M_INPUT = 2


def _bf(a):
    return np.asarray(a, bfloat16)


class LlamaQ4nxPrefill:
    """AIR realization of the Q4NX causal-LM prefill interface (Llama-3.1-8B)."""

    def __init__(self, seq_len=2048, n_layers=32, cache_dir=None, verbose=False):
        self.seq = seq_len
        self.n_layers = n_layers
        self.MAX_L = seq_len
        self.current_context_length = 0
        from llama31_8b_q4nx_weights import llama31_8b_config, generate_rope_lut
        from llama32_3b_prefill import compile_all_kernels, use_temporal_fa
        from shared.infra.backend_presets import LM_GEMV_BACKEND

        self.config = llama31_8b_config()
        self.config.n_layers = n_layers
        # seq_len-specific cache: the stitcher/flash-attn ELFs are compiled for this padded
        # context length, but the KernelCache is keyed by kernel NAME only. A shared dir would
        # let a shorter-context build (e.g. 256) be silently reused for a 2048 request -> the
        # prefill then returns all-zero logits for N beyond the built length. Isolate by seq_len.
        # The FA variant goes in the name for the same reason: the two wrappers disagree on
        # operand layout, so reusing one's `flash_attn` ELF under the other permutes the data.
        _fa = (
            "tfa"
            if use_temporal_fa(
                seq_len,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.head_dim,
            )
            else "hfa"
        )
        cache_dir = cache_dir or str(_HERE / f"_q4nx_cache_seq{seq_len}_{_fa}")
        self.cache = KernelCache(cache_dir=cache_dir, verbose=verbose)

        self.D = self.config.emb_dim
        self.KV_DIM = self.config.n_kv_heads * self.config.head_dim
        self._rope_lut = _bf(
            generate_rope_lut(config=self.config, seq_len=seq_len)
        )  # half-split, theta=5e5
        self._lm_backend = dict(LM_GEMV_BACKEND)

        # Compile (or reuse cached) the ELFs: the two llama32_3b stitchers
        # (rms_gemms_rope + o_ffn) plus flash_attn, and the 8-partition lm_head GEMV.
        # Q4NX_FORCE_COMPILE=1 forces recompile; otherwise a valid manifest is reused.
        force = os.environ.get("Q4NX_FORCE_COMPILE") == "1"
        if not force:
            self.cache.load_manifest()
        cached = set() if force else set(self.cache.artifacts)
        if not {"rms_gemms_rope", "o_ffn", "flash_attn"}.issubset(cached):
            compile_all_kernels(self.cache, self.config, seq_len, cpu_attn=False)
        else:
            print(
                "[q4nx_prefill] using cached stitcher ELFs (skip compile)", flush=True
            )
        # LM head on the NPU -- an 8-partition GEMV.
        if "lm_head_gemv" not in cached:
            from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module
            from shared.infra.external_kernels import compile_mv

            compile_mv(tile_m=_LM_TILE_M)  # mv.o, the GEMV micro-kernel linked below
            self.cache.compile_and_cache(
                "lm_head_gemv",
                build_lm_head_gemv_module(
                    self.D,
                    n_partitions=_LM_N_PARTITIONS,
                    n_part=_LM_N_PART,
                    herd_m=_LM_HERD_M,
                    tile_m=_LM_TILE_M,
                    m_input=_LM_M_INPUT,
                ),
                {"verbose": self.cache.verbose, **self._lm_backend},
            )
            self.cache._save_manifest()

        # Per-layer KV cache (roped K + raw V), [MAX_L, n_kv_heads*head_dim].
        self.kv_k = [
            np.zeros((self.MAX_L, self.KV_DIM), bfloat16) for _ in range(n_layers)
        ]
        self.kv_v = [
            np.zeros((self.MAX_L, self.KV_DIM), bfloat16) for _ in range(n_layers)
        ]
        self._weights = None
        self._embed = None
        self._final_norm = None
        self._lm_head = None
        self._dev_t = 0.0  # accumulated NPU dispatch time (s)
        self._op_t = {}  # per-op NPU dispatch time (s)

    def _dev(self, fn, *a, tag="?"):
        import time

        t = time.time()
        r = fn(*a)
        dt = time.time() - t
        self._dev_t += dt
        self._op_t[tag] = self._op_t.get(tag, 0.0) + dt
        return r

    # ---- causal_lm interface ----
    def load_weights(self, model=None):
        """Load Q4NX transformer weights + embed/lm_head/final-norm from the
        self-contained `model.q4nx` bundle (HF repo `Q4NX_MODEL_SOURCE`, or a
        local dir/file), dequant Q4NX->bf16 on the host at load, then pre-load
        them into per-layer resident BOs (written once; skipped at prefill)."""
        model = model or os.environ.get("Q4NX_MODEL_SOURCE", MODEL_DEFAULT)
        from llama31_8b_q4nx_weights import load_q4nx_weights
        from llama32_3b_prefill import preload_prefill_weights

        print(f"[q4nx_prefill] loading weights from model.q4nx ({model})", flush=True)
        w = load_q4nx_weights(model, config=self.config)
        self._weights = w
        self._embed = w.embed_table
        self._final_norm = np.asarray(w.final_norm, np.float32)
        self._lm_head = w.lm_head
        preload_prefill_weights(w, self.config, self.cache, self.seq, self._rope_lut)
        self._preload_lm_head_gemv()

    def _preload_lm_head_gemv(self):
        """Build the 8 padded bf16 lm_head partitions [16384, D] and write them
        into resident BOs once (static; skipped thereafter)."""
        self._lm_parts = []
        for p in range(_LM_N_PARTITIONS):
            n0 = p * _LM_N_PART
            n1 = min(n0 + _LM_N_PART, VOCAB)
            w = np.zeros((_LM_N_PART, self.D), bfloat16)
            if n1 > n0:
                w[: n1 - n0] = _bf(self._lm_head[n0:n1])
            self._lm_parts.append(w)
        self._lm_head_npu(np.zeros(self.D, bfloat16))  # warm/allocate resident BOs

    def _lm_head_npu(self, hidden_bf16):
        """On-device final logits from a single bf16 hidden row [D] -> [VOCAB]."""
        lm_inputs = [np.ascontiguousarray(hidden_bf16, bfloat16)]
        for p in range(_LM_N_PARTITIONS):
            lm_inputs.append(self._lm_parts[p])
            lm_inputs.append(np.zeros(_LM_N_PART, bfloat16))
        res = self.cache.load_and_run(
            "lm_head_gemv",
            self._lm_backend,
            *lm_inputs,
            output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
            static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
            intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
        )
        return np.concatenate(res, axis=0)[:VOCAB]

    def _run_layer(self, x, k, N):
        """One transformer block fully on-device via the llama32_3b stitchers
        (rms+qkv+rope+attn in rms_gemms_rope+flash_attn, o+residual+rms+gate/up/
        SiLU+down in o_ffn). Captures roped-K + raw-V into the KV cache."""
        from llama32_3b_prefill import run_transformer_block

        out, inter = self._dev(
            run_transformer_block,
            x,
            self._weights.layers[k],
            self._rope_lut,
            self.config,
            self.cache,
            k,
            False,
            False,
            tag="layer",
        )
        self.kv_k[k][:N] = np.asarray(inter["k_roped"], bfloat16)[:N]
        self.kv_v[k][:N] = np.asarray(inter["v"], bfloat16)[:N]
        return out

    def prefill(self, ids, payload=None):
        assert self._weights is not None, "call load_weights() first"
        N = len(ids)
        assert N <= self.seq, (N, self.seq)
        base = self.current_context_length
        x = np.zeros((self.seq, self.D), bfloat16)
        x[:N] = _bf(np.stack([self._embed[t] for t in ids]))
        for k in range(self.n_layers):
            x = self._run_layer(x, k, N)
        self.current_context_length = base + N
        # Final RMSNorm on the single prediction row (host, <1ms), then NPU LM head.
        xf = np.asarray(x[N - 1], np.float32)
        xn = xf / np.sqrt((xf * xf).mean() + 1e-5) * self._final_norm
        return self._dev(self._lm_head_npu, _bf(xn), tag="lm_head")

    # ---- KV cache (causal_lm) ----
    def get_k_cache(self, layer_idx, idx):
        """Roped-K vector at position idx of a layer."""
        return self.kv_k[layer_idx][idx]

    def get_v_cache(self, layer_idx, idx):
        """Raw-V vector at position idx of a layer."""
        return self.kv_v[layer_idx][idx]

    def kv_view(self, layer_idx):
        """(roped_K, raw_V) for the filled context [0:ctx] of a layer -> decode handoff."""
        c = self.current_context_length
        return self.kv_k[layer_idx][:c], self.kv_v[layer_idx][:c]

    def clear_context(self):
        self.current_context_length = 0
        for k in range(self.n_layers):
            self.kv_k[k][:] = 0
            self.kv_v[k][:] = 0

    def get_current_context_length(self):
        return self.current_context_length

    def set_context_length(self, L):
        self.current_context_length = L


def _main():
    ap = argparse.ArgumentParser(description="Llama-3.1-8B Q4NX prefill on NPU2")
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="build/cache the prefill ELFs and exit (no weights, no NPU dispatch)",
    )
    ap.add_argument(
        "--n-layers", type=int, default=int(os.environ.get("NLAYERS", "32"))
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=int(os.environ.get("Q4NX_SEQ_LEN", "2048")),
        help="padded prefill length",
    )
    ap.add_argument("--cache-dir", default=os.environ.get("Q4NX_CACHE_DIR") or None)
    ap.add_argument(
        "--bench-l",
        type=int,
        default=int(os.environ.get("Q4NX_BENCH_L", "0")),
        help="warm TTFT benchmark at this context length",
    )
    ap.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help="weight source: HF repo id (model.q4nx) or a local dir/file "
        f"(default: {MODEL_DEFAULT})",
    )
    args = ap.parse_args()

    print(
        f"[q4nx_prefill] constructing seq_len={args.seq_len} (compiling engines)...",
        flush=True,
    )
    model = LlamaQ4nxPrefill(
        seq_len=args.seq_len, n_layers=args.n_layers, cache_dir=args.cache_dir
    )
    # --compile-only: build/cache the ELFs and exit (no weights, no NPU dispatch).
    # CI-runnable without the external Q4NX weight data.
    if args.compile_only:
        print("Compilation passed.", flush=True)
        return 0
    print("[q4nx_prefill] loading Q4NX weights (host dequant)...", flush=True)
    model.load_weights(model=args.model)
    print(f"[q4nx_prefill] prefill prompt N={len(PROMPT)} ...", flush=True)
    logits = model.prefill(PROMPT)
    top = int(np.asarray(logits).argmax())
    print(
        f"[q4nx_prefill] first-token argmax={top} (expect {EXPECT_FIRST} ' Paris')",
        flush=True,
    )
    ok = top == EXPECT_FIRST
    print("[q4nx_prefill] *** PARIS ***" if ok else "[q4nx_prefill] MISS", flush=True)

    if args.bench_l:
        import time

        model.clear_context()
        ids = [
            int(t % VOCAB) for t in range(args.bench_l)
        ]  # synthetic prompt (timing only)
        print(f"[bench] warmup prefill L={args.bench_l}...", flush=True)
        model.prefill(ids)
        model.clear_context()
        model._dev_t = 0.0
        model._op_t.clear()
        # Per-ELF BO-Write (host->dev DMA) vs NPU-Run (on-device) vs BO-Read split.
        model.cache.profiler.enabled = True
        for d in (
            model.cache.profiler.kernel_times,
            model.cache.profiler.cpu_times,
            model.cache.profiler.kernel_breakdowns,
        ):
            d.clear()
        model.cache.profiler.layer_times.clear()
        print(f"[bench] timed prefill L={args.bench_l}...", flush=True)
        t0 = time.time()
        model.prefill(ids)
        wall = time.time() - t0
        npu = model._dev_t
        print(
            f"\n[bench] L={args.bench_l}: WALL={wall*1000:.0f}ms {args.bench_l/wall:.0f} tok/s prefill  |  "
            f"NPU-dispatch={npu*1000:.0f}ms {args.bench_l/npu:.0f} tok/s  |  host={(wall-npu)*1000:.0f}ms",
            flush=True,
        )
        # Machine-readable perf line for bench/extract_perf.py (TTFT = prefill wall;
        # decode tok/s is reported separately by the inference path).
        print(
            f"[q4nx_prefill] Inference: prompt_len={args.bench_l}, n_tokens=0",
            flush=True,
        )
        print(f"Time to first token (TTFT): {wall:.3f}s", flush=True)
        print(
            "[bench] per-op NPU: "
            + "  ".join(
                f"{k}={v*1000:.0f}ms"
                for k, v in sorted(model._op_t.items(), key=lambda x: -x[1])
            ),
            flush=True,
        )
        model.cache.profiler.report()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
