# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Llama-3.2-1B Q4NX prefill in mlir-air.
#
# Runs the prefill MECHANISM on the AMD NPU2: Q4NX weights, host dequant
# Q4NX->bf16 at load, then op-by-op bf16 GEMM /
# RMSNorm / RoPE / SwiGLU / causal-GQA-flash-attention all ON THE NPU with
# RESIDENT weight BOs, and the final LM head as an on-device GEMV. The whole
# transformer block runs through the two llama32_1b multi-launch stitchers
# (rms_gemms_rope + o_ffn) plus flash_attn, driven by KernelCache.load_and_run
# with per-layer resident BOs (weights written once).
#
# Gate: first prompt token argmax 12366 (" Paris") for "The capital of France is".
# Warm TTFT @ L=2048 ~= 0.93 s (TEMPORAL_CAUSAL_SKIP=1, Q4NX_BENCH_L=2048).
#
# Registry GEMM shapes exist only at M=2048, so the whole prefill runs at
# seq_len=2048 (pad the prompt, read the last real token's row for the logit).
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
_LLAMA1B = str(_HERE.parent / "llama32_1b")  # fused-stitcher prefill driver
for _p in (_PROG, _LLMS, _LLAMA1B, str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.infra.cache import KernelCache  # noqa: E402

# Q4NX unpack/dequant + model dims + numerics.
from llama32_1b_q4nx_weights import (  # noqa: E402
    _bf,
    D,
    DK,
    DV,
    DH,
    N_Q_HEADS,
    N_KV_HEADS,
)

# Default weight source: the self-contained model.q4nx bundle on the Hub. May be
# overridden with --model / Q4NX_MODEL_SOURCE (an HF repo id, or a local dir/file).
MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Llama-3.2-1B-NPU2")

VOCAB = 128256
PROMPT = [128000, 791, 6864, 315, 9822, 374]  # "The capital of France is"
EXPECT_FIRST = 12366  # " Paris"

# On-device LM head GEMV: 8 partitions of M=16384 (8*16384=131072 >= VOCAB), K=D.
_LM_N_PART = 16384
_LM_N_PARTITIONS = 8


class LlamaQ4nxPrefill:
    """AIR realization of the Q4NX causal-LM prefill interface."""

    def __init__(self, seq_len=2048, n_layers=16, cache_dir=None, verbose=False):
        self.seq = seq_len
        self.n_layers = n_layers
        self.MAX_L = seq_len
        self.current_context_length = 0
        self.wcache_dir = str(_HERE / ".wcache")
        # seq_len-specific cache: the stitcher/flash-attn ELFs are compiled for this padded
        # context length, but the KernelCache is keyed by kernel NAME only. A shared dir would
        # let a shorter-context build (e.g. 256) be silently reused for a 2048 request -> the
        # prefill then returns all-zero logits for N beyond the built length. Isolate by seq_len.
        cache_dir = cache_dir or str(_HERE / f"_q4nx_cache_seq{seq_len}")
        self.cache = KernelCache(cache_dir=cache_dir, verbose=verbose)

        from llama32_1b_weights import LlamaConfig, generate_rope_lut
        from llama32_1b_prefill import compile_all_kernels
        from shared.infra.backend_presets import LM_GEMV_BACKEND

        self.config = LlamaConfig(n_layers=n_layers)
        self._rope_lut = generate_rope_lut(
            self.config, seq_len=seq_len
        )  # half-split, theta=5e5
        self._lm_backend = dict(LM_GEMV_BACKEND)

        # Compile (or reuse cached) the 4 ELFs: rms_gemms_rope + flash_attn + o_ffn
        # (the two llama32_1b stitchers + attn) and the 8-partition lm_head GEMV.
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
        # LM head on the NPU — an 8-partition GEMV.
        if "lm_head_gemv" not in cached:
            from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module

            self.cache.compile_and_cache(
                "lm_head_gemv",
                build_lm_head_gemv_module(D),
                {"verbose": self.cache.verbose, **self._lm_backend},
            )
            self.cache._save_manifest()

        # Per-layer KV cache (roped K + raw V), [MAX_L, n_kv_heads*head_dim].
        self.kv_k = [np.zeros((self.MAX_L, DK), bfloat16) for _ in range(n_layers)]
        self.kv_v = [np.zeros((self.MAX_L, DV), bfloat16) for _ in range(n_layers)]
        self._w = None  # per-layer bf16 weights [K,N]
        self._rms = None  # per-layer (attn_norm, ffn_norm)
        self._final_norm = None
        self._lm_head = None
        self._embed = None
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
        local dir/file): per-layer Q4NX projections + bf16 norms + bf16 embed +
        (tied) Q4NX lm_head, dequant Q4NX->bf16 on the host at load."""
        model = model or os.environ.get("Q4NX_MODEL_SOURCE", MODEL_DEFAULT)
        from llama32_1b_q4nx_weights import Q4nxModel

        qm = Q4nxModel(model)
        print(f"[q4nx_prefill] loading weights from model.q4nx ({model})", flush=True)
        self._w = [qm.layer_weights(k) for k in range(self.n_layers)]
        self._rms = [qm.layer_rms(k) for k in range(self.n_layers)]
        self._embed, self._final_norm, self._lm_head = qm.embed_norm_lmhead()
        self._build_fused_weights_and_preload()

    def _build_fused_weights_and_preload(self):
        """Wrap the Q4NX-dequant'd bf16 [K,N] weights + norm vectors into the
        llama32_1b LlamaWeights container and pre-load them into per-layer
        resident BOs (weights written once; static_input_indices skips them at
        prefill)."""
        from llama32_1b_weights import LayerWeights, LlamaWeights
        from llama32_1b_prefill import preload_prefill_weights

        layers = []
        for k in range(self.n_layers):
            Wt = self._w[k]
            rms_in, rms_post = self._rms[k]
            layers.append(
                LayerWeights(
                    attn_norm=np.asarray(rms_in, bfloat16),
                    wq=Wt["q"],
                    wk=Wt["k"],
                    wv=Wt["v"],
                    wo=Wt["o"],
                    ffn_norm=np.asarray(rms_post, bfloat16),
                    w_gate=Wt["gate"],
                    w_up=Wt["up"],
                    w_down=Wt["down"],
                )
            )
        self._fused_weights = LlamaWeights(
            embed_table=None,
            layers=layers,
            final_norm=np.asarray(self._final_norm, bfloat16),
            lm_head=None,
        )
        preload_prefill_weights(
            self._fused_weights, self.config, self.cache, self.seq, self._rope_lut
        )
        self._preload_lm_head_gemv()

    def _preload_lm_head_gemv(self):
        """Build the 8 padded bf16 lm_head partitions [16384, D] and write them
        into resident BOs once (static; skipped thereafter)."""
        self._lm_parts = []
        for p in range(_LM_N_PARTITIONS):
            n0 = p * _LM_N_PART
            n1 = min(n0 + _LM_N_PART, VOCAB)
            w = np.zeros((_LM_N_PART, D), bfloat16)
            w[: n1 - n0] = np.asarray(self._lm_head[n0:n1], bfloat16)
            self._lm_parts.append(w)
        self._lm_head_npu(np.zeros(D, bfloat16))  # warm/allocate resident weight BOs

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
        """One transformer block fully on-device via the llama32_1b stitchers
        (rms+qkv+rope+attn in rms_gemms_rope+flash_attn, o+residual+rms+gate/up/
        SiLU+down in o_ffn). Captures roped-K + raw-V into the KV cache."""
        from llama32_1b_prefill import run_transformer_block

        out, inter = self._dev(
            run_transformer_block,
            x,
            self._fused_weights.layers[k],
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
        assert self._w is not None, "call load_weights() first"
        N = len(ids)
        assert N <= self.seq, (N, self.seq)
        base = self.current_context_length
        x = np.zeros((self.seq, D), bfloat16)
        x[:N] = _bf(np.stack([self._embed[t] for t in ids]))
        for k in range(self.n_layers):
            x = self._run_layer(x, k, N)
        self.current_context_length = base + N
        # Final RMSNorm on the single prediction row (host, <1ms), then NPU LM head.
        xf = x[N - 1].astype(np.float32)
        xn = xf / np.sqrt((xf * xf).mean() + 1e-6) * self._final_norm
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
    ap = argparse.ArgumentParser(description="Llama-3.2-1B Q4NX prefill on NPU2")
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="build/cache the prefill ELFs and exit (no weights, no NPU dispatch)",
    )
    ap.add_argument(
        "--n-layers", type=int, default=int(os.environ.get("NLAYERS", "16"))
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
        # decode tok/s is reported separately by the chatbot/inference path).
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
