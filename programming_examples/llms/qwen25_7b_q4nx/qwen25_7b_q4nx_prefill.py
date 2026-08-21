# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen2.5-7B Q4NX prefill in mlir-air.
#
# A thin driver over the config-parameterized qwen25_3b prefill builders -- same
# Qwen2.5 block shape (RMSNorm -> Q/K/V GEMM + bias -> RoPE -> causal GQA flash
# attention -> SwiGLU), different constants -- rather than a self-contained
# clone. Weights come from a FastFlowLM Q4NX bundle, dequantized on the host at
# load and written once into resident per-layer BOs; every op then runs on the
# NPU.
#
# Deltas vs the Qwen2.5-3B Q4_0 sibling, all model-driven:
#   codec   Q4NX (unsigned int4, affine scale+min) from a pre-quantized
#           FastFlowLM bundle, instead of quantizing an HF bf16 checkpoint to
#           Q4_0 on the fly. Same codec the fused decode consumes -> prefill and
#           decode agree bit-for-bit on the weights.
#   shape   28 layers, 28 heads x head_dim 128, 4 kv heads (GQA group 7),
#           hidden 18944, vocab 152064, rope_theta 1e6.
#   lm head UNTIED -- the bundle ships a separate Q4NX lm_head, where the 3B
#           reuses the embedding table.
#
# The GEMM registry shapes exist at M=2048, so the whole prefill runs at
# seq_len=2048 (pad the prompt, read the last real token's row for the logit).
#
# Weight source (env-overridable): MODEL_SOURCE -- an HF repo id, a local dir
# containing model.q4nx, or a direct model.q4nx path.
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_PROG = str(_HERE.parent.parent)  # programming_examples
_LLMS = str(_HERE.parent)  # llms
_QWEN = str(_HERE.parent / "qwen25_3b")  # prefill builders + block runner
for _p in (_PROG, _LLMS, _QWEN, str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.infra.cache import KernelCache  # noqa: E402

from qwen25_7b_q4nx_weights import (  # noqa: E402
    D,
    DK,
    DV,
    VOCAB,
    NUM_LAYERS,
    load_q4nx_weights,
    qwen25_7b_config,
)

MODEL_DEFAULT = os.environ.get("MODEL_SOURCE", "Qwen/Qwen2.5-7B-Instruct")
EPS = 1e-6

# On-device LM head GEMV: 10 partitions of M=16384 (10*16384=163840 >= VOCAB).
_LM_N_PART = 16384
_LM_N_PARTITIONS = 10
# K=3584 clears the L2 A-stage budget (herd_m*tile_m*(K+1)*2 <= 512 KiB allows
# herd_m*tile_m <= 73, so the default 64 fits -- unlike Qwen3-8B at K=4096, which
# misses it by 128 B and has to halve tile_m). Only m_input has to come down:
# m_input=4 would put the ping-pong L1 A tiles at 2*28 KiB of a 64 KiB L1.
# tile_m stays 8, so the default mv.o (DIM_M_OUTPUT=8) is reused as-is.
_LM_HERD_M = 8
_LM_TILE_M = 8
_LM_M_INPUT = 2

# "The capital of France is" (Qwen2.5 BPE). Gate: the next token is " Paris".
PROMPT = [785, 6722, 315, 9625, 374]
EXPECT_FIRST = 12095  # " Paris"


def _bf(a):
    return np.asarray(a, bfloat16)


class Qwen25Q4nxPrefill:
    """AIR realization of the Q4NX causal-LM prefill interface (Qwen2.5-7B)."""

    def __init__(
        self, seq_len=2048, n_layers=NUM_LAYERS, cache_dir=None, verbose=False
    ):
        self.seq = seq_len
        self.n_layers = n_layers
        self.MAX_L = seq_len
        self.current_context_length = 0
        self.D = D

        from qwen25_3b_weights import generate_rope_lut
        from qwen25_3b_prefill import compile_all_kernels
        from shared.infra.backend_presets import LM_GEMV_BACKEND

        self.config = qwen25_7b_config(n_layers=n_layers)
        # Attention on the NPU (head-first FA, head_dim=128). Q4NX_CPU_ATTN=1
        # falls back to the fp32 host reference -- correct but ~50x slower.
        self.cpu_attn = os.environ.get("Q4NX_CPU_ATTN") == "1"
        self._rope_lut = generate_rope_lut(self.config, seq_len=seq_len)  # theta 1e6
        self._lm_backend = dict(LM_GEMV_BACKEND)

        # seq_len-specific cache: the ELFs are compiled for this padded context
        # length but the KernelCache is keyed by kernel NAME only, so a shared dir
        # would let a shorter-context build be silently reused for a longer request
        # -> all-zero logits beyond the built length. Isolate by seq_len (and by
        # attention backend, since only one of the two builds flash_attn).
        _att = "cpu" if self.cpu_attn else "fa"
        cache_dir = cache_dir or str(_HERE / f"_q4nx_cache_seq{seq_len}_{_att}")
        self.cache = KernelCache(cache_dir=cache_dir, verbose=verbose)

        # Compile (or reuse cached) the prefill ELFs + flash_attn, plus the
        # 10-partition lm_head GEMV. Q4NX_FORCE_COMPILE=1 forces a rebuild.
        force = os.environ.get("Q4NX_FORCE_COMPILE") == "1"
        if not force:
            self.cache.load_manifest()
        cached = set() if force else set(self.cache.artifacts)
        need = {
            "rms_qkv_bias_rope",
            "o_res_norm",
            "gate",
            "up",
            "swiglu",
            "down_add",
        }
        if not self.cpu_attn:
            need.add("flash_attn")
        if not need.issubset(cached):
            compile_all_kernels(
                self.cache, self.config, seq_len, cpu_attn=self.cpu_attn
            )
        else:
            print("[q4nx_prefill] using cached prefill ELFs (skip compile)", flush=True)
        if "lm_head_gemv" not in cached:
            from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module

            self.cache.compile_and_cache(
                "lm_head_gemv",
                build_lm_head_gemv_module(
                    emb_dim=D,
                    n_partitions=_LM_N_PARTITIONS,
                    n_part=_LM_N_PART,
                    herd_m=_LM_HERD_M,
                    tile_m=_LM_TILE_M,
                    m_input=_LM_M_INPUT,
                ),
                {"verbose": self.cache.verbose, **self._lm_backend},
            )
            self.cache._save_manifest()

        # Per-layer KV cache (roped K + biased raw V), [MAX_L, n_kv*head_dim].
        self.kv_k = [np.zeros((self.MAX_L, DK), bfloat16) for _ in range(n_layers)]
        self.kv_v = [np.zeros((self.MAX_L, DV), bfloat16) for _ in range(n_layers)]
        self._weights = None
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
        """Load the Q4NX bundle, dequant to bf16 on the host, and pre-load every
        projection into resident per-layer BOs."""
        model = model or MODEL_DEFAULT
        self._weights = load_q4nx_weights(model, config=self.config)
        self._lm_head = self._weights.lm_head
        self._preload()

    def _preload(self):
        """Resident BOs: prefill block weights (written once, then skipped via
        static_input_indices) + the LM-head partitions."""
        from qwen25_3b_prefill import preload_prefill_weights

        for i, lw in enumerate(self._weights.layers):
            lw._layer_idx = i  # per-layer BO isolation
        preload_prefill_weights(
            self._weights, self.config, self.cache, self.seq, self._rope_lut
        )
        self._preload_lm_head_gemv()

    def _preload_lm_head_gemv(self):
        """Build the padded bf16 lm_head partitions [_LM_N_PART, D] and write them
        into resident BOs once (static; skipped thereafter)."""
        self._lm_parts = []
        for p in range(_LM_N_PARTITIONS):
            n0 = p * _LM_N_PART
            n1 = min(n0 + _LM_N_PART, VOCAB)
            w = np.zeros((_LM_N_PART, D), bfloat16)
            if n1 > n0:
                w[: n1 - n0] = _bf(self._lm_head[n0:n1])
            self._lm_parts.append(w)
        self._lm_head_npu(np.zeros(D, bfloat16))  # warm/allocate resident BOs

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
        return np.concatenate(
            [np.asarray(res[2 + 2 * p], np.float32) for p in range(_LM_N_PARTITIONS)]
        )[:VOCAB]

    def _run_layer(self, x, k, N):
        """One Qwen2.5 transformer block fully on-device. Captures roped-K and
        biased-V into the KV cache (the decode hand-off)."""
        from qwen25_3b_prefill import run_transformer_block_qwen25

        out, inter = self._dev(
            run_transformer_block_qwen25,
            x,
            self._weights.layers[k],
            self._rope_lut,
            self.config,
            self.cache,
            k,
            self.cpu_attn,
            False,  # verbose
            tag="layer",
        )
        self.kv_k[k][:N] = np.asarray(inter["k_roped"], bfloat16).reshape(-1, DK)[:N]
        self.kv_v[k][:N] = np.asarray(inter["v"], bfloat16).reshape(-1, DV)[:N]
        return out

    def prefill(self, ids, payload=None):
        assert self._weights is not None, "call load_weights() first"
        N = len(ids)
        assert N <= self.seq, (N, self.seq)
        # Single-shot prefill only: the embedding write and every layer's KV
        # write below are indexed from row 0, so a second call would silently
        # overwrite the cache from the start while current_context_length kept
        # accumulating. Fail loudly instead of corrupting.
        base = self.current_context_length
        assert base == 0, (
            f"incremental prefill is not supported (already prefilled {base} "
            f"tokens); construct a new instance or clear_context() first"
        )
        x = np.zeros((self.seq, D), bfloat16)
        x[:N] = np.asarray(self._weights.embed_table[list(ids)], bfloat16)
        for k in range(self.n_layers):
            x = self._run_layer(x, k, N)
        self.current_context_length = base + N
        # Final RMSNorm on the single prediction row (host, <1ms) + NPU LM head.
        xf = np.asarray(x[N - 1], np.float32)
        xn = (
            xf
            / np.sqrt((xf * xf).mean() + EPS)
            * np.asarray(self._weights.final_norm, np.float32)
        )
        self._last_hidden = xf
        return self._dev(self._lm_head_npu, _bf(xn), tag="lm_head")

    # ---- KV cache (causal_lm) ----
    def get_k_cache(self, layer_idx, idx):
        """Roped-K vector at position idx of a layer."""
        return self.kv_k[layer_idx][idx]

    def get_v_cache(self, layer_idx, idx):
        """Biased raw-V vector at position idx of a layer."""
        return self.kv_v[layer_idx][idx]

    def kv_stack(self):
        """(K, V) stacked over layers as [n_layers, ctx, kv_dim] -- the layout the
        fused decode's seed_kv consumes."""
        c = self.current_context_length
        return (
            np.stack([self.kv_k[L][:c] for L in range(self.n_layers)]),
            np.stack([self.kv_v[L][:c] for L in range(self.n_layers)]),
        )

    def kv_view(self, layer_idx):
        """(roped_K, biased_V) for the filled context [0:ctx] -> decode hand-off."""
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
    ap = argparse.ArgumentParser(description="Qwen2.5-7B Q4NX prefill on NPU2")
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="build/cache the prefill ELFs and exit (no weights, no NPU dispatch)",
    )
    ap.add_argument(
        "--n-layers", type=int, default=int(os.environ.get("NLAYERS", str(NUM_LAYERS)))
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
    ap.add_argument("--prompt", default=None, help="text prompt (needs transformers)")
    ap.add_argument(
        "--dump-kv",
        default=None,
        metavar="PATH.npz",
        help="write the prefill KV cache (roped K + biased V, per layer) plus the "
        "prompt ids to PATH.npz -- the hand-off to the fused NPU decode. Written "
        "as raw uint16 so the bf16 payload survives np.savez unchanged.",
    )
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()

    print(
        f"[q4nx_prefill] constructing seq_len={args.seq_len} "
        f"n_layers={args.n_layers} (compiling engines)...",
        flush=True,
    )
    model = Qwen25Q4nxPrefill(
        seq_len=args.seq_len, n_layers=args.n_layers, cache_dir=args.cache_dir
    )
    if args.compile_only:
        print("Compilation passed.", flush=True)
        return 0

    print("[q4nx_prefill] loading Q4NX weights (host dequant)...", flush=True)
    model.load_weights(model=args.model)

    ids = PROMPT
    tok = None
    if args.prompt:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        ids = tok(args.prompt, return_tensors=None)["input_ids"]
    print(f"[q4nx_prefill] prefill prompt N={len(ids)} ...", flush=True)
    logits = model.prefill(ids)
    top = int(np.asarray(logits).argmax())
    txt = f" ({tok.decode([top])!r})" if tok else ""
    print(f"[q4nx_prefill] first-token argmax={top}{txt}", flush=True)
    ok = True
    if ids == PROMPT:
        ok = top == EXPECT_FIRST
        print(
            (
                "[q4nx_prefill] *** PARIS ***"
                if ok
                else f"[q4nx_prefill] MISS (expect {EXPECT_FIRST})"
            ),
            flush=True,
        )

    if args.dump_kv:
        c = model.get_current_context_length()
        np.savez(
            args.dump_kv,
            ids=np.asarray(ids, np.int64),
            ctx=np.int64(c),
            # bf16 payloads move as raw uint16: np.savez does not round-trip the
            # ml_dtypes bf16 dtype tag.
            k=np.stack([model.kv_k[l][:c] for l in range(model.n_layers)]).view(
                np.uint16
            ),
            v=np.stack([model.kv_v[l][:c] for l in range(model.n_layers)]).view(
                np.uint16
            ),
            hidden=np.asarray(model._last_hidden, np.float32),
        )
        print(
            f"[q4nx_prefill] KV dumped -> {args.dump_kv} "
            f"({model.n_layers} layers x {c} tokens)",
            flush=True,
        )

    if args.bench_l:
        import time

        model.clear_context()
        bench_ids = [int(t % VOCAB) for t in range(args.bench_l)]  # timing only
        print(f"[bench] warmup prefill L={args.bench_l}...", flush=True)
        model.prefill(bench_ids)
        model.clear_context()
        model._dev_t = 0.0
        model._op_t.clear()
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
        model.prefill(bench_ids)
        wall = time.time() - t0
        npu = model._dev_t
        print(
            f"\n[bench] L={args.bench_l}: WALL={wall*1000:.0f}ms "
            f"{args.bench_l/wall:.0f} tok/s prefill  |  "
            f"NPU-dispatch={npu*1000:.0f}ms {args.bench_l/npu:.0f} tok/s  |  "
            f"host={(wall-npu)*1000:.0f}ms",
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
