# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Decode throughput vs KV depth, for the host-attention model family.

Shared by every bf16/int4 example whose decode attention runs on the host: the
NPU kernels there are per-token GEMVs with no context dependence, so a context
is set by sizing the KV cache and telling the step which position to attend
from -- no rebuild, no prefill of that length, one process for a whole curve.
(The fused-decode family compiles its context into the ELF and is swept by
bench/sweep_decode.py instead.)

Latency only, never a correctness gate -- `make verify` is that. The KV cache
contents are synthetic.

Lives here rather than in each driver because the measurement, not the model,
is what these examples share: nine copies of it drifted the moment one of them
needed a more robust statistic.
"""

import statistics
import time

import numpy as np
from ml_dtypes import bfloat16


def bench_contexts(args):
    """--bench-decode as a list of KV depths."""
    return [int(c) for c in args.bench_decode.split(",") if c.strip()]


def bench_rope_len(args):
    """RoPE LUT rows the decode sweep needs, or 0 when it is not running.

    The sweep attends from positions far past the prefill length, so a driver
    that sizes its LUT to seq_len alone would index off the end.
    """
    ctxs = bench_contexts(args) if getattr(args, "bench_decode", "") else []
    return max(ctxs) + 16 if ctxs else 0


def bench_decode(session, contexts, step_fn, iters=10, warmup=3):
    """Measure decode latency at each KV depth in `contexts`, in one session.

    `step_fn` is the model's run_npu_decode_step; everything else comes off the
    Session, so a model contributes no bench code of its own.

    Reports MEDIAN and MIN, never a mean or an sd. This decode allocates and
    upcasts the whole KV cache per token (1.9 GB/token at 8k on a 0.6B), and
    that occasionally collides with kernel reclaim: a measured 12-sample point
    ran 657-862 ms eleven times and 46846 ms once. A mean or an sd is entirely
    determined by that one sample -- the first version of this reported a mean,
    and the resulting cell moved 6x between runs. The median is the typical
    token and is stable to ~1%; min is the uncontended floor, so median >> min
    says the box was busy rather than the design being slow.
    """
    cfg = session.config
    k_cache = np.zeros(
        (cfg.n_layers, cfg.n_kv_heads, max(contexts) + iters + warmup, cfg.head_dim),
        dtype=bfloat16,
    )
    v_cache = np.zeros_like(k_cache)
    print(f"[bench] KV cache {2 * k_cache.nbytes / 1e9:.2f} GB", flush=True)
    x = session.weights.embed_table[1].astype(bfloat16)

    def _step(pos):
        step_fn(
            x,
            session.weights,
            cfg,
            session.decode_cache,
            session.rope_lut_bf16,
            k_cache,
            v_cache,
            pos,
        )

    for ctx in contexts:
        for i in range(warmup):
            _step(ctx + i)
        samples = []
        for i in range(iters):
            t0 = time.perf_counter()
            _step(ctx + warmup + i)
            samples.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(samples)
        # The line format bench/sweep_decode_runtime.py parses.
        print(
            f"[bench] decode ctx={ctx} median {med:.3f} ms min {min(samples):.3f} ms "
            f"({1000.0 / med:.2f} tok/s)",
            flush=True,
        )
