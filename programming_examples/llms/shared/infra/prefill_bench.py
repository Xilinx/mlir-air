# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Warm prefill TTFT at the length the engines were built for.

The other axis from decode_bench.py, and the sibling of it for the same reason
the two sweepers are separate files: decode throughput is a function of KV
depth, TTFT is a function of how many token rows the prefill engines compute.

The axis is the PADDED length, not the prompt length. These prefills pad the
prompt to the built length and read the last real token's row, so at a fixed
engine TTFT does not move with the prompt -- which is why the ids here are
synthetic and why every point in a sweep rebuilds the engines.

Latency only, never a correctness gate -- `make verify` is that.
"""

import statistics
import time


def bench_prefill(session, prefill_fn, cpu_attn=False, iters=3, warmup=1):
    """Time a warm prefill at session.seq_len and print it for the sweepers.

    `prefill_fn` is the model's run_npu_prefill; everything else comes off the
    Session, so a model contributes no bench code of its own.

    Median of a few runs rather than a single timed pass, and min printed
    beside it. This is not theoretical: the NPU is shared, and a single-pass
    revision of this file published 60550 ms for smollm2_1_7b at 4096 and got a
    33x "MHA cliff" written into a lit comment on the strength of it. The point
    re-measures at 3498 ms under active contention from an unrelated job. Same
    reasoning as decode_bench.py, at a smaller iteration count because each
    iteration here costs a whole prefill.

    min is the uncontended floor: median >> min says the box was busy, which is
    the check that would have caught the above before it reached a comment.

    Note on flock: an outer `flock /tmp/npu.lock` around this cannot be the
    answer. cache.py already takes that same lock per dispatch, and BSD flock(2)
    on one inode is not re-entrant, so holding it across the timed region
    deadlocks. Serialising the measurement is a scheduling question; making the
    number robust is this function's job.
    """
    ids = [int(t % session.config.vocab_size) for t in range(session.seq_len)]

    def _once():
        t0 = time.perf_counter()
        prefill_fn(
            ids,
            session.weights,
            session.config,
            session.prefill_cache,
            session.decode_cache,
            session.rope_lut_bf16,
            session.seq_len,
            tokenizer=session.tokenizer,
            cpu_attn=cpu_attn,
            quiet=True,
        )
        return time.perf_counter() - t0

    print(f"[bench] warmup prefill L={session.seq_len}...", flush=True)
    for _ in range(warmup):
        _once()
    print(f"[bench] timed prefill L={session.seq_len}...", flush=True)
    runs = [_once() for _ in range(iters)]
    wall = statistics.median(runs)

    # The line formats bench/extract_perf.py and bench/sweep_prefill.py parse.
    # "Inference: prompt_len=" is what extract_perf.py reads the context off.
    print(f"\nInference: prompt_len={session.seq_len}, n_tokens=0", flush=True)
    print(f"Time to first token (TTFT): {wall:.3f}s", flush=True)
    print(
        f"[bench] L={session.seq_len}: {session.seq_len / wall:.0f} tok/s prefill",
        flush=True,
    )
    # Separate line, so the established "[bench] L=..." format that
    # sweep_prefill.py and the q4nx drivers share stays byte-identical.
    print(
        f"[bench] prefill median {wall * 1000:.1f} ms min {min(runs) * 1000:.1f} ms",
        flush=True,
    )
