# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Complete-request timing of the production path per configuration.

For a configuration (prefill bucket P, capacity C), a prompt of exactly P
tokens (the gate prompts' text repeated, ids cut at P) and fresh greedy
requests of 128 output tokens (EOS ignored). Boundaries:
  TTFT     input ids on the host -> first token chosen on the host (embedding,
           prefill, final norm, lm_head, argmax);
  decode   the 127 following steps (decode throughput = 127 / decode time);
  request  TTFT + decode.
At least --warmup seconds of untimed requests, then timed requests until both
--min-runs and --min-seconds are met. Raw samples and host load are recorded.

  python3 bench.py --build BUILD --prefill P --capacity C --out timing.json

--prefill-only L1,L2,... times the prefill alone (the fused_prefill device in
--device) at each length: one warm-up, then --min-runs prefills, printing the
line bench/sweep_prefill.py --fused reads.

--decode-only C1,C2,... times one decode step at each context C: the
fused_decode_gdn ELF of capacity C in BUILD/fdg_C stepping at position C - 1,
attending over C cache rows. The rows are synthetic (zero), as in the other
models' decode sweeps, so a context needs no C-token prefill. A step is what a
generated token costs the host: the embedding row and rotary rows in, the
dispatch, the logits read. Prints the line bench/sweep_decode_runtime.py reads.
"""

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]


def prompt_ids(tok, P):
    from air_examples.llms.qwen35_0_8b_q4nx.bench_prompts import prompt_ids as ids_of

    return ids_of(tok, P)


def load():
    try:
        return os.getloadavg()
    except OSError:
        return None


def prefill_only(device, lengths, runs):
    """[bench] prefill L=<n> median <ms> ms min <ms> ms, per length."""
    from air_examples.llms.qwen35_0_8b_q4nx import q35_npu as N
    from air_examples.llms.qwen35_0_8b_q4nx import q35_weights as W
    from air_examples.llms.shared.fused_prefill.qwen35 import Qwen35Prefill

    tok = N.tokenizer()
    cfg = N.text_config()
    Wt = W.Q4nx()
    cos, sin = N.rotary_table(cfg, max(lengths) + 32)
    fp = Qwen35Prefill(device, Wt, cfg.layer_types[: cfg.num_hidden_layers], cos, sin)
    for n in lengths:
        ids = prompt_ids(tok, n)
        fp.prefill(ids)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fp.prefill(ids)
            ts.append((time.perf_counter() - t0) * 1e3)
        print(
            f"[bench] prefill L={n} median {statistics.median(ts):.1f} ms "
            f"min {min(ts):.1f} ms",
            flush=True,
        )


def decode_only(build, contexts, runs, warmup=5):
    """[bench] decode ctx=<n> median <ms> ms min <ms> ms (<tok/s> tok/s), per
    context."""
    import numpy as np
    from ml_dtypes import bfloat16

    from air_examples.llms.qwen35_0_8b_q4nx import q35_npu as N
    from air_examples.llms.qwen35_0_8b_q4nx import q35_weights as W
    from air_examples.llms.qwen35_0_8b_q4nx.q35_fdg import FdgDecode

    cfg = N.text_config()
    Wt = W.Q4nx()
    cos, sin = N.rotary_table(cfg, max(contexts) + 32)
    cos, sin = cos.astype(bfloat16), sin.astype(bfloat16)
    row = (np.random.default_rng(0).standard_normal(W.D) * 0.02).astype(bfloat16)
    for n in contexts:
        dec = FdgDecode(build, n, Wt, cos, sin)
        for _ in range(warmup):
            dec.step(row, n - 1)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            dec.step(row, n - 1)
            ts.append((time.perf_counter() - t0) * 1e3)
        med = statistics.median(ts)
        print(
            f"[bench] decode ctx={n} median {med:.2f} ms min {min(ts):.2f} ms "
            f"({1e3 / med:.2f} tok/s)",
            flush=True,
        )
        del dec  # its buffers and hardware context, before the next capacity


def main():
    from air_examples.llms.qwen35_0_8b_q4nx.q35_npu import Qwen35Npu, tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--build")
    ap.add_argument("--prefill", type=int)
    ap.add_argument("--capacity", type=int)
    ap.add_argument("--outputs", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--min-runs", type=int, default=3)
    ap.add_argument("--min-seconds", type=float, default=10.0)
    ap.add_argument("--out")
    ap.add_argument("--prefill-only", default=None, help="comma list of prompt lengths")
    ap.add_argument(
        "--device", default=None, help="the prefill device (--prefill-only)"
    )
    ap.add_argument("--decode-only", default=None, help="comma list of decode contexts")
    ap.add_argument(
        "--steps", type=int, default=40, help="timed steps per context (--decode-only)"
    )
    a = ap.parse_args()
    if a.decode_only:
        decode_only(a.build, [int(n) for n in a.decode_only.split(",")], a.steps)
        return
    if a.prefill_only:
        lengths = [int(n) for n in a.prefill_only.split(",")]
        prefill_only(
            a.device or str(Path(a.build) / "device"), lengths, max(a.min_runs, 5)
        )
        return
    tok = tokenizer()
    ids = prompt_ids(tok, a.prefill)
    assert len(ids) == a.prefill and a.prefill + a.outputs <= a.capacity
    rec = dict(
        prefill=a.prefill,
        capacity=a.capacity,
        outputs=a.outputs,
        prompt_tokens=len(ids),
        decode="fused_decode_gdn",
        boundary="complete request: ids on host -> 128 greedy tokens on host",
        load_before=load(),
        cpus=os.cpu_count(),
    )
    t0 = time.perf_counter()
    m = Qwen35Npu(a.build, a.capacity)
    rec["init_s"] = time.perf_counter() - t0
    warm, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < a.warmup or not warm:
        toks, _, ttft, dt = m.generate(ids, a.outputs)
        warm.append(dict(ttft_s=ttft, decode_s=dt))
    runs, t0 = [], time.perf_counter()
    while len(runs) < a.min_runs or time.perf_counter() - t0 < a.min_seconds:
        r0 = time.perf_counter()
        toks, _, ttft, dt = m.generate(ids, a.outputs)
        req = time.perf_counter() - r0
        assert len(toks) == a.outputs
        runs.append(
            dict(
                ttft_s=ttft,
                decode_s=dt,
                request_s=req,
                decode_tok_s=(a.outputs - 1) / dt,
                breakdown=dict(m.times),
            )
        )
        print(
            f"ttft {ttft * 1e3:.1f} ms  decode {(a.outputs - 1) / dt:.2f} tok/s  "
            f"request {req:.3f} s",
            flush=True,
        )
    rec["measured_s"] = time.perf_counter() - t0
    rec["load_after"] = load()
    rec["warmup"] = warm
    rec["runs"] = runs
    rec["text"] = tok.decode(toks)
    med = lambda k: statistics.median(r[k] for r in runs)  # noqa: E731
    rec["median"] = {
        k: med(k) for k in ("ttft_s", "decode_s", "request_s", "decode_tok_s")
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(
        f"P={a.prefill} C={a.capacity}: TTFT {rec['median']['ttft_s'] * 1e3:.1f} ms, "
        f"decode {rec['median']['decode_tok_s']:.2f} tok/s, request {rec['median']['request_s']:.2f} s "
        f"(n={len(runs)}, load {rec['load_before']} -> {rec['load_after']})"
    )


if __name__ == "__main__":
    main()
