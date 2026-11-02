# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Single-question decision latency per prompt length on the production path.

For each P a prompt of exactly P tokens: the request set's bucket-P request
with the longest `route` question, that question alone, its state grown with
filler text to P tokens. The timed span runs from the tokenized input ids on
the host to the option probabilities on the host (D3Npu.probabilities). Per
P: at least --warmup seconds of untimed runs, then timed runs until both
--min-runs and --min-seconds are met. Reports the median, P / median, the
raw samples, the last run's host / device split and the host load.

  python3 bench.py --build BUILD --model edge [--wform hilo] --out timing.json
"""

import argparse
import json
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

from air_examples.llms.d3_edge.d3_npu import D3Npu, add_model_args  # noqa: E402

FILLER = " The customer repeated the request and asked for an update on the ticket."


def prompt_of_length(m, P, req):
    """(state, question, input ids) of req's `route` question grown to
    exactly P tokens, or ids None when tokenization cannot land on P."""
    key = "route"
    q = {key: req["questions"][key]}
    state = req["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    while True:
        ((_, nq, ids),) = m.prepare(state, q)
        if len(ids) >= P:
            break
        state += FILLER
    # trim the state from the end until the prompt fits, then pad with single
    # characters to land exactly on P
    while len(ids) > P:
        state = state[:-8]
        ((_, nq, ids),) = m.prepare(state, q)
    while len(ids) < P:
        state += "."
        ((_, nq, ids),) = m.prepare(state, q)
    return state, nq, (ids if len(ids) == P else None)


def loadavg():
    return Path("/proc/loadavg").read_text().split()[:3]


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--requests", default=str(HERE / "requests.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="512,1024,2048,4096")
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--min-runs", type=int, default=3)
    ap.add_argument("--min-seconds", type=float, default=10.0)
    a = ap.parse_args()

    reqs = json.loads(Path(a.requests).read_text())["requests"]
    m = D3Npu(a.build, a.model, a.wform)
    results = {}
    for P in [int(v) for v in a.lengths.split(",")]:
        req = max(
            (r for r in reqs if r["bucket"] == P),
            key=lambda r: r["lengths"]["route"],
        )
        state, nq, ids = prompt_of_length(m, P, req)
        assert ids is not None, f"no prompt of exactly {P} tokens"
        count = len(nq.keys)
        load0 = loadavg()
        t_end = time.perf_counter() + a.warmup
        warm = 0
        while time.perf_counter() < t_end or warm < 1:
            m.probabilities(ids, count)
            warm += 1
        samples, t_start = [], time.perf_counter()
        while (
            len(samples) < a.min_runs or time.perf_counter() - t_start < a.min_seconds
        ):
            m.times.clear()
            m.pf.dev_t, m.pf.by_op = 0.0, {}
            t0 = time.perf_counter()
            p = m.probabilities(ids, count).tolist()
            samples.append(time.perf_counter() - t0)
        med = statistics.median(samples)
        results[str(P)] = dict(
            n_tokens=len(ids),
            request=req["id"],
            question="route",
            warmup_runs=warm,
            warmup_seconds=a.warmup,
            runs=len(samples),
            measured_seconds=sum(samples),
            samples_s=samples,
            median_s=med,
            tokens_per_s=P / med,
            last_host_s=dict(m.times),
            last_device_wait_s=dict(m.pf.by_op),
            probabilities=p,
            option_count=count,
            input_ids=ids,
            loadavg_before=load0,
            loadavg_after=loadavg(),
        )
        print(
            f"[bench] {a.model}/{m.wform} P={P} median {med * 1e3:.1f} ms "
            f"({P / med:.0f} tok/s) runs={len(samples)} load {load0} -> "
            f"{loadavg()} host={ {k: round(v * 1e3, 1) for k, v in m.times.items()} }",
            flush=True,
        )
        # the line bench/sweep_prefill.py --fused reads
        print(
            f"[bench] prefill L={P} median {med * 1e3:.1f} ms "
            f"min {min(samples) * 1e3:.1f} ms",
            flush=True,
        )
    Path(a.out).write_text(
        json.dumps(
            dict(
                boundary="tokenized input ids on host -> option probabilities on host",
                model=a.model,
                wform=m.wform,
                results=results,
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
