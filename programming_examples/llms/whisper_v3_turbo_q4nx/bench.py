# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Complete-request timing of the production path (whisper_npu.transcribe)
on the gate's first clips: warm-up requests for at least WARMUP_S seconds,
then rounds over the clips until at least MIN_ROUNDS rounds and MEASURE_S
seconds. Prints the mean front end (log-mel, convs) and encoder time, the
time to first token (from the audio), the decode throughput after the first
token and the real-time factor (request time over audio duration), in the
lines bench/extract_perf.py reads.

  python3 bench.py --build BUILD [--clips N] [--out OUT.json]
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

WARMUP_S, MEASURE_S, MIN_ROUNDS = 5.0, 10.0, 3


def main():
    from air_examples.llms.whisper_v3_turbo_q4nx import inputs
    from air_examples.llms.whisper_v3_turbo_q4nx.whisper_npu import (
        Decoding,
        WhisperNpu,
        transcribe,
    )
    from air_examples.llms.whisper_v3_turbo_q4nx.whisper_weights import Q4nx

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--clips", type=int, default=inputs.N_VERIFY)
    ap.add_argument("--out")
    a = ap.parse_args()
    npu = WhisperNpu(a.build, Q4nx())
    dec = Decoding()
    clips = inputs.clips(a.clips)

    def request(audio):
        t = {}
        out = transcribe(npu, audio, dec, times=t)
        t.update(tokens=len(out), audio=len(audio) / 16000)
        return t

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < WARMUP_S:
        for _, audio, _ in clips:
            request(audio)
    warm = time.perf_counter() - t0
    samples, rounds, t0 = [], 0, time.perf_counter()
    while rounds < MIN_ROUNDS or time.perf_counter() - t0 < MEASURE_S:
        for cid, audio, _ in clips:
            samples.append(dict(id=cid, **request(audio)))
        rounds += 1
    measured = time.perf_counter() - t0

    def mean(k):
        return float(np.mean([s[k] for s in samples]))

    after = sum(s["tokens"] - 1 for s in samples)
    decode_s = sum(s["total"] - s["first"] for s in samples)
    rtf = sum(s["total"] for s in samples) / sum(s["audio"] for s in samples)
    print(f"{len(samples)} requests ({rounds} rounds of {len(clips)} clips), "
          f"warm-up {warm:.1f} s, measured {measured:.1f} s")  # fmt: skip
    print(f"Audio: {mean('audio'):.1f} s, {mean('tokens'):.1f} tokens per request")
    print(f"Front end (log-mel, convs): {mean('front') * 1e3:.1f} ms")
    print(f"Encoder: {mean('encoder') * 1e3:.1f} ms")
    print(f"Time to first token (TTFT): {mean('first'):.4f} s")
    print(
        f"Decode: {decode_s / after * 1e3:.2f} ms per token ({after / decode_s:.2f} tok/s)"
    )
    print(f"Request: {mean('total') * 1e3:.1f} ms")
    print(f"Real-time factor: {rtf:.4f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(
            json.dumps(
                dict(warmup_s=warm, measured_s=measured, rounds=rounds, samples=samples)
            )
        )


if __name__ == "__main__":
    main()
