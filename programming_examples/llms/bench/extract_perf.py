# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Extract latency/throughput from `make profile` stdout into JSON.

Decoupled from model code: parses the human-readable summary printed by the
LLM inference scripts. Output is heterogeneous across models, so each metric is
extracted from any of the known line formats and left null when absent (a
missing metric is not fatal — some models legitimately don't emit all lines).
"""

import argparse
import datetime
import json
import re
import sys

# TTFT: qwen/llama3b/int4/smollm print "Time to first token (TTFT): X.XXs";
# llama32_1b prints "End-to-end (prefill, per query)  N ms".
TTFT_S_RE = re.compile(r"Time to first token \(TTFT\):\s*([\d.]+)\s*s")
PREFILL_MS_RE = re.compile(r"End-to-end \(prefill, per query\)\s+([\d.]+)\s*ms")

# Decode throughput: qwen family + llama3b print "(X.XX tok/s)";
# llama32_1b prints "End-to-end (per token)  N ms" (throughput = 1000/ms).
TOKS_RE = re.compile(r"\(([\d.]+)\s*tok/s\)")
DECODE_MS_RE = re.compile(r"End-to-end \(per token\)\s+([\d.]+)\s*ms")


def _ttft_ms(text):
    m = TTFT_S_RE.search(text)
    if m:
        return round(float(m.group(1)) * 1000.0, 2)
    m = PREFILL_MS_RE.search(text)
    if m:
        return round(float(m.group(1)), 2)
    return None


def _decode_tok_s(text):
    m = TOKS_RE.search(text)
    if m:
        return round(float(m.group(1)), 2)
    m = DECODE_MS_RE.search(text)
    if m:
        ms = float(m.group(1))
        return round(1000.0 / ms, 2) if ms else None
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="?", help="profile stdout file (default: stdin)")
    p.add_argument("--model", default="")
    p.add_argument("--runner", default="")
    p.add_argument("--verify-status", default="", choices=["", "pass", "fail"])
    p.add_argument("--mlir-air-sha", default="")
    p.add_argument("--mlir-aie-hash", default="")
    p.add_argument("--llvm-aie-version", default="")
    # Default None (-> JSON null) so an unset run param is not misreported as a
    # real value (e.g. n_tokens: 0). The profile lit tests don't pass these.
    p.add_argument("--n-tokens", type=int, default=None)
    p.add_argument("--prompt", default=None)
    args = p.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    data = {
        "model": args.model,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "runner": args.runner,
        "verify_status": args.verify_status,
        "metrics": {
            "ttft_ms": _ttft_ms(text),
            "decode_tokens_per_sec": _decode_tok_s(text),
        },
        "toolchain": {
            "mlir_air_sha": args.mlir_air_sha,
            "mlir_aie_hash": args.mlir_aie_hash,
            "llvm_aie_version": args.llvm_aie_version,
        },
        "run_params": {
            "n_tokens": args.n_tokens,
            "prompt": args.prompt,
        },
    }
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
