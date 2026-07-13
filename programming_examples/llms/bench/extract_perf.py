# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Extract end-to-end latency numbers from `make profile` stdout into JSON.

Decoupled from model code: parses the human-readable summary printed by the
LLM inference scripts. Exits non-zero if an expected metric line is missing,
so silent format drift fails the nightly instead of publishing empty numbers.
"""

import argparse
import datetime
import json
import re
import sys

PREFILL_RE = re.compile(r"End-to-end \(prefill, per query\)\s+([\d.]+)\s*ms")
DECODE_RE = re.compile(r"End-to-end \(per token\)\s+([\d.]+)\s*ms")


def _find(pattern, text, label):
    m = pattern.search(text)
    if m is None:
        sys.exit(f"error: could not find '{label}' latency line in profile output")
    return float(m.group(1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        nargs="?",
        help="profile stdout file (default: read stdin)",
    )
    p.add_argument("--model", default="llama32_1b")
    p.add_argument("--runner", default="")
    p.add_argument("--mlir-air-sha", default="")
    p.add_argument("--mlir-aie-hash", default="")
    p.add_argument("--llvm-aie-version", default="")
    p.add_argument("--model-variant", default="")
    p.add_argument("--n-tokens", type=int, default=0)
    p.add_argument("--prompt", default="")
    args = p.parse_args()

    text = open(args.input).read() if args.input else sys.stdin.read()

    # First token is produced at the end of NPU prefill, so prefill end-to-end
    # latency is the time-to-first-token. Decode throughput is the reciprocal
    # of per-token latency.
    ttft_ms = _find(PREFILL_RE, text, "prefill")
    decode_ms_per_token = _find(DECODE_RE, text, "decode")

    data = {
        "model": args.model,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "runner": args.runner,
        "metrics": {
            "ttft_ms": ttft_ms,
            "decode_tokens_per_sec": round(1000.0 / decode_ms_per_token, 2),
        },
        "toolchain": {
            "mlir_air_sha": args.mlir_air_sha,
            "mlir_aie_hash": args.mlir_aie_hash,
            "llvm_aie_version": args.llvm_aie_version,
        },
        "run_params": {
            "model_variant": args.model_variant,
            "n_tokens": args.n_tokens,
            "prompt": args.prompt,
        },
    }
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
