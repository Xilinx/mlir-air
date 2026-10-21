# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixed-prompt profile of NPU prefill and the DFlash decode loop."""

import argparse
import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from model_paths import load_tokenizer
from provenance import runtime_identity
from run import run_prompt
from runtime import load_pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="qwen3_8b_dflash_b8")
    p.add_argument("--prompt", default="Introduce me what is GPU")
    p.add_argument("--tokens", type=int, default=256)
    a = p.parse_args()
    tokenizer = load_tokenizer()
    loop, prefill = load_pipeline(
        a.build / "target",
        a.build / "draft",
        prepass_artifacts=a.build / "prepass",
        prefill_artifacts=a.build / "prefill_C2048",
    )
    # Both measurements call the function run/chat use; warmup is excluded.
    run_prompt(
        loop,
        tokenizer,
        a.prompt,
        SimpleNamespace(tokens=17, prefill="npu", no_stream=True),
        npu_prefill=prefill,
    )
    row = run_prompt(
        loop,
        tokenizer,
        a.prompt,
        SimpleNamespace(tokens=a.tokens, prefill="npu", no_stream=True),
        npu_prefill=prefill,
    )
    if row["decode_tokens"] <= 0 or row["decode_seconds"] <= 0:
        raise RuntimeError("Profile produced no timed decode tokens")
    result = dict(
        model=a.model,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        runner="",
        verify_status="",
        toolchain={},
        metrics=dict(
            ttft_ms=1000 * row["time_to_first_token_seconds"],
            decode_tokens_per_sec=row["decode_tps"],
            context_len=row["input_tokens"],
        ),
        run_params=dict(
            prompt=a.prompt,
            n_tokens=a.tokens,
            batch=8,
            thinking=False,
            protocol="dflash-npu-prefill-profile-v1",
        ),
        runtime=runtime_identity(),
        dflash=row,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
