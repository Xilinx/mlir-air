#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Image-turn latency at each image-token budget.

For each budget in --budgets: one image and a short prompt; one warm-up turn,
then --repeat timed turns with the vision tower, prefill and decoder loaded.
Records the median TTFT (vision tower + prefill), its two parts, and decode
tok/s over --n-tokens greedy tokens (EOS does not stop the decode, so every
turn times the same number of tokens).

Prints one `[bench] image_tokens=...` line per budget and writes
<--model>.image_sweep.json to the current directory.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gemma4_vision_inference as V  # noqa: E402
import gemma4_vision_reference as R  # noqa: E402

BUDGETS = (70, 140, 280, 560, 1120)
PROMPT = "Describe this image."


def measure(vlm, image, budget, repeat, n_tokens):
    pv, pid, ids = vlm.prompt(image, PROMPT, budget)
    n_image = sum(t == vlm.ids.image for t in ids)
    turns = []
    for r in range(repeat + 1):
        logits, t = vlm.prefill(pv, pid, ids)
        gen, t_dec, _ = vlm.generate(ids, logits, n_tokens, stop=False)
        if r == 0:
            continue  # warm-up: first use of this shape's ELFs
        turns.append(
            dict(
                vision_ms=t["vision"] * 1e3,
                prefill_ms=t["prefill"] * 1e3,
                ttft_ms=(t["vision"] + t["prefill"]) * 1e3,
                decode_tokens_per_sec=(len(gen) - 1) / t_dec,
            )
        )
    point = {k: float(np.median([x[k] for x in turns])) for k in turns[0]}
    point["min_ttft_ms"] = float(min(x["ttft_ms"] for x in turns))
    point.update(
        image_budget=budget,
        image_tokens=n_image,
        prompt_len=len(ids),
        n_tokens=n_tokens,
        status="ok",
    )
    return point


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gemma4_e2b_vision")
    ap.add_argument(
        "--budgets", default=",".join(map(str, BUDGETS)), help="soft-token budgets"
    )
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--n-tokens", type=int, default=64)
    ap.add_argument("--image", default=None, help="default: the shapes image")
    ap.add_argument("--fused-dir", default=str(V._TEXT / "_fused_prefill"))
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",") if b]
    if not budgets or args.repeat < 1 or args.n_tokens < 2:
        ap.error("need at least one budget, --repeat >= 1 and --n-tokens >= 2")

    image = R.load_image(args.image)
    vlm = V.VisionLanguageModel(args.fused_dir)
    points = []
    for budget in budgets:
        try:
            p = measure(vlm, image, budget, args.repeat, args.n_tokens)
        except Exception as e:  # one budget failing does not hide the others
            p = dict(image_budget=budget, status="error", detail=str(e)[:200])
            print(f"[bench] image_budget={budget} ERROR {e}", flush=True)
        else:
            print(
                f"[bench] image_budget={budget} image_tokens={p['image_tokens']} "
                f"prompt_len={p['prompt_len']} ttft_ms={p['ttft_ms']:.1f} "
                f"(vision {p['vision_ms']:.1f} + prefill {p['prefill_ms']:.1f}) "
                f"decode_tok_s={p['decode_tokens_per_sec']:.2f}",
                flush=True,
            )
        points.append(p)
    vlm.close()

    rec = dict(
        model=args.model,
        axis="image_tokens",
        metric="ttft_ms",
        prompt=PROMPT,
        image=args.image or "shapes",
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        points=points,
    )
    out = Path(f"{args.model}.image_sweep.json")
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {out}")
    return 0 if all(p["status"] == "ok" for p in points) else 1


if __name__ == "__main__":
    raise SystemExit(main())
