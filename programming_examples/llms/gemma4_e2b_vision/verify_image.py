#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Image-turn model gate: NPU vs HF bf16 Gemma4ForConditionalGeneration.

The shared verify runner's token-level method, on image prompts: both sides
generate N greedy tokens without stopping at EOS, and
verify.comparators.compute_topk_set_check compares the chosen tokens and checks
mutual top-k inclusion at the first divergence.

The HF reference builds its inputs with Gemma4Processor, independently of this
example's prompt builder, and the two prompts are asserted identical. It runs
on the CPU in bf16 and is cached on disk, keyed on its inputs.

    python3 verify_image.py                 # every case in CASES
    python3 verify_image.py --reference-only
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gemma4_vision_inference as V  # noqa: E402
import gemma4_vision_reference as R  # noqa: E402
from gemma4_vision_weights import (  # noqa: E402
    VisionWeights,
    checkpoint_identity,
    preprocess,
)

# (name, image, max soft tokens). The 640x480 image's patches do not fill whole
# attention key blocks, so it covers the padding-key path.
CASES = [
    ("shapes-280", None, 280),
    ("shapes640x480-70", (640, 480), 70),
]
_REF_DIR = _HERE / "_oracle_cache"


def case_image(size):
    img = R.make_test_image()
    return img.resize(size) if size else img


def hf_reference(image, prompt, mst, n_tokens, k, expect_ids):
    """-> (chosen [n], topk [n][k]) from HF bf16 on the CPU."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from air_examples.llms.verify.comparators import topk_token_ids

    path = VisionWeights().path.parent
    proc = AutoProcessor.from_pretrained(str(path))
    text = proc.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = proc(
        text=text,
        images=[image],
        return_tensors="pt",
        images_kwargs={"max_soft_tokens": mst},
    )
    ids = inputs["input_ids"][0].tolist()
    assert ids == expect_ids, (
        "HF's processor and build_prompt_ids disagree on the prompt",
        len(ids),
        len(expect_ids),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(path), dtype=torch.bfloat16
    ).eval()
    chosen, tops = [], []
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        for step in range(n_tokens):
            z = out.logits[0, -1].float().numpy()
            t = int(z.argmax())
            chosen.append(t)
            tops.append(topk_token_ids(z, k))
            if step + 1 == n_tokens:
                break
            out = model(
                input_ids=torch.tensor([[t]]),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    return chosen, tops


def cached_reference(name, image, prompt, mst, n_tokens, k, expect_ids):
    h = hashlib.sha256(
        json.dumps([expect_ids, prompt, n_tokens, k, checkpoint_identity()]).encode()
    )
    h.update(np.asarray(image, np.uint8).tobytes())
    path = _REF_DIR / f"hf_{name}_{h.hexdigest()[:12]}.json"
    if path.exists():
        return json.loads(path.read_text())
    chosen, tops = hf_reference(image, prompt, mst, n_tokens, k, expect_ids)
    _REF_DIR.mkdir(exist_ok=True)
    rec = dict(chosen=chosen, topk=tops)
    tmp = path.with_suffix(f".{os.getpid()}.json")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, path)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", default=V.DEFAULT_PROMPT)
    ap.add_argument("--n-tokens", type=int, default=32)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cases", default=",".join(c[0] for c in CASES))
    ap.add_argument("--fused-dir", default=str(V._TEXT / "_fused_prefill"))
    ap.add_argument("--reference-only", action="store_true")
    args = ap.parse_args()
    if args.n_tokens < 1:
        ap.error("--n-tokens must be at least 1")

    from air_examples.llms.verify.comparators import compute_topk_set_check

    tok, ids_cfg = V.load_tokenizer()
    wanted = [c for c in args.cases.split(",") if c]
    unknown = sorted(set(wanted) - {c[0] for c in CASES})
    if not wanted or unknown:
        ap.error(
            f"--cases must name cases from {[c[0] for c in CASES]}"
            + (f"; unknown: {unknown}" if unknown else "")
        )
    cases = [c for c in CASES if c[0] in wanted]
    plans = []
    for name, size, mst in cases:
        image = case_image(size)
        _, _, n_soft = preprocess(image, max_soft_tokens=mst)
        ids = V.build_prompt_ids(tok, ids_cfg, args.prompt, n_soft)
        ref = cached_reference(
            name, image, args.prompt, mst, args.n_tokens, args.topk, ids
        )
        print(f"[{name}] HF reference: {tok.decode(ref['chosen'])!r}", flush=True)
        plans.append((name, image, mst, ids, ref))
    if args.reference_only:
        return 0

    vlm = V.VisionLanguageModel(args.fused_dir)
    fails = 0
    for i, (name, image, mst, ids, ref) in enumerate(plans):
        pv, pid, ids2 = vlm.prompt(image, args.prompt, mst)
        assert ids2 == ids
        logits, _ = vlm.prefill(pv, pid, ids)
        gen, _, tops = vlm.generate(
            ids, logits, args.n_tokens, stop=False, topk=args.topk
        )
        n = args.n_tokens
        lens = [len(gen), len(tops), len(ref["chosen"]), len(ref["topk"])]
        if lens != [n] * 4:
            print(
                f"[{name}] FAIL: expected {n} tokens and top-k lists on both "
                f"sides, got {lens} (NPU tokens, NPU top-k, HF tokens, HF top-k)"
            )
            fails += 1
            continue
        rec = compute_topk_set_check(
            gen, tops, ref["chosen"], ref["topk"], k=args.topk, prompt_idx=i,
            prompt_text=name,
        )  # fmt: skip
        print(f"[{name}] NPU:          {tok.decode(gen)!r}")
        div = rec.divergence_step
        print(
            f"[{name}] {rec.status}: {rec.n_steps} steps compared, "
            + ("no divergence" if div is None else f"first divergence at step {div}")
            + (f" ({rec.fail_reason})" if rec.fail_reason else "")
        )
        fails += rec.status != "OK"
    vlm.close()
    print("PASS!" if not fails else f"FAIL ({fails} of {len(plans)} cases)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
