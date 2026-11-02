# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CPU BF16 reference of the model gate: the original HF checkpoint the Q4NX
bundle was quantized from (Qwen/Qwen3.5-0.8B @ 2fc06364..., see README.md,
"Correctness"), greedy, 32 tokens per prompt, full logits.

  python3 reference.py --out OUT_DIR [--prompts FILE] [--tokens 32]

Writes OUT_DIR/reference.json (ids, chosen tokens, top-5 per step, library
versions) and OUT_DIR/ref_logits_<i>.npy (bf16 bits, [tokens, vocab]).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

HF_REPO = "Qwen/Qwen3.5-0.8B"
HF_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PROMPTS = Path(__file__).resolve().parents[1] / "verify" / "prompts" / "instruct.txt"


def load_prompts(path=PROMPTS):
    return [
        l.strip()
        for l in Path(path).read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HF_REPO, revision=HF_REVISION)


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument(
        "--timing-prompts",
        default="",
        help="comma list of P: the timing prompts of those buckets (bench.prompt_ids) "
        "instead of the prompts file",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    tok = tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        HF_REPO, revision=HF_REVISION, dtype=torch.bfloat16
    ).eval()
    recs = []
    if a.timing_prompts:
        from bench_prompts import prompt_ids

        items = [
            (f"timing prompt P={P}", prompt_ids(tok, int(P)))
            for P in a.timing_prompts.split(",")
        ]
    else:
        items = [(p, tok(p)["input_ids"]) for p in load_prompts(a.prompts)]
    for i, (p, ids) in enumerate(items):
        t0 = time.time()
        g = model.generate(
            torch.tensor([ids]),
            max_new_tokens=a.tokens,
            do_sample=False,
            min_new_tokens=a.tokens,
            output_logits=True,
            return_dict_in_generate=True,
            eos_token_id=None,
        )
        lg = torch.stack(g.logits, 0)[:, 0].float()  # [T, V]
        assert torch.isfinite(lg).all()
        np.save(
            out / f"ref_logits_{i}.npy", lg.to(torch.bfloat16).view(torch.int16).numpy()
        )
        toks = g.sequences[0, len(ids) :].tolist()
        recs.append(
            {
                "prompt": p,
                "ids": ids,
                "from_ids": bool(a.timing_prompts),
                "tokens": toks,
                "top5": lg.topk(5, -1).indices.tolist(),
                "text": tok.decode(toks),
                "seconds": time.time() - t0,
            }
        )
        print(i, len(ids), repr(recs[-1]["text"][:80]), flush=True)
    meta = {
        "model": HF_REPO,
        "revision": HF_REVISION,
        "dtype": "bfloat16",
        "device": "cpu",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "greedy_tokens": a.tokens,
        "eos_ignored": True,
        "prompts_file": str(a.prompts),
    }
    (out / "reference.json").write_text(
        json.dumps({"meta": meta, "prompts": recs}, indent=1)
    )


if __name__ == "__main__":
    main()
