#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The accepted-length DISTRIBUTION, which is what decides the block size.

docs/DFlashFeasibility.md's section 6 reduces the whole block-size question to
one comparison: an extra verify slot costs a fixed number of milliseconds, so
slot *k* is worth building iff `P(produced >= k)` clears that cost expressed in
baseline token steps. Section 3 measured 1270 blocks and kept only the MEAN, and
a mean over 16 says nothing about the tail -- which is the entire thing a larger
block buys.

So this keeps the raw per-block lengths. Same models, same greedy settings, same
seeded dataset selection as `dflash_phase2_upstream_sweep_large.py`, via the
real unmodified upstream `dflash_generate`. CPU-only, no XRT.

What `acceptance_lengths` actually contains, since the name misleads: upstream
appends `produced = accepted + 1` -- the drafted tokens that matched, PLUS the
bonus token the verify pass emits for free. It is tokens per speculative step,
range 1..16 at block 16, never 0. That is the same quantity section 6's
break-even is stated in, so the two are directly comparable.

    python3 dflash_acceptance_hist.py --n 10                  # measure
    python3 dflash_acceptance_hist.py --analyze <json>        # re-read, no models

The measure pass appends to the JSON after every prompt, so it can be killed and
its partial result is still analyzable.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "_dflash_upstream"

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"
DEFAULT_OUT = _HERE / "dflash_acceptance_hist.json"

# Categories, and the upstream dataset each is drawn from. mt-bench is the
# open-ended chat axis -- section 3's chat number came from three hand-written
# prompts, this replaces them with the dataset upstream's own benchmark uses.
CATEGORIES = {
    "math": "gsm8k",
    "code": "humaneval",
    "chat": "mt-bench",
}

# Section 6, all [hw] on qwen3-4b, dispatch_time.py, median of 25, L=128.
BASELINE_MS = 56.946  # 36 layers + tied head, batch 1 -- one plain decode step
VERIFY_B1, VERIFY_B8 = 56.946, 177.511  # 36 layers + head
DRAFT_B1, DRAFT_B8 = 14.107, 40.412  # 5 layers + head (22.472 + 17.940 at b8)

MARG_VERIFY = (VERIFY_B8 - VERIFY_B1) / 7.0  # ms per extra verify slot
MARG_DRAFT = (DRAFT_B8 - DRAFT_B1) / 7.0  # ms per extra DRAFT slot


def _load_upstream(name):
    spec = importlib.util.spec_from_file_location(
        f"dflash_upstream_{name}", str(_UPSTREAM / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_ms(b):
    return VERIFY_B1 + MARG_VERIFY * (b - 1)


def draft_ms(b):
    return DRAFT_B1 + MARG_DRAFT * (b - 1)


def measure(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    m = _load_upstream("model")
    m._cuda_time = lambda: time.perf_counter()  # CPU-only
    b = _load_upstream("benchmark")  # real dataset loading + selection

    print(f"[hist] loading target {TARGET_ID} (bf16, CPU, sdpa)...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, attn_implementation="sdpa", dtype=torch.bfloat16
    )
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)

    print(f"[hist] loading draft {DRAFT_ID} via upstream model.py...", flush=True)
    config = AutoConfig.from_pretrained(DRAFT_ID)
    draft = m.DFlashDraftModel.from_pretrained(
        DRAFT_ID, config=config, attn_implementation="sdpa", dtype=torch.bfloat16
    )
    draft.eval()
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

    out = Path(args.out)
    state = {
        "config": {
            "target": TARGET_ID,
            "draft": DRAFT_ID,
            "block_size": args.block,
            "max_new_tokens": args.max_new_tokens,
            "n_per_dataset": args.n,
            "greedy": True,
            "enable_thinking": False,
            "quantity": "produced = accepted + bonus, per speculative step",
        },
        "runs": [],
    }
    if out.exists() and args.resume:
        state = json.loads(out.read_text())
        print(
            f"[hist] resuming, {len(state['runs'])} runs already recorded", flush=True
        )
    done = {(r["category"], r["index"]) for r in state["runs"]}

    for cat in args.categories:
        ds_name = CATEGORIES[cat]
        print(f"\n[hist] loading '{ds_name}' for category '{cat}'...", flush=True)
        sample = b._select_dataset(b.load_and_process_dataset(ds_name), args.n)
        print(
            f"[hist] {ds_name}: {len(sample)} prompts (seed 42, upstream order)",
            flush=True,
        )

        for i, instance in enumerate(sample):
            if (cat, i) in done:
                continue
            question = instance["turns"][0]
            messages = [{"role": "user", "content": question}]
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = torch.tensor(
                [tokenizer.encode(chat_text, add_special_tokens=False)],
                dtype=torch.long,
            )
            t0 = time.perf_counter()
            result = m.dflash_generate(
                draft,
                target,
                input_ids,
                args.max_new_tokens,
                stop_ids,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                block_size=args.block,
                return_stats=True,
            )
            lens = list(result.acceptance_lengths)
            state["runs"].append(
                {"category": cat, "dataset": ds_name, "index": i, "lengths": lens}
            )
            out.write_text(json.dumps(state))  # after EVERY prompt: killable
            mean = sum(lens) / len(lens) if lens else 0.0
            print(
                f"[hist] {cat} {i + 1}/{len(sample)}: {len(lens)} blocks, "
                f"mean={mean:.2f}/{args.block}, wall={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    return analyze(args)


def analyze(args):
    path = Path(args.analyze or args.out)
    if not path.exists():
        sys.exit(f"missing {path} -- run the measure pass first")
    state = json.loads(path.read_text())
    runs = state["runs"]
    if not runs:
        sys.exit(f"{path} has no runs yet")

    by_cat = {}
    for r in runs:
        by_cat.setdefault(r["category"], []).extend(r["lengths"])
    by_cat["ALL"] = [x for r in runs for x in r["lengths"]]

    cfg = state["config"]
    blk = cfg["block_size"]
    print(f"\n{'=' * 78}")
    print(f"accepted-length distribution  [{path.name}]")
    print(
        f"  {cfg['target']} + {cfg['draft']}, block {cfg['block_size']}, greedy, "
        f"thinking off  [cpu]"
    )
    print(f"  quantity: {cfg['quantity']}")
    print("=" * 78)

    n_runs = {c: sum(1 for r in runs if r["category"] == c) for c in by_cat}
    n_runs["ALL"] = len(runs)
    print(f"\n{'category':<10} {'prompts':>7} {'blocks':>7} {'mean/' + str(blk):>8}")
    for c, lens in by_cat.items():
        print(f"  {c:<8} {n_runs[c]:>7} {len(lens):>7} {sum(lens) / len(lens):>8.2f}")

    # The survival curve. P(produced >= k) is exactly the probability that verify
    # slot k emits a token, so it is the value side of the per-slot trade.
    full_marg = MARG_VERIFY + MARG_DRAFT
    thresh_v = MARG_VERIFY / BASELINE_MS
    thresh_f = full_marg / BASELINE_MS
    print(f"\nP(produced >= k) -- the chance verify slot k emits a token")
    print(
        f"  a slot pays iff P > {thresh_f:.3f}  ({full_marg:.2f} ms per slot, "
        f"verify {MARG_VERIFY:.2f} + draft {MARG_DRAFT:.2f}, against a "
        f"{BASELINE_MS:.2f} ms baseline step)"
    )
    hdr = "".join(f"{k:>6}" for k in range(1, blk + 1))
    print(f"\n  {'k':<8}{hdr}")
    for c, lens in by_cat.items():
        n = len(lens)
        row = "".join(
            f"{sum(1 for x in lens if x >= k) / n:>6.2f}" for k in range(1, blk + 1)
        )
        print(f"  {c:<8}{row}")
    print(
        f"  {'pays?':<8}"
        + "".join(
            f"{'y' if sum(1 for x in by_cat['ALL'] if x >= k) / len(by_cat['ALL']) > thresh_f else '.':>6}"
            for k in range(1, blk + 1)
        )
        + "   <- ALL"
    )

    # Throughput by block size. produced_B is approximated as min(produced_16, B):
    # the drafter's block-16 attention is non-causal over the whole block, so a
    # true block-B draft is not exactly the block-16 draft truncated. Stated, not
    # hidden -- it is the one modelled step in an otherwise measured chain.
    sizes = [b for b in (2, 4, 6, 8, 10, 12, 14, 16) if b <= blk]
    print(
        f"\nblock size, under produced_B ~= min(produced_{blk}, B)   [hw timings, cpu acceptance]"
    )
    print(
        f"  step(B) = verify({VERIFY_B1:.1f} + {MARG_VERIFY:.2f}(B-1)) + "
        f"draft({DRAFT_B1:.1f} + {MARG_DRAFT:.2f}(B-1))"
    )
    for c, lens in by_cat.items():
        print(f"\n  {c}:")
        print(
            f"    {'B':>3} {'tok/step':>9} {'step ms':>9} {'ms/tok':>8} {'tok/s':>7} "
            f"{'vs base':>8} {'P(>=B)':>7}"
        )
        best = None
        for bs in sizes:
            toks = sum(min(x, bs) for x in lens) / len(lens)
            step = verify_ms(bs) + draft_ms(bs)
            ms_tok = step / toks
            sp = BASELINE_MS / ms_tok
            p = sum(1 for x in lens if x >= bs) / len(lens)
            mark = ""
            if best is None or sp > best[1]:
                best, mark = (bs, sp), ""
            print(
                f"    {bs:>3} {toks:>9.2f} {step:>9.1f} {ms_tok:>8.2f} "
                f"{1000 / ms_tok:>7.2f} {sp:>7.2f}x {p:>7.2f}{mark}"
            )
        print(f"    best: B={best[0]} at {best[1]:.2f}x baseline")

    # And the constraint the checkpoint actually imposes: block_size is fixed at
    # 16 by the trained drafter, so a smaller verify block still pays the full
    # block-16 draft unless the drafter is re-trained.
    print(
        f"\n  if the draft stays at block {blk} (the checkpoint's fixed size) and only"
    )
    print(f"  verify shrinks -- draft cost pinned at {draft_ms(blk):.1f} ms:")
    print(f"    {'B':>3} " + " ".join(f"{c:>9}" for c in by_cat))
    for bs in sizes:
        cells = []
        for c, lens in by_cat.items():
            toks = sum(min(x, bs) for x in lens) / len(lens)
            step = verify_ms(bs) + draft_ms(blk)
            cells.append(f"{BASELINE_MS / (step / toks):>8.2f}x")
        print(f"    {bs:>3} " + " ".join(cells))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=10, help="prompts per dataset")
    ap.add_argument(
        "--block",
        type=int,
        default=16,
        help="verify block size. The checkpoint is TRAINED at 16; "
        "running it smaller measures whether a shorter block is "
        "usable at all, which truncating a block-16 run cannot say.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--resume", action="store_true", help="keep runs already in --out")
    ap.add_argument(
        "--analyze",
        nargs="?",
        const=str(DEFAULT_OUT),
        default=None,
        help="report on an existing JSON without loading any model",
    )
    args = ap.parse_args()
    return analyze(args) if args.analyze else measure(args)


if __name__ == "__main__":
    sys.exit(main())
