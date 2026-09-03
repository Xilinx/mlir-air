#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The acceptance rate with the DEVICE drafter in the loop.

This is the number the whole DFlash case rests on, and until now it has been an
upper bound. Section 3.1 measured the accepted-length distribution with a BF16
drafter on CPU and priced block 8 at 1.24x; the device drafter is an int4
pre-pass plus q4k decode layers and proposes different tokens, so 1.24x bounds
it rather than measures it.

WHAT IT RUNS. `dflash_loop.py`'s real loop, unchanged -- batch-8 taps verify,
24-launch pre-pass, bidirectional draft, greedy accept -- over the same prompt
sources section 3.1 used, so the two numbers are comparable. A different prompt
set would not be: the Paris prompt degenerates into "of the of the" by token 12
under plain greedy and gives a drafter nothing real to predict, which is why
`dflash_loop.py`'s own 1.409 prices nothing.

EVERY PROMPT IS PREFILLED BEFORE THE DEVICE IS OPENED. The numpy prefill holds
the whole dequantized model and each decoder then allocates a multi-GiB
host-only BO; holding the model across that dies during allocation with no
traceback. So DFlashLoop prefills the whole set, drops the model, and then
`select(i)` re-seeds the target's KV per prompt.

    python3 dflash_tokenize.py --dataset gsm8k --n 20 > prompts_gsm8k.json
    python3 dflash_acceptance_device.py --prompts prompts_gsm8k.json --n-tokens 48

`--exactness` additionally re-runs each prompt with the drafter off. With a
causal verify pass (section 3.9) the two token streams must AGREE except where
the top-2 margin is near zero, and that is the correctness gate on the loop --
not the acceptance number, which is a performance measurement.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Section 6, measured on device: a baseline batch-1 token step. The SPECULATIVE
# step is no longer a constant here -- the loop times its own target, draft and
# pre-pass, and this prices from that. It had to become a measurement: the
# pre-pass was a third PDI at a fixed 118.5 ms a block, and folding it onto the
# target's own program (dflash_prepass_waves.py) took it to a few ms, so a
# number written down in this file would now be wrong by a third of a step.
# `--spec-ms` still forces one, for pricing a hypothetical.
STEP_MS_SPEC = 217.9
STEP_MS_BASE = 56.9


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompts", help="JSON file: list of token-id lists")
    ap.add_argument("--dataset", choices=("gsm8k", "humaneval", "mtbench"))
    ap.add_argument("--n", type=int, default=10, help="prompts")
    ap.add_argument("--n-tokens", type=int, default=48)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--max-L", type=int, default=512)
    ap.add_argument("--stack", default="6080")
    ap.add_argument("--target-prefix", default="taps_b8_L")
    ap.add_argument("--draft-prefix", default="draft_b8_L")
    ap.add_argument("--exactness", action="store_true")
    # A CONTROL for the acceptance number, not a mode anyone should ship.
    # "cpu" runs the pre-pass in numpy off the same q4k bytes the waves
    # stream, so a difference between the two runs is the wave arithmetic and
    # nothing else. See dflash_prepass_waves.CpuPrepass.
    # "hybrid" keeps fc on the device (it rides the verify pass's tail) and
    # takes only the two ALTERNATE STREAMS to numpy. That is what makes
    # RMS_MEMTILE_REFEED=3 usable with the fold: a narrow UNI_WAVE range lowers
    # to a different device under mode 3, so those streams' insts.bin address
    # hardware the verify xclbin does not have. See HybridPrepass.
    ap.add_argument(
        "--prepass", choices=("waves", "cpu", "hybrid", "draft"), default="waves"
    )
    ap.add_argument("--out", default=None, help="write the raw per-block lengths")
    ap.add_argument("--spec-ms", type=float, default=STEP_MS_SPEC)
    ap.add_argument("--base-ms", type=float, default=STEP_MS_BASE)
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--target-env",
        action="append",
        default=[],
        help="K=V the TARGET template was built with (repeatable), e.g. "
        "--target-env VOCAB_CHUNK_I2=50 --target-env UNI_LM=6",
    )
    args = ap.parse_args()

    import numpy as np

    import dflash_loop as DL
    import qwen3_4b_q4nx_inference as INF

    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text())
    elif args.dataset:
        # Out of process: transformers and XRT cannot share one.
        prompts = json.loads(
            subprocess.run(
                [
                    sys.executable,
                    str(_HERE / "dflash_tokenize.py"),
                    "--dataset",
                    args.dataset,
                    "--n",
                    str(args.n),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    else:
        ap.error("need --prompts or --dataset")
    prompts = [list(p) for p in prompts][: args.n]
    print(
        f"[acceptance] {len(prompts)} prompts, lengths "
        f"{min(len(p) for p in prompts)}..{max(len(p) for p in prompts)}, "
        f"{args.n_tokens} tokens each, block {args.block}",
        flush=True,
    )

    loop = DL.DFlashLoop(
        args.model or INF.MODEL_DEFAULT,
        prompts,
        block=args.block,
        max_L=args.max_L,
        stack=args.stack,
        target_prefix=args.target_prefix,
        draft_prefix=args.draft_prefix,
        speculate=True,
        prepass=args.prepass,
        target_env=dict(kv.split("=", 1) for kv in args.target_env),
    )

    all_lens, rows, mismatches = [], [], 0
    t_step, n_step = 0.0, 0
    t_parts = {}
    for i in range(loop.n_prompts):
        loop.select(i)
        toks, acc, _t = loop.run(args.n_tokens, verbose=False)
        gen = toks[loop.P :]
        all_lens.extend(acc)
        # The step, MEASURED, rather than the constant below. The pre-pass used
        # to be a third PDI whose cost was fixed and enormous, so a number in
        # this file was as good as any; folded onto the target's own program it
        # is a few ms, and the step is now dominated by parts this loop times
        # itself. Prompt 0 is dropped -- its block 0 pre-pass covers the whole
        # prompt (ceil(P/B) dispatches, not one) and every device buffer is cold.
        if i:
            t_step += sum(_t.values())
            n_step += len(acc)
            # PER-COMPONENT, because the summed step hides the one number a
            # target-side optimization moves. RMS_MEMTILE_REFEED=3 acts on the
            # verify dispatch only; against a CPU pre-pass that dominates the
            # step, a 20% target win is a 6% step win and reads as noise.
            for _k, _v in _t.items():
                t_parts[_k] = t_parts.get(_k, 0.0) + _v
        note = ""
        if args.exactness:
            loop.select(i)
            ref, _, _ = loop.run(args.n_tokens, speculate=False, verbose=False)
            rgen = ref[loop.P :]
            k = min(len(gen), len(rgen))
            same = next((j for j in range(k) if gen[j] != rgen[j]), k)
            if same < k:
                mismatches += 1
            note = f"   exact for {same}/{k}"
        rows.append(dict(prompt_len=loop.P, produced=len(gen), acc=acc))
        print(
            f"  prompt {i:3d} (len {loop.P:4d}): {len(acc):3d} blocks, "
            f"{len(gen):3d} tokens, mean {np.mean(acc):.3f}{note}",
            flush=True,
        )

    a = np.array(all_lens, float)
    mean = float(a.mean())
    # THE MEAN IS A SAMPLE, AND IT MOVES BETWEEN RUNS OF THE SAME BINARIES.
    # A batch-8 dispatch is not reproducible on this machine (README section 4)
    # and this loop takes exactly one per verify block -- no retry, unlike the
    # verify gate -- so a flipped argmax changes where a block stops and the
    # whole downstream block structure with it. Measured 2026-09-02, two runs of
    # ONE set of templates: 3.529 over 119 blocks and 3.534 over 116, differing
    # on 3 of 12 prompts and agreeing exactly on the other 9.
    #
    # So print the standard error next to it. Without it a reader compares
    # 3.529 against a 3.614 written down last week and calls it a regression --
    # which is exactly what happened, and cost a device session. Both are within
    # half a standard error of each other. Blocks inside one prompt are
    # correlated, so this UNDERSTATES the true spread; treat it as a floor.
    sem = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")
    spec_ms = 1e3 * t_step / n_step if n_step else args.spec_ms
    src = "MEASURED here" if n_step else "section 6"
    speed = mean * args.base_ms / spec_ms
    # EVERY SUMMARY LINE FLUSHES, and that is not decoration. Tearing down the
    # device contexts at interpreter exit can segfault, and stdout is
    # block-buffered whenever this is redirected to a file -- so a run that
    # completed every prompt lost its entire summary and kept all twelve
    # per-prompt lines, which carry flush=True. It reads as "the run died
    # early". It did not; only the buffer did. Measured 2026-09-02 on the
    # 12-prompt gsm8k gate, where the mean had to be recomputed by hand from
    # the surviving lines.
    print(f"\n[acceptance] {len(a)} blocks over {loop.n_prompts} prompts", flush=True)
    print(
        "  accepted-length histogram (tokens committed per verify dispatch):",
        flush=True,
    )
    for k in range(1, args.block + 1):
        c = int((a == k).sum())
        if c:
            print(
                f"    {k:2d}: {c:5d}  {c / len(a):6.1%}  {'#' * int(60 * c / len(a))}",
                flush=True,
            )
    print(
        f"\n  mean tokens per verify dispatch : {mean:.3f} +/- {sem:.3f}"
        f"  (1 s.e.; a FLOOR -- see the comment above)\n"
        f"  speculative step                : {spec_ms:.1f} ms  ({src})\n"
        f"  break-even                      : {spec_ms / args.base_ms:.2f}\n"
        f"  SPEEDUP vs batch-1 decode       : {speed:.2f}x"
        f"   ({'worth it' if speed > 1 else 'NOT worth it'})\n"
        f"  (against a {args.base_ms:.1f} ms baseline token; the step is the "
        f"loop's own target + draft + pre-pass over {n_step} blocks)\n"
        f"  DO NOT read a difference of less than ~{2 * sem:.2f} tokens against "
        f"a previously recorded mean as a change.",
        flush=True,
    )
    if t_parts:
        print("\n  step breakdown, ms per verify block:", flush=True)
        for _k in sorted(t_parts, key=lambda k: -t_parts[k]):
            print(
                f"    {_k:>8s} : {1e3 * t_parts[_k] / n_step:8.2f}"
                f"  ({t_parts[_k] / t_step:5.1%})",
                flush=True,
            )
    if args.exactness:
        print(
            f"  exactness: {loop.n_prompts - mismatches}/{loop.n_prompts} prompts "
            f"reproduce the non-speculative stream",
            flush=True,
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps(dict(rows=rows, mean=mean, speedup=speed), indent=1)
        )
        print(f"  raw -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
