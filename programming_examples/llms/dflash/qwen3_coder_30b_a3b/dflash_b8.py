# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Greedy DFlash with blocks of 8: the z-lab drafter proposes 7 tokens after
# the last emitted one (drafter), the NPU verify pass checks all 8 rows
# (verify_b8), and the longest agreeing prefix plus the target's next token
# are emitted. The drafter attends to the target's hidden states at its tap
# layers for every committed token.
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model_paths  # noqa: E402
import model_spec  # noqa: E402
import verify_b8 as V  # noqa: E402

W, TOK = V.W, V.TOK
EOS = model_spec.EOS_IDS


class _Taps(dict):
    """The verify pass's taps; each one's share of the drafter's context work
    is queued for a later dispatch's wait as it arrives."""

    def __init__(self, dr, v, start):
        super().__init__(dict.fromkeys(dr.taps))
        self.dr, self.v, self.start = dr, v, start

    def __setitem__(self, k, rows):
        super().__setitem__(k, rows)
        i, dr, p = self.dr.taps.index(k), self.dr, self.start
        self.v.work.append(lambda: dr.tap(i, rows))
        if i == len(dr.taps) - 1:
            self.v.work += [lambda: dr.finish(p, 0), lambda: dr.finish(p, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompt", default="Write a Python function that checks if a number is prime."
    )
    ap.add_argument("--case", help="a benchmark prompt, e.g. humaneval:0 (the n-th)")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--reads", default="256,512,1024,2048,4096")
    ap.add_argument("--root", default="builds", help="the verify pass's builds")
    ap.add_argument("--draft", type=Path, help="the z-lab drafter checkpoint")
    ap.add_argument(
        "--draft-build",
        type=Path,
        help="the drafter's build (build_target.py --role drafter); default "
        "<root>/drafter",
    )
    ap.add_argument(
        "--draft-weights",
        type=Path,
        default=model_paths.default_draft_cache(),
        help="the drafter's packed weights (prepare_coder_draft.py)",
    )
    ap.add_argument(
        "--check", action="store_true", help="against the reference's greedy"
    )
    a = ap.parse_args()
    import drafter
    import torch

    sys.path.insert(0, str(V.M.CODER))
    import reference as RF

    torch.set_num_threads(12)
    w = W.Qwen3CoderQ4k()
    tok = w.tokenizer()
    prompt = a.prompt
    if a.case:
        import json

        ds, k = a.case.split(":")
        cases = json.loads(
            (HERE.parent / "common" / "benchmark" / "prompts.json").read_text()
        )
        prompt = [c for c in cases["cases"] if c["dataset"] == ds][int(k)]["messages"][
            0
        ]["content"]
    text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text).ids
    n = len(ids)

    reads = [int(r) for r in a.reads.split(",")]
    # a block's 8 rows must fit the longest read
    limit = max(reads)
    if -(-n // TOK) * TOK > limit:
        raise SystemExit(f"a prompt of {n} tokens does not fit {limit} positions")
    v = V.VerifyB8(w, a.root, 4096, reads)
    dr = drafter.Drafter(
        w,
        model_paths.draft_snapshot(a.draft),
        a.draft_build or Path(a.root) / "drafter",
        a.draft_weights,
    )
    t0 = time.time()
    first, taps0 = v.prefill(ids, dr.taps)
    print(
        f"prefill (NPU verify pass, 8 at a time) {n} tokens: {time.time() - t0:.1f} s"
    )
    ctx = np.concatenate([taps0[l] for l in dr.taps], -1)
    out, start, p0, rounds = [first], n, 0, []
    times = dict.fromkeys(("draft", "verify", "head"), 0.0)
    t_all = time.perf_counter()
    stamps = []
    while len(out) < a.gen and out[-1] not in EOS and start + TOK <= limit:
        t = time.perf_counter()
        dr.context(ctx, p0)
        props = dr.propose(out[-1], start)
        times["draft"] += time.perf_counter() - t
        block = [out[-1]] + props
        t = time.perf_counter()
        taps = _Taps(dr, v, start)
        hl = v.forward(w.embed(block).astype(np.float32), start, taps=taps)
        times["verify"] += time.perf_counter() - t
        t = time.perf_counter()
        post = v.head(hl).tolist()
        times["head"] += time.perf_counter() - t
        acc = 0
        while acc < TOK - 1 and props[acc] == post[acc]:
            acc += 1
        emitted = props[:acc] + [post[acc]]
        # stop at the first EOS, accepted or not
        acc = next((i for i, t in enumerate(emitted) if t in EOS), acc)
        out += emitted[: acc + 1]
        rounds.append(acc + 1)
        stamps.append((time.perf_counter(), len(out)))
        ctx = np.concatenate([taps[l][: acc + 1] for l in dr.taps], -1)
        p0 = start
        start += acc + 1
    wall = time.perf_counter() - t_all
    complete = len(out) >= a.gen or out[-1] in EOS
    if not complete:
        print(f"stopped at position {start}: the next block passes {limit}")
    print(
        "[draft] "
        + ", ".join(
            f"{k} {v_ / max(1, len(rounds)) * 1e3:.1f}" for k, v_ in dr.times.items()
        )
        + " ms/round"
    )
    out = out[: a.gen]
    tau = sum(rounds) / max(1, len(rounds))
    print(tok.decode(out))
    print(
        f"[dflash] {len(out)} tokens in {len(rounds)} rounds, tau {tau:.2f}, "
        f"{wall / max(1, len(out) - 1) * 1e3:.1f} ms/token; "
        + ", ".join(
            f"{k} {s_ / max(1, len(rounds)) * 1e3:.1f}" for k, s_ in times.items()
        )
        + " ms/round"
    )
    if len(stamps) > 3:
        (t2, n2), (t9, n9) = stamps[1], stamps[-1]
        print(
            f"[dflash] after the first 2 rounds: {(t9 - t2) / (n9 - n2) * 1e3:.1f} ms/token"
        )
    ok = True
    if a.check:
        lm = torch.from_numpy(w.lm_head)
        fn = np.asarray(w.final_norm(), np.float32)

        def head(x):
            x = V.M.rmsnorm(np.asarray(x, np.float32), fn)
            return (torch.from_numpy(np.ascontiguousarray(x)) @ lm.T).numpy()

        ref = RF.Reference(w)
        lg = head(np.asarray(ref.forward(ids, lm=False)[-1:]))[0]
        g, margins = [int(lg.argmax())], [np.diff(np.sort(lg)[-2:])[0]]
        for _ in range(len(out) - 1):
            lg = head(np.asarray(ref.forward([g[-1]], lm=False)[-1:]))[0]
            g.append(int(lg.argmax()))
            margins.append(np.diff(np.sort(lg)[-2:])[0])
        same = next((i for i, (p, q) in enumerate(zip(out, g)) if p != q), None)
        print(f"reference greedy {'matches' if same is None else f'differs at {same}'}")
        if same is not None:
            # how close the reference's own top two were there
            print(
                f"  reference top-2 logit margin there {margins[same]:.4f}; "
                f"smallest before it {min(margins[:same], default=0):.4f}"
            )
        # a capture cut short by the cache is no check of --gen tokens
        ok = same is None and complete
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
