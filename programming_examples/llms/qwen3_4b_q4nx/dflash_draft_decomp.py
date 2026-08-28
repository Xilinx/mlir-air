#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The two claims that let the DFlash drafter run on the decode engine.

docs/DFlashFeasibility.md section 3.2 got the drafter's MASK onto the array.
What is still missing is the rest of what a drafter layer does that a target
layer does not (`_dflash_upstream/model.py:340-397`):

    q      = q_proj(hidden_states)                          B rows
    k, v   = k/v_proj(cat[target_hidden, hidden_states])    ctx + B rows
    target_hidden = hidden_norm(fc(taps))    12800 -> 2560, ONCE per block

Two structural claims make that fit the engine, and neither is obvious enough
to build on unchecked. This checks both against the real checkpoint.

CLAIM 1 -- fc DECOMPOSES ONTO THE EXISTING PROJECTION SHAPE. `fc` is a single
12800 -> 2560 linear over the 5 concatenated taps, a shape the engine has no
phase for. But a linear over a concatenation is a SUM of linears over the
parts:

    fc(cat[h1..h5]) = sum_i W[:, i*2560:(i+1)*2560] @ h_i

which is five 2560 -> 2560 projections accumulated -- exactly what the proj
cores already do across weight phases. If this holds, `fc` needs no new kernel,
only a new wave.

CLAIM 2 -- THE CONTEXT K/V IS LAYER-INVARIANT INPUT, so it can be computed
BEFORE the layer loop and written into the KV cache, after which the drafter is
an ordinary bidirectional decode. `target_hidden` never flows through the stack
and never sees `input_layernorm` (model.py:446-453 hands it to `self_attn`
raw), so every layer re-projects the SAME vector with its own k/v weights. If
that is right, all 5 layers' context K/V are computable up front from the taps
alone.

Checked by hooking `k_proj`, which each layer calls TWICE -- first on
`target_hidden`, then on `hidden_states`. The first call's INPUT must be
byte-identical across all five layers (that is claim 2), and must equal
`hidden_norm(fc(taps))` computed independently (that is claim 1 feeding it).

Nothing here reimplements attention. dflash_phase2_replay.py records why: a
hand-rolled attention loop was not bit-exact (max abs diff 1.25) and was
discarded rather than debugged. This only observes the real model.

    python3 dflash_draft_decomp.py
    python3 dflash_draft_decomp.py --ctx 8 --block 8
"""

import argparse
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "_dflash_upstream"

DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"


def _load_upstream(name):
    spec = importlib.util.spec_from_file_location(
        f"dflash_upstream_{name}", str(_UPSTREAM / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ctx",
        type=int,
        default=8,
        help="context rows (<= block after the first block)",
    )
    ap.add_argument("--block", type=int, default=8, help="DFlash block size")
    ap.add_argument(
        "--start", type=int, default=128, help="absolute position of block token 0"
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig

    m = _load_upstream("model")
    cfg = AutoConfig.from_pretrained(DRAFT_ID)
    print(
        f"[decomp] loading {DRAFT_ID} (fp32 on CPU, so the check measures the\n"
        f"         decomposition and not bf16 rounding)...",
        flush=True,
    )
    draft = m.DFlashDraftModel.from_pretrained(
        DRAFT_ID, config=cfg, attn_implementation="sdpa", dtype=torch.float32
    )
    draft.eval()

    H = draft.config.hidden_size
    ntap = len(draft.target_layer_ids)
    print(
        f"[decomp] hidden {H}, {ntap} taps -> fc {draft.fc.weight.shape[1]} -> "
        f"{draft.fc.weight.shape[0]}, {len(draft.layers)} layers"
    )
    print(
        f"[decomp] ctx {args.ctx}, block {args.block}, block token 0 at position {args.start}"
    )

    g = torch.Generator().manual_seed(0)
    taps = torch.randn(1, args.ctx, ntap * H, generator=g)
    noise = torch.randn(1, args.block, H, generator=g)
    # ctx occupies [start-ctx, start), the block [start, start+B) -- contiguous,
    # which is what model.py:246 slices and what lets the engine keep one
    # position-per-slot convention.
    pos = torch.arange(args.start - args.ctx, args.start + args.block)[None]

    bad = 0

    # ---- CLAIM 1: fc over a concatenation is a sum of per-tap projections ----
    W = draft.fc.weight  # [2560, 12800]
    with torch.no_grad():
        ref_fc = draft.fc(taps)
        acc = torch.zeros_like(ref_fc)
        for i in range(ntap):
            acc += taps[..., i * H : (i + 1) * H] @ W[:, i * H : (i + 1) * H].T
    err = (acc - ref_fc).abs().max().item()
    scale = ref_fc.abs().max().item()
    ok1 = err <= 1e-4 * max(scale, 1.0)
    bad += not ok1
    print(f"\n  CLAIM 1  fc(cat) == sum of {ntap} per-tap 2560->2560 projections")
    print(
        f"           max abs diff {err:.3e} against a max |fc| of {scale:.3f}"
        f"   {'OK' if ok1 else '<-- MISMATCH'}"
    )

    # ---- CLAIM 2: the context K/V input is the same for every layer ----
    seen = []  # (layer_idx, call_idx) -> input tensor

    def hook(mod, inp, out):
        seen.append(inp[0].detach().clone())

    handles = [l.self_attn.k_proj.register_forward_hook(hook) for l in draft.layers]
    with torch.no_grad():
        draft(
            position_ids=pos,
            noise_embedding=noise,
            target_hidden=taps,
            past_key_values=None,
            use_cache=False,
        )
    for h in handles:
        h.remove()

    nlayer = len(draft.layers)
    if len(seen) != 2 * nlayer:
        sys.exit(f"expected {2 * nlayer} k_proj calls, saw {len(seen)}")
    ctx_in = seen[0::2]  # first call of each layer: target_hidden
    blk_in = seen[1::2]  # second: that layer's normed hidden_states

    same = all(torch.equal(ctx_in[0], c) for c in ctx_in[1:])
    bad += not same
    print(f"\n  CLAIM 2  every layer's context input is the SAME tensor")
    print(
        f"           {nlayer} layers, all equal: {same}"
        f"   {'OK' if same else '<-- MISMATCH'}"
    )

    with torch.no_grad():
        mine = draft.hidden_norm(draft.fc(taps))
    err2 = (ctx_in[0] - mine).abs().max().item()
    ok2 = err2 == 0.0
    bad += not ok2
    print(
        f"           and equals hidden_norm(fc(taps)) computed outside the "
        f"loop: max abs diff {err2:.3e}   {'OK' if ok2 else '<-- MISMATCH'}"
    )

    # The contrast that makes claim 2 meaningful: hidden_states DOES evolve, so
    # a check that merely found "some input repeats" would prove nothing.
    moves = sum(1 for c in blk_in[1:] if not torch.equal(blk_in[0], c))
    print(
        f"           (control: the BLOCK input changes in {moves} of "
        f"{nlayer - 1} layer steps, as it must)"
    )
    bad += moves != nlayer - 1

    print(
        "\n"
        + (
            "BOTH CLAIMS HOLD -- fc is 5 accumulated 2560->2560 projections, and\n"
            "all 5 layers' context K/V can be built from the taps before the\n"
            "layer loop runs."
            if not bad
            else f"{bad} CHECK(S) FAILED -- the engine mapping in section 3.3 does not hold."
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
