#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Weight traffic per DFlash iteration, read out of the builder.

Decode on NPU2 is weight-streaming bound, so a speculative decoding loop costs
what its two passes read from DDR:

    draft   5 drafter layers + LM head
    verify  36 target layers + LM head      (a batch of block_size tokens)
    ------------------------------------
    accept  tau tokens

This reports both against the plain one-token-per-call baseline, taking every
size from fused_decode.py rather than restating it (see llms/bench/decode_geometry.py
for the same idiom). tau is an input, not a prediction: it depends on the drafter
and on how much quantization moved the target away from what the drafter was
trained against, and neither is measured here.

KV traffic is excluded. It is unchanged per call, and a batched call reads it
ONCE for the whole block, so leaving it out understates DFlash -- increasingly
so at long context, where KV dominates. Weights are the conservative view.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BF16 = 2


def load(model, vocab_chunk_i2, ctx=2048, env_extra=None):
    for k in list(os.environ):
        if k.startswith("DECODE_") or k in (
            "VOCAB_CHUNK_I2",
            "LM_HEAD",
            "NLAYERS",
            "UNIFIED",
            "W_DUAL_CHAN",
        ):
            os.environ.pop(k, None)
    os.environ.update(
        DECODE_MODEL=model,
        VOCAB_CHUNK_I2=str(vocab_chunk_i2),
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        UNIFIED="1",
        DECODE_GOLDEN_L=str(ctx),
        W_DUAL_CHAN="1",
    )
    os.environ.update(env_extra or {})
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        "_fd_" + model.replace("-", "_").replace(".", "_"), HERE / "fused_decode.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def head_bytes(fd):
    """LM-head weight bytes: UNI_LM chunks of VOCAB_W_BLOCKS packed q4k blocks."""
    return fd.UNI_LM * fd.VOCAB_W_BLOCKS * fd.BLOCK_BF16 * BF16


def body_bytes(fd):
    return fd.UNI_DEC * fd.W_LAYER * BF16


def fc_bytes(fd, n_taps, q4nx=True):
    """DFlash's context-fusion linear: Linear(n_taps * hidden, hidden).

    Easy to leave out and not small. At n_taps=5 and hidden=2560 that is 32.8 M
    parameters -- a third of a whole Qwen3-4B layer -- so it is 6.5% of a
    5-layer q4nx drafter body, or 21% of one if it ships in bf16.

    It also decomposes for free: fc(concat(h1..h5)) == sum_i W_i @ h_i, so it is
    five accumulating hidden->hidden projections (I2=5, J2=5 on qwen3-4b).

    BOTH forms are expressible -- the undecomposed 12800-wide contraction gives
    J2 = 12800/512 = 25, a legal integer. The reason to decompose is L1, not
    legality: RCACHE_LEN is 2*max(J2P)*8, so a J2=25 phase would raise the
    per-core reduce cache from 304 to 400 elements, +3.0 KB at batch 16 on a
    core that already sits at 41.5 KB of 54. The decomposed form reuses a J2 the
    model already has and costs nothing.
    """
    params = n_taps * fd.K * fd.K
    return int(params * (0.625 if q4nx else 2.0))


def footprint(tgt, drf, n_taps, draft_q4nx, ctx):
    """Resident DDR a DFlash deployment needs, beyond a plain decode.

    Two things a traffic model does not show. First, DFlash runs TWO models, so
    both weight sets are resident at once. Second, both keep a KV cache, and the
    drafter's is a real allocation rather than a rounding error.

    KV is `ATTN_MAXL * KVSZ_TOK` bf16 per layer, both read straight out of the
    builder -- the same slab the decode already allocates, just twice over.
    """
    kv_layer = tgt.ATTN_MAXL * tgt.KVSZ_TOK * BF16
    return dict(
        ctx=tgt.ATTN_MAXL,
        t_w=body_bytes(tgt) + head_bytes(tgt),
        d_w=body_bytes(drf) * (1 if draft_q4nx else int(2.0 / 0.625))
        + fc_bytes(drf, n_taps, draft_q4nx),
        t_kv=tgt.UNI_DEC * kv_layer,
        d_kv=drf.UNI_DEC * kv_layer,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", default="qwen3-4b")
    ap.add_argument("--draft", default="qwen3-4b-draft")
    ap.add_argument("--vocab-chunk-i2", default="30")
    ap.add_argument(
        "--tau",
        type=float,
        default=6.0,
        help="accepted tokens per block (INPUT, not a prediction)",
    )
    ap.add_argument(
        "--bw",
        type=float,
        default=46.0,
        help="sustained weight-stream GB/s [measured 39-59 across models]",
    )
    ap.add_argument(
        "--draft-bf16",
        action="store_true",
        help="drafter body in bf16 as z-lab ships it, not q4nx",
    )
    ap.add_argument(
        "--taps",
        type=int,
        default=5,
        help="target_layer_ids the drafter fuses (5 for Qwen3-4B-DFlash-b16); "
        "sizes the fc linear",
    )
    ap.add_argument(
        "--draft-head-frac",
        type=float,
        default=1.0,
        help="fraction of the vocabulary the DRAFT pass computes logits over. "
        "The drafter only has to propose tokens, so a frequency-pruned head is "
        "an option; the target still scores the full vocabulary. Cannot break "
        "correctness -- a token outside the subset just never gets proposed -- "
        "but it lowers acceptance by an amount this does not model.",
    )
    args = ap.parse_args()

    tgt = load(args.target, args.vocab_chunk_i2)
    t_body, t_head = body_bytes(tgt), head_bytes(tgt)
    drf = load(args.draft, args.vocab_chunk_i2)
    d_body = body_bytes(drf)
    if args.draft_bf16:
        # q4nx is 0.625 B/param, bf16 is 2.0 -> 3.2x the bytes for the same weights.
        d_body = int(d_body * 2.0 / 0.625)
    d_fc = fc_bytes(drf, args.taps, q4nx=not args.draft_bf16)
    # Tied: the drafter uses the target's head, optionally only part of it.
    d_head = int(t_head * args.draft_head_frac)

    g = 1e9
    verify = t_body + t_head
    draft = d_body + d_fc + d_head
    per_iter = verify + draft
    ms = lambda b: b / g / args.bw * 1e3

    print(
        f"\nDFlash weight traffic  [target {args.target}, draft {args.draft}"
        f"{', bf16 body' if args.draft_bf16 else ''}]"
    )
    print(f"  sustained {args.bw} GB/s, tau = {args.tau} accepted tokens/block\n")
    print(f"  {'':22s}{'GB':>8}{'ms':>9}")
    print(f"  {'-'*39}")
    print(
        f"  {'target body (%2d layers)' % tgt.UNI_DEC:22s}"
        f"{t_body/g:8.3f}{ms(t_body):9.1f}"
    )
    print(f"  {'LM head':22s}{t_head/g:8.3f}{ms(t_head):9.1f}")
    print(f"  {'= verify pass':22s}{verify/g:8.3f}{ms(verify):9.1f}")
    print()
    print(
        f"  {'draft body (%2d layers)' % drf.UNI_DEC:22s}"
        f"{d_body/g:8.3f}{ms(d_body):9.1f}"
    )
    _hf = (
        "" if args.draft_head_frac == 1.0 else f", {args.draft_head_frac:.0%} of vocab"
    )
    print(f"  {'fc  (%d taps -> 1)' % args.taps:22s}{d_fc/g:8.3f}{ms(d_fc):9.1f}")
    print(f"  {'LM head (tied%s)' % _hf:22s}{d_head/g:8.3f}{ms(d_head):9.1f}")
    print(f"  {'= draft pass':22s}{draft/g:8.3f}{ms(draft):9.1f}")
    print()
    print(f"  {'per DFlash iteration':22s}{per_iter/g:8.3f}{ms(per_iter):9.1f}")
    print()

    base_ms = ms(verify)  # one token, one full target pass
    dflash_ms = ms(per_iter) / args.tau
    print(f"  baseline   {base_ms:6.1f} ms/token   ({1e3/base_ms:5.1f} tok/s)")
    print(
        f"  DFlash     {dflash_ms:6.1f} ms/token   ({1e3/dflash_ms:5.1f} tok/s)"
        f"   -> {base_ms/dflash_ms:.2f}x"
    )
    print()
    print(
        f"  drafter is {100*draft/per_iter:.0f}% of the iteration; "
        f"of that the LM head is {100*d_head/draft:.0f}%"
    )
    # tau at which the loop stops paying for itself.
    print(f"  break-even at tau = {per_iter/verify:.2f}")
    print()

    fp = footprint(tgt, drf, args.taps, not args.draft_bf16, args.vocab_chunk_i2)
    mb = lambda b: b / 2**20
    tot = fp["t_w"] + fp["d_w"] + fp["t_kv"] + fp["d_kv"]
    print(f"  resident DDR at context {fp['ctx']} (both models live at once):")
    print(f"    {'target weights':22s}{mb(fp['t_w']):8.0f} MB")
    print(f"    {'drafter weights + fc':22s}{mb(fp['d_w']):8.0f} MB")
    print(f"    {'target KV':22s}{mb(fp['t_kv']):8.0f} MB")
    print(f"    {'drafter KV':22s}{mb(fp['d_kv']):8.0f} MB")
    print(f"    {'total':22s}{mb(tot):8.0f} MB")
    print(
        f"    the drafter adds {100*(fp['d_w']+fp['d_kv'])/(fp['t_w']+fp['t_kv']):.0f}%"
        f" to what a plain decode already holds"
    )
    print()


if __name__ == "__main__":
    main()
