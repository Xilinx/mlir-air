#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""L1 budget for running the decode superkernel at batch > 1.

Speculative decoding (DFlash) needs the superkernel to process a block of tokens
per call instead of one. Every ACTIVATION buffer then scales with the batch while
every WEIGHT buffer stays put, so the question for each tile is simply: what is
the largest batch that still fits 64 KB, and where does the batch have to be
tiled instead?

It reads the real buffer shapes out of fused_decode.py for the selected model
rather than restating them, the same way llms/bench/decode_geometry.py reads BO
sizes -- a restated shape is a shape that silently goes stale.

SCOPE. Two tile roles are reported exactly, because their buffer sets are
unambiguous in the builder:

  proj core       xblk, wblk, yacc, rcache, ypair
  attention CU    aq, ak, av, as, ao

The rms / rope / glu buffers are listed individually but NOT summed into a tile
total: they sit on different cores (RMS_PCOL, the rope tile, GLU_PCOL) and this
script does not model that assignment. Use them to see which buffers need row
tiling, not as a per-tile budget.

Nothing here is a cycle count -- see bench_q4k_mm.py for the compute side.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L1_BYTES = 64 * 1024
BF16 = 2
F32 = 4


def load_builder(model, vocab_chunk_i2, ctx=2048, env_extra=None):
    """Import fused_decode.py configured for `model` and hand back the module."""
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
    spec = importlib.util.spec_from_file_location("_fd_l1", HERE / "fused_decode.py")
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    return fd


def proj_core(fd, batch, scratch_rows=0, scratch_cols=0):
    """[(name, bytes, scales_with_batch)] for one projection core.

    scratch_rows/cols size the UNPACKED-WEIGHT buffer that kernels/q4k_mm.h
    needs and today's GEMV does not: `q4k_unpack_block` materialises a
    rows x cols bf16 tile before `aie::mmul` can touch it. Pass 0 for today's
    kernel. It does not scale with the batch -- that is the whole point of
    unpacking once -- but it is the largest single buffer on the core once it
    exists, so it decides the batch ceiling rather than the activations do.

    The tile need not be the whole 32x256 block that bench_q4k_mm.py measures:
    the mmul contracts over the columns, so it can be chunked and accumulated at
    the same total unpack cost. Sweep --scratch-cols to price that.
    """
    rows = [
        ("xblk   activation chunk", fd.COL_BLOCK * BF16 * batch, True),
        ("wblk   weight block", fd.BLOCK_BF16 * BF16, False),
        ("yacc   accumulator", fd.ROW_BLOCK * F32 * batch, True),
        ("rcache reduce cache", fd.RCACHE_LEN * BF16 * batch, True),
        # Only the payload widens; the 16-element wire header does not, so it is
        # a separate fixed row rather than a term inside a scaling one.
        ("ypair  wire header", 16 * BF16, False),
        ("ypair  egress payload", fd.PAIR_ROWS * fd.ROW_BLOCK * BF16 * batch, True),
    ]
    if scratch_rows and scratch_cols:
        rows.append(
            (
                f"wscr   unpacked {scratch_rows}x{scratch_cols}",
                scratch_rows * scratch_cols * BF16,
                False,
            )
        )
    return rows


def attn_cu(fd, qtile):
    """[(name, bytes, scales_with_qtile)] for one attention compute unit.

    ak/av hold the KV BLOCK, which every query in the tile shares -- they do not
    scale. That is the whole reason batching attention is cheap: the KV read is
    amortised over the tile.
    """
    return [
        ("aq  query", fd.DQ_PADDED_PER_CU * BF16 * qtile, True),
        ("ak  k block", 16 * fd.KVPC_DH * BF16, False),
        ("av  v block", 16 * fd.KVPC_DH * BF16, False),
        ("as  scores", fd.SSZ_BLK * BF16 * qtile, True),
        ("ao  out", fd.DQ_PADDED_PER_CU * BF16 * qtile, True),
    ]


def other_buffers(fd, batch):
    """rms / rope / glu buffers. Reported individually, not summed (see module docstring)."""
    return [
        ("rms_l1     activation", fd.K * BF16 * batch, True),
        ("rms_w2k_l1 norm weights", 2 * fd.K * BF16, False),
        ("qkv_l1     qkv out", fd.M * BF16 * batch, True),
        ("ropeq_l1   roped q", fd.DQ_PADDED * BF16 * batch, True),
        ("ropekv_l1  roped kv", fd.DK * BF16 * batch, True),
        ("ropelut_l1 cos/sin", fd.ROPE_W_LEN * BF16 * batch, True),
        ("glu_x_l1   up|gate", fd.GLU_SLICE * BF16 * batch, True),
        ("glu_hid_l1 silu*up", fd.GLU_HID * BF16 * batch, True),
    ]


def _largest_fit(fixed, per_unit, budget):
    """Largest n with fixed + n*per_unit <= budget (0 if even n=1 fails)."""
    if per_unit <= 0:
        return float("inf")
    return max(0, (budget - fixed) // per_unit)


def report(fd, model, batch, stack, verbose, scratch_rows=0, scratch_cols=0):
    budget = L1_BYTES - stack
    print(
        f"\n=== {model}  batch {batch}  "
        f"(L1 {L1_BYTES//1024} KB - {stack//1024} KB stack = {budget//1024} KB)"
    )
    print(
        f"    K={fd.K} M={fd.M} DH={fd.DH} DQ_PADDED_PER_CU={fd.DQ_PADDED_PER_CU} "
        f"KVPC_DH={fd.KVPC_DH} SSZ_BLK={fd.SSZ_BLK}"
    )

    for title, rows, unit in (
        ("proj core", proj_core(fd, batch, scratch_rows, scratch_cols), "batch"),
        ("attention CU", attn_cu(fd, batch), "query tile"),
    ):
        fixed = sum(b for _, b, s in rows if not s)
        per = sum(b for _, b, s in rows if s) // batch
        tot = fixed + per * batch
        fits = "FITS" if tot <= budget else "OVER"
        print(f"\n  {title}: {tot/1024:6.1f} KB at {unit} {batch}   [{fits}]")
        if verbose:
            for n, b, s in rows:
                print(f"      {n:26s} {b/1024:7.2f} KB  {'x batch' if s else 'fixed'}")
        cap = _largest_fit(fixed, per, budget)
        print(
            f"      fixed {fixed/1024:.1f} KB + {per/1024:.2f} KB per {unit}"
            f"  ->  max {unit} that fits: {cap}"
        )

    print("\n  rms / rope / glu buffers (per buffer; different cores, not summed):")
    for n, b, s in other_buffers(fd, batch):
        if s:
            cap = _largest_fit(0, b // batch, budget)
            print(
                f"      {n:26s} {b/1024:8.2f} KB at batch {batch}"
                f"   max batch alone: {cap}"
            )
        else:
            print(f"      {n:26s} {b/1024:8.2f} KB   fixed")


# (model, VOCAB_CHUNK_I2, extra builder env) -- the pairs fused_decode.py asserts.
MODELS = {
    "llama-3.2-1b": ("18", None),
    "qwen3-4b": ("30", None),
    "llama-3.2-3b": ("9", None),
    "qwen3-8b": ("8", {"DECODE_STACK": "6144", "DECODE_WGROUP": "9"}),
    "gemma3-4b": ("5", None),
    "phi4-mini": ("18", None),
    "qwen2.5-7b": ("7", None),
    "llama-3.1-8b": ("16", {"DECODE_STACK": "6144", "DECODE_WGROUP": "8"}),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.2-1b", choices=sorted(MODELS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--stack",
        type=int,
        default=10240,
        help="DECODE_STACK bytes carved out of the same 64 KB",
    )
    ap.add_argument(
        "--scratch-rows",
        type=int,
        default=0,
        help="unpacked-weight tile rows for kernels/q4k_mm.h (0 = today's GEMV, "
        "which needs none; 32 = the shape bench_q4k_mm.py measures)",
    )
    ap.add_argument(
        "--scratch-cols",
        type=int,
        default=0,
        help="unpacked-weight tile columns; chunking the contraction shrinks this "
        "at no extra unpack cost",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="per-buffer detail")
    args = ap.parse_args()

    i2, extra = MODELS[args.model]
    fd = load_builder(args.model, i2, env_extra=extra)
    if extra and "DECODE_STACK" in extra:
        args.stack = int(extra["DECODE_STACK"])
    report(
        fd,
        args.model,
        args.batch,
        args.stack,
        args.verbose,
        args.scratch_rows,
        args.scratch_cols,
    )
    print()


if __name__ == "__main__":
    main()
