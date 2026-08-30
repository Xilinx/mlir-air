#!/usr/bin/env python3
"""What does the DRAFTER's LM head cost, priced by deletion?

The drafter runs the same 151936-wide vocab projection the target does, over 5
layers instead of 36. In the loop that is 37.5 ms per block against the target's
157.5 -- 7.5 ms per drafter layer against 4.4 per target layer -- and the vocab
waves are the obvious candidate for the difference.

UNI_WAVE_LO/HI keep the ABI and the CDO fixed, so three instruction streams of
the same build dispatch against ONE xclbin:

    draft_b8_L511.insts.bin   waves [0,15)   the shipping stream
    _dnolm_L511.insts.bin     waves [0,5)    the decode layers alone
    _dlmonly_L511.insts.bin   waves [5,15)   the vocab waves alone

full - nolm IS the LM head, and lmonly is an independent read of the same
number: they must agree inside the ~1.5 ms run-to-run spread or the split is
not measuring what it claims.

All three go through `dispatch_insts`, which does no L patching -- so all three
run at the L the stream was compiled for, and the attention work they share
cancels in the difference.

WHAT IT FOUND, ctx 96, median of 7 `[hw]` (docs/DFlashFeasibility.md 3.17):
4.242 ms per drafter layer against the target's 4.375, so the drafter has no
per-layer anomaly at all -- the whole 7.5 ms/layer average was the LM head,
which is 13.8 ms, 39% of a drafter dispatch and 6.4% of a step.

AND THE REASON TO KEEP IT. `--batch 1` runs the same three ranges of the same
design against a batch-1 build. The vocab range is the smallest workload this
engine has -- a weight feed, a projection and an rms pass, with no attention, no
KV, no rope and no GLU -- and it still loses 1.90x going to batch 8 (7.278 ->
13.800 ms on identical weights; both `lmonly` streams are byte-for-byte 34576).
That makes it the minimal reproducer for the open batched number, and a much
better probe vehicle than a whole decode layer.
"""

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import argparse
import os

os.environ.setdefault("Q4NX_QWEN3_4B_DECODE_DIR", str(_HERE))


def _load_builder():
    """fused_decode at the DRAFTER's geometry -- the module the templates came
    from, so every extent here is the builder's arithmetic and not a copy."""
    import importlib.util

    keep = dict(os.environ)
    os.environ.update(
        DECODE_MODEL="qwen3-4b-draft",
        VOCAB_CHUNK_I2="30",
        W_DUAL_CHAN="1",
        FUSED_DECODE_EMIT_ONLY="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        UNIFIED="1",
        DECODE_BATCH="8",
        DECODE_GOLDEN_L="511",
        DECODE_MASK_BIDIR="1",
        DECODE_NO_LM_WAVES="0",
        # Without it the batch-8 load dies on the rms core's L1 ceiling before
        # W_LAYER is defined, and the only symptom is a missing attribute.
        DECODE_STACK="6080",
    )
    fdir = _HERE.parent.parent / "fused_decode"
    if str(fdir) not in sys.path:
        sys.path.insert(0, str(fdir))
    p = fdir / "fused_decode.py"
    s = importlib.util.spec_from_file_location("fd_geom_probe", str(p))
    m = importlib.util.module_from_spec(s)
    try:
        s.loader.exec_module(m)
    except SystemExit:
        pass
    os.environ.clear()
    os.environ.update(keep)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=511)
    ap.add_argument("--max-L", type=int, default=512)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument(
        "--batch",
        type=int,
        default=8,
        help="1 measures the SAME vocab dataflow at batch 1 -- the batched "
        "penalty in a workload with no attention, KV, rope or GLU in it",
    )
    args = ap.parse_args()

    import numpy as np

    import dflash_draft_decoder as DD
    import qwen3_4b_q4nx_inference as INF

    fd = _load_builder()

    dec = DD.build_draft_decoder(
        INF.MODEL_DEFAULT,
        max_L=args.max_L,
        batch=args.batch,
        template_prefix=f"draft_b{args.batch}_L",
    )
    B = dec.batch
    n = dec.UNI_DEC
    Z = np.zeros((n, args.ctx, dec.DK_TOT_A), np.float32)
    dec.seed_kv(Z, Z, args.ctx)

    sfx = "" if args.batch == 8 else f"_b{args.batch}"
    streams = {
        "full  [0,15)": f"draft_b{args.batch}_L{args.L}.insts.bin",
        "nolm  [0,5) ": f"_dnolm{sfx}_L{args.L}.insts.bin",
        "lmonly[5,15)": f"_dlmonly{sfx}_L{args.L}.insts.bin",
    }
    blobs = {}
    for k, f in streams.items():
        p = _HERE / f
        if not p.exists():
            raise SystemExit(
                f"missing {f} -- build it with _build_draft_split_streams.sh"
            )
        blobs[k] = np.fromfile(p, np.uint8)

    toks = DD.block_ids(9707, B, 151669) if B > 1 else 9707
    dec.dispatch(toks, args.ctx)  # warm the device program and the BOs

    # The builder's OWN byte accounting, not a restatement of it. BLOCK_BF16 is
    # an ELEMENT count (proj_qmm_pack.py:20; the weight memref is bf16), so a
    # packed q4k block is 2*BLOCK_BF16 bytes -- and W_LAYER is already in those
    # elements. Getting this wrong by 2x is what made the whole engine look like
    # it ran at 8 GB/s.
    MB = 2**20
    lay_mb = dec.UNI_DEC * fd.W_LAYER * 2 / MB
    voc_mb = fd.UNI_LM * fd.VOCAB_W_BLOCKS * fd.BLOCK_BF16 * 2 / MB
    wmb = {
        "full  [0,15)": lay_mb + voc_mb,
        "nolm  [0,5) ": lay_mb,
        "lmonly[5,15)": voc_mb,
    }

    print(f"\n  drafter, batch {B}, ctx {args.ctx}, median of {args.reps}")
    print(f"    stream          bytes   weight MB       ms     GB/s")
    med = {}
    for k, blob in blobs.items():
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            dec.dispatch_insts(blob)
            ts.append((time.perf_counter() - t0) * 1e3)
        med[k] = float(np.median(ts))
        gbs = wmb[k] * MB / 1e9 / (med[k] * 1e-3)
        print(
            f"    {k}  {blob.nbytes:7d}  {wmb[k]:9.1f}  {med[k]:7.3f}  {gbs:6.2f}"
            f"   (min {min(ts):.3f})"
        )

    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        dec.dispatch(toks, args.ctx)
        ts.append((time.perf_counter() - t0) * 1e3)
    print(
        f"    dispatch()       (host)  {float(np.median(ts)):7.3f}   (min {min(ts):.3f})"
    )

    lm_diff = med["full  [0,15)"] - med["nolm  [0,5) "]
    print(
        f"\n  LM head by deletion : {lm_diff:6.3f} ms"
        f"   ({100*lm_diff/med['full  [0,15)']:.0f}% of the drafter's dispatch)"
        f"\n  LM head standalone  : {med['lmonly[5,15)']:6.3f} ms"
        f"\n  5 decode layers     : {med['nolm  [0,5) ']:6.3f} ms"
        f"  ({med['nolm  [0,5) ']/5:.3f} ms/layer)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
