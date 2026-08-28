# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The DFlash drafter's DECODE half, driven by the target's own FusedDecoder.
#
# `qwen3-4b-draft` is qwen3-4b's per-layer geometry with UNI_DEC=5
# (fused_decode.py:427), so nothing in that driver is target-specific once the
# weight source, the requant cache and the template family are arguments -- see
# `FusedDecoder.__init__`'s `decode_model` / `weights` / `npz` / `artifact_dir`.
# This module is the drafter's half of that: the cache naming, the block
# arithmetic, and the KV seeding from the pre-pass.
#
# THE MASK. A draft pass is BIDIRECTIONAL -- every query in the block sees the
# whole block, not just its own past (docs/DFlashFeasibility.md section 3.2).
# The engine serves both from one program: `_tok_L(L, t)` is `L + t*S`, and with
# S=0 every token's context length is L + B - 1, constant in t, while the KV
# append slot stays (L-1)+t. So for a block of B tokens at positions
# ctx..ctx+B-1 with ctx context rows already in the cache:
#
#     RTP-L = ctx + 1     ->  every token sees ctx + B keys
#                             AND token t's K/V lands at slot (L-1)+t = ctx+t
#
# which is one value doing both jobs. `dispatch(toks, p)` takes p = L-1 = ctx.
#
# THE BLOCK'S TOKENS. Slot 0 is the token the target already committed; slots
# 1.. are the MASK TOKEN (_dflash_upstream/model.py:197 fills output_ids with
# mask_token_id before anything writes to them). The drafter is trained to
# predict from target_hidden plus those noise slots, which is why they are not
# left as whatever was there before. Only slots 1.. are predictions.

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


# The drafter's own requant cache. The name carries W_DUAL_CHAN for the same
# reason the target's does: that flag reorders the DDR weight cascade, so a
# cache packed under the other setting feeds the xclbin the wrong blocks and
# the only symptom is a drafter nobody accepts.
def draft_cache_path(dual=None, cache_dir=None):
    dual = int(os.environ.get("W_DUAL_CHAN", "1")) if dual is None else int(dual)
    return str(Path(cache_dir or _HERE) / f"_draft_q4nx{'_w2ch' if dual else ''}.npz")


def ensure_draft_cache(target_source, dual=None, cache_dir=None, draft_path=None):
    """Build the drafter's requant cache if it is not already there."""
    import importlib.util
    import sys

    path = draft_cache_path(dual, cache_dir)
    if os.path.exists(path):
        return path

    import qwen3_4b_draft_requant as rq

    fdir = _HERE.parent.parent / "fused_decode"
    for k in list(os.environ):
        if k.startswith("DECODE_"):
            os.environ.pop(k, None)
    os.environ.update(
        DECODE_MODEL="qwen3-4b-draft",
        VOCAB_CHUNK_I2="30",
        LM_HEAD="0",
        NLAYERS="1",
        UNIFIED="1",
        DECODE_GOLDEN="1",
        DECODE_GOLDEN_L="128",
    )
    if dual is not None:
        os.environ["W_DUAL_CHAN"] = str(int(dual))
    if str(fdir) not in sys.path:
        sys.path.insert(0, str(fdir))
    spec = importlib.util.spec_from_file_location(
        "fused_decode_draft_pack", str(fdir / "fused_decode.py")
    )
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    rq.build_draft_requant_cache(fd, path, target_source, draft_path=draft_path)
    return path


def build_draft_decoder(
    target_source,
    max_L,
    batch=8,
    stack="6080",
    template_prefix="draft_b8_L",
    artifact_dir=None,
    npz=None,
    draft_path=None,
    extra_env=None,
):
    """A `FusedDecoder` running the 5-layer drafter with a BIDIRECTIONAL mask.

    `template_prefix` is deliberately not `decode_b<B>_L`: the drafter's
    templates live in the same directory as the target's and are a different
    xclbin for a different model at the same context length, so sharing the name
    would make the scan pick whichever was built last.
    """
    import qwen3_4b_q4nx_inference as INF
    from qwen3_4b_draft_weights import DraftWeights

    env = {"DECODE_STACK": stack, "DECODE_MASK_BIDIR": "1"}
    env.update(extra_env or {})
    return INF.FusedDecoder(
        max_L=max_L,
        batch=batch,
        decode_model="qwen3-4b-draft",
        weights=DraftWeights(draft_path, target_source=target_source),
        npz=npz or ensure_draft_cache(target_source, draft_path=draft_path),
        template_prefix=template_prefix,
        artifact_dir=artifact_dir or _HERE,
        env_extra=env,
    )


def block_ids(known_token, block_size, mask_token_id):
    """[known, mask, mask, ...] -- the drafter's input for one block."""
    return [int(known_token)] + [int(mask_token_id)] * (block_size - 1)


def seed_context_kv(dec, k_ctx, v_ctx, ctx):
    """Write the pre-pass's per-layer context K/V into the drafter's KV cache.

    `k_ctx`/`v_ctx` are [n_layers, >=ctx, 1024] -- the pre-pass's output, K
    already normed and rotated. They land at slots 0..ctx-1, which is where the
    positions they were rotated for live.
    """
    import numpy as np

    n = dec.UNI_DEC
    fk = np.ascontiguousarray(np.asarray(k_ctx, np.float32)[:n, :ctx])
    fv = np.ascontiguousarray(np.asarray(v_ctx, np.float32)[:n, :ctx])
    assert fk.shape == (n, ctx, dec.DK_TOT_A), (fk.shape, n, ctx, dec.DK_TOT_A)
    dec.seed_kv(fk, fv, ctx)


def append_context_kv(dec, k_new, v_new, p0):
    """Write `k_new`/`v_new` ([n_layers, m, 1024]) at KV slots p0..p0+m-1.

    `seed_context_kv` is the round-0 form: it zeroes the cache and lays down the
    whole prefix. Every later round appends, because the drafter's cache is
    cropped to `start` rather than rebuilt (_dflash_upstream/model.py:249) --
    round n's context rows are positions `start_{n-1}..start_n-1`, which no
    round has written before. Writing only those keeps the whole prefix live at
    the cost of one m-row DMA per layer per group.

    Rows p0.. also happen to be where the PREVIOUS round's draft block appended
    its own K/V. That is not a hazard: the accepted prefix's rows are exactly
    the ones being overwritten here with the target's (better) values, and the
    rejected tail is overwritten by the next draft dispatch before anything
    reads it.
    """
    import numpy as np

    n, RW, NG = dec.UNI_DEC, dec.REGION_W, dec.NGRP
    m = int(np.asarray(k_new).shape[1])
    RS = dec.cur_maxl * RW
    assert p0 + m <= dec.cur_maxl, (p0, m, dec.cur_maxl)
    fk = np.asarray(k_new, np.float32)[:n]
    fv = np.asarray(v_new, np.float32)[:n]
    for Lyr in range(n):
        for gi in range(NG):
            for src, base in ((fk, gi), (fv, NG + gi)):
                off = base * RS + p0 * RW
                dec.KV[Lyr, off : off + m * RW].reshape(m, RW)[:] = src[
                    Lyr, :, gi * RW : (gi + 1) * RW
                ].astype(dec.bf16)
    TO = dec.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    _lreg = dec._geom.lreg(dec.cur_maxl)
    for Lyr in range(n):
        boff = Lyr * _lreg * 2
        dec.kvc.write(dec.KV[Lyr, :_lreg].view(np.int16), boff)
        dec.kvc.sync(TO, _lreg * 2, boff)
    dec._kv_dirty = False


def read_block_kv(dec, ctx, B):
    """The block's own K/V back out of the device KV cache: ([5,B,1024], same).

    The inverse of `FusedDecoder.seed_kv`'s region-major write: layer Lyr's
    group gi occupies `[gi*RS, gi*RS + maxl*RW)` for K and `[(NG+gi)*RS, ...)`
    for V, with RS = ATTN_MAXL * REGION_W. Reading it back is the only way to
    see what the engine actually appended -- a wrong append slot or a wrong
    block rope is invisible in the logits, which the mask and five layers of
    attention have already mixed.
    """
    import numpy as np

    xrt = dec.xrt
    FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    n, RW, NG = dec.UNI_DEC, dec.REGION_W, dec.NGRP
    maxl, lreg = dec.cur_maxl, dec._geom.lreg(dec.cur_maxl)
    dec.kvc.sync(FROM, n * lreg * 2, 0)
    buf = np.frombuffer(dec.kvc.map(), dtype=dec.bf16, count=n * lreg).reshape(n, lreg)
    RS = maxl * RW
    k = np.zeros((n, B, NG * RW), np.float32)
    v = np.zeros((n, B, NG * RW), np.float32)
    for L in range(n):
        for gi in range(NG):
            k[L, :, gi * RW : (gi + 1) * RW] = (
                buf[L, gi * RS : gi * RS + maxl * RW]
                .reshape(maxl, RW)[ctx : ctx + B]
                .astype(np.float32)
            )
            v[L, :, gi * RW : (gi + 1) * RW] = (
                buf[L, (NG + gi) * RS : (NG + gi) * RS + maxl * RW]
                .reshape(maxl, RW)[ctx : ctx + B]
                .astype(np.float32)
            )
    return k, v


def draft_block(dec, toks, ctx):
    """One bidirectional draft dispatch. Returns [B, VOCAB] logits.

    `ctx` is how many context rows are already in the KV cache; the block's B
    tokens are at positions ctx..ctx+B-1. RTP-L = ctx+1 is both the bidirectional
    key count and the append slot base (see the module docstring), and
    `dispatch` derives it as p+1.
    """
    return dec.dispatch(toks, ctx)
