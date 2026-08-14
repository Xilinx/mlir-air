#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared machinery for the decode staircase: run each token on the smallest KV window.

The compiled KV readback streams ATTN_MAXL positions whatever the real context length is
(`RB_ROUNDS` feeds the outer size of the readback nd-DMA and is a build constant), so a
2048-window template moves the whole padded cache per token even at L=50. Building a
template pair per ATTN_MAXL and dispatching on the smallest one covering L removes the
difference.

Every model's decoder keeps its own BO set and dispatch; what lives here is the part that
must not diverge between them -- the window bookkeeping and the KV re-space.
"""

import numpy as np
from ml_dtypes import bfloat16


class KVGeometry:
    """Region-major KV cache shape.

    Per layer: NGRP K regions then NGRP V regions, each ATTN_MAXL*REGION_W elements, with
    position p at p*REGION_W inside its region. `lreg_max` sizes the BO, which is allocated
    for the largest window and shared by all of them.
    """

    def __init__(self, n_layers, kvsz_tok, region_w, ngrp, lreg_max):
        self.n_layers = int(n_layers)
        self.kvsz_tok = int(kvsz_tok)
        self.region_w = int(region_w)
        self.ngrp = int(ngrp)
        self.lreg_max = int(lreg_max)

    def lreg(self, maxl):
        return maxl * self.kvsz_tok

    @property
    def n_regions(self):
        return 2 * self.ngrp


def resolve_windows(gen, staircase):
    """Windows to hold open: the staircase up to the active one, or just the active one.

    `gen.attn_maxl` is what `DecodeInstsGen.select(max_L)` already resolved to -- the
    smallest calibrated window covering the caller's reach. Anything above it can never be
    selected for L <= that reach, so opening it would only cost init time and an unused
    hw_context. With `max_L=None` the active window is the largest and the full staircase
    is kept.
    """
    if not staircase:
        return [gen.attn_maxl]
    return [m for m in gen.calibrated_windows() if m <= gen.attn_maxl]


def open_windows(dev, xrt, gen, windows, kernel_match="MLIR_AIE"):
    """One hw_context + kernel per window, all opened up front.

    Host-only BOs are host memory registered with the device rather than bound to a
    context slot, so one BO set stays valid on every window's kernel; that is what keeps a
    switch to a kernel swap instead of a multi-GB weight re-upload. Returns
    {maxl: (ctx, kernel)} -- the context is kept in the tuple so it outlives the kernel.
    """
    out = {}
    for m in windows:
        xb = xrt.xclbin(gen.xclbin_for_maxl(m))
        dev.register_xclbin(xb)
        ctx = xrt.hw_context(dev, xb.get_uuid())
        xks = [k for k in xb.get_kernels() if kernel_match in k.get_name()]
        if not xks:
            raise RuntimeError(
                f"no kernel matching {kernel_match!r} in "
                f"{gen.xclbin_for_maxl(m)} (found: "
                f"{[k.get_name() for k in xb.get_kernels()]})"
            )
        out[m] = (ctx, xrt.kernel(ctx, xks[0].get_name()))
    return out


def make_insts_states(gen, xrt, dev, group_id, windows):
    """Per-window instruction stream: base, L-slope, host buffer and its own cacheable BO.

    `lo`/`hi` bound the L-dependent words so a per-token patch can rewrite and sync just
    that slice. `primed` tracks whether the full base stream has been written yet.
    """
    st = {}
    exact = getattr(gen, "exact", False)
    for m in windows:
        i1 = gen.insts_for(m, 1)
        if exact:
            # A dynseq build computes the stream, so there is nothing to calibrate
            # -- and nothing that could be: its readback length steps with
            # ceil(L/16), which no two-point slope reproduces. Keep the generator
            # and rewrite the whole stream per token.
            st[m] = dict(
                exact=True,
                gen=gen,
                maxl=m,
                buf=i1.astype(np.uint32).copy(),
                size=int(i1.size),
                ib=xrt.bo(dev, i1.nbytes, xrt.bo.cacheable, group_id),
                primed=False,
            )
            continue
        i2 = gen.insts_for(m, 2)
        ld = np.where(i1 != i2)[0]
        st[m] = dict(
            ld=ld,
            lo=int(ld.min()),
            hi=int(ld.max()) + 1,
            base=i1[ld].astype(np.int64),
            slope=i2[ld].astype(np.int64) - i1[ld].astype(np.int64),
            buf=i1.astype(np.uint32).copy(),
            size=int(i1.size),
            ib=xrt.bo(dev, i1.nbytes, xrt.bo.cacheable, group_id),
            primed=False,
        )
    return st


def patch_insts(state, L, xrt, to_dir):
    """Write the L-dependent words of `state` into its BO and return the stream length.

    The base stream is written whole once; afterwards only the [lo:hi] slice is rewritten
    and synced.
    """
    if state.get("exact"):
        # Whole stream: the L-dependent words are scattered across it and cost
        # far less to rewrite than to locate.
        state["buf"][:] = state["gen"].insts_for(state["maxl"], L)
        state["ib"].write(state["buf"], 0)
        state["ib"].sync(to_dir)
        return state["size"]
    state["buf"][state["ld"]] = (state["base"] + (L - 1) * state["slope"]).astype(
        np.uint32
    )
    if not state["primed"]:
        state["ib"].write(state["buf"], 0)
        state["ib"].sync(to_dir)
        state["primed"] = True
    else:
        lo, hi = state["lo"], state["hi"]
        state["ib"].write(state["buf"][lo:hi], lo * 4)
        state["ib"].sync(to_dir, (hi - lo) * 4, lo * 4)
    return state["size"]


def respace_kv(kvc, geom, old_maxl, new_maxl, live, xrt):
    """Re-lay the `live` filled positions from one window's KV layout into another's.

    Region stride is ATTN_MAXL*REGION_W, so changing window re-spaces the regions while
    each region's live prefix is unchanged. Gathered to a compact temp first, so growing
    and shrinking are both safe. Cost is proportional to `live`, and a crossing happens
    exactly when `live` is small.

    Reads back from the device: the kernel appends each token's K/V in place, so any host
    mirror of the cache is stale after the first dispatch.
    """
    if old_maxl == new_maxl or live <= 0:
        return
    TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    R, NR = geom.region_w, geom.n_regions
    kvc.sync(FROM)
    src = np.frombuffer(kvc.map(), dtype=bfloat16, count=geom.n_layers * geom.lreg_max)
    dst = np.zeros(geom.n_layers * geom.lreg_max, dtype=bfloat16)
    n = live * R
    for lay in range(geom.n_layers):
        so = lay * geom.lreg(old_maxl)
        do = lay * geom.lreg(new_maxl)
        for r in range(NR):
            dst[do + r * new_maxl * R : do + r * new_maxl * R + n] = src[
                so + r * old_maxl * R : so + r * old_maxl * R + n
            ]
    kvc.write(dst.view(np.int16), 0)
    kvc.sync(TO)
