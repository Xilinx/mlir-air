#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The DFlash pre-pass as WAVES of the projection engine, not a third PDI.

WHY. The pre-pass costs 82.0 ms of dispatch plus 36.5 ms of ELF load/unload per
block -- 36% of a 328.3 ms speculative step -- and
`dflash_prepass_cost.py` shows it is purely weight-bandwidth-bound at
0.46 GB/s: it streams 30.9 MB of constant weights through a generic int4 GEMM on
FOUR cores, once per block, to produce at most eight rows. The batch-8 verify
pass moves the model at ~11 GB/s on the same silicon a few milliseconds earlier.
Widening that GEMM is not a parameter -- `dflash_int4.HERD_N` is the ROW extent
and NPU2 has four compute rows. See docs/DFlashFeasibility.md section 3.13.

So the pre-pass should run where the bandwidth already is: the 16 projection
tiles (`fused_decode.PCOL` x rows 2-5), fed by the same per-column dual-channel
q4k weight stream every decode layer uses.

WHAT THIS FILE OWNS. Every DFlash-specific decision, so `fused_decode.py` --
the conventional-LLM engine -- needs only a generic "extra waves" hook and
carries no DFlash code:

  * where the extra weights live in their own BO, and in what order;
  * which row-iterations of the drafter's phase-0 slab are K and V;
  * the wave descriptors (I2 / J2 / dest / X source / weight offset);
  * the layout gate, below, which is the thing that fails SILENTLY.

TEN WAVES, and section 3.3's decomposition was right after all:

  fc      is FIVE accumulating 2560x2560 projections, I2=5 J2=5, one per tap.
          A single 2560x12800 wave is the tidier description and it was tried;
          it has nowhere to keep its X. The taps can only reach @xnorm through
          the rms core (the one producer there with a route to DDR), and at the
          shipping RMS_BAND_STREAM=0 that core's @rmsX get is ONE op outside the
          refeed loop, landing a resident BATCH*K row -- 40 KB of a 64 KB tile
          for ONE tap. Split, each wave's X is one resident tap and the
          cross-wave sum lands where the model already puts a cross-phase sum:
          the rms core's residual, which accumulates through DDR.
          The pack is NOT redone -- `fc_slab_perm` shows the five-slab layout is
          a bijection on the shipped `W_fc` blocks, so it is a gather.
  ctx K/V is the drafter's OWN k_proj/v_proj, already inside each drafter
          layer's phase-0 `concat([Wq;Wk;Wv])` slab. Nothing new is packed --
          only a row window is selected.

Every wave now reads exactly ONE X slot, which is what made the split worth
taking on its own terms.

A wave is a whole launch iteration running ONE phase, which is the shape the
LM-head waves already have (`nph_v = _sel(idx(1), ... idx(NPH))`). That is why
`qwen3_4b_draft_requant.py`'s note about `fc` not fitting as a 5th PHASE (it
would break `FULL4`) does not apply here: this is a 5th kind of WAVE.

    python3 dflash_prepass_waves.py             # the layout gate
    python3 dflash_prepass_waves.py --format    # q4k vs AWQ-int4 vs bf16
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "fused_decode"))

import numpy as np

# The block layout, imported rather than restated so this file cannot drift
# from the packer it is the inverse of.
from proj_qmm_pack import BLOCK_BF16, COL_BLOCK, GROUP, N_GROUPS, PARALLEL, ROW_BLOCK


# ---------------------------------------------------------------------------
# The cascade layout, as arithmetic instead of a comment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CascadeGeom:
    """Where `pack_q4k_cascade(iter_major=True, dual_chan=...)` puts each block.

    The packer emits `for cx: for (i, j, cy) in order:`, and with `dual_chan`
    the order hoists the ROW HALF `h` outermost so each of a column's two shim
    MM2S channels reads one contiguous DDR run. So a column's stream is

        [h=0: (i,j) in step order, cy in 0..half-1]
        [h=1: (i,j) in step order, cy in half..NCY-1]

    and that is what makes a row window cheap: restricting to `i` in
    [iter_lo, iter_lo+n_iter) is a CONTIGUOUS sub-run of each (cx, h) share,
    because `i` is the outer loop of `steps`. `_feed_wcol` already issues one
    put per channel, so selecting a window is a base offset and a length --
    no new descriptor shape.
    """

    NCX: int
    NCY: int
    nbi_pc: int  # row-block iterations per core
    nbj: int  # column-blocks = K / COL_BLOCK
    dual_chan: bool

    @property
    def half(self):
        return self.NCY // 2 if self.dual_chan else self.NCY

    @property
    def n_chan(self):
        return 2 if self.dual_chan else 1

    @property
    def blocks_per_step(self):
        """Blocks one (cx, h) sub-run emits per (i, j) step: one per cy."""
        return self.half

    @property
    def steps_per_chan(self):
        return self.nbi_pc * self.nbj

    @property
    def blocks_per_chan(self):
        return self.steps_per_chan * self.blocks_per_step

    @property
    def blocks_per_col(self):
        return self.blocks_per_chan * self.n_chan

    @property
    def n_blocks(self):
        return self.blocks_per_col * self.NCX

    def iter_window(self, row_lo, row_hi):
        """(iter_lo, n_iter) for matrix rows [row_lo, row_hi).

        Under `iter_major` the block for (cx, i, j, cy) is global row-block
        `gi = i*(NCX*NCY) + cx*NCY + cy`, so iteration `i` owns exactly the
        contiguous span of `NCX*NCY` row-blocks starting at `i*NCX*NCY` --
        `NCX*NCY*ROW_BLOCK` matrix rows. A window that is not iteration-aligned
        cannot be expressed as a feed offset, so refuse it here rather than
        emit a plausible wrong slab.
        """
        span = self.NCX * self.NCY * ROW_BLOCK
        if row_lo % span or row_hi % span:
            raise ValueError(
                f"rows [{row_lo}, {row_hi}) are not aligned to the {span}-row "
                f"iteration granule (NCX*NCY*ROW_BLOCK); a q/k/v split that "
                f"lands mid-iteration cannot be selected by a feed offset"
            )
        return row_lo // span, (row_hi - row_lo) // span

    def chan_window(self, iter_lo, n_iter):
        """(first_block, n_blocks) of that window WITHIN one (cx, h) sub-run."""
        return iter_lo * self.nbj * self.blocks_per_step, (
            n_iter * self.nbj * self.blocks_per_step
        )

    def block_coords(self):
        """(gi, j) per block, in emission order. The inverse of the packer."""
        out = []
        for cx in range(self.NCX):
            for h in range(self.n_chan):
                for i in range(self.nbi_pc):
                    for j in range(self.nbj):
                        for cyl in range(self.blocks_per_step):
                            cy = h * self.half + cyl
                            out.append(
                                (i * (self.NCX * self.NCY) + cx * self.NCY + cy, j)
                            )
        return out


def geom_for(M, K, fd):
    """The geometry `pack_q4k_cascade` will use for an [M, K] matrix."""
    nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
    n_cores = fd.NCX * fd.NCY
    if M % ROW_BLOCK or K % COL_BLOCK or nbi % n_cores:
        raise ValueError(
            f"[{M},{K}] does not tile: needs M % {ROW_BLOCK} == 0, "
            f"K % {COL_BLOCK} == 0 and (M/{ROW_BLOCK}) % {n_cores} == 0"
        )
    return CascadeGeom(
        NCX=fd.NCX,
        NCY=fd.NCY,
        nbi_pc=nbi // n_cores,
        nbj=nbj,
        dual_chan=bool(getattr(fd, "W_DUAL_CHAN", 0)),
    )


# ---------------------------------------------------------------------------
# The inverse of the packer -- the layout gate's instrument
# ---------------------------------------------------------------------------
def unpack_blocks(packed):
    """(q, scale, mins) for every block in a packed slab, vectorized.

    `pack_q4k_block` is a per-block Python triple loop; inverting it the same
    way takes minutes on fc's 4000 blocks, and a gate nobody runs is not a gate.
    Layout (proj_qmm_pack.py:5-9):
        bf16  scales[8 groups][32 rows]   offset 0
        bf16  mins  [8 groups][32 rows]   offset 512
        uint8 qs[2 rowgrp][256 col][8]    offset 1024, byte = (w_odd<<4)|w_even
    """
    from ml_dtypes import bfloat16

    u8 = np.ascontiguousarray(packed).view(np.uint8).reshape(-1, BLOCK_BF16 * 2)
    nb = u8.shape[0]
    sm = 2 * N_GROUPS * ROW_BLOCK
    sc = u8[:, :sm].view(bfloat16).reshape(nb, N_GROUPS, ROW_BLOCK)
    mn = u8[:, sm : 2 * sm].view(bfloat16).reshape(nb, N_GROUPS, ROW_BLOCK)
    qs = u8[:, 2 * sm :].reshape(nb, ROW_BLOCK // PARALLEL, COL_BLOCK, PARALLEL // 2)
    # row = g*PARALLEL + kk*2 + p, p=0 low nibble (even row), p=1 high (odd)
    pair = np.stack([qs & 0xF, qs >> 4], axis=-1)  # [nb, g, col, kk, p]
    q = pair.transpose(0, 1, 3, 4, 2).reshape(nb, ROW_BLOCK, COL_BLOCK)
    # scales/mins come back group-major; the caller wants [block, row, group]
    return q, sc.transpose(0, 2, 1), mn.transpose(0, 2, 1)


def dequant_cascade(packed, M, K, geom, rows=None):
    """Reconstruct [M, K] (float32) from a cascade-packed slab.

    `rows` optionally restricts the output to a row range (lo, hi) so a gate on
    fc's 2560x12800 does not materialize 131 MB it will not look at.
    """
    q, sc, mn = unpack_blocks(packed)
    coords = geom.block_coords()
    if q.shape[0] != len(coords):
        raise ValueError(
            f"slab holds {q.shape[0]} blocks, geometry expects {len(coords)} "
            f"-- M/K or dual_chan disagree with how this was packed"
        )
    lo, hi = (0, M) if rows is None else rows
    out = np.zeros((hi - lo, K), np.float32)
    sc_e = np.repeat(sc.astype(np.float32), GROUP, axis=2)
    mn_e = np.repeat(mn.astype(np.float32), GROUP, axis=2)
    w = q.astype(np.float32) * sc_e + mn_e
    for b, (gi, j) in enumerate(coords):
        r0 = gi * ROW_BLOCK
        if r0 + ROW_BLOCK <= lo or r0 >= hi:
            continue
        out[r0 - lo : r0 - lo + ROW_BLOCK, j * COL_BLOCK : (j + 1) * COL_BLOCK] = w[b]
    return out


# ---------------------------------------------------------------------------
# The two waves
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WaveSpec:
    """One extra launch iteration of the projection engine.

    `i2` / `j2` / `dest` are the scalars the proj core's arm already selects per
    phase (`_psw`/`_sel`); `iter_lo` and `w_off` are the two additions, and both
    are feed-side offsets rather than new descriptor shapes.
    """

    name: str
    m: int  # projection output rows
    k: int  # contraction width
    i2: int  # row-block iterations this wave runs
    j2: int  # column-block PAIRS: 2*j2 == k / COL_BLOCK
    iter_lo: int  # first row-block iteration of the source slab
    w_off: int  # element offset of the slab's column-0 base in the extra BO
    x_slot: int  # first X-buffer slot this wave contracts over
    x_stride: int  # slots between them (fc's taps are every 8th)
    dest: str  # "rms" -- see the `dest` note in `wave_specs`
    # Which M-row band of its OWN projection's output this wave holds, counted
    # from that projection's first row rather than from the slab's. It is
    # `iter_lo` for fc, whose slab IS the projection, and `iter_lo - 8` for a
    # context-K/V wave, whose slab starts with q. The engine never reads either:
    # an i2=1 wave always deposits into band 0 of its own output slot, so this
    # is the host's map from slots back to rows.
    out_band: int = 0
    # The PROJECTION this wave is a band of -- "fc", or one drafter layer's
    # context K/V. Carried rather than parsed back out of `name`, because the
    # assembler below turns output slots into projections and a naming
    # convention is not a thing to make that depend on.
    group: str = ""

    def as_config(self):
        """The subset fused_decode's DECODE_EXTRA_WAVES takes.

        `m` and `k` stay here and are not passed: the engine derives K from
        `j2` and the X extent from that, so a wave cannot claim an X of one
        width and weights of another.
        """
        return {
            k_: getattr(self, k_)
            for k_ in (
                "name",
                "i2",
                "j2",
                "iter_lo",
                "w_off",
                "x_slot",
                "x_stride",
                "dest",
            )
        }

    @property
    def blocks(self):
        return self.i2 * (2 * self.j2) * 16


def _kv_rows(fd_draft):
    """(row_lo, row_hi) of k_proj+v_proj inside the phase-0 QKV slab.

    Derived from the checkpoint's own shapes, not hardcoded: phase 0 is
    `concat([Wq; Wk; Wv], axis=0)` (qwen3_4b_draft_requant.py:77), so K starts
    where Q ends. Getting this wrong is the failure this file's gate exists for
    -- it packs and runs and returns another projection's answer.
    """
    from qwen3_4b_draft_weights import DraftWeights

    q_rows = DraftWeights._PROJ["q"][1]
    k_rows = DraftWeights._PROJ["k"][1]
    v_rows = DraftWeights._PROJ["v"][1]
    return q_rows, q_rows + k_rows + v_rows


def _target_x_slots():
    """X_SLOTS of the TARGET's build -- one per layer, plus the input.

    Read out of a fused_decode loaded at the target's geometry rather than
    written down, because it is the number the extra waves have to sit above
    and the two must not be able to drift apart.
    """
    # qwen3_4b_draft_requant._load_fd pins DECODE_MODEL to the drafter, so load
    # the target the same way it does rather than trying to steer that one.
    import importlib.util

    saved = {k: v for k, v in os.environ.items() if k.startswith("DECODE_")}
    for k in list(os.environ):
        if k.startswith("DECODE_"):
            os.environ.pop(k, None)
    os.environ.update(
        DECODE_MODEL="qwen3-4b",
        VOCAB_CHUNK_I2="30",
        LM_HEAD="0",
        NLAYERS="1",
        UNIFIED="1",
        DECODE_GOLDEN="1",
        DECODE_GOLDEN_L="128",
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "fused_decode_target_geom",
            str(_HERE.parent.parent / "fused_decode" / "fused_decode.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.UNI_DEC + 1
    finally:
        for k in list(os.environ):
            if k.startswith("DECODE_"):
                os.environ.pop(k, None)
        os.environ.update(saved)


def _tap_slots(fd_draft):
    """(first slot, slot stride) of the target's hidden-state taps.

    The taps live at X slot `lid+1` for each tapped layer id, and the tapped
    layers are evenly spaced, so the five of them are an arithmetic progression
    -- which is what lets fc read all 12800 columns as ONE strided descriptor
    instead of five puts. Asserted rather than assumed: if a checkpoint ever
    taps unevenly, this has to become several waves and the caller should find
    out here rather than from a wrong answer.
    """
    from dflash_phase2_replay import TARGET_LAYER_IDS

    slots = [lid + 1 for lid in TARGET_LAYER_IDS]
    step = {b - a for a, b in zip(slots, slots[1:])}
    if len(step) != 1:
        raise ValueError(
            f"tap slots {slots} are not evenly spaced; fc cannot be one "
            f"strided X read"
        )
    return slots[0], step.pop()


def fc_slab_perm(fd_draft):
    """Block permutation taking the shipped one-slab `W_fc` pack to five slabs.

    `pack_q4k_cascade(iter_major=True)` emits `for cx: for h: for i: for j: for
    cy`, so a column-block range is NOT contiguous -- it is interleaved with the
    row iteration and the channel half. But it IS a permutation: every block of
    the 2560x12800 pack appears exactly once across the five 2560x2560 slabs,
    because the split is on column-block boundaries and `block_coords` carries
    `(gi, j)` for both geometries. The assertion below is the whole argument --
    a bijection means no block is dropped, duplicated, or re-quantized.

    Returns an index array `perm` such that
    `blocks[perm]` is the five-slab layout, laid out slab-major.
    """
    from qwen3_4b_draft_weights import D, FC_IN

    g_one = geom_for(D, FC_IN, fd_draft)
    g_slab = geom_for(D, D, fd_draft)
    pos = {c: k for k, c in enumerate(g_one.block_coords())}
    if len(pos) != g_one.n_blocks:
        raise AssertionError("one-slab block coords are not unique")
    coords = g_slab.block_coords()
    perm = np.asarray(
        [pos[(gi, j + s * g_slab.nbj)] for s in range(FC_IN // D) for (gi, j) in coords]
    )
    if len(set(perm.tolist())) != g_one.n_blocks:
        raise AssertionError(
            f"the five-slab layout is not a permutation of the one-slab pack: "
            f"{len(set(perm.tolist()))} distinct blocks of {g_one.n_blocks}"
        )
    return perm


def wave_specs(fd_draft):
    """The extra waves, and the compact extra-BO layout they read from.

    THE EXTRA WEIGHTS GET THEIR OWN BO. They are the DRAFTER's, and the pre-pass
    runs in the TARGET's program, so they cannot ride the target's weight cache.
    A separate buffer is also the shape `W_SPLIT` already established
    (fused_decode.py:1893-1904: extra weight buffers are appended AFTER the
    existing args, so x/w/rms/y/kvc binding positions do not move).

    It is packed COMPACTLY -- fc's slab, then only the K/V window of each
    drafter layer -- so it is 36.9 MB rather than the drafter's whole ~300 MB.
    The cost is that the per-wave base is not an affine `a_iv * slab`; with six
    waves a nested cmpi/select chain is the right shape anyway, and that is what
    `W_SPLIT` uses for exactly this reason (fused_decode.py:3080-3098: the
    post-unroll fold set carries cmpi/select and not div/rem).
    """
    from qwen3_4b_draft_weights import D, FC_IN

    n_layers = fd_draft.UNI_DEC
    kv_lo, kv_hi = _kv_rows(fd_draft)
    qkv_rows = kv_hi  # phase 0 is [q|k|v] stacked

    g_fc = geom_for(D, FC_IN, fd_draft)
    g_kv = geom_for(qkv_rows, D, fd_draft)
    kv_iter_lo, kv_n_iter = g_kv.iter_window(kv_lo, kv_hi)

    # fc contracts over the five hidden-state taps, which sit at X slots
    # `lid+1` for the tapped layer ids -- an arithmetic progression, because the
    # tapped layers are evenly spaced. That is what lets it be one strided read
    # rather than five: TAP_SLOTS = [2, 10, 18, 26, 34] is slot 2, stride 8.
    tap0, tap_stride = _tap_slots(fd_draft)
    # target_hidden -- fc's output and every context-K/V wave's input -- needs a
    # slot of its own past the layer outputs. The engine grows the X buffer to
    # whatever the wave list names, so this is a choice made here and nowhere
    # else.
    #
    # PAST THE TARGET'S SLOTS, not the drafter's. These waves are built from the
    # drafter's geometry but they run inside the TARGET's program, whose X
    # buffer has one slot per target layer plus the input; the drafter's five
    # layers would put target_hidden on top of target layer 5's output.
    th_slot = _target_x_slots()

    # fc is FIVE 2560x2560 waves, not one 2560x12800 one -- and that is forced
    # by an L1 budget, not chosen for tidiness. The rms core is the only producer
    # on @xnorm with a route to DDR, so the taps have to arrive through its
    # @rmsX get; at the shipping RMS_BAND_STREAM=0 that get is ONE op outside the
    # refeed loop, landing a resident BATCH*K row. One tap is 8*2560*2 = 40 KB of
    # a 64 KB tile, so five cannot be resident and a 12800-wide wave has nowhere
    # to hold its X. Moving the get inside the loop would change the DECODE
    # path's structure and its transfer count, which is the one thing this fold
    # may not do.
    #
    # Split, each wave's X is one resident tap and the cross-wave sum lands
    # where the model already puts a cross-phase sum: the rms core's residual,
    # which accumulates through DDR (read, add, write back). That is
    # section 3.3's own decomposition, fc(concat) = sum_i W_i . h_i, which
    # section 3.14 had overridden because the shipped pack is one slab.
    #
    # The pack does not have to be redone. `fc_slab_perm` shows the five-slab
    # layout is a BIJECTION on the shipped W_fc's blocks -- the q4k groups are 32
    # columns and the split is at multiples of 2560, so the quantized bytes are
    # identical and only re-ordered.
    #
    # Measured cost of the split, on the real fc against a bf16 reference:
    # 1.090e-01 where one wave gives 1.080e-01 (cos 0.996578 vs 0.996583). The
    # four extra bf16 roundings are 5.6e-03 between the two forms, two orders
    # below the 1.08e-01 quantization floor. It is free.
    #
    # AND EACH SLAB IS ONE WAVE PER ROW-BLOCK ITERATION, not one wave. That is a
    # DEADLOCK bound, measured on device: an extra wave is the only arm on which
    # the rms core is both the X producer and the output consumer, in that
    # order, so i2-1 egress rounds have to sit in the fabric while it finishes
    # feeding. The fabric holds four. i2=3 completes and i2=5 hangs, everything
    # else held fixed. fc's 12800 rows divide evenly only at i2 in {1, 5, 25},
    # so i2=1 it is -- 25 waves of 512 rows, and each is a single-iteration
    # window of a slab, which is the same `chan_window` slice the context K/V
    # waves already take.
    n_slab = FC_IN // D
    g_slab = geom_for(D, D, fd_draft)
    sub_first, sub_nblk = g_slab.chan_window(0, 1)
    sub_blocks = sub_nblk * g_slab.NCX * g_slab.n_chan
    sub_bytes = sub_blocks * BLOCK_BF16
    sub_rows = g_slab.NCX * g_slab.NCY * ROW_BLOCK
    waves = [
        WaveSpec(
            name=f"fc{s}i{t}",
            group="fc",
            m=sub_rows,
            k=D,
            i2=1,
            j2=g_slab.nbj // 2,
            iter_lo=t,
            out_band=t,
            w_off=(s * g_slab.nbi_pc + t) * sub_bytes,
            x_slot=tap0 + s * tap_stride,
            x_stride=1,
            dest="rms",
        )
        for s in range(n_slab)
        for t in range(g_slab.nbi_pc)
    ]
    assert n_slab * g_slab.nbi_pc * sub_blocks == g_fc.n_blocks, (
        f"{n_slab}x{g_slab.nbi_pc} sub-waves of {sub_blocks} blocks is not "
        f"fc's {g_fc.n_blocks}"
    )
    off = g_fc.n_blocks * BLOCK_BF16
    # AND THE CONTEXT K/V WAVES ARE THE SAME SHAPE, for both of fc's reasons and
    # a third that removes the last piece of engine work this fold was waiting
    # on.
    #
    #   i2=1, because the deadlock bound above is not about which core consumes
    #   the output -- it is about the rms core having to finish FEEDING before
    #   the first egress round can land, and that core feeds every extra wave.
    #
    #   dest="rms", because a "rope" dest needs the rope core to grow a
    #   context-K/V body (k_norm, RoPE, appendK/appendV at the drafter's KV
    #   base) and it buys NOTHING: the host already owns that arithmetic. The
    #   PDI pre-pass returns k_ctx already normed and rotated only because it
    #   had a rope stage to spare; `seed_context_kv` takes whatever it is given,
    #   and k_norm + RoPE on [5, 8, 1024] is microseconds of numpy against the
    #   ~3 ms of weight streaming this wave exists to do. So the K/V projection
    #   comes back raw on @layerOut, exactly as fc's partials do, and every line
    #   of engine support these waves need is already on silicon.
    #
    # What that costs is one X slot's worth of readback per wave instead of a
    # KV-cache write -- 20 x BATCH*K bf16 -- and what it saves is the entire
    # rope arm, which is the only reason `dest` still has a second value.
    sub_kv_first, sub_kv_nblk = g_kv.chan_window(0, 1)
    sub_kv_blocks = sub_kv_nblk * g_kv.NCX * g_kv.n_chan
    for L in range(n_layers):
        for t in range(kv_n_iter):
            waves.append(
                WaveSpec(
                    name=f"ctxkv{L}i{t}",
                    group=f"ctxkv{L}",
                    m=sub_rows,
                    k=D,
                    i2=1,
                    j2=g_kv.nbj // 2,
                    iter_lo=kv_iter_lo + t,
                    out_band=t,
                    w_off=off,
                    x_slot=th_slot,
                    x_stride=1,
                    dest="rms",
                )
            )
            off += sub_kv_blocks * BLOCK_BF16
    assert kv_n_iter * sub_rows == kv_hi - kv_lo, (
        f"{kv_n_iter} sub-waves of {sub_rows} rows is not the "
        f"{kv_hi - kv_lo}-row K/V window"
    )
    return waves, off


def fc_extra_bo(fd_draft, npz):
    """The fc region of the extra BO: 25 single-iteration sub-slabs, in order.

    Factored out of `build_extra_weights` so the device gate reads the SAME
    bytes the BO is built from rather than a second copy of the gather.
    """
    from qwen3_4b_draft_weights import D, FC_IN

    g_slab = geom_for(D, D, fd_draft)
    _, sub_nblk = g_slab.chan_window(0, 1)
    fc = np.asarray(npz["W_fc"]).reshape(-1, BLOCK_BF16)[fc_slab_perm(fd_draft)]
    parts = []
    for s in range(FC_IN // D):
        sl = fc[s * g_slab.n_blocks : (s + 1) * g_slab.n_blocks]
        for t in range(g_slab.nbi_pc):
            for cx in range(g_slab.NCX):
                for h in range(g_slab.n_chan):
                    b = (
                        cx * g_slab.blocks_per_col
                        + h * g_slab.blocks_per_chan
                        + t * sub_nblk
                    )
                    parts.append(sl[b : b + sub_nblk].reshape(-1))
    return np.concatenate(parts)


def ctxkv_extra_bo(fd_draft, npz):
    """The context-K/V region of the extra BO: 4 sub-slabs per drafter layer.

    The same gather as `fc_extra_bo`, over the K/V row window of each layer's
    phase-0 slab instead of over fc's five taps -- and factored out for the same
    reason, so the device gate reads the bytes the BO is built from.

    A sub-wave is ONE row-block iteration, so within a (cx, h) sub-run its
    blocks are `nbj * blocks_per_step` contiguous ones starting at iteration
    `it_lo + t`. `chan_window` states that; this only walks it.
    """
    from qwen3_4b_draft_weights import D

    kv_lo, kv_hi = _kv_rows(fd_draft)
    g_kv = geom_for(kv_hi, D, fd_draft)
    it_lo, n_it = g_kv.iter_window(kv_lo, kv_hi)
    _, sub_nblk = g_kv.chan_window(0, 1)

    W = np.asarray(npz["W"]).reshape(-1)
    parts = []
    for L in range(fd_draft.UNI_DEC):
        lay = W[L * fd_draft.W_LAYER : (L + 1) * fd_draft.W_LAYER]
        for t in range(n_it):
            for cx in range(g_kv.NCX):
                for h in range(g_kv.n_chan):
                    b = (
                        cx * g_kv.blocks_per_col
                        + h * g_kv.blocks_per_chan
                        + (it_lo + t) * sub_nblk
                    )
                    parts.append(lay[b * BLOCK_BF16 : (b + sub_nblk) * BLOCK_BF16])
    return np.concatenate(parts)


def build_extra_weights(fd_draft, npz, verbose=True):
    """The compact extra-weight BO: [fc slab | K/V window per drafter layer].

    Each wave's slab is gathered as `NCX * n_chan` contiguous runs -- one per
    (column, shim channel) -- and re-emitted in the same (cx, h) order the feed
    walks, so `_feed_wcol`'s two puts per column need only a base and a length.
    A single-iteration window is not contiguous until it is gathered, which is
    what both halves below are doing.
    """
    waves, total = wave_specs(fd_draft)
    out = np.concatenate([fc_extra_bo(fd_draft, npz), ctxkv_extra_bo(fd_draft, npz)])
    if out.size != total:
        raise AssertionError(f"extra BO is {out.size} elements, layout says {total}")
    if verbose:
        print(
            f"[prepass waves] extra BO {out.size} elements = "
            f"{out.size * 2 / 1e6:.1f} MB over {len(waves)} waves"
        )
    return out, waves


# ---------------------------------------------------------------------------
# Reading the waves back
# ---------------------------------------------------------------------------
# WHAT A WAVE LEAVES BEHIND IS NOT `W . x`. The X reaches the projection through
# the rms core, whose regen multiplies by the norm weight and a per-row scale,
# and whose residual pass adds the result into the same buffer the X arrived in.
# The norm weight is fed as ONES -- that is what makes the forwarding need no
# kernel change -- so the wave's output slot holds
#
#     x + W . (x / rms(x))
#
# and rms(x) is a per-row scalar the HOST can compute, because the host wrote x.
# The correction is therefore exact rather than a fit:
#
#     W . x  =  (readback - x) * rms(x)
#
# The eps below is the kernel's, and it has to stay that: a different one is a
# per-row relative error of eps/(2*rms^2), which on a hidden state of norm ~1
# is small enough to pass a cosine gate and wrong.
RMS_EPS = 1e-6


def rms_rows(x, eps=RMS_EPS):
    """The per-row scale the rms core divided out, as float32."""
    xf = np.asarray(x, np.float32)
    return np.sqrt((xf * xf).mean(-1, keepdims=True) + eps)


def assemble(fd, waves, xall, x_rows, eps=RMS_EPS):
    """{group: [B, nband*M]} -- every wave's projection, out of the X buffer.

    `xall` is the whole X BO as bf16, `x_rows` maps X slot to the [B, K] row
    block the host wrote there. Bands are placed by `out_band` and NOT by where
    the values landed: an i2=1 wave always deposits into columns [0, M) of its
    own output slot, because residual1 puts egress round r at column band r and
    such a wave has only round 0. The rms core does not know which output rows
    the wave computed; the wave descriptor does.
    """
    B, K = fd.BATCH, fd.K
    M = waves[0].m
    nband, out = {}, {}
    for w in waves:
        nband[w.group] = max(nband.get(w.group, 0), w.out_band + 1)
    for g, n in nband.items():
        out[g] = np.zeros((B, n * M), np.float32)
    for k, w in enumerate(waves):
        x = np.asarray(x_rows[w.x_slot], np.float32).reshape(B, K)
        sl = fd.EXTRA_OUT_SLOT[k]
        got = xall[sl * B * K : (sl + 1) * B * K].astype(np.float32).reshape(B, K)
        out[w.group][:, w.out_band * M : (w.out_band + 1) * M] += (
            got[:, :M] - x[:, :M]
        ) * rms_rows(x, eps)
    return out


def target_hidden(fd, waves, xall, taps, hn_w):
    """fc's answer, normed -- the drafter's context feature.

    `taps` is [B, 5*K] in TAP_SLOTS order, exactly what the target's
    `last_taps` holds and what the fc waves' X slots were filled from.
    """
    import dflash_sumnorm

    K = fd.K
    tap0, stride = _tap_slots(_load_draft_fd())
    x_rows = {
        tap0 + s * stride: np.asarray(taps, np.float32)[:, s * K : (s + 1) * K]
        for s in range(np.asarray(taps).shape[1] // K)
    }
    fc = assemble(fd, [w for w in waves if w.group == "fc"], xall, x_rows)["fc"]
    return dflash_sumnorm.reference([fc], np.asarray(hn_w))


def context_kv(fd, waves, xall, th, kn_w, positions):
    """(k_ctx, v_ctx), each [n_layers, B, KV_DIM] -- k_norm'd and rotated.

    The waves return the K/V projection RAW; k_norm and RoPE are the host's,
    which is why these waves need no rope-core arm. The K/V window is the phase-0
    slab's rows [q_rows, q_rows+2*KVD), so the first KVD columns of each layer's
    assembled band-run are K and the rest are V -- the packer's own row order,
    not a convention chosen here.
    """
    import dflash_ctxkv_int4_builder as CK
    from dflash_ctxkv_int4_gate import rope_ref

    kv = [w for w in waves if w.group != "fc"]
    got = assemble(fd, kv, xall, {kv[0].x_slot: th})
    HD, KVD = CK.HEAD_DIM, CK.KV_DIM
    ks, vs = [], []
    for L in range(len({w.group for w in kv})):
        a = got[f"ctxkv{L}"]
        k = a[:, :KVD].reshape(-1, HD)
        k = (k / np.sqrt((k**2).mean(-1, keepdims=True) + RMS_EPS)) * np.asarray(
            kn_w[L], np.float32
        )
        ks.append(rope_ref(k, positions).reshape(-1, KVD))
        vs.append(a[:, KVD : 2 * KVD])
    return np.stack(ks), np.stack(vs)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def _load_draft_fd():
    from qwen3_4b_draft_requant import _load_fd

    return _load_fd()


def gate_layout(cache=None, verbose=True):
    """Does the SHIPPED slab hold the matrix the wave descriptor claims?

    THIS IS THE ONE THAT MATTERS. The 4-bit re-quant is lossy, so this cannot be
    an equality check -- it is a check that the RIGHT rows were selected, in the
    right order. A layout error moves rows and shows up as a relative error of
    order 1; the quantization floor is order one step. The negative control
    below prints what that looks like, so the two are not confused.

    It reads the drafter's requant cache rather than re-packing. That is both
    faster -- `pack_q4k_block` is a per-block Python triple loop and fc is 4000
    blocks -- and a stronger claim: it tests the bytes the device will actually
    be handed, not a second copy made by the same code.
    """
    from qwen3_4b_draft_weights import D, DraftWeights, FC_IN

    fd = _load_draft_fd()
    dw = DraftWeights()
    bad = 0
    step = 1.0 / 15.0

    if cache is None:
        from dflash_draft_decoder import draft_cache_path

        cache = draft_cache_path(bool(fd.W_DUAL_CHAN))
    npz = np.load(cache)
    if verbose:
        print(f"  cache {Path(cache).name}  (W_DUAL_CHAN={int(bool(fd.W_DUAL_CHAN))})")

    # 1. fc: one row band. Checking every row costs 131 MB of float32 and proves
    #    nothing more -- a mis-ordered pack moves other rows INTO this band.
    fc = np.asarray(dw.fc(), np.float32)
    g_fc = geom_for(D, FC_IN, fd)
    band = (0, 512)
    got = dequant_cascade(
        np.asarray(npz["W_fc"]).reshape(-1), D, FC_IN, g_fc, rows=band
    )
    ref = fc[band[0] : band[1]]
    rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
    ok = rel <= 2.0 * step
    bad += not ok
    if verbose:
        print(
            f"  fc      [{D}x{FC_IN}] rows {band[0]}..{band[1]}: "
            f"max rel {rel:.3e}  (one step {step:.3e})  {'OK' if ok else 'WRONG LAYOUT'}"
        )

    # 1b. the five per-tap slabs the waves actually read, which are the shipped
    #     pack REORDERED by fc_slab_perm. The bijection assertion in that
    #     function proves no block is lost; this proves each slab holds the
    #     columns its wave will contract against, which is the part a
    #     permutation cannot tell you on its own.
    g_slab = geom_for(D, D, fd)
    reord = np.asarray(npz["W_fc"]).reshape(-1, BLOCK_BF16)[fc_slab_perm(fd)]
    worst, worst_s = 0.0, -1
    for s in range(FC_IN // D):
        sl = reord[s * g_slab.n_blocks : (s + 1) * g_slab.n_blocks].reshape(-1)
        got = dequant_cascade(sl, D, D, g_slab, rows=band)
        ref = fc[band[0] : band[1], s * D : (s + 1) * D]
        r = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
        if r > worst:
            worst, worst_s = r, s
    ok = worst <= 2.0 * step
    bad += not ok
    if verbose:
        # The control that makes it a gate: slab 0 against the NEXT tap's
        # columns. A permutation that is a bijection but wrong reads like this.
        sl0 = reord[: g_slab.n_blocks].reshape(-1)
        g0 = dequant_cascade(sl0, D, D, g_slab, rows=band)
        r1 = fc[band[0] : band[1], D : 2 * D]
        wrong = np.abs(g0 - r1).max() / max(np.abs(r1).max(), 1e-9)
        print(
            f"  fc slabs 5x[{D}x{D}] rows {band[0]}..{band[1]}: "
            f"worst rel {worst:.3e} (slab {worst_s})  "
            f"{'OK' if ok else 'WRONG SPLIT'}"
        )
        print(f"          negative control, slab 0 vs tap 1: {wrong:.3e}")

    # 1c. and the SUB-WAVE split of each slab: 25 single-iteration windows, in
    #     the (cx, h) run order the feed reads them back in. The slab gate above
    #     cannot see this -- it checks a permutation of whole slabs, and the
    #     sub-wave gather re-interleaves inside one. A wrong window reads the
    #     right slab's wrong 512 rows, which is O(1), not O(quant step).
    _, sub_nblk = g_slab.chan_window(0, 1)
    sub_rows = g_slab.NCX * g_slab.NCY * ROW_BLOCK
    g_sub = geom_for(sub_rows, D, fd)
    worst, worst_w = 0.0, ""
    for sl_i in range(FC_IN // D):
        sl = reord[sl_i * g_slab.n_blocks : (sl_i + 1) * g_slab.n_blocks]
        for t in range(g_slab.nbi_pc):
            run = np.concatenate(
                [
                    sl[b : b + sub_nblk].reshape(-1)
                    for cx in range(g_slab.NCX)
                    for h in range(g_slab.n_chan)
                    for b in [
                        cx * g_slab.blocks_per_col
                        + h * g_slab.blocks_per_chan
                        + t * sub_nblk
                    ]
                ]
            )
            got = dequant_cascade(run, sub_rows, D, g_sub)
            ref = fc[t * sub_rows : (t + 1) * sub_rows, sl_i * D : (sl_i + 1) * D]
            r = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
            if r > worst:
                worst, worst_w = r, f"fc{sl_i}i{t}"
    ok = worst <= 2.0 * step
    bad += not ok
    if verbose:
        print(
            f"  fc sub-waves 25x[{sub_rows}x{D}]: worst rel {worst:.3e} "
            f"({worst_w})  {'OK' if ok else 'WRONG WINDOW'}"
        )

    # 2. the K/V window of drafter layer 0's phase-0 slab, from the same cache.
    kv_lo, kv_hi = _kv_rows(fd)
    R = {
        nm: dw.dequant(f"layers.0.{t}.weight", M, Kc)
        for nm, (t, M, Kc) in DraftWeights._PROJ.items()
        if nm in ("q", "k", "v")
    }
    qkv = np.concatenate([R["q"], R["k"], R["v"]], 0).astype(np.float32)
    g_kv = geom_for(qkv.shape[0], D, fd)
    if g_kv.n_blocks != fd.PER_COL_PH[0] * fd.NCX:
        raise AssertionError(
            f"phase-0 geometry disagrees with the builder: {g_kv.n_blocks} "
            f"blocks vs PER_COL_PH[0]*NCX = {fd.PER_COL_PH[0] * fd.NCX}"
        )
    p0 = np.asarray(npz["W"]).reshape(-1)[: g_kv.n_blocks * BLOCK_BF16]
    it_lo, n_it = g_kv.iter_window(kv_lo, kv_hi)
    got = dequant_cascade(p0, qkv.shape[0], D, g_kv, rows=(kv_lo, kv_hi))
    ref = qkv[kv_lo:kv_hi]
    rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
    ok = rel <= 2.0 * step
    bad += not ok
    if verbose:
        print(
            f"  ctx K/V rows {kv_lo}..{kv_hi} = iters {it_lo}..{it_lo + n_it}: "
            f"max rel {rel:.3e}  (one step {step:.3e})  "
            f"{'OK' if ok else 'WRONG WINDOW'}"
        )
        # And the negative control: the window must NOT be q.
        wrong = np.abs(got - qkv[: kv_hi - kv_lo]).max() / max(np.abs(ref).max(), 1e-9)
        print(
            f"          against q's rows instead: max rel {wrong:.3e} "
            f"-- a window error looks like THIS, not like the line above"
        )

    # 3. the compact extra BO reproduces the same window.
    first, n_blk = g_kv.chan_window(it_lo, n_it)
    gathered = np.concatenate(
        [
            p0[
                (cx * g_kv.blocks_per_col + h * g_kv.blocks_per_chan + first)
                * BLOCK_BF16 : (
                    cx * g_kv.blocks_per_col + h * g_kv.blocks_per_chan + first + n_blk
                )
                * BLOCK_BF16
            ]
            for cx in range(g_kv.NCX)
            for h in range(g_kv.n_chan)
        ]
    )
    g_win = CascadeGeom(
        NCX=g_kv.NCX,
        NCY=g_kv.NCY,
        nbi_pc=n_it,
        nbj=g_kv.nbj,
        dual_chan=g_kv.dual_chan,
    )
    got2 = dequant_cascade(gathered, kv_hi - kv_lo, D, g_win)
    rel2 = np.abs(got2 - ref).max() / max(np.abs(ref).max(), 1e-9)
    ok2 = rel2 <= 2.0 * step
    bad += not ok2
    if verbose:
        print(
            f"  compact gather of that window: max rel {rel2:.3e}  "
            f"{'OK' if ok2 else 'GATHER WRONG'}"
        )
    return bad


def _prepass_chain(fc_w, hn_w, kv_w, kn_w, taps, positions):
    """The pre-pass in numpy: fc -> hidden_norm -> k/v_proj -> k_norm -> RoPE.

    Exactly the chain `dflash_draft_prepass_gate.py` references the device
    against, and the ORDER of it is what `dflash_prepass_oracle_gate.py` already
    proved against the real drafter's own KV rows. Nothing here re-litigates
    that; the only variable is which numbers the weights hold.
    """
    import dflash_ctxkv_int4_builder as CK
    import dflash_sumnorm
    from dflash_ctxkv_int4_gate import rope_ref

    HD, NKV = CK.HEAD_DIM, CK.N_KV_HEADS
    th = dflash_sumnorm.reference([np.asarray(taps, np.float32) @ fc_w.T], hn_w)
    out = []
    for L, (kw, vw) in enumerate(kv_w):
        k = (th @ kw.T).reshape(-1, HD)
        v = th @ vw.T
        kn = (k / np.sqrt((k**2).mean(-1, keepdims=True) + 1e-6)) * np.asarray(
            kn_w[L], np.float32
        )
        out.append((rope_ref(kn, positions), v))
    return th, out


def gate_format(rows=8, verbose=True):
    """Does q4k instead of AWQ int4 change the pre-pass's answer?

    The waves stream q4k because that is what the projection engine eats
    (`w = q*scale + min` per 32 columns); the current pre-pass is AWQ
    (`w = (q-z)*scale` per 128). The two conventions are kept in separate files
    with nothing shared (dflash_int4.py:25-30) precisely because mixing them
    produces plausible garbage, so this compares OUTPUTS, not encodings.

    q4k's group of 32 is FINER than AWQ's 128, so the expectation is that q4k is
    at least as close to bf16 as int4 is. bf16 is the reference; the number that
    matters is q4k-vs-bf16 against int4-vs-bf16, not q4k-vs-int4.
    """
    import dflash_ctxkv_int4_builder as CK
    import dflash_int4 as I
    from qwen3_4b_draft_weights import DraftWeights
    from qwen3_4b_q4nx_requant import _requant_q4k

    fd = _load_draft_fd()
    dw = DraftWeights()
    G = fd.GROUP

    def _q4k_dq(W):
        q, s, m = _requant_q4k(np.asarray(W, np.float32), G)
        return (
            q.reshape(W.shape[0], -1, G).astype(np.float32) * s[..., None]
            + m[..., None]
        ).reshape(W.shape)

    def _awq_dq(W):
        return I.awq_dequantize(*I.awq_quantize(np.asarray(W, np.float32)))

    fc = np.asarray(dw.fc(), np.float32)
    hn = np.asarray(dw.hidden_norm())
    kv = [CK.layer_kv_weights(dw, L) for L in range(fd.UNI_DEC)]
    kn = [
        np.asarray(dw.bf16(f"layers.{L}.self_attn.k_norm.weight"))
        for L in range(fd.UNI_DEC)
    ]
    rng = np.random.default_rng(0)
    taps = rng.normal(0, 1, (rows, fc.shape[1])).astype(np.float32)
    pos = np.arange(137, 137 + rows)

    ref = _prepass_chain(fc, hn, kv, kn, taps, pos)
    q4k = _prepass_chain(
        _q4k_dq(fc), hn, [(_q4k_dq(k), _q4k_dq(v)) for k, v in kv], kn, taps, pos
    )
    awq = _prepass_chain(
        _awq_dq(fc), hn, [(_awq_dq(k), _awq_dq(v)) for k, v in kv], kn, taps, pos
    )

    def _rel(a, b):
        return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9))

    def _cos(a, b):
        a, b = a.reshape(-1), b.reshape(-1)
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))

    bad = 0
    if verbose:
        print(f"  {rows} context rows, {fd.UNI_DEC} layers; bf16 is the reference")
        print(f"  {'stage':<14} {'q4k rel':>9} {'int4 rel':>9} {'q4k cos':>9}")
    rowsout = [("target_hidden", q4k[0], awq[0], ref[0])]
    for L in range(fd.UNI_DEC):
        rowsout.append((f"k_ctx L{L}", q4k[1][L][0], awq[1][L][0], ref[1][L][0]))
        rowsout.append((f"v_ctx L{L}", q4k[1][L][1], awq[1][L][1], ref[1][L][1]))
    for name, a, b, r in rowsout:
        ea, eb, c = _rel(a, r), _rel(b, r), _cos(a, r)
        # q4k must not be WORSE than the int4 path already shipping, with a
        # little slack for the two codecs rounding differently on one outlier.
        ok = ea <= max(1.25 * eb, 1e-3) and c >= 0.99
        bad += not ok
        if verbose:
            print(
                f"  {name:<14} {ea:>9.3e} {eb:>9.3e} {c:>9.5f}"
                + ("" if ok else "   <-- q4k is worse")
            )
    return bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--format", action="store_true", help="q4k vs int4 vs bf16")
    ap.add_argument("--layout", action="store_true", help="the packing/window gate")
    ap.add_argument("--specs", action="store_true", help="print the wave table")
    ap.add_argument("--cache", default=None, help="drafter requant .npz")
    args = ap.parse_args()
    if not (args.layout or args.specs or args.format):
        args.layout = args.specs = args.format = True

    rc = 0
    if args.specs:
        fd = _load_draft_fd()
        waves, total = wave_specs(fd)
        print(
            f"\n[prepass waves] {len(waves)} extra waves, extra BO "
            f"{total} elements = {total * 2 / 1e6:.1f} MB "
            f"({total * 2 / (fd.W_LAYER * 2):.2f} of a drafter layer slab)"
        )
        print(
            f"  {'wave':<8} {'M':>6} {'K':>6} {'I2':>4} {'J2':>4} "
            f"{'iter_lo':>7} {'blocks':>7} {'MB':>6}  {'X slots':<12} "
            f"{'dest':<5} w_off"
        )
        for w in waves:
            print(
                f"  {w.name:<8} {w.m:>6} {w.k:>6} {w.i2:>4} {w.j2:>4} "
                f"{w.iter_lo:>7} {w.blocks:>7} "
                f"{w.blocks * BLOCK_BF16 * 2 / 1e6:>6.1f}  "
                f"{f'{w.x_slot}+{w.k // 2560}x{w.x_stride}':<12} "
                f"{w.dest:<5} {w.w_off}"
            )
    if args.layout:
        print("\n[prepass waves] layout gate")
        rc += gate_layout(args.cache)
    if args.format:
        print("\n[prepass waves] format gate: q4k against the shipping int4 path")
        rc += gate_format()
    print("\n" + ("PASS" if not rc else f"FAIL ({rc})"))
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
