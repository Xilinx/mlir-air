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

TWO WAVES, NOT TEN. This corrects an earlier reading of section 3.3:

  fc      is ONE 2560x12800 projection, I2=5 J2=25 -- not five accumulating
          2560x2560 ones. Accumulating across input column-blocks is exactly
          what the proj cores already do (`_gemv`/`_mm` loop 2*J2 col-blocks
          into one `yacc`), and `qwen3_4b_draft_requant.py` already packs it
          that way as `W_fc`.
  ctx K/V is the drafter's OWN k_proj/v_proj, already inside each drafter
          layer's phase-0 `concat([Wq;Wk;Wv])` slab. Nothing new is packed --
          only a row window is selected.

A wave is a whole launch iteration running ONE phase, which is the shape the
LM-head waves already have (`nph_v = _sel(idx(1), ... idx(NPH))`). That is why
`qwen3_4b_draft_requant.py`'s note about `fc` not fitting as a 5th PHASE (it
would break `FULL4`) does not apply here: this is a 5th kind of WAVE.

    python3 dflash_prepass_waves.py             # the layout gate
    python3 dflash_prepass_waves.py --format    # q4k vs AWQ-int4 vs bf16
"""

import argparse
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
    x_src: str  # "taps" (raw DDR) or "xnorm" (the rms core's output)
    dest: str  # "rms" (hidden_norm) or "rope" (k_norm + RoPE)

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

    waves = [
        WaveSpec(
            name="fc",
            m=D,
            k=FC_IN,
            i2=g_fc.nbi_pc,
            j2=g_fc.nbj // 2,
            iter_lo=0,
            w_off=0,
            x_src="taps",
            dest="rms",
        )
    ]
    off = g_fc.n_blocks * BLOCK_BF16
    kv_blocks = kv_n_iter * g_kv.nbj * g_kv.blocks_per_step * g_kv.NCX * g_kv.n_chan
    for L in range(n_layers):
        waves.append(
            WaveSpec(
                name=f"ctxkv{L}",
                m=kv_hi - kv_lo,
                k=D,
                i2=kv_n_iter,
                j2=g_kv.nbj // 2,
                iter_lo=kv_iter_lo,
                w_off=off,
                x_src="xnorm",
                dest="rope",
            )
        )
        off += kv_blocks * BLOCK_BF16
    return waves, off


def build_extra_weights(fd_draft, npz, verbose=True):
    """The compact extra-weight BO: [fc slab | K/V window per drafter layer].

    Each layer's window is gathered as `NCX * n_chan` contiguous runs -- one per
    (column, shim channel) -- and re-emitted in the same (cx, h) order the feed
    walks, so `_feed_wcol`'s two puts per column need only a base and a length.
    """
    waves, total = wave_specs(fd_draft)
    from qwen3_4b_draft_weights import D

    kv_lo, kv_hi = _kv_rows(fd_draft)
    g_kv = geom_for(kv_hi, D, fd_draft)
    it_lo, n_it = g_kv.iter_window(kv_lo, kv_hi)
    first, n_blk = g_kv.chan_window(it_lo, n_it)

    W = np.asarray(npz["W"]).reshape(-1)
    fc = np.asarray(npz["W_fc"]).reshape(-1)
    parts = [fc]
    for L in range(fd_draft.UNI_DEC):
        lay = W[L * fd_draft.W_LAYER : (L + 1) * fd_draft.W_LAYER]
        for cx in range(g_kv.NCX):
            for h in range(g_kv.n_chan):
                base = cx * g_kv.blocks_per_col + h * g_kv.blocks_per_chan + first
                parts.append(lay[base * BLOCK_BF16 : (base + n_blk) * BLOCK_BF16])
    out = np.concatenate(parts)
    if out.size != total:
        raise AssertionError(f"extra BO is {out.size} elements, layout says {total}")
    if verbose:
        print(
            f"[prepass waves] extra BO {out.size} elements = "
            f"{out.size * 2 / 1e6:.1f} MB over {len(waves)} waves"
        )
    return out, waves


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
            f"{'iter_lo':>7} {'blocks':>7} {'MB':>6}  {'X':<6} {'dest':<5} w_off"
        )
        for w in waves:
            print(
                f"  {w.name:<8} {w.m:>6} {w.k:>6} {w.i2:>4} {w.j2:>4} "
                f"{w.iter_lo:>7} {w.blocks:>7} "
                f"{w.blocks * BLOCK_BF16 * 2 / 1e6:>6.1f}  "
                f"{w.x_src:<6} {w.dest:<5} {w.w_off}"
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
