# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8 dense MLP for two tokens at once, the speculative decode's verify
# pass: mlp_block.py's tile map, each weight block read once for both tokens
# (kernels/q4k_mvm2.cc).
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array: gate/up, then down
#   column 2      the assemble memtile: x [2][D], then each K half of the GLU
#                 output [2][I / 2]
#   (5, 5)        GLU: SiLU(gate) * up of each round, a K half at a time
#   (2, 2)        the down rows, to the output [2][D]
#
# Two tokens' x for the down (2 x 17408) does not fit a core's L1, so the down
# runs its K halves outermost: each core keeps an accumulator per row group
# and flushes the rows in the second half.
import argparse
import os
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import filelock
import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))
# llms/ is imported as the air_examples package, not through sys.path
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

import common  # noqa: E402
import mlp_block as M1  # noqa: E402
import qwen38_27b_q4nx_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.moe.array import HDR, MvmArray  # noqa: E402

R, RB, CB, BB = M1.R, M1.RB, M1.CB, M1.BB
D, I = W.D, W.I
NCB_X, NCB_H = M1.NCB_X, M1.NCB_H
RG_GU, RG_DN = M1.RG_GU, M1.RG_DN
NCOL, COLUMNS, ASM = M1.NCOL, M1.COLUMNS, M1.ASM
GLU, OUT = M1.GLU, M1.OUT
GLU_AT, OUT_AT = M1.GLU_AT, M1.OUT_AT
HALVES = 2
IH = I // HALVES  # a K half of the down's input, per token
# the element-wise atol, over max |want|
ATOL = 2e-2


@dataclass
class Phase2:
    """One projection for both tokens: ``nrg`` row groups per core of
    ``ncb`` column blocks, its K in ``halves`` parts of ``xw`` per token, and
    the consumer ``dest`` of its rows."""

    xw: int
    nrg: int
    ncb: int
    halves: int
    dest: int


class TwoTokenArray(MvmArray):
    """The GEMV array with q4k_step2: a packet carries both tokens' rows of
    its row block (row_block = 2 x 32), and a projection whose K comes in
    parts keeps an accumulator per row group."""

    def __init__(self, step, flush, **kw):
        super().__init__(**kw)
        self.step2, self.flush2 = step, flush

    def _phases(self, phases, xs, wab, yacc, y, flush, tx, ty):
        rb, x = self.row_block, xs[0]
        for ph in phases:
            ncbh = ph.ncb // ph.halves
            nblk = ph.nrg * ncbh
            assert nblk % 2 == 0
            for h in range(ph.halves):
                self.x_b.get(x[0 : 2 * ph.xw], indices=[tx, ty])
                for k in air.sequential(0, nblk // 2):
                    for half, wb in enumerate(wab):
                        i = k * 2 + half
                        cbl = i % ncbh
                        slot = i // ncbh if ph.halves > 1 else 0
                        self.w_l1.get(wb, indices=[tx, ty])
                        self.step2(x, wb, yacc, cbl + h * ncbh, ph.xw, h * ncbh, slot)
                        if h == ph.halves - 1:
                            with air.ops.branch(cbl == ncbh - 1):
                                self.flush2(yacc, y, slot)
                                self.out_a.put(
                                    y[16 - HDR : 16 + rb],
                                    indices=[tx, ty],
                                    dest=ph.dest,
                                )


def build_module():
    obj = dict(link_with="q4k_mvm2.o")
    step = air.extern("q4k_step2", scalars=[i32] * 4, **obj)
    flush = air.extern("q4k_flush2", scalars=[i32], **obj)
    g_ = lambda n: air.extern(n, link_with="q38_ffn.o", scalars=[i32, i32])  # noqa
    glu2, copy2 = g_("glu2_round"), g_("copy2_round")

    arr = TwoTokenArray(
        step,
        flush,
        row_block=2 * RB,
        block_bytes=BB,
        assemble_col=ASM,
        acc_words=4 * RB * (RG_DN // NCOL),
        columns=COLUMNS,
    )

    # each shim channel's share one run: [column][channel][...][2 row blocks]
    WGU = air.tensor([NCOL, 2, RG_GU // NCOL * NCB_X * 2 * BB], ui8)
    WDN = air.tensor([NCOL, 2, RG_DN // NCOL * NCB_H * 2 * BB], ui8)
    X = air.tensor([2 * D], bf16)
    OUT_T = air.tensor([2 * D], bf16)

    x_l2 = air.channel("XL2")
    arr.channels(n_dests=2)
    glu_out = air.channel("GluOut")
    out_ch = air.channel("Out")

    with air.launch(name="mlp2") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    x_l2.put(X)
                    out_ch.get(OUT_T)
                    arr.put_weights(lambda i, h: [WGU[i, h, :]])
                    arr.put_weights(lambda i, h: [WDN[i, h, :]])

                    def l2(shape):
                        return air.alloc(shape, bf16, scope=seg.private(), column=ASM)

                    # assemble memtile: x, then the GLU output a K half at a time
                    l2x, hs = l2([2 * D]), l2([2 * IH])
                    x_l2.get(l2x)
                    for _h in range(HALVES):
                        glu_out.get(hs)
                    arr.broadcast(lambda i: [l2x] + [hs] * HALVES)
                    arr.gather(seg, RG_GU // NCOL + RG_DN // NCOL)
                    arr.feed(seg, RG_GU // NCOL * NCB_X + RG_DN // NCOL * NCB_H)

                    with air.herd([range(1)], name="glu", shape=(1,), at=GLU_AT) as hg:

                        @hg.body
                        def _(tx):
                            p = hg.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            hh = air.alloc([2 * IH], bf16, scope=p)
                            nr = RG_GU // NCOL // HALVES
                            for _h in range(HALVES):
                                for j in air.sequential(0, nr):
                                    arr.out_y.get(u, indices=[0, GLU])
                                    glu2(hh, u, j, IH)
                                glu_out.put(hh)

                    with air.herd([range(1)], name="out", shape=(1,), at=OUT_AT) as ho:

                        @ho.body
                        def _(tx):
                            p = ho.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            y = air.alloc([2 * D], bf16, scope=p)
                            for j in air.sequential(0, RG_DN // NCOL):
                                arr.out_y.get(u, indices=[0, OUT])
                                copy2(y, u, j, D)
                            out_ch.put(y)

                    def phases():
                        yield Phase2(D, RG_GU // NCOL, NCB_X, 1, GLU)
                        yield Phase2(IH, RG_DN // NCOL, NCB_H, HALVES, OUT)

                    arr.herd(2 * IH, None, phases)

    return launch.build(target="npu2")


def per_channel(t, halves=1):
    """A projection's row groups [rg, ncb, 4, BB] -> [column, channel, ...]:
    column i takes row groups i, i + 4, ...; channel h row blocks 2h, 2h + 1;
    in each, the blocks in the order the cores take them: K half, row group,
    column block."""
    rg, ncb = t.shape[:2]
    t = t.reshape(rg // NCOL, NCOL, halves, ncb // halves, 2, 2, BB)
    # [col, ch, half, rg', cb', 2 row blocks, BB]
    t = t.transpose(1, 4, 2, 0, 3, 5, 6)
    return np.ascontiguousarray(t).reshape(NCOL, 2, -1)


def weights(w, layer):
    m = lambda n: np.asarray(w.raw(w.name(layer, f"mlp.{n}.weight")))  # noqa: E731
    gu = per_channel(M1.interleave(m("up_proj"), m("gate_proj")))
    dn = per_channel(M1.down(m("down_proj")), HALVES)
    return gu, dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", default="9338,760")
    ap.add_argument("--bench", type=int, default=0, help="time n dispatches")
    a = ap.parse_args()
    module = build_module()
    if a.print_module_only:
        print(module)
        return 0
    for k in ("q4k_mvm2", "q38_ffn"):
        common.compile_kernel(k)
    w = W.Qwen38Q4nx()
    toks = [int(v) for v in a.tokens.split(",")]
    e = w.embed(toks)
    n2 = w.norms(a.layer)[1]
    x = (e / np.sqrt((e * e).mean(-1, keepdims=True) + W.EPS) * n2).astype(bfloat16)
    gu, dn = weights(w, a.layer)
    be = XRTBackend(instance_name="mlp2", output_format="elf")
    art = be.compile(module)
    if a.bench:
        return M1.bench(
            art, [gu, dn, x.ravel(), np.zeros(2 * D, bfloat16)], a.bench, (0, 1)
        )
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        out = run(gu, dn, x.ravel(), np.zeros(2 * D, bfloat16))[3]
        be.unload()

    from reference import Reference, t

    p = Reference(w, layers=[a.layer]).weights(a.layer)
    want = Reference.mlp(t(x.astype(np.float32)), p).double().numpy()
    out = np.asarray(out).reshape(2, D)
    ok = all(common.check(f"token {k}", out[k], want[k], ATOL) for k in range(2))
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
