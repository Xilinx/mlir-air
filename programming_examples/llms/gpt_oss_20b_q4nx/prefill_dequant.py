# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS prefill expert dequant: a layer's MXFP4 expert weights to bf16, laid
# out for the prefill GEMM, as FastFlowLM's
# FLM_Xclbin/GPTOSS/gpt_dequant_mxfp4_npu_bin does it on the 8x4 array: every
# column on its own, the shim streaming blocks into the memtile, which hands
# one block to each of the column's four cores and gathers what they return.
#
# The input is the layer's row groups [group][column block][4 row blocks]
# [2560 bytes]; column x takes a contiguous share of the groups. Group g's
# 128 rows become panel slot g of the output, whose steps the GEMM reads (the
# bias steps in front of them are the host's).
import argparse
import functools
import os
import sys
import tempfile
import time
from pathlib import Path

import filelock
import numpy as np
import torch
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))

from air import api as air  # noqa: E402
from air.api import bf16, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402

import prefill_gemm as pg  # noqa: E402

COLS, ROWS = 8, 4
BLOCK = 2560  # bytes of one 32 x 128 block
STEP = pg.KC * pg.TN  # elements of one GEMM step


def panel_elems(k):
    return (pg.LEAD + k // pg.KC) * STEP


def build_module(groups, ncb):
    """`groups` row groups of `ncb` column blocks each."""
    assert groups % COLS == 0
    gpc = groups // COLS
    tiles = gpc * ncb
    k = ncb * pg.TK
    pe = panel_elems(k)

    dq = air.extern("dqp_block", link_with="dequant_prefill.o")

    QW = air.tensor([groups, ncb * ROWS * BLOCK], ui8)
    OUT = air.tensor([groups + 1, pe], bf16)

    q_l2 = air.channel("QL2", size=[COLS])
    q_l1 = air.channel("QL1", size=[COLS, ROWS])
    o_l1 = air.channel("OL1", size=[COLS, ROWS])
    o_l3 = air.channel("OL3", size=[COLS])

    with air.launch(name="dq", repeat=1) as launch:

        @launch.body
        def _(w):
            for x in range(COLS):
                q_l2.put(QW[x * gpc : (x + 1) * gpc, :], indices=[x])
                o_l3.get(OUT[x * gpc : (x + 1) * gpc, pg.LEAD * STEP : pe], indices=[x])

            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    sp = seg.private()
                    for x in range(COLS):
                        for _t in air.sequential(0, tiles):
                            li = air.alloc([ROWS * BLOCK], ui8, scope=sp, column=x)
                            q_l2.get(li, indices=[x])
                            for r in range(ROWS):
                                q_l1.put(
                                    li[r * BLOCK : (r + 1) * BLOCK], indices=[x, r]
                                )
                            air.dealloc(li)
                    for x in range(COLS):
                        for _t in air.sequential(0, tiles):
                            lo = air.alloc(
                                [4, ROWS, STEP // 4], bf16, scope=sp, column=x
                            )
                            for r in range(ROWS):
                                o_l1.get(lo[:, r : r + 1, :], indices=[x, r])
                            o_l3.put(lo, indices=[x])
                            air.dealloc(lo)

                    with air.herd(
                        [range(COLS), range(ROWS)],
                        name="dq",
                        shape=(COLS, ROWS),
                        at=(0, 2),
                    ) as he:

                        @he.body
                        def _(tx, ty):
                            p = he.private()
                            ib = [air.alloc([BLOCK], ui8, scope=p) for _ in range(2)]
                            ob = [air.alloc([STEP], bf16, scope=p) for _ in range(2)]
                            for _t in air.sequential(0, tiles // 2):
                                for h in range(2):
                                    q_l1.get(ib[h], indices=[tx, ty])
                                    dq(ib[h], ob[h])
                                    o_l1.put(ob[h], indices=[tx, ty])

    return launch.build(target="npu2")


def compile_kernel():
    import common

    common.compile_kernel("dequant_prefill")


def biases(blocks):
    """The bf16 bias of each row group, [groups, 128], from its blocks."""
    b = np.ascontiguousarray(blocks[:, 0, :, 128:192]).view(bfloat16)
    return b.reshape(len(b), pg.TN)


class DequantElf:
    """The layer dequant as a full ELF on its own context: write the layer's
    blocks into `qw`, call, and the GEMMs read panel slots of `out`."""

    def __init__(self, build_dir, groups, ncb):
        import pyxrt as xrt

        self.xrt = xrt
        d = Path(build_dir) / f"dq_g{groups}_c{ncb}"
        d.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(d)
        try:
            compile_kernel()
            art = XRTBackend(output_format="elf", instance_name="dq").compile(
                build_module(groups, ncb)
            )
        finally:
            os.chdir(cwd)
        self.pe = panel_elems(ncb * pg.TK)
        self.dev = xrt.device(0)
        self.ctx = xrt.hw_context(self.dev, xrt.elf(str(d / art.output_binary)))
        self.kern = xrt.ext.kernel(self.ctx, art.kernel)
        self.qw = xrt.ext.bo(self.dev, groups * ncb * ROWS * BLOCK)
        self.out = xrt.ext.bo(self.dev, (groups + 1) * self.pe * 2)
        self.qm = np.frombuffer(self.qw.map(), np.uint8)
        self.om = np.frombuffer(self.out.map(), bfloat16).reshape(groups + 1, self.pe)
        self.om[:] = 0
        self.out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.run_ = xrt.run(self.kern)
        self.device_s = 0.0  # start to completion, summed
        self.run_.set_arg(0, self.qw)
        self.run_.set_arg(1, self.out)

    def __call__(self, blocks=None, src=None, bias=None):
        """Dequantize a layer: `blocks`, its uint8 [groups, ncb, 4, 2560], or
        `src`, a buffer already holding them. Each group's bias (the bf16 in
        column block 0's bytes 128..191, or `bias` [groups, 128]) goes into
        its first step."""
        self.start(blocks, src, bias)
        self.wait()

    def start(self, blocks=None, src=None, bias=None):
        """Queue the dequant; wait() collects it."""
        xrt = self.xrt
        if src is None:
            self.qm[:] = blocks.reshape(-1)
            self.qw.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            src = self.qw
        self.run_.set_arg(0, src)
        if bias is None:
            bias = biases(blocks)
        # the bias steps are the host's, apart from what the dequant writes
        self.om[:-1, : pg.TN] = bias
        for g in range(len(bias)):
            self.out.sync(
                xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                pg.TN * 2,
                g * self.pe * 2,
            )
        self.t0 = time.perf_counter()
        self.run_.start()

    def wait(self):
        state = self.run_.wait(60000)
        self.device_s += time.perf_counter() - self.t0
        if state != self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"prefill dequant: {state}")

    @functools.lru_cache(maxsize=None)
    def panels(self, slot, count):
        """A buffer of `count` panel slots from `slot` on, for a GEMM."""
        return self.xrt.bo(self.out, count * self.pe * 2, slot * self.pe * 2)


def reference(qw, groups, ncb):
    """The packed panels pg.pack_weights would make of the dequantized blocks,
    without the bias steps."""
    import gpt_oss_20b_q4nx_weights as g

    blocks = torch.from_numpy(qw.reshape(groups, ncb, ROWS, BLOCK))
    w, _ = g.dequant_mxfp4(g._ungroup_rows(blocks, ROWS))  # [groups*128, ncb*128]
    packed = pg.pack_weights(w.numpy().astype(bfloat16))  # [groups, steps, STEP]
    return packed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--ncb", type=int, default=23)
    ap.add_argument("-p", "--print-module-only", action="store_true")
    args = ap.parse_args()

    mod = build_module(args.groups, args.ncb)
    if args.print_module_only:
        print(mod)
        sys.exit(0)

    compile_kernel()
    rng = np.random.default_rng(0)
    qw = rng.integers(0, 256, (args.groups, args.ncb, ROWS, BLOCK), dtype=np.uint8)
    qw[..., :128] = rng.integers(118, 128, qw[..., :128].shape)  # 2^-9 .. 2^0
    pe = panel_elems(args.ncb * pg.TK)
    be = XRTBackend(output_format="elf", instance_name="dq")
    art = be.compile(mod)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        out = np.full((args.groups + 1, pe), 9.0, bfloat16)
        got = run(qw.reshape(args.groups, -1), out)[1].reshape(args.groups + 1, pe)
        be.unload()
    want = reference(qw, args.groups, args.ncb)
    body = got[: args.groups, pg.LEAD * STEP :].astype(np.float32)
    ok = np.array_equal(body, want.reshape(args.groups, -1).astype(np.float32))
    lead = np.all(got[:, : pg.LEAD * STEP].astype(np.float32) == 9.0)
    if not ok:
        d = np.abs(body - want.reshape(args.groups, -1).astype(np.float32))
        print("mismatches:", int((d > 0).sum()), "first at", np.argwhere(d > 0)[:4])
    print(f"weights exact: {ok}; bias steps untouched: {lead}")
    print("PASS!" if ok and lead else "failed.")
    sys.exit(0 if ok and lead else 1)
