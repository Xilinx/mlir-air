# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS prefill GEMM, as FastFlowLM's FLM_Xclbin/MM/host7_mt_variant_bias
# lays it out on the 8x4 array: C[M, N] = mode(A[M, K] @ W[N, K]^T + bias).
#
#   - core (x, y) owns an m x 128 tile of C; a round covers 4 m-row blocks
#     (one per row) by 8 column panels (one per column);
#   - row y's A blocks come in through memtile 2y and are broadcast along the
#     row, 32 columns of K at a time;
#   - column x's weight panel streams through memtile x and down the column,
#     once per round, with the bias in front of it;
#   - the column memtile gathers its four tiles and writes them out.
#
# The number of rounds is a run-time scalar: one build serves every prompt
# length up to the A/C buffers' rows.
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import filelock
import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kernel_tests"))

from air import api as air  # noqa: E402
from air.api import bf16, f32, i32  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402

COLS, ROWS = 8, 4
TN, TK, KC = 128, 128, 32  # tile n, memtile k, K per core step
R, S, T = 8, 8, 8  # mmul shape
LEAD = 2  # bias steps in front of a panel


def kernel_obj(tm):
    return f"mm_prefill_m{tm}.o"


def compile_kernel(tm):
    """kernels/mm_prefill.cc for m-row tiles, into ./kernel_obj(tm)."""
    import common

    common.compile_kernel("mm_prefill", [f"-DMM_M={tm}"])
    os.replace("mm_prefill.o", kernel_obj(tm))


def pack_weights(w, bias=None):
    """W [N, K] (+ bias [N]) -> per 128-column panel, the steps a core takes:
    [panel][step][N/t][KC/s][s][t], led by two steps (the cores' buffer pair)
    whose first row is the bias."""
    n, k = w.shape
    assert n % TN == 0 and k % KC == 0
    b = w.reshape(n // TN, TN // T, T, k // KC, KC // S, S)
    b = b.transpose(0, 3, 1, 4, 5, 2).reshape(n // TN, k // KC, KC * TN)
    if bias is not None:
        lead = np.zeros((n // TN, LEAD, KC * TN), w.dtype)
        lead[:, 0, :TN] = bias.reshape(n // TN, TN)
        b = np.concatenate([lead, b], axis=1)
    return np.ascontiguousarray(b)


def build_module(k, n, max_m, tm=64, bias=True, pstride=1):
    """Column panel j of W is panel slot j * pstride."""
    assert k % TK == 0 and n % (COLS * TN) == 0
    round_m = ROWS * tm
    assert max_m % round_m == 0
    nn, kb, steps = n // (COLS * TN), k // TK, k // KC
    lead = LEAD if bias else 0
    panel = (lead + steps) * KC * TN
    max_rounds = max_m // round_m

    ex = lambda name, s=(): air.extern(  # noqa: E731
        name, link_with=kernel_obj(tm), scalars=list(s)
    )
    mm_init, mm_step = ex("mmp_init", [i32]), ex("mmp_step")
    mm_out = ex("mmp_out", [i32, i32])

    A = air.tensor([max_m, k], bf16)
    slots = (n // TN - 1) * pstride + 1
    W = air.tensor([slots, panel], bf16)
    C = air.tensor([max_m, n], bf16)
    NR = air.tensor([], i32, name="rounds")
    MODE = air.tensor([], i32, name="mode")

    a_l2 = air.channel("AL2", size=[ROWS])
    a_l1 = air.channel("AL1", size=[1, ROWS], broadcast_shape=[COLS, ROWS])
    b_l2 = air.channel("BL2", size=[COLS])
    b_l1 = air.channel("BL1", size=[COLS, 1], broadcast_shape=[COLS, ROWS])
    c_l1 = air.channel("CL1", size=[COLS, ROWS])
    c_l3 = air.channel("CL3", size=[COLS])

    # one dispatch of the launch per block of 8 column panels: a shim tile
    # then holds one task per channel
    with air.launch(name="mm", repeat=nn) as launch:

        @launch.body
        def _(nb):
            nr = air.rtp(NR.value)
            md = air.rtp(MODE.value)
            av = A.reshape([max_rounds, round_m, kb, TK])
            cv = C.reshape([max_rounds, 1, round_m, n])
            wv = W.reshape([slots, 1, lead + steps, KC * TN])
            for y in range(ROWS):
                a_l2.put(
                    av[:, y * tm : (y + 1) * tm, :, :].transpose(0, 2, 1, 3),
                    indices=[y],
                    extent=NR,
                )
            for x in range(COLS):
                p = nb * COLS + x
                s = p * pstride
                b_l2.put(
                    wv[s : s + 1, :, :, :].broadcast_to(
                        [max_rounds, 1, lead + steps, KC * TN]
                    ),
                    indices=[x],
                    extent=NR,
                )
                c_l3.get(
                    cv[:, :, :, p * TN : (p + 1) * TN],
                    indices=[x],
                    extent=NR,
                )
            with air.segment(name="seg", params=[nr, md]) as seg:

                @seg.body
                def _():
                    sp = seg.private()
                    nrounds = nr.as_index()
                    for y in range(ROWS):
                        for _b in air.sequential(0, nrounds * kb):
                            la = air.alloc([tm, TK], bf16, scope=sp, column=2 * y)
                            a_l2.get(la, indices=[y])
                            lv = la.reshape([tm // R, R, TK // S, S])
                            for c in range(TK // KC):
                                q = KC // S
                                a_l1.put(
                                    lv[:, :, c * q : (c + 1) * q, :].transpose(
                                        0, 2, 1, 3
                                    ),
                                    indices=[0, y],
                                )
                            air.dealloc(la)
                    for x in range(COLS):
                        for _b in air.sequential(0, nrounds * (lead + steps)):
                            lb = air.alloc([KC * TN], bf16, scope=sp, column=x)
                            b_l2.get(lb, indices=[x])
                            b_l1.put(lb, indices=[x, 0])
                            air.dealloc(lb)
                    for x in range(COLS):
                        for _r in air.sequential(0, nrounds):
                            lc = air.alloc([round_m * TN], bf16, scope=sp, column=x)
                            for y in range(ROWS):
                                c_l1.get(
                                    lc[y * tm * TN : (y + 1) * tm * TN], indices=[x, y]
                                )
                            c_l3.put(lc, indices=[x])
                            air.dealloc(lc)

                    with air.herd(
                        [range(COLS), range(ROWS)],
                        name="mm",
                        shape=(COLS, ROWS),
                        at=(0, 2),
                        params=[nr, md],
                    ) as he:

                        @he.body
                        def _(tx, ty):
                            p = he.private()
                            ab = [air.alloc([tm * KC], bf16, scope=p) for _ in range(2)]
                            bb = [air.alloc([KC * TN], bf16, scope=p) for _ in range(2)]
                            acc = air.alloc([tm * TN], f32, scope=p)
                            ob = [air.alloc([R * TN], bf16, scope=p) for _ in range(2)]
                            for _r in air.sequential(0, nr.as_index()):
                                if bias:
                                    for h in range(2):
                                        b_l1.get(bb[h], indices=[tx, ty])
                                mm_init(bb[0], acc, 1 if bias else 0)
                                for _s in air.sequential(0, steps // 2):
                                    for h in range(2):
                                        a_l1.get(ab[h], indices=[tx, ty])
                                        b_l1.get(bb[h], indices=[tx, ty])
                                        mm_step(ab[h], bb[h], acc)
                                for i in air.sequential(0, tm // R // 2):
                                    for h in range(2):
                                        mm_out(acc, ob[h], i * 2 + h, md.value)
                                        c_l1.put(ob[h], indices=[tx, ty])

    return launch.build(target="npu2")


BACKEND = {}


def build_gemm(build_dir, k, n, max_m, tm=64, pstride=1):
    """Compile one GEMM shape; returns (its directory, ELF, kernel name)."""
    out = Path(build_dir) / f"mm_k{k}_n{n}_m{max_m}_t{tm}_s{pstride}"
    out.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(out)
    try:
        compile_kernel(tm)
        be = XRTBackend(output_format="elf", instance_name="mm", **BACKEND)
        art = be.compile(build_module(k, n, max_m, tm, pstride=pstride))
    finally:
        os.chdir(cwd)
    return out, out / art.output_binary, art.kernel


class GemmElf:
    """One GEMM shape as a full ELF on its own context, with A and C resident;
    weights are BOs the caller makes once (weight_bo) and passes per call."""

    def __init__(self, build_dir, k, n, max_m, tm=64, pstride=1):
        import pyxrt as xrt
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )
        from air.backend.xrt import _HostSide

        self.xrt, self.k, self.n, self.max_m, self.tm = xrt, k, n, max_m, tm
        out, elf, kernel = build_gemm(build_dir, k, n, max_m, tm, pstride)
        self.dev = xrt.device(0)
        self.ctx = xrt.hw_context(self.dev, xrt.elf(str(elf)))
        self.kern = xrt.ext.kernel(self.ctx, kernel)
        self.a = xrt.ext.bo(self.dev, max_m * k * 2)
        self.c = xrt.ext.bo(self.dev, max_m * n * 2)
        self.am = np.frombuffer(self.a.map(), bfloat16).reshape(max_m, k)
        self.cm = np.frombuffer(self.c.map(), bfloat16).reshape(max_m, n)
        self.am[:] = 0
        self.a.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.run_ = xrt.run(self.kern)
        self.device_s = 0.0  # start to completion, summed
        self.last = None
        self.run_.set_arg(0, self.a)
        self.run_.set_arg(2, self.c)
        self.params = ParameterScratchpad(self.run_, str(out / "params.txt"))
        self.rules = _HostSide(str(out / "air_project" / "host")).rules(
            str(out / "params.txt")
        )
        self.panel = (LEAD + k // KC) * KC * TN

    def weight_bo(self, w, bias):
        """w [N', K'] and bias [N'] padded with zeros to this shape, packed."""
        wp = np.zeros((self.n, self.k), bfloat16)
        wp[: w.shape[0], : w.shape[1]] = w
        bp = np.zeros(self.n, bfloat16)
        if bias is not None:
            bp[: len(bias)] = bias
        packed = pack_weights(wp, bp)
        bo = self.xrt.ext.bo(self.dev, packed.nbytes)
        np.frombuffer(bo.map(), bfloat16)[:] = packed.ravel()
        bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return bo

    def __call__(self, a, w_bo, mode=0):
        """mode(a @ w.T + bias) for a [m, K' <= K]: m rows padded to a round.
        An int `a` is the row count of the A this ELF holds from its last call."""
        self.start(a, w_bo, mode)
        return self.wait()

    def start(self, a, w_bo, mode=0):
        """Queue the GEMM; wait() collects it."""
        xrt = self.xrt
        m = a if isinstance(a, int) else a.shape[0]
        rounds = -(-m // (ROWS * self.tm))
        self.rows = rows = rounds * ROWS * self.tm
        assert rows <= self.max_m
        if not isinstance(a, int):
            # pad rows only make output rows nobody reads; pad columns stay 0
            self.am[:m, : a.shape[1]] = a
            # only the rows this call moves
            self.a.sync(
                xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, rows * self.k * 2, 0
            )
        self.run_.set_arg(1, w_bo)
        vals = {3: rounds, 4: mode}
        if vals != self.last:  # the scratchpad keeps the last call's
            for name, arg, scale, addend in self.rules:
                self.params.write(name, np.int32(scale * vals[arg] + addend))
            self.params.sync()
            self.last = vals
        self.t0 = time.perf_counter()
        self.run_.start()

    def wait(self):
        """The started GEMM's output rows, a view of this ELF's C."""
        xrt = self.xrt
        state = self.run_.wait(60000)
        self.device_s += time.perf_counter() - self.t0
        if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"prefill GEMM: {state}")
        self.c.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
            self.rows * self.n * 2,
            0,
        )
        return self.cm[: self.rows]


def reference(a, w, bias, mode):
    y = a.astype(np.float32) @ w.astype(np.float32).T
    if bias is not None:
        y = y + bias.astype(np.float32)
    y = y.astype(bfloat16).astype(np.float32)
    if mode == 1:
        y = np.minimum(y, 7.0)
        y = y / (1 + np.exp(-1.702 * y))
    elif mode == 2:
        y = np.clip(y, -7.0, 7.0) + 1
    return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2944)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--max-m", type=int, default=1024)
    ap.add_argument("--tm", type=int, default=64)
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--m", type=str, default="256,768,512,1024")
    ap.add_argument("--build-dir", default="build_prefill_gemm")
    ap.add_argument("-p", "--print-module-only", action="store_true")
    args = ap.parse_args()

    if args.print_module_only:
        print(build_module(args.k, args.n, args.max_m, args.tm, not args.no_bias))
        sys.exit(0)

    rng = np.random.default_rng(0)
    w = (rng.standard_normal((args.n, args.k)) * 0.05).astype(bfloat16)
    b = None if args.no_bias else (rng.standard_normal(args.n) * 0.5).astype(bfloat16)
    ok = True
    build_gemm(args.build_dir, args.k, args.n, args.max_m, args.tm)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        mm = GemmElf(args.build_dir, args.k, args.n, args.max_m, args.tm)
        wb = mm.weight_bo(w, b)
        for i, m in enumerate(int(v) for v in args.m.split(",")):
            mode = i % 3
            a = (rng.standard_normal((m, args.k)) * 0.5).astype(bfloat16)
            mm.cm[:] = 7.5
            mm.c.sync(mm.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            got = mm(a, wb, mode)
            want = reference(a, w, b, mode)
            err = np.abs(got[:m].astype(np.float32) - want).max()
            tol = 0.02 * np.abs(want).max() + 1e-2
            untouched = np.all(mm.cm[len(got) :].astype(np.float32) == 7.5)
            good = err <= tol and untouched
            ok &= good
            print(
                f"M={m} mode {mode}: max err {err:.4f} (tol {tol:.4f}) rest "
                f"untouched={untouched} {'ok' if good else 'MISMATCH'}"
            )
    print("PASS!" if ok else "failed.")
    sys.exit(0 if ok else 1)
