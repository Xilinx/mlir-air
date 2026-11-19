# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The parts of a decode on FastFlowLM's Qwen3.5 tile map that do not depend
# on its mixer or FFN: one-core herds, memtile buffers, the wave's x traffic,
# and the host side's LM head run and timing loop.
import time

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import bf16


def wave_loop(n):
    return air.sequential(0, n)


def herd(name, at, params=()):
    """A one-core herd at `at`."""
    return air.herd([range(1)], name=name, shape=(1,), at=at, params=list(params))


def l2(seg, shape, col, dt=bf16):
    return air.alloc(shape, dt, scope=seg.private(), column=col)


def wave_x(m_in, out_ch, x, w, pw):
    """x in, then the wave's layers' outputs, each one run of DDR."""
    # x in before the outputs: with the wave a loop index the two regions of X
    # are not known apart, and a read after the write would wait for the whole
    # wave
    m_in.put(x[w, pw - 1, :])
    out_ch.get(x[w + 1, :, :])


class Decoder:
    """The LM head's ELF on the last layer's output, and the timing loop.
    Subclasses build self.bos (bos[1] = X) and self.run, then call
    attach_lm."""

    def __init__(self, waves, pw, d, unpack_logits):
        import pyxrt as xrt

        self.xrt, self.waves, self.pw, self.d = xrt, waves, pw, d
        self.dev = xrt.device(0)
        self.unpack_logits = unpack_logits

    def bo(self, t, extra=0):
        xrt = self.xrt
        b = np.ascontiguousarray(t).view(np.uint8).ravel()
        o = xrt.ext.bo(self.dev, b.nbytes + extra)
        m = np.frombuffer(o.map(), np.uint8)
        m[: b.nbytes] = b
        m[b.nbytes :] = 0
        o.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return o

    def kernel(self, art):
        xrt = self.xrt
        ctx = xrt.hw_context(self.dev, xrt.elf(art.output_binary))
        try:
            return ctx, xrt.ext.kernel(ctx, art.kernel)
        except RuntimeError:  # a repeated launch's kernel sits in the segment
            return ctx, xrt.ext.kernel(ctx, art.kernel.replace("main:", "seg:"))

    def attach_lm(self, lart, lm_args):
        waves, pw, d = self.waves, self.pw, self.d
        self.xmap = np.frombuffer(self.bos[1].map(), bfloat16).reshape(waves + 1, pw, d)
        xsub = self.xrt.bo(self.bos[1], d * 2, (waves * pw + pw - 1) * d * 2)
        self.lbos = [xsub] + [self.bo(t) for t in lm_args]
        self.lctx, lkern = self.kernel(lart)
        self.lrun = self.xrt.run(lkern)
        for i, b in enumerate(self.lbos):
            self.lrun.set_arg(i, b)

    def _go(self, runs, L):
        xrt = self.xrt
        for r in runs:
            r.start()
            st = r.wait(120000)
            if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"dispatch at L={L} ended in {st}")
        yb = self.lbos[-1]
        yb.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return self.unpack_logits(np.frombuffer(yb.map(), bfloat16)).astype(np.float32)

    def head(self, x, L):
        """The LM head alone on the last layer's output x, at position L - 1."""
        self.xmap[self.waves, self.pw - 1] = x
        self.bos[1].sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return self._go((self.lrun,), L)

    def bench(self, x, L, n):
        """n tokens at context length L, the layers' run then the LM head's;
        the state is not carried (the same inputs every time)."""
        self._set(x, L)
        t = []
        for _ in range(n + 1):
            t0 = time.perf_counter()
            self.run.start()
            self.run.wait(120000)
            t1 = time.perf_counter()
            self.lrun.start()
            self.lrun.wait(120000)
            t.append((t1 - t0, time.perf_counter() - t1))
        t = np.array(t[1:]) * 1e3
        tot = t.sum(1)
        med = float(np.median(tot))
        print(
            f"L={L}, {n} tokens: layers {np.median(t[:, 0]):.3f} ms + LM head"
            f" {np.median(t[:, 1]):.3f} ms = {med:.3f} ms median"
        )
        print(
            f"[bench] decode ctx={L} median {med:.2f} ms min {tot.min():.2f} ms "
            f"({1e3 / med:.2f} tok/s)"
        )
