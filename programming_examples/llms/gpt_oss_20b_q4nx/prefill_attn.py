# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS prefill attention with sinks, as FastFlowLM's
# FLM_Xclbin/GPTOSS/gpt_prefill_attn_v1_npu_bin lays it out: four units of
# two columns by four rows, a unit on one query head at a time, a core on 16
# of the round's 128 query rows; the K/V blocks of the head's KV head stream
# once through one memtile to all 32 cores.
#
# One dispatch is one round of 128 query rows over all 16 head groups (FLM
# loops its rounds inside one dispatch, its runtime sequence built per call).
# Its scalars: the round's first query row, the first key block, the number
# of key blocks, and the window (FLM's L_begin, kv_begin, kv_length, window).
# The two head groups of a KV head take its blocks together, so each block
# streams once.
#
# Host layouts (pack_q, pack_kv, unpack_o):
#   q   [column][head group][row][16 x 64 in 8x8 blocks, sink, pad]
#   kv  [KV head][128-key block][K then V: 8 steps of 16 keys each]
#   o   [column][head group][row][16 x 64 in 8x8 blocks]
import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kernel_tests"))

from air import api as air  # noqa: E402
from air.api import bf16, f32, i32  # noqa: E402

COLS, ROWS = 8, 4
CUS = COLS // 2
HEADS, KV_HEADS, DH = 64, 8, 64
GROUPS = HEADS // CUS  # head groups: one head per unit
LQ = 16  # query rows per core
ROUND = 128  # query rows per round, keys per block
QL = LQ * DH + 32  # a core's q, then its head's sink as f32
BLK = 2 * ROUND * DH  # one key block: K then V


def build_module(max_blocks):
    """Key blocks per KV head of the cache: max_blocks."""
    ex = lambda n, s=(): air.extern(  # noqa: E731
        n, link_with="attn_prefill.o", scalars=list(s)
    )
    f_init, f_qk = ex("apf_init"), ex("apf_qk", [i32, i32, i32, i32])
    f_sm, f_fv, f_out = ex("apf_softmax"), ex("apf_fv", [i32]), ex("apf_out")

    Q = air.tensor([COLS, GROUPS * ROWS * QL], bf16)
    KV = air.tensor([KV_HEADS, max_blocks, BLK], bf16)
    O = air.tensor([COLS, GROUPS * ROWS * LQ * DH], bf16)
    Q0 = air.tensor([], i32, name="q0")
    B0 = air.tensor([], i32, name="b0")
    NB = air.tensor([], i32, name="nb")
    WIN = air.tensor([], i32, name="window")

    q_l2 = air.channel("QL2", size=[COLS])
    q_l1 = air.channel("QL1", size=[COLS, ROWS])
    kv_l2 = air.channel("KVL2", size=[1])
    kv_l1 = air.channel("KVL1", size=[1, 1], broadcast_shape=[COLS, ROWS])
    o_l1 = air.channel("OL1", size=[COLS, ROWS])
    o_l3 = air.channel("OL3", size=[COLS])
    steps = ROUND // LQ

    with air.launch(name="attn", repeat=1) as launch:

        @launch.body
        def _(w):
            q0, b0, nb, win = (air.rtp(t.value) for t in (Q0, B0, NB, WIN))
            with air.segment(name="seg", params=[q0, b0, nb, win]) as seg:

                @seg.body
                def _():
                    sp = seg.private()
                    first = b0.as_index()
                    kv_l2.put(KV[:, first : first + nb.as_index(), :], indices=[0])
                    for x in range(COLS):
                        q_l2.put(Q[x : x + 1, :], indices=[x])
                        o_l3.get(O[x : x + 1, :], indices=[x])
                    for x in range(COLS):
                        for _g in air.sequential(0, GROUPS):
                            lq = air.alloc([ROWS * QL], bf16, scope=sp, column=x)
                            q_l2.get(lq, indices=[x])
                            for y in range(ROWS):
                                q_l1.put(lq[y * QL : (y + 1) * QL], indices=[x, y])
                            air.dealloc(lq)
                    for _b in air.sequential(0, nb.as_index() * KV_HEADS):
                        lkv = air.alloc([BLK], bf16, scope=sp, column=3)
                        kv_l2.get(lkv, indices=[0])
                        for st in range(2 * steps):
                            kv_l1.put(
                                lkv[st * LQ * DH : (st + 1) * LQ * DH], indices=[0, 0]
                            )
                        air.dealloc(lkv)
                    for x in range(COLS):
                        for _g in air.sequential(0, GROUPS):
                            lo = air.alloc([ROWS * LQ * DH], bf16, scope=sp, column=x)
                            for y in range(ROWS):
                                o_l1.get(
                                    lo[y * LQ * DH : (y + 1) * LQ * DH], indices=[x, y]
                                )
                            o_l3.put(lo, indices=[x])
                            air.dealloc(lo)

                    with air.herd(
                        [range(COLS), range(ROWS)],
                        name="attn",
                        shape=(COLS, ROWS),
                        at=(0, 2),
                        params=[q0, b0, nb, win],
                    ) as he:

                        @he.body
                        def _(tx, ty):
                            p = he.private()

                            def state():
                                return dict(
                                    q=air.alloc([QL], bf16, scope=p),
                                    s=air.alloc([steps * LQ * LQ], bf16, scope=p),
                                    mm=air.alloc([LQ * LQ + 32], bf16, scope=p),
                                    cl=air.alloc([2 * LQ], f32, scope=p),
                                    y=air.alloc([LQ * DH], f32, scope=p),
                                    o=air.alloc([LQ * DH], bf16, scope=p),
                                )

                            hg = [state(), state()]  # the KV head's two groups
                            kv = [air.alloc([LQ * DH], bf16, scope=p) for _ in range(2)]
                            qrow = q0.as_index() + ty * 32 + (tx % 2) * LQ
                            for _h in air.sequential(0, KV_HEADS):
                                for a in hg:
                                    q_l1.get(a["q"], indices=[tx, ty])
                                    f_init(a["q"], a["mm"], a["cl"], a["y"])
                                for b in air.sequential(0, nb.as_index()):
                                    k0 = (b0.as_index() + b) * ROUND
                                    for j2 in range(steps // 2):
                                        for h in range(2):
                                            kv_l1.get(kv[h], indices=[tx, ty])
                                            for a in hg:
                                                f_qk(
                                                    a["q"],
                                                    kv[h],
                                                    a["s"],
                                                    a["mm"],
                                                    j2 * 2 + h,
                                                    qrow,
                                                    k0,
                                                    win.value,
                                                )
                                    for a in hg:
                                        f_sm(a["s"], a["mm"], a["cl"], a["y"])
                                    for j2 in range(steps // 2):
                                        for h in range(2):
                                            kv_l1.get(kv[h], indices=[tx, ty])
                                            for a in hg:
                                                f_fv(a["s"], kv[h], a["y"], j2 * 2 + h)
                                for a in hg:
                                    f_out(a["q"], a["mm"], a["cl"], a["y"], a["o"])
                                    o_l1.put(a["o"], indices=[tx, ty])

    return launch.build(target="npu2")


def _blocks8(x):
    """[..., 16, 64] -> 8x8 blocks [..., 2, 8, 8, 8]."""
    s = x.shape[:-2]
    return x.reshape(*s, 2, 8, 8, 8).swapaxes(-3, -2)


def pack_q(q, sinks, q0):
    """q [rows, HEADS, DH] bf16 (rows from 0), sinks [HEADS]: the round of 128
    rows from q0, as the cores take it."""
    out = np.zeros((COLS, GROUPS, ROWS, QL), bfloat16)
    # rows y * 32 + half * 16 + r, heads group * CUS + cu; column 2 cu + half
    t = q[q0 : q0 + ROUND].reshape(ROWS, 2, 2, 8, GROUPS, CUS, 8, 8)
    # -> [cu][half][group][y][row block][d block][r][d]
    t = t.transpose(5, 1, 4, 0, 2, 6, 3, 7)
    out[..., : LQ * DH] = t.reshape(COLS, GROUPS, ROWS, LQ * DH)
    s = sinks.astype(np.float32).reshape(GROUPS, CUS).T  # [cu][group]
    out[..., LQ * DH : LQ * DH + 2] = np.repeat(s, 2, axis=0)[:, :, None, None].view(
        bfloat16
    )
    return out.reshape(COLS, -1)


def pack_kv(k, v, max_blocks):
    """k, v [KV_HEADS, keys, DH] bf16 -> the cache's blocks."""
    out = np.zeros((KV_HEADS, max_blocks, 2, ROUND // LQ, LQ * DH), bfloat16)
    n = k.shape[1] // ROUND
    for i, t in enumerate((k, v)):
        steps = t[:, : n * ROUND].reshape(KV_HEADS, n, ROUND // LQ, 2, 8, 8, 8)
        # [dh block][key block][key][dh]
        out[:, :n, i] = steps.transpose(0, 1, 2, 5, 3, 4, 6).reshape(
            KV_HEADS, n, ROUND // LQ, -1
        )
    return out.reshape(KV_HEADS, max_blocks, BLK)


def unpack_o(o):
    """The round's output -> [128 rows, HEADS, DH]."""
    # [cu][half][group][y][row block][d block][r][d] -> rows, heads, d
    o = o.reshape(CUS, 2, GROUPS, ROWS, 2, 8, 8, 8)
    o = o.transpose(3, 1, 4, 6, 2, 0, 5, 7)
    return o.reshape(ROUND, HEADS, DH)


def reference(q, k, v, sinks, q0, window):
    """Rows q0..q0+127 of attention with sinks; [128, HEADS, DH] float32."""
    rep = HEADS // KV_HEADS
    qq = q[q0 : q0 + ROUND].astype(np.float32).transpose(1, 0, 2)
    kk = np.repeat(k.astype(np.float32), rep, 0)
    vv = np.repeat(v.astype(np.float32), rep, 0)
    s = qq @ kk.transpose(0, 2, 1) * 0.125
    qp = np.arange(q0, q0 + ROUND)[:, None]
    kp = np.arange(k.shape[1])[None]
    keep = (kp <= qp) & (kp > qp - window)
    s = np.where(keep, s, -np.inf)
    sk = sinks.astype(np.float32)[:, None, None]
    m = np.maximum(s.max(-1, keepdims=True), sk)
    p = np.exp(s - m)
    o = (p @ vv) / (p.sum(-1, keepdims=True) + np.exp(sk - m))
    return o.transpose(1, 0, 2)


class AttnElf:
    """The attention as a full ELF on its own context. load_kv() once a
    layer, then one call per round of 128 query rows."""

    def __init__(self, build_dir, max_blocks):
        import pyxrt as xrt
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        from air.backend.xrt import XRTBackend, _HostSide

        import common

        self.xrt, self.mb = xrt, max_blocks
        d = Path(build_dir) / f"attn_b{max_blocks}"
        d.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(d)
        try:
            common.compile_kernel("attn_prefill")
            art = XRTBackend(output_format="elf", instance_name="attn").compile(
                build_module(max_blocks)
            )
        finally:
            os.chdir(cwd)
        self.dev = xrt.device(0)
        self.ctx = xrt.hw_context(self.dev, xrt.elf(str(d / art.output_binary)))
        self.kern = xrt.ext.kernel(self.ctx, art.kernel)
        nq, no = COLS * GROUPS * ROWS * QL, COLS * GROUPS * ROWS * LQ * DH
        self.q = xrt.ext.bo(self.dev, nq * 2)
        self.kv = xrt.ext.bo(self.dev, KV_HEADS * max_blocks * BLK * 2)
        self.o = xrt.ext.bo(self.dev, no * 2)
        self.qm = np.frombuffer(self.q.map(), bfloat16)
        self.kvm = np.frombuffer(self.kv.map(), bfloat16)
        self.om = np.frombuffer(self.o.map(), bfloat16)
        self.run_ = xrt.run(self.kern)
        self.device_s = 0.0  # start to completion, summed
        self.last = None
        for i, bo in enumerate((self.q, self.kv, self.o)):
            self.run_.set_arg(i, bo)
        self.params = ParameterScratchpad(self.run_, str(d / "params.txt"))
        self.rules = _HostSide(str(d / "air_project" / "host")).rules(
            str(d / "params.txt")
        )

    def load_kv(self, k, v):
        """k, v [KV_HEADS, keys, DH] bf16 (keys a multiple of 128)."""
        self.kvm[:] = pack_kv(k, v, self.mb).ravel()
        self.kv.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def __call__(self, q, sinks, q0, window):
        """Rows q0..q0+127 of q [rows, HEADS, DH] -> [128, HEADS, DH]."""
        xrt = self.xrt
        b0 = max(0, q0 - window) // ROUND
        nb = q0 // ROUND + 1 - b0
        self.qm[:] = pack_q(q, sinks, q0).ravel()
        self.q.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        vals = {3: q0, 4: b0, 5: nb, 6: window}
        if vals != self.last:  # the scratchpad keeps the last call's
            for name, arg, scale, addend in self.rules:
                self.params.write(name, np.int32(scale * vals[arg] + addend))
            self.params.sync()
            self.last = vals
        t0 = time.perf_counter()
        self.run_.start()
        state = self.run_.wait(60000)
        self.device_s += time.perf_counter() - t0
        if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"prefill attention: {state}")
        self.o.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return unpack_o(self.om)


if __name__ == "__main__":
    import argparse
    import tempfile

    import filelock

    from air.backend.xrt import XRTBackend

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-blocks", type=int, default=4)
    ap.add_argument("--rounds", type=str, default="0,2,3")
    ap.add_argument("--window", type=int, default=0, help="0: full")
    args = ap.parse_args()

    import common

    common.compile_kernel("attn_prefill")
    mb = args.max_blocks
    L = mb * ROUND
    rng = np.random.default_rng(0)
    q = rng.standard_normal((L, HEADS, DH)).astype(bfloat16)
    k = rng.standard_normal((KV_HEADS, L, DH)).astype(bfloat16)
    v = rng.standard_normal((KV_HEADS, L, DH)).astype(bfloat16)
    sinks = rng.standard_normal(HEADS).astype(bfloat16)
    kvp = pack_kv(k, v, mb)
    win = args.window or L  # FLM: L_end; a core scalar has 30 bits
    be = XRTBackend(output_format="elf", instance_name="attn")
    art = be.compile(build_module(mb))
    ok = True
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        for r in [int(v) for v in args.rounds.split(",")]:
            q0 = r * ROUND
            b0 = max(0, q0 - win) // ROUND if args.window else 0
            nb = r + 1 - b0
            o = run(
                pack_q(q, sinks, q0),
                kvp,
                np.zeros((COLS, GROUPS * ROWS * LQ * DH), bfloat16),
                q0,
                b0,
                nb,
                win,
            )[2]
            got = unpack_o(o).astype(np.float32)
            want = reference(q, k, v, sinks, q0, win)
            cos = (got * want).sum() / np.linalg.norm(got) / np.linalg.norm(want)
            err = np.abs(got - want).max()
            good = cos > 0.99  # bf16 scores and probabilities, as FLM
            ok &= good
            print(f"round {r} (b0 {b0}, nb {nb}): cos {cos:.5f} max err {err:.3f}")
        be.unload()
    print("PASS!" if ok else "failed.")
