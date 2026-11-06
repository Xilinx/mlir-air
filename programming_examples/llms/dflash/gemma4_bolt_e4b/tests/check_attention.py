#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""attention_graph.GemmaAttention on its own, as the decoder graph uses it:
Q frames and whole KV rows from DDR, the output gathered in a memory tile.
Random inputs against a float32 reference, at several cache lengths. With a
window, q, k and v have the decoder's spread sliding-head layout. With
--requests 2, the two tokens are two requests, each with its own cache and
length (the second request's length differs from the first's).

  python3 tests/check_attention.py --output DIR [--window 512]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from attention_graph import GemmaAttention, Q_PER_KV  # noqa: E402

EXAMPLES = HERE.parents[2]
DH, KV, TOK = 512, 2, 8
REQ = 1  # requests; with more than one, a token each
QROW = KV * Q_PER_KV * DH
# BF16 output against float32: absolute plus relative to the value
ATOL, RTOL = 0.1, 0.016


def namespace():
    from air.ir import (
        BF16Type,
        F32Type,
        IndexType,
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        MemRefType,
        StringAttr,
        UnitAttr,
    )
    from air.dialects.air import (
        Channel,
        ChannelGet,
        ChannelPut,
        MemorySpace,
        T,
        herd,
        launch,
        module_builder,
        segment,
    )
    from air.dialects.air import channel as channel_decl
    from air.dialects.func import FuncOp, CallOp
    from air.dialects.memref import AllocOp, DeallocOp
    from air.dialects import arith
    from air.dialects.scf import for_, yield_, index_switch, IfOp

    return SimpleNamespace(
        BATCH=TOK, REQUESTS=REQ, **{k: v for k, v in locals().items()}
    )


def module(cap, window):
    f = namespace()

    @f.module_builder
    def build():
        bf = f.BF16Type.get()
        i32 = f.IntegerType.get_signless(32)
        l2 = f.IntegerAttr.get(f.T.i32(), f.MemorySpace.L2)
        att = GemmaAttention(f)
        f.channel_decl("outO", size=[1])

        def mem(n):
            return f.MemRefType.get([n], bf)

        def const(v):
            return f.arith.ConstantOp(f.IntegerAttr.get(i32, v), None).result

        def lau(lx, ly, lsx, lsy, q, kv, o, *lengths):
            ix = f.arith.ConstantOp.create_index
            att.put_query(q, ix(0))
            for r, length in enumerate(lengths):
                base = r * 2 * cap * KV * DH
                att.put_kv(
                    kv, ix(base), ix(base + cap * KV * DH), length, const(window), r
                )
            f.ChannelGet("outO", o)

            @f.segment(name="seg", operands=list(lengths))
            def seg(*Ls):
                att.emit(list(Ls), const(1), const(window))
                omtb = f.AllocOp(
                    f.MemRefType.get([TOK * QROW], bf, memory_space=l2), [], []
                )
                omtb.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(i32, 3)
                att.gather(omtb, QROW)
                f.ChannelPut("outO", omtb)
                f.DeallocOp(omtb)

        @f.FuncOp.from_py_func(
            mem(Q_ROWS * QROW),
            mem(REQ * 2 * cap * KV * DH),
            mem(TOK * QROW),
            *([i32] * REQ),
        )
        def attention(Q, KVC, O, *Ls):
            f.launch(sizes=[1, 1], operands=[Q, KVC, O, *Ls])(lau)

    return build()


def compile_kernel(out):
    clang = Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/clang++"
    cmd = [
        str(clang),
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-deprecated-declarations",
        "-DNDEBUG",
        "-DMODEL_TYPE=GEMMA4_E4B",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-I" + str(Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include"),
        "-I" + str(EXAMPLES / "fused_decode/kernels"),
        "-I" + str(EXAMPLES / "fused_decode/models"),
        "-O2",
        "-c",
        str(HERE / "kernels/attn_b8.cc"),
        "-o",
        str(out / "attn_b8.o"),
    ]
    subprocess.run(cmd, check=True)


def build(out, cap, window):
    out.mkdir(parents=True, exist_ok=True)
    compile_kernel(out)
    mod = module(cap, window)
    assert mod.operation.verify()
    from air.backend.xrt import XRTBackend

    old = Path.cwd()
    try:
        os.chdir(out)
        art = XRTBackend(
            omit_while_true_loop=False,
            kernel_name="MLIR_AIE",
            stack_size=6080,
            omit_pingpong="all",
            target_device="npu2",
            use_lock_race_condition_fix_v2=True,
            emit_txn_cpp=True,
        ).compile(mod, output_binary_name="attention", insts="attention.insts.bin")
        shutil.copy2(out / art.txn_header, out / "attention.txn.h")
    finally:
        os.chdir(old)


def reference(q, k, v, p, window, tokens=None):
    """q [tokens, 8, DH], k/v [cap, 2, DH]; token i sees keys [0, p + i], or
    the last `window` of them."""
    tokens = TOK if tokens is None else tokens
    kk = np.repeat(k.astype(np.float32), Q_PER_KV, axis=1)
    vv = np.repeat(v.astype(np.float32), Q_PER_KV, axis=1)
    s = np.einsum("thd,lhd->thl", q.astype(np.float32), kk)
    vis = p + 1 + np.arange(tokens)
    j = np.arange(len(k))[None, None]
    lo = vis - window if window else np.zeros_like(vis)
    s = np.where((j < vis[:, None, None]) & (j >= lo[:, None, None]), s, -np.inf)
    pr = np.exp(s - s.max(-1, keepdims=True))
    pr /= pr.sum(-1, keepdims=True)
    return np.einsum("thl,lhd->thd", pr, vv)


def run(out, cap, prefixes, window):
    import pyxrt as xrt
    from air.backend.txn_builder import TxnBuilder

    dev = xrt.device(0)
    xb = xrt.xclbin(str(out / "attention.xclbin"))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kern = xrt.kernel(ctx, "MLIR_AIE")
    tb = TxnBuilder(str(out / "attention.txn.h"))
    (fn,) = tb.function_names
    longest = [cap - TOK // REQ + 1] * REQ
    ib = xrt.bo(dev, tb(fn, *longest).nbytes * 2, xrt.bo.cacheable, kern.group_id(1))
    sizes = [Q_ROWS * QROW, REQ * 2 * cap * KV * DH, TOK * QROW]
    bos = [
        xrt.bo(dev, n * 2, xrt.bo.host_only, kern.group_id(i + 3))
        for i, n in enumerate(sizes)
    ]
    TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    def call(q, ks, vs, ps):
        Ls = [p + 1 for p in ps]
        inst = tb(fn, *Ls)
        ib.write(inst, 0)
        ib.sync(TO)
        qf = q.reshape(TOK, KV, Q_PER_KV, DH).transpose(0, 2, 1, 3).reshape(-1)
        qf = np.concatenate([qf, np.zeros((Q_ROWS - TOK) * QROW, qf.dtype)])
        kvc = np.concatenate(
            [np.concatenate([k.reshape(-1), v.reshape(-1)]) for k, v in zip(ks, vs)]
        )
        # the output starts as NaN, so a row the graph never writes fails
        for bo, a in zip(bos, (qf, kvc, np.full(sizes[2], 0x7FC1, np.uint16))):
            bo.write(np.ascontiguousarray(a).view(np.int16).reshape(-1), 0)
            bo.sync(TO)
        t0 = time.perf_counter()
        st = kern(3, ib, inst.size, *bos, *Ls).wait(10000)
        ms = (time.perf_counter() - t0) * 1e3
        assert str(st).endswith("COMPLETED"), st
        bos[2].sync(FROM)
        r = np.frombuffer(bos[2].map(), np.uint16).copy()
        # the gather's half-major columns back to [token][head][DH]
        o = r.view(bfloat16).astype(np.float32).reshape(TOK, 2, -1, DH // 2)
        return o.transpose(0, 2, 1, 3).reshape(TOK, -1, DH), ms

    rows = []
    for p in prefixes:
        rng = np.random.default_rng(p)
        q = rng.normal(0, 0.7, (TOK, KV * Q_PER_KV, DH)).astype(np.float32)
        q = (q / np.sqrt(DH)).astype(bfloat16)
        ks = [rng.normal(0, 0.7, (cap, KV, DH)).astype(bfloat16) for _ in range(REQ)]
        vs = [rng.normal(0, 0.7, (cap, KV, DH)).astype(bfloat16) for _ in range(REQ)]
        if window:
            for a in [q] + ks:
                a[..., DH // 4 : DH // 2] = 0
                a[..., 3 * DH // 4 :] = 0
            for a in vs:
                a[..., DH // 2 :] = 0
        # a second request at another length: shorter, or past the first
        ps = [p] + [(p + 211 * r) % (cap - 1) for r in range(1, REQ)]
        if REQ == 1:
            exp = reference(q, ks[0], vs[0], p, window)
        else:
            exp = np.concatenate([
                reference(q[r : r + 1], ks[r], vs[r], ps[r], window, tokens=1)
                for r in range(REQ)
            ])  # fmt: skip
        got, ms = call(q, ks, vs, ps)
        err = np.abs(got - exp)
        bad = int(np.count_nonzero(~(err <= ATOL + RTOL * np.abs(exp))))
        rows.append(dict(prefix=p, max_abs_error=float(err.max()), out_of_tol=bad))
        at = "/".join(f"{x:4d}" for x in ps)
        print(f"p={at} {ms:.2f} ms max|err| {err.max():.4f} out-of-tol {bad}")
    return rows


Q_ROWS = TOK


def main():
    global TOK, REQ, Q_ROWS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--capacity", type=int, default=1024)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--prefixes", type=int, nargs="+", default=[0, 7, 100, 505, 1000])
    ap.add_argument("--batch", type=int, default=TOK, help="tokens per dispatch")
    ap.add_argument("--requests", type=int, default=REQ, help="a token each")
    a = ap.parse_args()
    TOK, REQ = a.batch, a.requests
    if REQ > 1:
        TOK = REQ
    # the last request's frame reads one token's rows past the Q buffer
    Q_ROWS = TOK + (REQ > 1)
    out = a.output.resolve()
    build(out, a.capacity, a.window)
    rows = run(out, a.capacity, a.prefixes, a.window)
    passed = all(r["out_of_tol"] == 0 for r in rows)
    (out / "result.json").write_text(
        json.dumps(dict(window=a.window, rows=rows, passed=passed), indent=2) + "\n"
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
