# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared-L1 communication between cores within a single air.herd.

A column of NC cores is expressed as ONE air.herd [1, NC]. Every core runs the
same body (a herd is a single program replicated across its cores) and learns
its position from the tile index ty. Data flows DOWN the column entirely through
shared L1: neighbor cores read/write a common L1 buffer, so there is no DMA --
and no AIE hardware cascade link -- between the cores. Exercising that shared-L1
hand-off is the whole point of the example.

    core 0   : v0 = g(in0)                    -> hop[0]
    core k   : vk = g(ink) + hop[k-1]         -> hop[k]        (1 <= k < NC-1)
    core NC-1: out = g(in_{NC-1}) + hop[NC-2] -> @outY

  * Each hop[k] is an L1 buffer shared between neighbor cores k and k+1: core k
    writes it, core k+1 reads it (intra-herd neighbor shared L1).
  * Each core picks its role (first / middle / last) from its tile index ty with
    a single scf.index_switch.
  * g() also runs a short loop whose per-iteration constant is chosen by a SECOND
    scf.index_switch keyed on the loop counter -- so the example shows
    scf.index_switch used both for tile-index role selection and for loop-index
    value selection.

The per-core compute here is just a trivial vectorized accumulate (a placeholder
for whatever real work a core would do); input/output move over simple L3<->L1
channels so the example runs standalone on NPU1.

Result: out = sum(in[0..NC-1]) + NC * sum(STEP_ADDENDS).
"""

import argparse
import numpy as np

np.random.seed(42)

import air
from air.ir import *
from air.dialects.air import *
from air.dialects import memref, vector, arith, scf
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_, index_switch
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp
from air.backend.xrt_runner import XRTRunner
from ml_dtypes import bfloat16

NC = 4  # cores in the column; NC-1 shared-L1 hops
T = 64  # elements per core tile
VEC = 16  # vector width

STEP_ADDENDS = [1.0, 10.0]  # per-step constant selected via scf.index_switch
NSTEP = len(STEP_ADDENDS)

range_ = for_


def parse_args():
    p = argparse.ArgumentParser(description="Single-column shared-L1 relay")
    p.add_argument("-p", "--print-ir", action="store_true", help="Print IR and exit")
    p.add_argument(
        "--output-format",
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    return p.parse_args()


@module_builder
def build_module():
    itype = IndexType.get()
    bf16 = air.ir.Type.parse("bf16")

    def idx(v):
        return ConstantOp(itype, v)

    l1 = Attribute.parse("2")
    l2 = Attribute.parse("1")
    l1_t = MemRefType.get([T], bf16, memory_space=l1)
    l2_in_t = MemRefType.get([NC, T], bf16, memory_space=l2)
    l2_out_t = MemRefType.get([1, T], bf16, memory_space=l2)
    l3_in_t = MemRefType.get([NC, T], bf16)
    l3_out_t = MemRefType.get([1, T], bf16)

    channel("inX", size=[NC])  # per-core input feed (L2 -> L1)
    channel("outY", size=[1])  # last core's result (L1 -> L2)

    am = AffineMapAttr.get(AffineMap.get_identity(1))

    def _rd(buf, j):
        cst0 = arith.ConstantOp(bf16, 0.0)
        return vector.transfer_read(
            VectorType.get([VEC], bf16),
            memref.subview(buf, [j], [VEC], [1]),
            [idx(0)],
            am,
            cst0,
            [True],
        )

    def _wr(vec, buf, j):
        vector.transfer_write(
            None, vec, memref.subview(buf, [j], [VEC], [1]), [idx(0)], am, [True]
        )

    def emit_axpy(buf, scalar_val):
        """buf[:] += scalar_val (bf16 SSA value)."""
        for j in range_(idx(0), idx(T), idx(VEC)):
            vs = vector.BroadcastOp(VectorType.get([VEC], bf16), scalar_val)
            _wr(arith.AddFOp(_rd(buf, j), vs), buf, j)
            yield_([])

    def emit_add_inplace(dst, src):
        """dst[:] += src[:]. Chunked so a producer's shared-buffer write and this
        consumer's read carry MATCHING per-chunk lock acquire/release counts."""
        for j in range_(idx(0), idx(T), idx(VEC)):
            _wr(arith.AddFOp(_rd(dst, j), _rd(src, j)), dst, j)
            yield_([])

    def emit_copy(src, dst):
        """dst[:] = src[:], chunked (same lock-count reasoning as emit_add_inplace)."""
        for j in range_(idx(0), idx(T), idx(VEC)):
            _wr(_rd(src, j), dst, j)
            yield_([])

    @FuncOp.from_py_func(l3_in_t, l3_out_t)
    def col_relay(l3_in, l3_out):
        @launch(operands=[l3_in, l3_out], sizes=[1, 1])
        def launch_body(lx, ly, lsx, lsy, gin, gout):
            @segment(name="seg", operands=[gin, gout])
            def segment_body(sin, sout):
                # Relay input L3 -> L2 (one DMA) -> fan per-core via @inX.
                in_l2 = AllocOp(l2_in_t, [], [])
                dma_memcpy_nd(
                    in_l2.result,
                    sin,
                    dst_offsets=[idx(0), idx(0)],
                    dst_sizes=[idx(NC), idx(T)],
                    dst_strides=[idx(T), idx(1)],
                    src_offsets=[idx(0), idx(0)],
                    src_sizes=[idx(NC), idx(T)],
                    src_strides=[idx(T), idx(1)],
                )
                for c in range(NC):
                    ChannelPut(
                        "inX",
                        in_l2.result,
                        indices=[idx(c)],
                        offsets=[idx(c), idx(0)],
                        sizes=[idx(1), idx(T)],
                        strides=[idx(T), idx(1)],
                    )

                # NC-1 shared L1 hop buffers, one per relay edge (core k -> k+1).
                shbuf = [AllocOp(l1_t, [], []) for _ in range(NC - 1)]
                shbuf_ops = [b.result for b in shbuf]

                @herd(name="col_relay", sizes=[1, NC], operands=shbuf_ops)
                def herd_body(tx, ty, sx, sy, *hops):
                    # ---- input + step loop (all cores run this common body) ----
                    # local = in + sum over steps of STEP_ADDENDS[step], where
                    # each step's addend is picked by scf.index_switch(step).
                    local = AllocOp(l1_t, [], [])
                    ChannelGet("inX", local.result, indices=[ty])
                    a_consts = [arith.ConstantOp(bf16, v) for v in STEP_ADDENDS]
                    for step in range_(idx(0), idx(NSTEP), idx(1)):
                        addend = index_switch(
                            [bf16],
                            step,
                            list(range(NSTEP - 1)),
                            case_body_builder=lambda op, i, cv: yield_(
                                [a_consts[i].result]
                            ),
                            default_body_builder=lambda op: yield_(
                                [a_consts[-1].result]
                            ),
                        )
                        emit_axpy(local.result, addend)
                        yield_([])

                    # ---- relay role dispatch by ty via scf.index_switch ----
                    # case k (0..NC-2): if k>0 fold in the incoming hop, then
                    #                   publish to hop[k].
                    # default (k=NC-1): fold in the last hop, emit the result.
                    def emit_role(k):
                        if k > 0:
                            emit_add_inplace(local.result, hops[k - 1])
                        if k < NC - 1:
                            emit_copy(local.result, hops[k])
                        else:
                            ChannelPut("outY", local.result, indices=[idx(0)])

                    index_switch(
                        [],
                        ty,
                        list(range(NC - 1)),
                        case_body_builder=lambda op, k, cv: (
                            emit_role(k),
                            yield_([]),
                        )[-1],
                        default_body_builder=lambda op: (
                            emit_role(NC - 1),
                            yield_([]),
                        )[-1],
                    )

                # Collect the last core's result L1 -> L2 -> L3.
                out_l2 = AllocOp(l2_out_t, [], [])
                ChannelGet(
                    "outY",
                    out_l2.result,
                    indices=[idx(0)],
                    offsets=[idx(0), idx(0)],
                    sizes=[idx(1), idx(T)],
                    strides=[idx(T), idx(1)],
                )
                dma_memcpy_nd(
                    sout,
                    out_l2.result,
                    dst_offsets=[idx(0), idx(0)],
                    dst_sizes=[idx(1), idx(T)],
                    dst_strides=[idx(T), idx(1)],
                    src_offsets=[idx(0), idx(0)],
                    src_sizes=[idx(1), idx(T)],
                    src_strides=[idx(T), idx(1)],
                )


def main():
    args = parse_args()
    mlir_module = build_module()
    if args.print_ir:
        print(str(mlir_module))
        return

    A = np.random.rand(NC, T).astype(bfloat16)
    s = float(sum(STEP_ADDENDS))
    C = (A.astype(np.float32).sum(axis=0) + NC * s).astype(bfloat16).reshape(1, T)

    runner = XRTRunner(
        omit_while_true_loop=False,
        verbose=False,
        output_format=args.output_format,
        instance_name="col_relay",
        debug_ir=True,
    )
    exit(runner.run_test(mlir_module, inputs=[A], expected_outputs=[C], rtol=3e-2))


if __name__ == "__main__":
    main()
