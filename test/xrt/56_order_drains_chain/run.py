# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Test: a launch that reads back, through host memory, what its own drains wrote
# (air.order_drains).
#
# One [ROUNDS + 1, N] i32 buffer in host memory: row 0 is the input, the other rows start at a
# sentinel. Round r streams row r - 1 to the core, which adds one and sends the result back to
# host memory as row r. Round r therefore reads what round r - 1's drain wrote. air-to-std
# defers every drain wait to the launch end unless the drains are marked air.order_drains, in
# which case they stay in program order and a drain group is awaited right after the next
# group is armed, so a read of group g's rows has to be issued after group g + 1's drain
# (the order below). Without the marker the later reads race the earlier drains.
#
# The output must be row 0 + r in every row r. Pass --no-order to leave the drains unmarked
# (reproduces the failure on a compiler without the ordering).

import argparse

import filelock
import numpy as np

import air.backend.xrt as xrt_backend
from air.dialects import arith, memref
from air.dialects.air import *
from air.dialects.func import FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
from air.ir import *

N = 64
ROUNDS = 3  # four un-awaited feeds on one shim channel exceed its task queue
SENTINEL = -12345


@module_builder
def build_module(order_drains):
    i32 = T.i32()
    buf_ty = MemRefType.get([ROUNDS + 1, N], i32)
    Channel("ChanIn")
    Channel("ChanOut")

    @FuncOp.from_py_func(buf_ty)
    def chain(buf):
        @launch(operands=[buf])
        def launch_body(b):
            for r in range(1, ROUNDS + 1):
                # Round r's drain is armed before the read that depends on round r - 1's:
                # a group is awaited right after the next group is armed.
                drain = ChannelGet(
                    "ChanOut", b, offsets=[r, 0], sizes=[1, N], strides=[N, 1]
                )
                if order_drains:
                    drain.operation.attributes["air.order_drains"] = UnitAttr.get()
                ChannelPut(
                    "ChanIn", b, offsets=[r - 1, 0], sizes=[1, N], strides=[N, 1]
                )

            @segment(name="segment_0")
            def segment_body():
                @herd(name="herd_0", sizes=[1, 1])
                def herd_body(x, y, sx, sy):
                    l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
                    tile_ty = MemRefType.get([N], i32, memory_space=l1)
                    one = arith.ConstantOp(i32, 1).result
                    for _ in for_(ROUNDS):
                        tile = AllocOp(tile_ty, [], [])
                        ChannelGet("ChanIn", tile)
                        for i in for_(N):
                            v = memref.LoadOp(tile, [i]).result
                            memref.StoreOp(arith.addi(v, one), tile, [i])
                            yield_([])
                        ChannelPut("ChanOut", tile)
                        DeallocOp(tile)
                        yield_([])


def run_test(order_drains):
    module = build_module(order_drains)
    data = np.full((ROUNDS + 1, N), SENTINEL, dtype=np.int32)
    data[0] = np.arange(N, dtype=np.int32)
    ref = data.copy()
    for r in range(1, ROUNDS + 1):
        ref[r] = ref[0] + r

    backend = xrt_backend.XRTBackend(verbose=False)
    compiled = backend.compile(module)
    with filelock.FileLock("/tmp/npu.lock"):
        fn = backend.load(compiled)
        (out,) = fn(data)
        out = out.reshape(ROUNDS + 1, N)
        backend.unload()

    bad = [r for r in range(ROUNDS + 1) if not np.array_equal(out[r], ref[r])]
    if bad:
        print("rows differing from row0 + r:", bad)
        print("failed.")
        return 0
    print("PASS!")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-order", action="store_true", help="leave the drains unmarked")
    args = ap.parse_args()
    raise SystemExit(0 if run_test(not args.no_order) else 1)
