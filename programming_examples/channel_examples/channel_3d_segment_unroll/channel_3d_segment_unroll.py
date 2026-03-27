# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dual-Column Cascade Accumulate with Segment Unroll Example

Demonstrates direct L3->L1 data distribution combined with cascade
accumulation, stamped out to two physical copies via segment unroll.

Pattern:
  1. The segment is unrolled with sizes=[NUM_SEGMENTS, 1], creating
     NUM_SEGMENTS independent physical copies of the herd.
  2. Each segment copy contains a [NUM_TILES, NUM_COLS] herd (4x2),
     giving two cascade columns per segment.
  3. Input data is distributed via a 3D channel [NUM_SEGMENTS, NUM_TILES, NUM_COLS].
     Each core (segment_idx, tx, ty) gets its own unique tile from L3
     directly to L1 (no memtile, no broadcast).
  4. Within each segment, two independent cascade chains flow along tx:
       Column ty: core (0,ty) -> (1,ty) -> (2,ty) -> (3,ty)
  5. The bottom cores (tx=3) from all segments write accumulated results
     to L3 via NUM_SEGMENTS * NUM_COLS = 4 shim DMA channels.

Net effect per cascade column:
  output[seg, ty] = sum_{tx=0}^{3} input[seg, tx, ty]
"""

from air.ir import *
from air.dialects.air import *
from air.dialects import arith, linalg, scf
from air.dialects.memref import AllocOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, XRTBackend, type_mapper, make_air_parser, run_on_npu

import numpy as np

np.random.seed(42)

range_ = for_

NUM_TILES = 4  # Cascade depth (cores per column)
NUM_COLS = 2  # Cascade columns per segment
NUM_SEGMENTS = 2  # Segment unroll factor
DATA_SIZE = 1024

TOTAL_IN = NUM_SEGMENTS * NUM_TILES * NUM_COLS * DATA_SIZE  # 16384
TOTAL_OUT = NUM_SEGMENTS * NUM_COLS * DATA_SIZE  # 4096


@module_builder
def build_module():
    xrt_dtype = T.i32()
    tile_shape = [DATA_SIZE]

    # L3 types: flat arrays for input and output
    l3MemrefTyIn = MemRefType.get([TOTAL_IN], xrt_dtype)
    l3MemrefTyOut = MemRefType.get([TOTAL_OUT], xrt_dtype)

    # L1 type: one tile per core
    l1MemrefTy = l1_memref_type(tile_shape, xrt_dtype)

    # 3D input channel: [NUM_SEGMENTS, NUM_TILES, NUM_COLS].
    # Each core (seg, tx, ty) gets its own unique tile.
    # Direct L3->L1 (no broadcast, no memtile).
    Channel("chan_in", size=[NUM_SEGMENTS, NUM_TILES, NUM_COLS])

    # Cascade channel: per-segment, two independent chains.
    channel(
        "chan_cascade", size=[NUM_SEGMENTS, NUM_TILES, NUM_COLS], channel_type="cascade"
    )

    # Output channel: one per cascade column across all segments.
    Channel("chan_out", size=[NUM_SEGMENTS, NUM_COLS])

    @FuncOp.from_py_func(l3MemrefTyIn, l3MemrefTyOut)
    def channel_3d_segment_unroll(arg0, arg1):

        launch_size = [1, 1]

        @launch(operands=[arg0, arg1], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_in,
            l3_out,
        ):
            # Distribute unique tiles to each core across all segments.
            # Layout: core (seg, tx, ty) gets input at offset
            # (seg * NUM_TILES * NUM_COLS + tx * NUM_COLS + ty) * DATA_SIZE.
            for seg in range(NUM_SEGMENTS):
                for tx in range(NUM_TILES):
                    for ty in range(NUM_COLS):
                        offset = (
                            seg * NUM_TILES * NUM_COLS + tx * NUM_COLS + ty
                        ) * DATA_SIZE
                        ChannelPut(
                            "chan_in",
                            l3_in,
                            indices=[seg, tx, ty],
                            offsets=[offset],
                            sizes=[DATA_SIZE],
                            strides=[1],
                        )

            # Segment unroll: create NUM_SEGMENTS physical copies.
            @segment(name="segment_0", sizes=[NUM_SEGMENTS, 1])
            def segment_body(seg_x, seg_y, seg_sx, seg_sy):

                # Each segment contains a [NUM_TILES, NUM_COLS] herd.
                # Cascade flows west-to-east (along tx) within each row (ty).
                @herd(
                    name="herd_0",
                    sizes=[NUM_TILES, NUM_COLS],
                    operands=[seg_x],
                )
                def herd_body(tx, ty, sx, sy, herd_seg_x):
                    c0 = arith.ConstantOp.create_index(0)
                    last_tile = arith.ConstantOp.create_index(NUM_TILES - 1)

                    # Receive this core's unique input tile from L3
                    in_buf = AllocOp(l1MemrefTy, [], [])
                    ChannelGet("chan_in", in_buf, indices=[herd_seg_x, tx, ty])

                    # Accumulation buffer
                    acc_buf = AllocOp(l1MemrefTy, [], [])

                    # Core (tx=0, ty): first in cascade chain.
                    cmp_first = arith.CmpIOp(arith.CmpIPredicate.eq, tx, c0)
                    if_first = scf.IfOp(cmp_first, has_else=True)
                    with InsertionPoint(if_first.then_block):
                        linalg.copy(in_buf, outs=[acc_buf])
                        ChannelPut(
                            "chan_cascade",
                            acc_buf,
                            indices=[herd_seg_x, tx, ty],
                        )
                        yield_([])

                    # Cores (tx>0, ty): receive cascade, add input, pass along.
                    with InsertionPoint(if_first.else_block):
                        c1_idx = arith.ConstantOp.create_index(1)
                        prev_tx = arith.SubIOp(tx, c1_idx)

                        ChannelGet(
                            "chan_cascade",
                            acc_buf,
                            indices=[herd_seg_x, prev_tx, ty],
                        )

                        linalg.add(in_buf, acc_buf, outs=[acc_buf])

                        # Last core: write to output. Others: cascade forward.
                        cmp_last = arith.CmpIOp(arith.CmpIPredicate.eq, tx, last_tile)
                        if_last = scf.IfOp(cmp_last, has_else=True)
                        with InsertionPoint(if_last.then_block):
                            ChannelPut(
                                "chan_out",
                                acc_buf,
                                indices=[herd_seg_x, ty],
                            )
                            yield_([])
                        with InsertionPoint(if_last.else_block):
                            ChannelPut(
                                "chan_cascade",
                                acc_buf,
                                indices=[herd_seg_x, tx, ty],
                            )
                            yield_([])

                        yield_([])

            # Receive output from bottom core of each column in each segment.
            # NUM_SEGMENTS * NUM_COLS = 4 shim DMA channels.
            for seg in range(NUM_SEGMENTS):
                for ty in range(NUM_COLS):
                    offset = (seg * NUM_COLS + ty) * DATA_SIZE
                    ChannelGet(
                        "chan_out",
                        l3_out,
                        indices=[seg, ty],
                        offsets=[offset],
                        sizes=[DATA_SIZE],
                        strides=[1],
                    )


if __name__ == "__main__":
    parser = make_air_parser("Builds, runs, and tests the 3D channel with segment unroll example")
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Input: 16 unique tiles (2 segments × 4 rows × 2 cols), one per core.
    # Layout: core (seg, tx, ty) gets
    # input[(seg*NUM_TILES*NUM_COLS + tx*NUM_COLS + ty)*DATA_SIZE : ... + DATA_SIZE].
    input_a = np.arange(0, TOTAL_IN, dtype=np.int32)

    # Expected output: per cascade column (seg, ty), sum of all tiles in that column.
    # output[(seg*NUM_COLS+ty)*DATA_SIZE : ... + DATA_SIZE] =
    #   sum_{tx=0}^{3} input_tile(seg, tx, ty)
    expected_output = np.zeros(TOTAL_OUT, dtype=np.int32)
    for seg in range(NUM_SEGMENTS):
        for ty in range(NUM_COLS):
            out_start = (seg * NUM_COLS + ty) * DATA_SIZE
            for tx in range(NUM_TILES):
                in_start = (seg * NUM_TILES * NUM_COLS + tx * NUM_COLS + ty) * DATA_SIZE
                expected_output[out_start : out_start + DATA_SIZE] += input_a[
                    in_start : in_start + DATA_SIZE
                ]

    exit(run_on_npu(args, mlir_module, inputs=[input_a], instance_name="channel_3d_segment_unroll", expected_outputs=[expected_output]))
