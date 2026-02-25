# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cascade Reduction Example

Demonstrates a 4-tile cascade reduction pattern using AIR channels.

Data flows through a 1x4 herd in a pipeline:
  1. Launch puts input data on @chan_in
  2. Tile 0: gets from @chan_in, adds 1, puts on @chan_cascade[0]
  3. Tile 1: gets from @chan_cascade[0], adds 1, puts on @chan_cascade[1]
  4. Tile 2: gets from @chan_cascade[1], adds 1, puts on @chan_cascade[2]
  5. Tile 3: gets from @chan_cascade[2], adds 1, puts on @chan_out
  6. Launch gets output from @chan_out

Final result: output = input + 4
"""

import argparse

from air.ir import *
from air.dialects.air import *
from air.dialects import arith, linalg, memref, scf
from air.dialects.memref import AllocOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

NUM_TILES = 4
DATA_SIZE = 2048


@module_builder
def build_module():
    xrt_dtype = T.i32()
    data_shape = [1, 1, DATA_SIZE]

    # L3 types
    l3MemrefTy = MemRefType.get(data_shape, xrt_dtype)
    # L1 types
    l1MemrefTy = MemRefType.get(
        data_shape,
        xrt_dtype,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    # Channels: chan_in/chan_out use DMA (L3<->L1), chan_cascade uses
    # direct core-to-core cascade connections between adjacent tiles.
    channel("chan_in", size=[1])
    channel("chan_cascade", size=[NUM_TILES], channel_type="cascade")
    channel("chan_out", size=[1])

    @FuncOp.from_py_func(l3MemrefTy, l3MemrefTy)
    def cascade_reduce(arg0, arg1):

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
            # Send input to tile 0
            ChannelPut("chan_in", l3_in)

            @segment(name="segment_0")
            def segment_body():

                # Herd oriented as NUM_TILES columns x 1 row so cascade flows
                # go West-to-East between adjacent columns.
                @herd(name="herd_0", sizes=[NUM_TILES, 1])
                def herd_body(tx, ty, sx, sy):
                    c0 = arith.ConstantOp.create_index(0)
                    c1_i32 = arith.ConstantOp(IntegerAttr.get(T.i32(), 1), None)
                    last_tile = arith.ConstantOp.create_index(NUM_TILES - 1)

                    # Each tile has a local buffer initialized to 1
                    local_buf = AllocOp(l1MemrefTy, [], [])
                    linalg.fill(c1_i32, outs=[local_buf])

                    # Receive buffer
                    recv_buf = AllocOp(l1MemrefTy, [], [])

                    # Tile 0: read from chan_in
                    cmp_first = arith.CmpIOp(arith.CmpIPredicate.eq, tx, c0)
                    if_first = scf.IfOp(cmp_first, hasElse=True)
                    with InsertionPoint(if_first.then_block):
                        ChannelGet("chan_in", recv_buf)
                        linalg.add(recv_buf, local_buf, outs=[local_buf])
                        ChannelPut("chan_cascade", local_buf, indices=[tx])
                        yield_([])

                    # Tiles 1..N-1: read from previous tile's cascade channel
                    with InsertionPoint(if_first.else_block):
                        c1_idx = arith.ConstantOp.create_index(1)
                        prev_tx = arith.SubIOp(tx, c1_idx)
                        ChannelGet("chan_cascade", recv_buf, indices=[prev_tx])
                        linalg.add(recv_buf, local_buf, outs=[local_buf])

                        # Last tile: write to chan_out; others: write to chan_cascade
                        cmp_last = arith.CmpIOp(arith.CmpIPredicate.eq, tx, last_tile)
                        if_last = scf.IfOp(cmp_last, hasElse=True)
                        with InsertionPoint(if_last.then_block):
                            ChannelPut("chan_out", local_buf)
                            yield_([])
                        with InsertionPoint(if_last.else_block):
                            ChannelPut("chan_cascade", local_buf, indices=[tx])
                            yield_([])

                        yield_([])

            # Receive output from last tile
            ChannelGet("chan_out", l3_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the cascade reduction example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, DATA_SIZE, dtype=np.int32).reshape(1, 1, DATA_SIZE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.zeros(num_samples, dtype=int),
                np.zeros(num_samples, dtype=int),
                np.random.randint(0, DATA_SIZE, num_samples),
            ]
        )

        sampled_values = np.array(
            [input_a[i, j, k] + NUM_TILES for i, j, k in zip(*sampled_indices)],
            dtype=np.int32,
        )

        sampled_data = {
            "shape": (1, 1, DATA_SIZE),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="cascade_reduce",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
