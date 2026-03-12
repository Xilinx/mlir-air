# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Test: DMA padding during memtile-to-tile transfer
# Demonstrates memtile DMA BD padding for matrix A distribution
#
# Input: [64x500] i32 from host (a chunk of a [768x500] matrix)
# L3->L2: Move into memtile
# L2->L1: Memtile DMA pads columns 500->512 (12 zeros after),
#          distributes to 4 cores in a column, each [64x128]
# Passthrough (identity)
# L1->L2: Collected back appended as [64x512]
# L2->L3: Copy back to host as [64x512]

import argparse

import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.backend.xrt_runner import XRTRunner, type_mapper

INPUT_ROWS = 64
INPUT_COLS = 500
PADDED_COLS = 512
PAD_AMOUNT = PADDED_COLS - INPUT_COLS  # 12

NUM_CORES = 4
TILE_COLS = PADDED_COLS // NUM_CORES  # 128

# Real columns in the last core: 500 - 3*128 = 116
LAST_CORE_REAL_COLS = INPUT_COLS - (NUM_CORES - 1) * TILE_COLS  # 116

INOUT_DATATYPE = np.int32


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)

    input_type = MemRefType.get([INPUT_ROWS, INPUT_COLS], xrt_dtype)
    output_type = MemRefType.get([INPUT_ROWS, PADDED_COLS], xrt_dtype)

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    mem_space_l2 = IntegerAttr.get(T.i32(), MemorySpace.L2)

    tile_type_l1 = MemRefType.get(
        shape=[INPUT_ROWS, TILE_COLS],
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    l2_in_type = MemRefType.get(
        shape=[INPUT_ROWS, INPUT_COLS],
        element_type=xrt_dtype,
        memory_space=mem_space_l2,
    )

    l2_out_type = MemRefType.get(
        shape=[INPUT_ROWS, PADDED_COLS],
        element_type=xrt_dtype,
        memory_space=mem_space_l2,
    )

    Channel("L3ToL2")
    Channel("L2ToL1", size=[1, NUM_CORES])
    Channel("L1ToL2", size=[1, NUM_CORES])
    Channel("L2ToL3")

    @FuncOp.from_py_func(input_type, output_type)
    def pad_passthrough(input_buf, output_buf):

        @launch(operands=[input_buf, output_buf])
        def launch_body(l3_in, l3_out):

            ChannelPut("L3ToL2", l3_in)
            ChannelGet("L2ToL3", l3_out)

            @segment(name="seg")
            def segment_body():

                l2_in = AllocOp(l2_in_type, [], [])
                l2_out = AllocOp(l2_out_type, [], [])

                # L3 -> L2
                ChannelGet("L3ToL2", l2_in)

                # L2 -> L1: distribute columns with padding on last core
                for c in range(NUM_CORES):
                    col_off = c * TILE_COLS
                    if c < NUM_CORES - 1:
                        ChannelPut(
                            "L2ToL1",
                            l2_in,
                            indices=[0, c],
                            offsets=[0, col_off],
                            sizes=[INPUT_ROWS, TILE_COLS],
                            strides=[INPUT_COLS, 1],
                        )
                    else:
                        # Last core: pad 116 -> 128 columns
                        ChannelPut(
                            "L2ToL1",
                            l2_in,
                            indices=[0, c],
                            offsets=[0, col_off],
                            sizes=[INPUT_ROWS, LAST_CORE_REAL_COLS],
                            strides=[INPUT_COLS, 1],
                            pad_before=[0, 0],
                            pad_after=[0, PAD_AMOUNT],
                        )

                # 4-core herd: passthrough
                @herd(name="compute", sizes=[1, NUM_CORES])
                def herd_body(tx, ty, sx, sy):
                    tile_buf = AllocOp(tile_type_l1, [], [])
                    ChannelGet("L2ToL1", tile_buf, indices=[tx, ty])
                    ChannelPut("L1ToL2", tile_buf, indices=[tx, ty])
                    DeallocOp(tile_buf)

                # L1 -> L2: collect
                for c in range(NUM_CORES):
                    col_off = c * TILE_COLS
                    ChannelGet(
                        "L1ToL2",
                        l2_out,
                        indices=[0, c],
                        offsets=[0, col_off],
                        sizes=[INPUT_ROWS, TILE_COLS],
                        strides=[PADDED_COLS, 1],
                    )

                # L2 -> L3
                ChannelPut("L2ToL3", l2_out)

                DeallocOp(l2_in)
                DeallocOp(l2_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Tests DMA padding during memtile-to-tile transfer",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
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

    # Generate input: [64x500] with sequential values mod 1000
    input_data = (
        (np.arange(INPUT_ROWS * INPUT_COLS, dtype=np.int64) % 1000)
        .astype(INOUT_DATATYPE)
        .reshape(INPUT_ROWS, INPUT_COLS)
    )

    # Expected output: [64x512] with original data + 12 zero-padded columns
    expected = np.zeros((INPUT_ROWS, PADDED_COLS), dtype=INOUT_DATATYPE)
    expected[:, :INPUT_COLS] = input_data

    # Stochastically sample for verification
    num_samples = 200
    np.random.seed(42)
    sampled_row = np.random.randint(0, INPUT_ROWS, num_samples)
    sampled_col = np.random.randint(0, PADDED_COLS, num_samples)
    sampled_indices = np.vstack([sampled_row, sampled_col])
    sampled_values = np.array(
        [expected[r, c] for r, c in zip(sampled_row, sampled_col)],
        dtype=INOUT_DATATYPE,
    )

    sampled_data = {
        "shape": (INPUT_ROWS, PADDED_COLS),
        "indices": sampled_indices,
        "values": sampled_values,
    }

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="pad_passthrough",
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_data],
            stochastic_expected_outputs=[sampled_data],
        )
    )
