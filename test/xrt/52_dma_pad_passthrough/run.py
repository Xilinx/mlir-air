# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Test: DMA padding via air.dma_memcpy_nd lowered through -air-dma-to-channel
#
# This test validates that padding attributes on air.dma_memcpy_nd propagate
# correctly to air.channel.put during the -air-dma-to-channel conversion.
#
# Input: [64x480] i32 from host
# L3->L2: DMA into memtile
# L2->L1: Per-core DMA, each core reads 120 cols padded to 128
# Passthrough (identity)
# L1->L2: DMA back to memtile as [64x512]
# L2->L3: DMA back to host

import argparse

import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.backend.xrt_runner import XRTRunner, type_mapper

INPUT_ROWS = 64
INPUT_COLS = 480
PADDED_COLS = 512

NUM_CORES = 4
TILE_COLS = PADDED_COLS // NUM_CORES  # 128
REAL_COLS_PER_CORE = INPUT_COLS // NUM_CORES  # 120
PAD_PER_CORE = TILE_COLS - REAL_COLS_PER_CORE  # 8

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

    @FuncOp.from_py_func(input_type, output_type)
    def pad_passthrough(input_buf, output_buf):

        @launch(operands=[input_buf, output_buf])
        def launch_body(l3_in, l3_out):

            @segment(name="seg", operands=[l3_in, l3_out])
            def segment_body(seg_in, seg_out):

                l2_in = AllocOp(l2_in_type, [], [])
                l2_out = AllocOp(l2_out_type, [], [])

                # L3 -> L2: DMA input from host to memtile
                dma_memcpy_nd(l2_in, seg_in)

                # 4-core herd: each core reads 120 cols with 8-col padding
                @herd(
                    name="compute",
                    sizes=[1, NUM_CORES],
                    operands=[l2_in, l2_out],
                )
                def herd_body(tx, ty, sx, sy, h_l2_in, h_l2_out):
                    tile_buf = AllocOp(tile_type_l1, [], [])

                    # Compute column offset: ty * REAL_COLS_PER_CORE
                    c_real = arith.ConstantOp(IndexType.get(), REAL_COLS_PER_CORE)
                    col_off = arith.MulIOp(ty, c_real)

                    # Constants for DMA parameters
                    c0 = arith.ConstantOp.create_index(0)
                    c1 = arith.ConstantOp.create_index(1)
                    c_rows = arith.ConstantOp.create_index(INPUT_ROWS)
                    c_real_cols = arith.ConstantOp.create_index(REAL_COLS_PER_CORE)
                    c_tile_cols = arith.ConstantOp.create_index(TILE_COLS)
                    c_in_cols = arith.ConstantOp.create_index(INPUT_COLS)
                    c_out_cols = arith.ConstantOp.create_index(PADDED_COLS)

                    # L2 -> L1: read 120 cols, pad to 128
                    dma_memcpy_nd(
                        tile_buf,
                        h_l2_in,
                        src_offsets=[c0, col_off],
                        src_sizes=[c_rows, c_real_cols],
                        src_strides=[c_in_cols, c1],
                        pad_before=[0, 0],
                        pad_after=[0, PAD_PER_CORE],
                    )

                    # L1 -> L2: write 128 cols
                    # Output column offset: ty * TILE_COLS
                    c_tile = arith.ConstantOp(IndexType.get(), TILE_COLS)
                    out_col_off = arith.MulIOp(ty, c_tile)
                    dma_memcpy_nd(
                        h_l2_out,
                        tile_buf,
                        dst_offsets=[c0, out_col_off],
                        dst_sizes=[c_rows, c_tile_cols],
                        dst_strides=[c_out_cols, c1],
                    )

                    DeallocOp(tile_buf)

                # L2 -> L3: DMA output from memtile to host
                dma_memcpy_nd(seg_out, l2_out)

                DeallocOp(l2_in)
                DeallocOp(l2_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Tests DMA padding via dma_memcpy_nd through -air-dma-to-channel",
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

    # Generate input: [64x480] with sequential values mod 1000
    input_data = (
        (np.arange(INPUT_ROWS * INPUT_COLS, dtype=np.int64) % 1000)
        .astype(INOUT_DATATYPE)
        .reshape(INPUT_ROWS, INPUT_COLS)
    )

    # Expected output: [64x512] with original data interleaved with padding
    # Each core's 120-col slice gets 8 zero-padded cols appended
    expected = np.zeros((INPUT_ROWS, PADDED_COLS), dtype=INOUT_DATATYPE)
    for c in range(NUM_CORES):
        src_start = c * REAL_COLS_PER_CORE
        dst_start = c * TILE_COLS
        expected[:, dst_start : dst_start + REAL_COLS_PER_CORE] = input_data[
            :, src_start : src_start + REAL_COLS_PER_CORE
        ]

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
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_data],
            stochastic_expected_outputs=[sampled_data],
        )
    )
