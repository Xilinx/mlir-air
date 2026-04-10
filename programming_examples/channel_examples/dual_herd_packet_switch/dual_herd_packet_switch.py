# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Dual-herd example with packet-switched shim DMA channels.
#
# Two 8x1 herds coexist on the same NPU2 device:
#   - add_herd: element-wise add (C_add = A0 + B0)
#   - mul_herd: element-wise mul (C_mul = A1 * B1)
#
# All four L3-to-L1 input channels use channel_type="dma_packet" so that
# the shim DMA MM2S ports are time-multiplexed via packet-switched routing.
# Without packet switching, 4 channels x 8 tiles = 32 circuit-switched flows
# would exceed the 16 available shim MM2S channels on NPU2. With dma_packet,
# the ShimDMAAllocator reuses channels on the same shim tile via packet IDs.

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.backend.xrt_runner import XRTRunner, type_mapper
import air.dialects.linalg.opdsl.lang as linalg_lang

TILE_SIZE = 1024
HERD_SIZE = 8
TOTAL_SIZE = TILE_SIZE * HERD_SIZE  # 8192

INOUT_DATATYPE = bfloat16


# elemwise_binary is deprecated from upstream; define as custom linalg op.
@linalg_lang.linalg_structured_op
def elemwise_binary(
    lhs=linalg_lang.TensorDef(linalg_lang.TV.T1),
    rhs=linalg_lang.TensorDef(linalg_lang.TV.T2),
    O=linalg_lang.TensorDef(linalg_lang.U, output=True),
    fun=linalg_lang.BinaryFnAttrDef(default=linalg_lang.BinaryFn.add),
    cast=linalg_lang.TypeFnAttrDef(default=linalg_lang.TypeFn.cast_signed),
):
    O[None] = fun(cast(linalg_lang.U, lhs[None]), cast(linalg_lang.U, rhs[None]))


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)

    # L3 memref types
    l3_type = MemRefType.get([TOTAL_SIZE], xrt_dtype)

    # L1 tile memref types
    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_type = MemRefType.get(
        shape=[TILE_SIZE],
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    # --- Input channels: packet-switched for shim DMA sharing ---
    channel("A0_to_L1", size=[HERD_SIZE, 1], channel_type="dma_packet")
    channel("B0_to_L1", size=[HERD_SIZE, 1], channel_type="dma_packet")
    channel("A1_to_L1", size=[HERD_SIZE, 1], channel_type="dma_packet")
    channel("B1_to_L1", size=[HERD_SIZE, 1], channel_type="dma_packet")

    # --- Output channels: circuit-switched (default dma_stream) ---
    Channel("Add_out", size=[HERD_SIZE, 1])
    Channel("Mul_out", size=[HERD_SIZE, 1])

    @FuncOp.from_py_func(l3_type, l3_type, l3_type, l3_type, l3_type, l3_type)
    def dual_herd_elemwise(a0, b0, a1, b1, c_add, c_mul):

        @launch(operands=[a0, b0, a1, b1, c_add, c_mul])
        def launch_body(a0_, b0_, a1_, b1_, c_add_, c_mul_):

            # L3 -> L1 puts for all four inputs (packet-switched)
            for tile_idx in range(HERD_SIZE):
                offset = tile_idx * TILE_SIZE
                ChannelPut(
                    "A0_to_L1",
                    a0_,
                    offsets=[offset],
                    sizes=[TILE_SIZE],
                    strides=[1],
                    indices=[tile_idx, 0],
                )
                ChannelPut(
                    "B0_to_L1",
                    b0_,
                    offsets=[offset],
                    sizes=[TILE_SIZE],
                    strides=[1],
                    indices=[tile_idx, 0],
                )
                ChannelPut(
                    "A1_to_L1",
                    a1_,
                    offsets=[offset],
                    sizes=[TILE_SIZE],
                    strides=[1],
                    indices=[tile_idx, 0],
                )
                ChannelPut(
                    "B1_to_L1",
                    b1_,
                    offsets=[offset],
                    sizes=[TILE_SIZE],
                    strides=[1],
                    indices=[tile_idx, 0],
                )

            # L1 -> L3 gets for outputs (circuit-switched)
            for tile_idx in range(HERD_SIZE):
                offset = tile_idx * TILE_SIZE
                ChannelGet(
                    "Add_out",
                    c_add_,
                    offsets=[offset],
                    sizes=[TILE_SIZE],
                    strides=[1],
                    indices=[tile_idx, 0],
                )
                ChannelGet(
                    "Mul_out",
                    c_mul_,
                    offsets=[offset],
                    sizes=[TILE_SIZE],
                    strides=[1],
                    indices=[tile_idx, 0],
                )

            @segment(name="seg")
            def segment_body():

                # --- Herd 0: element-wise add ---
                @herd(name="add_herd", sizes=[HERD_SIZE, 1])
                def add_herd_body(tx, ty, sx, sy):
                    tile_a = AllocOp(l1_type, [], [])
                    tile_b = AllocOp(l1_type, [], [])
                    tile_c = AllocOp(l1_type, [], [])

                    ChannelGet("A0_to_L1", tile_a, indices=[tx, 0])
                    ChannelGet("B0_to_L1", tile_b, indices=[tx, 0])

                    elemwise_binary(
                        tile_a,
                        tile_b,
                        outs=[tile_c],
                        fun=linalg_lang.BinaryFn.add,
                        cast=linalg_lang.TypeFn.cast_signed,
                    )

                    ChannelPut("Add_out", tile_c, indices=[tx, 0])

                    DeallocOp(tile_a)
                    DeallocOp(tile_b)
                    DeallocOp(tile_c)

                # --- Herd 1: element-wise mul ---
                @herd(name="mul_herd", sizes=[HERD_SIZE, 1])
                def mul_herd_body(tx, ty, sx, sy):
                    tile_a = AllocOp(l1_type, [], [])
                    tile_b = AllocOp(l1_type, [], [])
                    tile_c = AllocOp(l1_type, [], [])

                    ChannelGet("A1_to_L1", tile_a, indices=[tx, 0])
                    ChannelGet("B1_to_L1", tile_b, indices=[tx, 0])

                    elemwise_binary(
                        tile_a,
                        tile_b,
                        outs=[tile_c],
                        fun=linalg_lang.BinaryFn.mul,
                        cast=linalg_lang.TypeFn.cast_signed,
                    )

                    ChannelPut("Mul_out", tile_c, indices=[tx, 0])

                    DeallocOp(tile_a)
                    DeallocOp(tile_b)
                    DeallocOp(tile_c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="dual_herd_packet_switch.py",
        description="Dual 8x1 herd elemwise add/mul with packet-switched shim DMA",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
        dest="output_format",
        help="Output format for the compiled binary (default: elf)",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Golden reference computation using small integers (exact in bf16)
    np.random.seed(42)
    a0 = np.random.randint(-8, 9, size=TOTAL_SIZE).astype(bfloat16)
    b0 = np.random.randint(-8, 9, size=TOTAL_SIZE).astype(bfloat16)
    a1 = np.random.randint(-4, 5, size=TOTAL_SIZE).astype(bfloat16)
    b1 = np.random.randint(-4, 5, size=TOTAL_SIZE).astype(bfloat16)

    c_add_ref = (a0.astype(np.float32) + b0.astype(np.float32)).astype(bfloat16)
    c_mul_ref = (a1.astype(np.float32) * b1.astype(np.float32)).astype(bfloat16)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="dual_herd_elemwise",
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[a0, b0, a1, b1],
            expected_outputs=[c_add_ref, c_mul_ref],
        )
    )
