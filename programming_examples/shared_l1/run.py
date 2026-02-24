# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AIR Shared L1 Buffer Dataflow Example

This script demonstrates data flowing through air herds via shared L1 buffers.
The AIE architecture features shared L1 buffer across neighboring AIE tiles.

A shared L1 buffer is represented via a memref with memory_space=2, that is
allocated outside of both herds, but used by both herds via their args list.

Dataflow:
  1. herd_producer: Reads input from external → local L1 buffer A
                    Computes A + 1 → shared L1 buffer B
  2. herd_consumer: Reads from shared L1 buffer B
                    Computes B + 2 → local L1 buffer C
                    Writes output from C → external

Final result: output = input + 3
"""

import argparse
import numpy as np

import air
from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import memref, vector, arith, scf
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp
from air.backend.xrt_runner import XRTRunner
from ml_dtypes import bfloat16

# Constants for buffer sizes
L1_BUFFER_SIZE_M = 64
L1_BUFFER_SIZE_N = 64
L2_BUFFER_SIZE_M = L1_BUFFER_SIZE_M
L2_BUFFER_SIZE_N = L1_BUFFER_SIZE_N
# Number of elements processed in a single vector operation
VECTOR_SIZE = 16

range_ = for_


def parse_args():
    parser = argparse.ArgumentParser(description="AIR Shared L1 Buffer Example")
    parser.add_argument(
        "--m-size",
        type=int,
        default=64,
        help="Number of rows (M dimension)",
    )
    parser.add_argument(
        "--n-size",
        type=int,
        default=64,
        help="Number of columns (N dimension)",
    )
    parser.add_argument(
        "-p", "--print-ir", action="store_true", help="Print MLIR IR and exit"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    args = parser.parse_args()
    return args


@module_builder
def build_module(M_SIZE, N_SIZE):
    """
    Build the AIR dataflow module demonstrating shared L1 buffer communication.

    The key feature is that a shared L1 buffer (memory_space=2) is allocated
    at the segment level (outside both herds) and passed to both herds via
    their operands list.

    Returns:
        The constructed MLIR module.
    """
    index_type = IndexType.get()
    bf16 = air.ir.Type.parse("bf16")

    # Memory types
    # L3 memory (external/DDR)
    memrefMxN = MemRefType.get((M_SIZE, N_SIZE), bf16)
    # L2 memory (memory_space=1)
    memrefL2 = MemRefType.get(
        (L2_BUFFER_SIZE_M, L2_BUFFER_SIZE_N), bf16, memory_space=Attribute.parse("1")
    )
    # L1 memory (memory_space=2) - used for both local and shared buffers
    memrefL1 = MemRefType.get(
        (L1_BUFFER_SIZE_M, L1_BUFFER_SIZE_N), bf16, memory_space=Attribute.parse("2")
    )

    # AIR channels for external data movement
    # Input: L2 to herd_producer
    channel("InputToProducer", size=[1, 1])
    # Output: herd_consumer to L2
    channel("ConsumerToOutput", size=[1, 1])

    @FuncOp.from_py_func(memrefMxN, memrefMxN)
    def func1(arg0, arg1):
        """
        Top-level function demonstrating shared L1 buffer dataflow.

        Args:
            arg0: Input memory reference (L3)
            arg1: Output memory reference (L3)

        Returns:
            None. Operates via side effects on MLIR memory references.
        """
        launch_x_size = ConstantOp(index_type, M_SIZE // L1_BUFFER_SIZE_M)
        launch_y_size = ConstantOp(index_type, N_SIZE // L1_BUFFER_SIZE_N)

        @launch(operands=[arg0, arg1], sizes=[launch_x_size, launch_y_size])
        def launch_body(
            launch_ivx, launch_ivy, launch_sizex, launch_sizey, l3_in, l3_out
        ):
            @segment(operands=[launch_ivx, launch_ivy, l3_in, l3_out])
            def segment_body(lau_ivx, lau_ivy, seg_in, seg_out):
                c0 = ConstantOp(index_type, 0)
                c1 = ConstantOp(index_type, 1)
                cML1 = ConstantOp(index_type, L1_BUFFER_SIZE_M)
                cNL1 = ConstantOp(index_type, L1_BUFFER_SIZE_N)
                cM = ConstantOp(index_type, M_SIZE)
                cN = ConstantOp(index_type, N_SIZE)

                # Allocate L2 buffers for input and output staging
                l2_in = AllocOp(memrefL2, [], [])
                l2_out = AllocOp(memrefL2, [], [])

                # ============================================================
                # KEY FEATURE: Shared L1 buffer allocated at SEGMENT level
                # This buffer will be passed to BOTH herds and represents
                # the shared L1 memory between neighboring AIE tiles.
                # ============================================================
                shared_l1_buffer = AllocOp(memrefL1, [], [])

                # Compute offsets based on launch indices
                mul_m_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(L1_BUFFER_SIZE_M),
                        )
                    ],
                )
                mul_n_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(L1_BUFFER_SIZE_N),
                        )
                    ],
                )
                offset_m = affine_apply(mul_m_map, [lau_ivx])
                offset_n = affine_apply(mul_n_map, [lau_ivy])

                # Stage 1: DMA from L3 to L2
                dma_memcpy_nd(
                    l2_in.result,
                    seg_in,
                    dst_offsets=[c0, c0],
                    dst_sizes=[cML1, cNL1],
                    dst_strides=[cNL1, c1],
                    src_offsets=[offset_m, offset_n],
                    src_sizes=[cML1, cNL1],
                    src_strides=[cN, c1],
                )

                # Stage 2: Send L2 buffer to herd_producer via channel
                ChannelPut(
                    "InputToProducer",
                    l2_in.result,
                    indices=[c0, c0],
                    offsets=[c0, c0],
                    sizes=[cML1, cNL1],
                    strides=[cNL1, c1],
                )

                # ============================================================
                # HERD_PRODUCER:
                # - Receives input data from channel → local L1 buffer A
                # - Computes A + 1 → writes to SHARED L1 buffer
                # ============================================================
                @herd(
                    name="herd_producer",
                    sizes=[c1, c1],
                    operands=[shared_l1_buffer.result],  # Pass shared buffer
                )
                def herd_producer_body(
                    hx,
                    hy,
                    hx_size,
                    hy_size,
                    shared_buf,  # Shared L1 buffer received as argument
                ):
                    # Allocate LOCAL L1 buffer A (inside this herd)
                    local_buf_a = AllocOp(memrefL1, [], [])

                    # Receive input data from channel into local buffer A
                    ChannelGet("InputToProducer", local_buf_a.result, indices=[hx, hy])

                    # Compute: shared_buf = local_buf_a + 1
                    c0 = ConstantOp(index_type, 0)
                    c1_idx = ConstantOp(index_type, 1)
                    c16 = ConstantOp(index_type, VECTOR_SIZE)
                    cML1 = ConstantOp(index_type, L1_BUFFER_SIZE_M)
                    cNL1 = ConstantOp(index_type, L1_BUFFER_SIZE_N)

                    # Add 1 to each element and write to shared buffer
                    cst_one = arith.ConstantOp(bf16, 1.0)
                    for i in range_(c0, cML1, c1_idx):
                        for j in range_(c0, cNL1, c16):
                            sub_a = memref.subview(
                                local_buf_a.result, [i, j], [1, VECTOR_SIZE], [1, 1]
                            )
                            sub_shared = memref.subview(
                                shared_buf, [i, j], [1, VECTOR_SIZE], [1, 1]
                            )
                            layout = StridedLayoutAttr.get(
                                ShapedType.get_dynamic_size(), [1]
                            )
                            collapsed_type = MemRefType.get(
                                (VECTOR_SIZE,),
                                bf16,
                                memory_space=Attribute.parse("2"),
                                layout=layout,
                            )
                            collapse_a = memref.collapse_shape(
                                collapsed_type, sub_a, [[0, 1]]
                            )
                            collapse_shared = memref.collapse_shape(
                                collapsed_type, sub_shared, [[0, 1]]
                            )
                            cst0 = arith.ConstantOp(bf16, 0.0)
                            v_a = vector.transfer_read(
                                VectorType.get([VECTOR_SIZE], bf16),
                                collapse_a,
                                [c0],
                                AffineMapAttr.get(AffineMap.get_identity(1)),
                                cst0,
                                [True],
                            )
                            # Create vector of 1.0
                            v_one = vector.BroadcastOp(
                                VectorType.get([VECTOR_SIZE], bf16), cst_one
                            )
                            # Compute A + 1
                            v_result = arith.AddFOp(v_a, v_one)
                            # Write to shared buffer
                            vector.transfer_write(
                                None,
                                v_result,
                                collapse_shared,
                                [c0],
                                AffineMapAttr.get(AffineMap.get_identity(1)),
                                [True],
                            )
                            yield_([])
                        yield_([])

                # ============================================================
                # HERD_CONSUMER:
                # - Reads from SHARED L1 buffer (produced by herd_producer)
                # - Computes shared_buf + 2 → writes to local L1 buffer C
                # - Sends output from local buffer C to channel
                # ============================================================
                @herd(
                    name="herd_consumer",
                    sizes=[c1, c1],
                    operands=[shared_l1_buffer.result],  # Pass same shared buffer
                )
                def herd_consumer_body(
                    hx,
                    hy,
                    hx_size,
                    hy_size,
                    shared_buf,  # Same shared L1 buffer received as argument
                ):
                    # Allocate LOCAL L1 buffer C (inside this herd)
                    local_buf_c = AllocOp(memrefL1, [], [])

                    # Compute: local_buf_c = shared_buf + 2
                    c0 = ConstantOp(index_type, 0)
                    c1_idx = ConstantOp(index_type, 1)
                    c16 = ConstantOp(index_type, VECTOR_SIZE)
                    cML1 = ConstantOp(index_type, L1_BUFFER_SIZE_M)
                    cNL1 = ConstantOp(index_type, L1_BUFFER_SIZE_N)

                    # Add 2 to each element from shared buffer, write to local C
                    cst_two = arith.ConstantOp(bf16, 2.0)
                    for i in range_(c0, cML1, c1_idx):
                        for j in range_(c0, cNL1, c16):
                            sub_shared = memref.subview(
                                shared_buf, [i, j], [1, VECTOR_SIZE], [1, 1]
                            )
                            sub_c = memref.subview(
                                local_buf_c.result, [i, j], [1, VECTOR_SIZE], [1, 1]
                            )
                            layout = StridedLayoutAttr.get(
                                ShapedType.get_dynamic_size(), [1]
                            )
                            collapsed_type = MemRefType.get(
                                (VECTOR_SIZE,),
                                bf16,
                                memory_space=Attribute.parse("2"),
                                layout=layout,
                            )
                            collapse_shared = memref.collapse_shape(
                                collapsed_type, sub_shared, [[0, 1]]
                            )
                            collapse_c = memref.collapse_shape(
                                collapsed_type, sub_c, [[0, 1]]
                            )
                            cst0 = arith.ConstantOp(bf16, 0.0)
                            # Read from shared buffer
                            v_shared = vector.transfer_read(
                                VectorType.get([VECTOR_SIZE], bf16),
                                collapse_shared,
                                [c0],
                                AffineMapAttr.get(AffineMap.get_identity(1)),
                                cst0,
                                [True],
                            )
                            # Create vector of 2.0
                            v_two = vector.BroadcastOp(
                                VectorType.get([VECTOR_SIZE], bf16), cst_two
                            )
                            # Compute shared + 2
                            v_result = arith.AddFOp(v_shared, v_two)
                            # Write to local buffer C
                            vector.transfer_write(
                                None,
                                v_result,
                                collapse_c,
                                [c0],
                                AffineMapAttr.get(AffineMap.get_identity(1)),
                                [True],
                            )
                            yield_([])
                        yield_([])

                    # Send output from local buffer C to channel
                    ChannelPut("ConsumerToOutput", local_buf_c.result, indices=[hx, hy])

                # Stage: Receive output from herd_consumer via channel
                ChannelGet(
                    "ConsumerToOutput",
                    l2_out.result,
                    indices=[c0, c0],
                    offsets=[c0, c0],
                    sizes=[cML1, cNL1],
                    strides=[cNL1, c1],
                )

                # Stage: DMA from L2 to L3 (output)
                dma_memcpy_nd(
                    seg_out,
                    l2_out.result,
                    dst_offsets=[offset_m, offset_n],
                    dst_sizes=[cML1, cNL1],
                    dst_strides=[cN, c1],
                    src_offsets=[c0, c0],
                    src_sizes=[cML1, cNL1],
                    src_strides=[cNL1, c1],
                )


def main():
    """Main entry point for running the AIR Shared L1 Buffer example."""
    args = parse_args()
    M_SIZE = args.m_size
    N_SIZE = args.n_size

    # Build the MLIR module
    mlir_module = build_module(M_SIZE, N_SIZE)

    if args.print_ir:
        # Print the MLIR IR and exit
        print(str(mlir_module))
        return

    # Prepare input and expected output data
    # Final result: output = input + 1 (producer) + 2 (consumer) = input + 3
    A = np.random.rand(M_SIZE, N_SIZE).astype(bfloat16)
    C = (A + 3.0).astype(bfloat16)

    # Run the module using XRTRunner
    runner = XRTRunner(
        omit_while_true_loop=False,
        verbose=False,
        runtime_loop_tiling_sizes=[1, 1],
        output_format=args.output_format,
        instance_name="func1",
        debug_ir=True,
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[A],
            expected_outputs=[C],
            rtol=1e-2,
        )
    )


if __name__ == "__main__":
    main()
