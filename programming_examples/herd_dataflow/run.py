# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AIR Herd Dataflow Example

This script demonstrates building and running a dataflow module using AIR and MLIR constructs.
"""

import argparse
import numpy as np

import air
from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import memref, vector, arith, scf
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store, subview
from air.backend.xrt_runner import XRTRunner
from ml_dtypes import bfloat16

# Constants for buffer sizes and loop bounds
L1_BUFFER_SIZE_M = 64
L2_BUFFER_SIZE_M = L1_BUFFER_SIZE_M
# Number of columns used in the hardware array
NUM_COLUMNS = 4
L1_BUFFER_SIZE_N = 64
L2_BUFFER_SIZE_N = L1_BUFFER_SIZE_N * NUM_COLUMNS
# Stride (step size) in the column dimension for channel and DMA operations; 1 means data is contiguous in memory
CHANNEL_STRIDE = 1
# Number of elements processed in a single vector operation (matches hardware vector width)
VECTOR_SIZE = 16

range_ = for_


def parse_args():
    parser = argparse.ArgumentParser(description="AIR Herd Dataflow Example")
    parser.add_argument(
        "--m-size",
        type=int,
        default=256,
        help="Number of rows (M dimension) for L2 buffer",
    )
    parser.add_argument(
        "--n-size",
        type=int,
        default=256,
        help="Number of columns (N dimension) for L2 buffer",
    )
    parser.add_argument(
        "-p", "--print-ir", action="store_true", help="Print MLIR IR and exit"
    )
    parser.add_argument(
        "--mlir-source",
        choices=["python", "file"],
        default="python",
        help="How to obtain the MLIR module: 'python' (build with Python bindings) or 'file' (load and parse air.mlir; NOTE: air.mlir is only valid for M=N=256)",
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
    Build the AIR dataflow module with memory allocation, channel setup, and computation kernels.

    Returns:
        The constructed MLIR module.
    """

    # External function for the final compute stage (herd_2)
    # This function will be linked from "extern_func.o" and called in herd_2 for further computation.
    input_type = bfloat16
    output_type = bfloat16
    index_type = IndexType.get()
    bf16 = air.ir.Type.parse("bf16")
    memrefMxN = MemRefType.get((M_SIZE, N_SIZE), bf16)
    memrefL1xN_l2 = MemRefType.get(
        (L1_BUFFER_SIZE_M, L2_BUFFER_SIZE_N), bf16, memory_space=Attribute.parse("1")
    )
    memrefL1xL1_l1 = MemRefType.get(
        (L1_BUFFER_SIZE_M, L1_BUFFER_SIZE_N), bf16, memory_space=Attribute.parse("2")
    )

    add_3_bf16_ty = FunctionType.get([memrefL1xL1_l1, memrefL1xL1_l1], [])
    add_3_func = FuncOp("add_3_bf16", add_3_bf16_ty, visibility="private")
    add_3_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
    add_3_func.attributes["link_with"] = StringAttr.get("extern_func.o")

    # AIR channels model hardware FIFOs for inter-stage communication
    # Default channel_type is "dma_stream", representing data movement performed using DMA streaming interconnects
    channel("L2ToL1Chan1", size=[NUM_COLUMNS, 1])  # L2 to L1, input A
    channel("L2ToL1Chan2", size=[NUM_COLUMNS, 1])  # L2 to L1, input B
    channel("L1ToL1Chan1", size=[NUM_COLUMNS, 1])  # Between herd_0 and herd_1
    channel(
        "L1ToL1Chan2", size=[NUM_COLUMNS, 1], channel_type="cascade"
    )  # Between herd_1 and herd_2; channel_type="cascade" means peer-to-peer communication between compute tiles
    channel("L1ToL2Chan1", size=[NUM_COLUMNS, 1])  # Output from herd_2 to L2

    @FuncOp.from_py_func(memrefMxN, memrefMxN, memrefMxN)
    def func1(arg0, arg1, arg2):
        """
        Top-level function: runtime dispatch over a "launch" iteration space (not necessarily hardware parallelism).
        Compute the three-stage dataflow using vector addition, copy, and external function call kernels.

        Args:
            arg0, arg1, arg2: Input and output memory references.

        Returns:
            None. Operates via side effects on MLIR memory references.
        """
        launch_x_size = ConstantOp(index_type, M_SIZE // L1_BUFFER_SIZE_M)
        launch_y_size = ConstantOp(index_type, N_SIZE // L2_BUFFER_SIZE_N)

        # air.launch: runtime dispatch over a "launch" iteration space (may be sequential or parallel depending on runtime)
        @launch(operands=[arg0, arg1, arg2], sizes=[launch_x_size, launch_y_size])
        def launch_body(
            launch_ivx, launch_ivy, launch_sizex, launch_sizey, l3_a, l3_b, l3_c
        ):

            # Each segment is a program mapped to a hardware tile of resources, including the L1 and L2 memories and compute cores
            @segment(operands=[launch_ivx, launch_ivy, l3_a, l3_b, l3_c])
            def segment_body(lau_ivx, lau_ivy, seg_a, seg_b, seg_c):
                c0 = ConstantOp(index_type, 0)
                c1 = ConstantOp(index_type, 1)
                cML1 = ConstantOp(index_type, L1_BUFFER_SIZE_M)
                cNL1 = ConstantOp(index_type, L1_BUFFER_SIZE_N)
                cML2 = ConstantOp(index_type, L2_BUFFER_SIZE_M)
                cNL2 = ConstantOp(index_type, L2_BUFFER_SIZE_N)
                cM = ConstantOp(index_type, M_SIZE)
                cN = ConstantOp(index_type, N_SIZE)
                # Allocate L2 buffers for tile-local computation (memory space 1 = L2)
                alloc_1 = AllocOp(memrefL1xN_l2, [], [])
                alloc_2 = AllocOp(memrefL1xN_l2, [], [])
                alloc_3 = AllocOp(memrefL1xN_l2, [], [])

                mul_m_l2_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(L2_BUFFER_SIZE_M),
                        )
                    ],
                )
                mul_n_l2_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(L2_BUFFER_SIZE_N),
                        )
                    ],
                )
                mul_n_l1_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(L1_BUFFER_SIZE_M),
                        )
                    ],
                )
                pid_x_offset = affine_apply(mul_m_l2_map, [lau_ivx])
                pid_y_offset = affine_apply(mul_n_l2_map, [lau_ivy])

                mac_n_l2_map = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(L1_BUFFER_SIZE_N),
                            ),
                            AffineSymbolExpr.get(1),
                        )
                    ],
                )
                # Stage 1: DMA from L2 to L1 for both inputs, in parallel for each sub-tile
                par_1 = scf.ForallOp(
                    lower_bounds=[c0], upper_bounds=[NUM_COLUMNS], steps=[c1]
                )
                with InsertionPoint(par_1.body):
                    apply_l1 = affine_apply(
                        mul_n_l1_map, [par_1.induction_variables[0]]
                    )
                    apply_l2 = affine_apply(
                        mac_n_l2_map, [par_1.induction_variables[0], pid_y_offset]
                    )
                    dma_memcpy_nd(
                        alloc_1.result,
                        seg_a,
                        dst_offsets=[c0, apply_l1],
                        dst_sizes=[cML1, cNL1],
                        dst_strides=[cNL2, c1],
                        src_offsets=[pid_x_offset, apply_l2],
                        src_sizes=[cML1, cNL1],
                        src_strides=[cN, c1],
                    ),
                    dma_memcpy_nd(
                        alloc_2.result,
                        seg_b,
                        dst_offsets=[c0, apply_l1],
                        dst_sizes=[cML1, cNL1],
                        dst_strides=[cNL2, c1],
                        src_offsets=[pid_x_offset, apply_l2],
                        src_sizes=[cML1, cNL1],
                        src_strides=[cN, c1],
                    )
                    scf.InParallelOp()

                # Stage 2: Send L1 buffers to next stage via AIR channels
                par_2 = scf.ForallOp(
                    lower_bounds=[c0], upper_bounds=[NUM_COLUMNS], steps=[c1]
                )
                with InsertionPoint(par_2.body):
                    apply = affine_apply(mul_n_l1_map, [par_2.induction_variables[0]])
                    ChannelPut(
                        "L2ToL1Chan1",
                        alloc_1.result,
                        indices=[par_2.induction_variables[0], c0],
                        offsets=[c0, apply],
                        sizes=[cML1, cNL1],
                        strides=[cNL2, c1],
                    ),
                    ChannelPut(
                        "L2ToL1Chan2",
                        alloc_2.result,
                        indices=[par_2.induction_variables[0], c0],
                        offsets=[c0, apply],
                        sizes=[cML1, cNL1],
                        strides=[cNL2, c1],
                    ),
                    scf.InParallelOp()

                # Stage 3: First compute herd (herd_0): NUM_COLUMNS x 1 shape. Each of the NUM_COLUMNS tiles asynchronously receives two input tiles via channels, performs vector addition, and sends the result to the next stage.
                @herd(name="herd_0", sizes=[NUM_COLUMNS, c1], operands=[])
                def herd0_body(h0_x, h0_y, h0_x_size, h0_y_size):
                    alloc_a = AllocOp(
                        memrefL1xL1_l1, [], []
                    )  # L1 memory (memory space 2)
                    alloc_b = AllocOp(
                        memrefL1xL1_l1, [], []
                    )  # L1 memory (memory space 2)
                    alloc_c = AllocOp(
                        memrefL1xL1_l1, [], []
                    )  # L1 memory (memory space 2)
                    ChannelGet("L2ToL1Chan1", alloc_a.result, indices=[h0_x, h0_y])
                    ChannelGet("L2ToL1Chan2", alloc_b.result, indices=[h0_x, h0_y])
                    c0 = ConstantOp(index_type, 0)
                    c1 = ConstantOp(index_type, 1)
                    c16 = ConstantOp(index_type, VECTOR_SIZE)
                    cML1 = ConstantOp(index_type, L1_BUFFER_SIZE_M)
                    cNL1 = ConstantOp(index_type, L1_BUFFER_SIZE_N)
                    for i in range_(c0, cML1, c1):
                        sub_a = memref.subview(
                            alloc_a.result,
                            [i, 0],
                            [1, L1_BUFFER_SIZE_M],
                            [1, 1],
                        )
                        sub_b = memref.subview(
                            alloc_b.result,
                            [i, 0],
                            [1, L1_BUFFER_SIZE_M],
                            [1, 1],
                        )
                        sub_c = memref.subview(
                            alloc_c.result,
                            [i, 0],
                            [1, L1_BUFFER_SIZE_M],
                            [1, 1],
                        )
                        for j in range_(c0, cNL1, c16):
                            sub_a_16 = memref.subview(
                                sub_a,
                                [0, j],
                                [1, VECTOR_SIZE],
                                [1, 1],
                            )
                            sub_b_16 = memref.subview(
                                sub_b,
                                [0, j],
                                [1, VECTOR_SIZE],
                                [1, 1],
                            )
                            sub_c_16 = memref.subview(
                                sub_c,
                                [0, j],
                                [1, VECTOR_SIZE],
                                [1, 1],
                            )
                            layout = StridedLayoutAttr.get(
                                ShapedType.get_dynamic_size(),
                                [
                                    1,
                                ],
                            )
                            collapsed_type = MemRefType.get(
                                (VECTOR_SIZE,),
                                bf16,
                                memory_space=Attribute.parse("2"),
                                layout=layout,
                            )
                            collapse_dims = [[0, 1]]
                            collapse_a = memref.collapse_shape(
                                collapsed_type, sub_a_16, collapse_dims
                            )
                            collapse_b = memref.collapse_shape(
                                collapsed_type, sub_b_16, collapse_dims
                            )
                            collapse_c = memref.collapse_shape(
                                collapsed_type, sub_c_16, collapse_dims
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
                            v_b = vector.TransferReadOp(
                                VectorType.get([VECTOR_SIZE], bf16),
                                collapse_b,
                                [c0],
                                AffineMapAttr.get(AffineMap.get_identity(1)),
                                cst0,
                                [True],
                            )
                            v_c = arith.AddFOp(v_a, v_b)
                            vector.transfer_write(
                                None,
                                v_c,
                                collapse_c,
                                [c0],
                                AffineMapAttr.get(AffineMap.get_identity(1)),
                                [True],
                            )
                            yield_([])
                        yield_([])
                    ChannelPut("L1ToL1Chan1", alloc_c.result, indices=[h0_x, h0_y])

                # Stage 4: Second herd (herd_1): NUM_COLUMNS x 1 shape. Each of the NUM_COLUMNS tiles asynchronously receives a tile via channel, copies its contents, and sends it to the next stage.
                @herd(name="herd_1", sizes=[NUM_COLUMNS, c1], operands=[])
                def herd1_body(h1_x, h1_y, h1_x_size, h1_y_size):
                    c0 = ConstantOp(index_type, 0)
                    c1 = ConstantOp(index_type, 1)
                    cML1 = ConstantOp(index_type, L1_BUFFER_SIZE_M)
                    cNL1 = ConstantOp(index_type, L1_BUFFER_SIZE_N)
                    alloc_a = AllocOp(
                        memrefL1xL1_l1, [], []
                    )  # L1 memory (memory space 2)
                    alloc_c = AllocOp(
                        memrefL1xL1_l1, [], []
                    )  # L1 memory (memory space 2)
                    ChannelGet("L1ToL1Chan1", alloc_a.result, indices=[h1_x, h1_y])
                    for i in range_(c0, cML1, c1):
                        for j in range_(c0, cNL1, c1):
                            store(load(alloc_a.result, [i, j]), alloc_c.result, [i, j])
                            yield_([])
                        yield_([])
                    ChannelPut("L1ToL1Chan2", alloc_c.result, indices=[h1_x, h1_y])

                # Stage 5: Third herd (herd_2): NUM_COLUMNS x 1 shape. Each of the NUM_COLUMNS tiles asynchronously receives a tile via channel, calls an external function for further computation, and sends the result to the output channel.
                @herd(name="herd_2", sizes=[NUM_COLUMNS, c1], operands=[])
                def herd2_body(h2_x, h2_y, h2_x_size, h2_y_size):
                    alloc_a = AllocOp(
                        memrefL1xL1_l1, [], []
                    )  # L1 memory (memory space 2)
                    alloc_c = AllocOp(
                        memrefL1xL1_l1, [], []
                    )  # L1 memory (memory space 2)
                    ChannelGet("L1ToL1Chan2", alloc_a.result, indices=[h2_x, h2_y])
                    CallOp(add_3_func, [alloc_a.result, alloc_c.result])
                    ChannelPut("L1ToL2Chan1", alloc_c.result, indices=[h2_x, h2_y])

                herd2_body.attributes["link_with"] = StringAttr.get("extern_func.o")

                # Stage 6: Gather results from all tiles and DMA back to L2
                par_3 = scf.ForallOp(
                    lower_bounds=[c0], upper_bounds=[NUM_COLUMNS], steps=[c1]
                )
                with InsertionPoint(par_3.body):
                    apply_l1 = affine_apply(
                        mul_n_l1_map, [par_3.induction_variables[0]]
                    )
                    apply_l2 = affine_apply(
                        mac_n_l2_map, [par_3.induction_variables[0], pid_y_offset]
                    )
                    ChannelGet(
                        "L1ToL2Chan1",
                        alloc_3.result,
                        indices=[par_3.induction_variables[0], c0],
                        offsets=[c0, apply_l1],
                        sizes=[cML1, cNL1],
                        strides=[cNL2, c1],
                    ),
                    dma_memcpy_nd(
                        seg_c,
                        alloc_3.result,
                        dst_offsets=[pid_x_offset, apply_l2],
                        dst_sizes=[cML1, cNL1],
                        dst_strides=[cN, c1],
                        src_offsets=[c0, apply_l1],
                        src_sizes=[cML1, cNL1],
                        src_strides=[cNL2, c1],
                    )
                    scf.InParallelOp()


def main():
    """Main entry point for running the AIR Herd Dataflow example."""
    args = parse_args()
    M_SIZE = args.m_size
    N_SIZE = args.n_size

    # Obtain the MLIR module either by building with Python or loading from file
    if args.mlir_source == "python":
        mlir_module = build_module(M_SIZE, N_SIZE)
    else:
        import os

        script_dir = os.path.dirname(os.path.abspath(__file__))
        mlir_path = os.path.join(script_dir, "air.mlir")
        with open(mlir_path, "r") as f:
            mlir_text = f.read()
        mlir_module = Module.parse(mlir_text, context=air.ir.Context())

    if args.print_ir:
        # Print the MLIR IR and exit
        print(str(mlir_module))
        return

    # Prepare input and expected output data
    A = np.random.rand(M_SIZE, N_SIZE).astype(bfloat16)
    B = np.random.rand(M_SIZE, N_SIZE).astype(bfloat16)
    C = (A + B + 3.0).astype(bfloat16)

    # Run the module using XRTRunner
    runner = XRTRunner(
        omit_while_true_loop=False,
        verbose=False,
        runtime_loop_tiling_sizes=[2, 2],
        output_format=args.output_format,
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[A, B],
            expected_outputs=[C],
            rtol=1e-2,
        )
    )


if __name__ == "__main__":
    main()
