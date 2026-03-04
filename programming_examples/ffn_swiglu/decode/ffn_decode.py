# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FFN SwiGLU Decode — Multi-Column Feed-Forward Network

Implements the LLaMA-style FFN for single-token decode:
  output = W_down @ SwiGLU(W_gate @ x, W_up @ x)

Architecture (matching IRON):
  4 sequential launches, each using [1, num_cols] herd:
  1. Gate GEMV: each column computes dim_m rows of gate = W_gate @ x
  2. Up GEMV: each column computes dim_m rows of up = W_up @ x
  3. SwiGLU: each column computes dim_m elements of SiLU(gate) * up
  4. Down GEMV: each column reads full intermediate, computes dim_m output rows

Intermediates passed as function arguments (required for multi-launch).
Weight matrices pre-transposed to 32-bit-word layout on host.

Target: AIE2P (NPU2). Requires ELF output format.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


def transpose_32bit_words(A, M, K):
    """Convert A[M,K] row-major to 32-bit-word-transposed layout."""
    if M % 16 != 0:
        raise ValueError(f"M must be divisible by 16; got M={M}.")
    if K % 8 != 0:
        raise ValueError(f"K must be divisible by 8; got K={K}.")
    return A.reshape(M, K // 2, 2).transpose(1, 0, 2).reshape(-1)


@module_builder
def build_module(dim, num_cols, np_dtype):
    xrt_dtype = type_mapper(np_dtype)
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    assert (
        dim % num_cols == 0
    ), f"dim ({dim}) must be divisible by num_cols ({num_cols})"
    assert (
        dim % 16 == 0
    ), f"dim ({dim}) must be divisible by 16 (matvec kernel constraint)"
    assert (
        dim % 8 == 0
    ), f"dim ({dim}) must be divisible by 8 (matvec kernel constraint)"
    dim_m = dim // num_cols
    assert (
        dim_m % 16 == 0
    ), f"dim/num_cols ({dim_m}) must be divisible by 16 (matvec kernel constraint)"
    mat_size = dim * dim

    # L3 types
    l3_x_ty = MemRefType.get([dim], xrt_dtype)
    l3_w_ty = MemRefType.get([3 * mat_size], xrt_dtype)
    l3_out_ty = MemRefType.get([dim], xrt_dtype)
    l3_vec_ty = MemRefType.get([dim], xrt_dtype)

    # L1 types
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_x_ty = MemRefType.get([dim], xrt_dtype, memory_space=l1_space)
    l1_w_part_ty = MemRefType.get([dim_m * dim], xrt_dtype, memory_space=l1_space)
    l1_vec_part_ty = MemRefType.get([dim_m], xrt_dtype, memory_space=l1_space)
    l1_vec_full_ty = MemRefType.get([dim], xrt_dtype, memory_space=l1_space)

    # External kernels
    matvec_func = FuncOp(
        "matvec_vectorized_bf16_bf16",
        ([l1_w_part_ty, l1_x_ty, l1_vec_part_ty], []),
        visibility="private",
    )
    zero_func = FuncOp(
        "zero_vectorized_bf16", ([l1_vec_part_ty], []), visibility="private"
    )
    swiglu_func = FuncOp(
        "swiglu_bf16",
        ([l1_vec_part_ty, l1_vec_part_ty, l1_vec_part_ty, i32], []),
        visibility="private",
    )
    for func in [matvec_func, zero_func, swiglu_func]:
        func.attributes["link_with"] = StringAttr.get("ffn_kernels.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # Intermediates as function arguments (required for multi-launch)
    @FuncOp.from_py_func(l3_x_ty, l3_w_ty, l3_vec_ty, l3_vec_ty, l3_vec_ty, l3_out_ty)
    def ffn_swiglu(arg_x, arg_w, arg_gate, arg_up, arg_inter, arg_out):

        # Helper to build a GEMV launch
        def make_gemv_launch(
            launch_id, seg_name, herd_name, l3_x, l3_w, l3_result, w_base_offset
        ):
            @launch(operands=[l3_x, l3_w, l3_result])
            def gemv_launch(lx, lw, lr):

                @segment(name=seg_name, operands=[lx, lw, lr])
                def seg(sx, sw, sr):

                    @herd(
                        name=herd_name,
                        sizes=[1, num_cols],
                        operands=[sx, sw, sr],
                    )
                    def h(_tx, _ty, _sx, _sy, hx, hw, hr):
                        l1_x = AllocOp(l1_x_ty, [], [])
                        l1_w = AllocOp(l1_w_part_ty, [], [])
                        l1_out = AllocOp(l1_vec_part_ty, [], [])

                        dma_memcpy_nd(l1_x, hx)
                        part_size = ConstantOp(index_type, dim_m * dim)
                        w_base = ConstantOp(index_type, w_base_offset)
                        col_off = arith.muli(_ty, part_size)
                        w_off = arith.addi(w_base, col_off)
                        dma_memcpy_nd(
                            l1_w,
                            hw,
                            src_offsets=[w_off],
                            src_sizes=[dim_m * dim],
                            src_strides=[1],
                        )
                        CallOp(zero_func, [l1_out])
                        CallOp(matvec_func, [l1_w, l1_x, l1_out])
                        out_off = arith.muli(_ty, ConstantOp(index_type, dim_m))
                        dma_memcpy_nd(
                            hr,
                            l1_out,
                            dst_offsets=[out_off],
                            dst_sizes=[dim_m],
                            dst_strides=[1],
                        )
                        DeallocOp(l1_x)
                        DeallocOp(l1_w)
                        DeallocOp(l1_out)

                    h.attributes["link_with"] = StringAttr.get("ffn_kernels.o")

        # Launch 1: gate = W_gate @ x
        make_gemv_launch(1, "gate_seg", "gate_h", arg_x, arg_w, arg_gate, 0)

        # Launch 2: up = W_up @ x
        make_gemv_launch(2, "up_seg", "up_h", arg_x, arg_w, arg_up, mat_size)

        # Launch 3: intermediate = SwiGLU(gate, up)
        @launch(operands=[arg_gate, arg_up, arg_inter])
        def swiglu_launch(lg, lu, li):

            @segment(name="swiglu_seg", operands=[lg, lu, li])
            def seg(sg, su, si):

                @herd(
                    name="swiglu_h",
                    sizes=[1, num_cols],
                    operands=[sg, su, si],
                )
                def h(_tx, _ty, _sx, _sy, hg, hu, hi):
                    l1_g = AllocOp(l1_vec_part_ty, [], [])
                    l1_u = AllocOp(l1_vec_part_ty, [], [])
                    l1_i = AllocOp(l1_vec_part_ty, [], [])

                    part_off = arith.muli(_ty, ConstantOp(index_type, dim_m))
                    dma_memcpy_nd(
                        l1_g,
                        hg,
                        src_offsets=[part_off],
                        src_sizes=[dim_m],
                        src_strides=[1],
                    )
                    dma_memcpy_nd(
                        l1_u,
                        hu,
                        src_offsets=[part_off],
                        src_sizes=[dim_m],
                        src_strides=[1],
                    )
                    dm = ConstantOp(i32, dim_m)
                    CallOp(swiglu_func, [l1_g, l1_u, l1_i, dm])
                    dma_memcpy_nd(
                        hi,
                        l1_i,
                        dst_offsets=[part_off],
                        dst_sizes=[dim_m],
                        dst_strides=[1],
                    )
                    DeallocOp(l1_g)
                    DeallocOp(l1_u)
                    DeallocOp(l1_i)

                h.attributes["link_with"] = StringAttr.get("ffn_kernels.o")

        # Launch 4: out = W_down @ intermediate
        @launch(operands=[arg_inter, arg_w, arg_out])
        def down_launch(li, lw, lo):

            @segment(name="down_seg", operands=[li, lw, lo])
            def seg(si, sw, so):

                @herd(
                    name="down_h",
                    sizes=[1, num_cols],
                    operands=[si, sw, so],
                )
                def h(_tx, _ty, _sx, _sy, hi, hw, ho):
                    l1_inter = AllocOp(l1_vec_full_ty, [], [])
                    l1_w = AllocOp(l1_w_part_ty, [], [])
                    l1_out = AllocOp(l1_vec_part_ty, [], [])

                    # Full intermediate (broadcast)
                    dma_memcpy_nd(l1_inter, hi)
                    # W_down partition
                    two_mat = ConstantOp(index_type, 2 * mat_size)
                    part_size = ConstantOp(index_type, dim_m * dim)
                    col_off = arith.muli(_ty, part_size)
                    w_off = arith.addi(two_mat, col_off)
                    dma_memcpy_nd(
                        l1_w,
                        hw,
                        src_offsets=[w_off],
                        src_sizes=[dim_m * dim],
                        src_strides=[1],
                    )
                    CallOp(zero_func, [l1_out])
                    CallOp(matvec_func, [l1_w, l1_inter, l1_out])
                    out_off = arith.muli(_ty, ConstantOp(index_type, dim_m))
                    dma_memcpy_nd(
                        ho,
                        l1_out,
                        dst_offsets=[out_off],
                        dst_sizes=[dim_m],
                        dst_strides=[1],
                    )
                    DeallocOp(l1_inter)
                    DeallocOp(l1_w)
                    DeallocOp(l1_out)

                h.attributes["link_with"] = StringAttr.get("ffn_kernels.o")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="FFN SwiGLU decode — multi-column",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--dim", type=int, default=128, help="Dimension")
    parser.add_argument("--num-cols", type=int, default=4, help="AIE columns")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
        dest="output_format",
    )
    args = parser.parse_args()

    dim = args.dim
    num_cols = args.num_cols
    dim_m = dim // num_cols
    INPUT_DATATYPE = bfloat16

    mlir_module = build_module(dim, num_cols, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    x = (np.random.randn(dim) * 0.1).astype(INPUT_DATATYPE)
    W_gate = (np.random.randn(dim, dim) * 0.1).astype(INPUT_DATATYPE)
    W_up = (np.random.randn(dim, dim) * 0.1).astype(INPUT_DATATYPE)
    W_down = (np.random.randn(dim, dim) * 0.1).astype(INPUT_DATATYPE)

    def pack_weights_partitioned(W, dim, dim_m, num_cols):
        parts = []
        for col in range(num_cols):
            W_part = W[col * dim_m : (col + 1) * dim_m, :]
            parts.append(transpose_32bit_words(W_part, dim_m, dim))
        return np.concatenate(parts)

    W_gate_packed = pack_weights_partitioned(W_gate, dim, dim_m, num_cols)
    W_up_packed = pack_weights_partitioned(W_up, dim, dim_m, num_cols)
    W_down_packed = pack_weights_partitioned(W_down, dim, dim_m, num_cols)
    packed_weights = np.concatenate([W_gate_packed, W_up_packed, W_down_packed]).astype(
        INPUT_DATATYPE
    )

    # Intermediate buffers (function arguments, not memref.alloc)
    gate_buf = np.zeros(dim, dtype=INPUT_DATATYPE)
    up_buf = np.zeros(dim, dtype=INPUT_DATATYPE)
    inter_buf = np.zeros(dim, dtype=INPUT_DATATYPE)

    # Reference (f32)
    x_f32 = x.astype(np.float32)
    gate = W_gate.astype(np.float32) @ x_f32
    up = W_up.astype(np.float32) @ x_f32
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
    silu_gate = gate * sigmoid_gate
    intermediate = silu_gate * up
    ref_out = (W_down.astype(np.float32) @ intermediate).astype(INPUT_DATATYPE)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="ffn_swiglu",
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[x, packed_weights, gate_buf, up_buf, inter_buf],
            expected_outputs=[ref_out],
            rtol=1e0,
            atol=0.5,
        )
    )
