# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FFN SwiGLU Prefill — Multi-Column Feed-Forward Network (GEMM-based)

Implements the LLaMA-style FFN for multi-token prefill:
  output = SwiGLU(x @ W_gate^T, x @ W_up^T) @ W_down^T

Architecture (matching IRON):
  4 sequential launches, each using [1, num_cols] herd:
  1. Gate GEMM: each column computes dim_n rows of gate = x @ W_gate^T
  2. Up GEMM: each column computes dim_n rows of up = x @ W_up^T
  3. SwiGLU: each column computes seq_len * dim_n elements of SiLU(gate) * up
  4. Down GEMM: each column reads full intermediate, computes dim_n output rows

Intermediates passed as function arguments (required for multi-launch).
Weight matrices in row-major layout (transposed on host for x @ W^T).

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


@module_builder
def build_module(seq_len, dim, num_cols, np_dtype):
    xrt_dtype = type_mapper(np_dtype)
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    assert dim % num_cols == 0
    assert dim % 8 == 0
    dim_n = dim // num_cols  # Output rows per column
    assert dim_n % 8 == 0
    assert seq_len % 8 == 0

    # Weight partition size: each column handles dim_n output rows
    # W_part[dim_n, dim] stored as flat [dim_n * dim]
    w_part_size = dim_n * dim
    mat_size = dim * dim  # Full weight matrix size

    # L3 types (flat 1D)
    l3_x_ty = MemRefType.get([seq_len * dim], xrt_dtype)
    l3_w_ty = MemRefType.get([3 * mat_size], xrt_dtype)  # gate + up + down
    l3_out_ty = MemRefType.get([seq_len * dim], xrt_dtype)
    l3_vec_ty = MemRefType.get([seq_len * dim], xrt_dtype)

    # L1 types
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_x_ty = MemRefType.get([seq_len * dim], xrt_dtype, memory_space=l1_space)
    l1_w_part_ty = MemRefType.get([w_part_size], xrt_dtype, memory_space=l1_space)
    l1_out_part_ty = MemRefType.get([seq_len * dim_n], xrt_dtype, memory_space=l1_space)
    l1_vec_full_ty = MemRefType.get([seq_len * dim], xrt_dtype, memory_space=l1_space)

    # External kernels
    matmul_func = FuncOp(
        "matmul_bf16",
        ([l1_x_ty, l1_w_part_ty, l1_out_part_ty], []),
        visibility="private",
    )
    zero_func = FuncOp(
        "zero_vectorized_bf16",
        ([l1_out_part_ty], []),
        visibility="private",
    )
    swiglu_func = FuncOp(
        "swiglu_bf16",
        ([l1_out_part_ty, l1_out_part_ty, l1_out_part_ty, i32], []),
        visibility="private",
    )
    for func in [matmul_func, zero_func, swiglu_func]:
        func.attributes["link_with"] = StringAttr.get("ffn_kernels.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # Intermediates as function arguments (required for multi-launch)
    @FuncOp.from_py_func(l3_x_ty, l3_w_ty, l3_vec_ty, l3_vec_ty, l3_vec_ty, l3_out_ty)
    def ffn_swiglu(arg_x, arg_w, arg_gate, arg_up, arg_inter, arg_out):

        def make_gemm_launch(seg_name, herd_name, l3_x, l3_w, l3_result, w_base_offset):
            """Build a GEMM launch: result = x @ W_part^T for each column."""

            @launch(operands=[l3_x, l3_w, l3_result])
            def gemm_launch(lx, lw, lr):

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
                        l1_out = AllocOp(l1_out_part_ty, [], [])

                        # DMA x (full input, broadcast to all columns)
                        dma_memcpy_nd(l1_x, hx)

                        # DMA weight partition for this column
                        part_size = ConstantOp(index_type, w_part_size)
                        w_base = ConstantOp(index_type, w_base_offset)
                        col_off = arith.muli(_ty, part_size)
                        w_off = arith.addi(w_base, col_off)
                        dma_memcpy_nd(
                            l1_w,
                            hw,
                            src_offsets=[w_off],
                            src_sizes=[w_part_size],
                            src_strides=[1],
                        )

                        # Zero output, compute matmul
                        CallOp(zero_func, [l1_out])
                        CallOp(matmul_func, [l1_x, l1_w, l1_out])

                        # DMA output back (column-partitioned)
                        # Output layout: [seq_len, dim] where this column writes
                        # columns [col*dim_n : (col+1)*dim_n] of each row.
                        # Since output is flat [seq_len * dim], we need strided write:
                        # for each of seq_len rows, write dim_n elements at offset col*dim_n
                        out_col_off = arith.muli(_ty, ConstantOp(index_type, dim_n))
                        dma_memcpy_nd(
                            hr,
                            l1_out,
                            dst_offsets=[out_col_off],
                            dst_sizes=[seq_len, dim_n],
                            dst_strides=[dim, 1],
                        )

                        DeallocOp(l1_x)
                        DeallocOp(l1_w)
                        DeallocOp(l1_out)

                    h.attributes["link_with"] = StringAttr.get("ffn_kernels.o")

        # Launch 1: gate = x @ W_gate^T
        make_gemm_launch("gate_seg", "gate_h", arg_x, arg_w, arg_gate, 0)

        # Launch 2: up = x @ W_up^T
        make_gemm_launch("up_seg", "up_h", arg_x, arg_w, arg_up, mat_size)

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
                    l1_g = AllocOp(l1_out_part_ty, [], [])
                    l1_u = AllocOp(l1_out_part_ty, [], [])
                    l1_i = AllocOp(l1_out_part_ty, [], [])

                    part_off = arith.muli(_ty, ConstantOp(index_type, dim_n))
                    # Read seq_len rows of dim_n elements each from gate/up
                    dma_memcpy_nd(
                        l1_g,
                        hg,
                        src_offsets=[part_off],
                        src_sizes=[seq_len, dim_n],
                        src_strides=[dim, 1],
                    )
                    dma_memcpy_nd(
                        l1_u,
                        hu,
                        src_offsets=[part_off],
                        src_sizes=[seq_len, dim_n],
                        src_strides=[dim, 1],
                    )
                    dm = ConstantOp(i32, seq_len * dim_n)
                    CallOp(swiglu_func, [l1_g, l1_u, l1_i, dm])
                    dma_memcpy_nd(
                        hi,
                        l1_i,
                        dst_offsets=[part_off],
                        dst_sizes=[seq_len, dim_n],
                        dst_strides=[dim, 1],
                    )
                    DeallocOp(l1_g)
                    DeallocOp(l1_u)
                    DeallocOp(l1_i)

                h.attributes["link_with"] = StringAttr.get("ffn_kernels.o")

        # Launch 4: out = inter @ W_down^T
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
                    l1_out = AllocOp(l1_out_part_ty, [], [])

                    # Full intermediate (broadcast)
                    dma_memcpy_nd(l1_inter, hi)
                    # W_down partition
                    two_mat = ConstantOp(index_type, 2 * mat_size)
                    part_size = ConstantOp(index_type, w_part_size)
                    col_off = arith.muli(_ty, part_size)
                    w_off = arith.addi(two_mat, col_off)
                    dma_memcpy_nd(
                        l1_w,
                        hw,
                        src_offsets=[w_off],
                        src_sizes=[w_part_size],
                        src_strides=[1],
                    )
                    CallOp(zero_func, [l1_out])
                    CallOp(matmul_func, [l1_inter, l1_w, l1_out])
                    out_col_off = arith.muli(_ty, ConstantOp(index_type, dim_n))
                    dma_memcpy_nd(
                        ho,
                        l1_out,
                        dst_offsets=[out_col_off, 0],
                        dst_sizes=[seq_len, dim_n],
                        dst_strides=[dim, 1],
                    )
                    DeallocOp(l1_inter)
                    DeallocOp(l1_w)
                    DeallocOp(l1_out)

                h.attributes["link_with"] = StringAttr.get("ffn_kernels.o")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="FFN SwiGLU prefill — multi-column GEMM-based",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--dim", type=int, default=128, help="Model dimension")
    parser.add_argument("--num-cols", type=int, default=4, help="AIE columns")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
        dest="output_format",
    )
    args = parser.parse_args()

    seq_len = args.seq_len
    dim = args.dim
    num_cols = args.num_cols
    dim_n = dim // num_cols
    INPUT_DATATYPE = bfloat16

    mlir_module = build_module(seq_len, dim, num_cols, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    x = (np.random.randn(seq_len, dim) * 0.1).astype(INPUT_DATATYPE)
    W_gate = (np.random.randn(dim, dim) * 0.1).astype(INPUT_DATATYPE)
    W_up = (np.random.randn(dim, dim) * 0.1).astype(INPUT_DATATYPE)
    W_down = (np.random.randn(dim, dim) * 0.1).astype(INPUT_DATATYPE)

    # Pack weights: for GEMM C = A @ B where B = W^T[K, N].
    # Kernel indexes B as B[kk * N + j], i.e., B is [K, N] row-major.
    # B = W^T has shape [dim, dim]. Partition column c gets N=dim_n columns:
    #   B_part = W^T[:, c*dim_n : (c+1)*dim_n] = W[c*dim_n:(c+1)*dim_n, :]^T
    # Stored as [dim, dim_n] row-major = W[:, c*dim_n:(c+1)*dim_n].
    def pack_weights(W, dim, dim_n, num_cols):
        parts = []
        for col in range(num_cols):
            # W columns [col*dim_n : (col+1)*dim_n] → [dim, dim_n] row-major
            W_part = W[:, col * dim_n : (col + 1) * dim_n]
            parts.append(W_part.reshape(-1))
        return np.concatenate(parts)

    W_gate_packed = pack_weights(W_gate, dim, dim_n, num_cols)
    W_up_packed = pack_weights(W_up, dim, dim_n, num_cols)
    W_down_packed = pack_weights(W_down, dim, dim_n, num_cols)
    packed_weights = np.concatenate([W_gate_packed, W_up_packed, W_down_packed]).astype(
        INPUT_DATATYPE
    )

    # Intermediate buffers (function arguments)
    gate_buf = np.zeros(seq_len * dim, dtype=INPUT_DATATYPE)
    up_buf = np.zeros(seq_len * dim, dtype=INPUT_DATATYPE)
    inter_buf = np.zeros(seq_len * dim, dtype=INPUT_DATATYPE)

    # Reference (f32)
    x_f32 = x.astype(np.float32)
    gate = x_f32 @ W_gate.astype(np.float32).T  # [seq_len, dim]
    up = x_f32 @ W_up.astype(np.float32).T
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
    silu_gate = gate * sigmoid_gate
    intermediate = silu_gate * up
    ref_out = (intermediate @ W_down.astype(np.float32).T).astype(INPUT_DATATYPE)

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
            inputs=[
                x.reshape(-1),
                packed_weights,
                gate_buf,
                up_buf,
                inter_buf,
            ],
            expected_outputs=[ref_out.reshape(-1)],
            rtol=1e0,
            atol=0.5,
        )
    )
