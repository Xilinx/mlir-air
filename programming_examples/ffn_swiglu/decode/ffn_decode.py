# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FFN SwiGLU Decode — Feed-Forward Network with SwiGLU activation

Implements the LLaMA-style FFN for single-token decode:
  output = W_down @ SwiGLU(W_gate @ x, W_up @ x)

where SwiGLU(gate, up) = SiLU(gate) * up.

All three weight matrices are packed into one buffer [3*dim*dim] to
reduce DMA channels. Uses a single AIE tile with external kernels.

Weight matrices must be pre-transposed to 32-bit-word layout on the host.

Target: AIE2P (NPU2). Requires dim % 16 == 0 and dim % 8 == 0.
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
    return A.reshape(M, K // 2, 2).transpose(1, 0, 2).reshape(-1)


@module_builder
def build_module(dim, np_dtype):
    xrt_dtype = type_mapper(np_dtype)
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    mat_size = dim * dim

    # L3: x[dim], packed_weights[3*mat_size], out[dim]
    l3_x_ty = MemRefType.get([dim], xrt_dtype)
    l3_w_ty = MemRefType.get([3 * mat_size], xrt_dtype)
    l3_out_ty = MemRefType.get([dim], xrt_dtype)

    # L1
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_vec_ty = MemRefType.get([dim], xrt_dtype, memory_space=l1_space)
    l1_mat_ty = MemRefType.get([mat_size], xrt_dtype, memory_space=l1_space)

    # External kernels
    matvec_func = FuncOp(
        "matvec_vectorized_bf16_bf16",
        ([l1_mat_ty, l1_vec_ty, l1_vec_ty], []),
        visibility="private",
    )
    zero_func = FuncOp("zero_vectorized_bf16", ([l1_vec_ty], []), visibility="private")
    swiglu_func = FuncOp(
        "swiglu_bf16",
        ([l1_vec_ty, l1_vec_ty, l1_vec_ty, i32], []),
        visibility="private",
    )
    for func in [matvec_func, zero_func, swiglu_func]:
        func.attributes["link_with"] = StringAttr.get("ffn_kernels.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_x_ty, l3_w_ty, l3_out_ty)
    def ffn_swiglu(arg_x, arg_w, arg_out):

        @herd(
            name="ffn_herd",
            sizes=[1, 1],
            operands=[arg_x, arg_w, arg_out],
        )
        def herd_body(_tx, _ty, _sx, _sy, l3_x, l3_w, l3_out):
            l1_x = AllocOp(l1_vec_ty, [], [])
            l1_w = AllocOp(l1_mat_ty, [], [])
            l1_gate = AllocOp(l1_vec_ty, [], [])
            l1_up = AllocOp(l1_vec_ty, [], [])
            l1_intermediate = AllocOp(l1_vec_ty, [], [])
            l1_out = AllocOp(l1_vec_ty, [], [])

            # DMA x
            dma_memcpy_nd(l1_x, l3_x)

            # Step 1: gate = W_gate @ x (W_gate at offset 0)
            dma_memcpy_nd(
                l1_w,
                l3_w,
                src_offsets=[0],
                src_sizes=[mat_size],
                src_strides=[1],
            )
            CallOp(zero_func, [l1_gate])
            CallOp(matvec_func, [l1_w, l1_x, l1_gate])

            # Step 2: up = W_up @ x (W_up at offset mat_size)
            mat_size_idx = ConstantOp(index_type, mat_size)
            dma_memcpy_nd(
                l1_w,
                l3_w,
                src_offsets=[mat_size_idx],
                src_sizes=[mat_size],
                src_strides=[1],
            )
            CallOp(zero_func, [l1_up])
            CallOp(matvec_func, [l1_w, l1_x, l1_up])

            # Step 3: intermediate = SwiGLU(gate, up)
            dim_const = ConstantOp(i32, dim)
            CallOp(swiglu_func, [l1_gate, l1_up, l1_intermediate, dim_const])

            # Step 4: out = W_down @ intermediate (W_down at offset 2*mat_size)
            two_mat_idx = ConstantOp(index_type, 2 * mat_size)
            dma_memcpy_nd(
                l1_w,
                l3_w,
                src_offsets=[two_mat_idx],
                src_sizes=[mat_size],
                src_strides=[1],
            )
            CallOp(zero_func, [l1_out])
            CallOp(matvec_func, [l1_w, l1_intermediate, l1_out])

            # DMA output
            dma_memcpy_nd(l3_out, l1_out)

            DeallocOp(l1_x)
            DeallocOp(l1_w)
            DeallocOp(l1_gate)
            DeallocOp(l1_up)
            DeallocOp(l1_intermediate)
            DeallocOp(l1_out)

        herd_body.attributes["link_with"] = StringAttr.get("ffn_kernels.o")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="FFN SwiGLU decode (weights in 32-bit-word-transposed layout)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--dim", type=int, default=32, help="Embedding/hidden dimension"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    dim = args.dim
    INPUT_DATATYPE = bfloat16

    mlir_module = build_module(dim, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    x = (np.random.randn(dim) * 0.1).astype(INPUT_DATATYPE)
    W_gate = (np.random.randn(dim, dim) * 0.1).astype(INPUT_DATATYPE)
    W_up = (np.random.randn(dim, dim) * 0.1).astype(INPUT_DATATYPE)
    W_down = (np.random.randn(dim, dim) * 0.1).astype(INPUT_DATATYPE)

    # Pack weights: [W_gate_t, W_up_t, W_down_t]
    W_gate_t = transpose_32bit_words(W_gate, dim, dim)
    W_up_t = transpose_32bit_words(W_up, dim, dim)
    W_down_t = transpose_32bit_words(W_down, dim, dim)
    packed_weights = np.concatenate([W_gate_t, W_up_t, W_down_t]).astype(INPUT_DATATYPE)

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
            inputs=[x, packed_weights],
            expected_outputs=[ref_out],
            rtol=1e0,
            atol=0.5,
        )
    )
