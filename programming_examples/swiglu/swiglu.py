# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized SwiGLU (Swish-Gated Linear Unit) Example

Implements element-wise SwiGLU on 1D inputs [N]:
  SwiGLU(x, gate, up) = SiLU(x * gate) * (x * up)

where SiLU(z) = z * sigmoid(z) = z * 0.5 * (tanh(z/2) + 1).

Uses the tanh-based sigmoid identity to avoid exp and division, which
have precision and correctness issues on AIE2P. The hardware tanh
intrinsic (__builtin_aie2p_tanh) is used directly.

The gate and up weights are packed into a single rank-2 buffer
[2, N] to reduce the number of DMA channels needed (AIE2P tiles
have only 2 S2MM channels). The L1 buffer is a flat [2*tile_n]
to allow simple 1D subview/transfer_read operations.

Uses a single AIE tile with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write, BroadcastOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in, vector_size=16):
    xrt_dtype_in = type_mapper(np_dtype_in)
    assert n % tile_n == 0
    assert tile_n % vector_size == 0
    VECTOR_SIZE = vector_size
    index_type = IndexType.get()

    l3memrefTy = MemRefType.get([n], xrt_dtype_in)
    # gate and up packed as [2, N]: row 0 = gate, row 1 = up
    l3GateUpTy = MemRefType.get([2, n], xrt_dtype_in)

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTy = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    # L1 buffer for gate+up tile: flat [2*tile_n] for simple 1D indexing
    l1GateUpTy = MemRefType.get(
        shape=[2 * tile_n],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )

    vecTy = VectorType.get([VECTOR_SIZE], xrt_dtype_in)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3memrefTy, l3GateUpTy, l3memrefTy)
    def swiglu(arg0, arg1, arg2):
        # arg0 = x [N], arg1 = gate_up [2, N], arg2 = output [N]

        @herd(
            name="herd_0",
            sizes=[1, 1],
            operands=[arg0, arg1, arg2],
        )
        def herd_body(
            _tx,
            _ty,
            _sx,
            _sy,
            _l3_x,
            _l3_gate_up,
            _l3_out,
        ):
            l1_x = AllocOp(l1MemrefTy, [], [])
            l1_gate_up = AllocOp(l1GateUpTy, [], [])
            l1_out = AllocOp(l1MemrefTy, [], [])

            c0 = ConstantOp(index_type, 0)

            for _l_ivx in range_(0, n, tile_n):
                # DMA: load x tile
                dma_memcpy_nd(
                    l1_x,
                    _l3_x,
                    src_offsets=[_l_ivx],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                # DMA: load gate and up tiles from [2, N] L3 buffer
                # into flat [2*tile_n] L1 buffer
                dma_memcpy_nd(
                    l1_gate_up,
                    _l3_gate_up,
                    src_offsets=[0, _l_ivx],
                    src_sizes=[2, tile_n],
                    src_strides=[n, 1],
                )

                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)
                cTileNIdx = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)
                half_const = arith.ConstantOp(xrt_dtype_in, 0.5)
                one_const = arith.ConstantOp(xrt_dtype_in, 1.0)
                v_half = BroadcastOp(vecTy, half_const)
                v_one = BroadcastOp(vecTy, one_const)

                for j in range_(c0, cTileN, cVecSize):
                    sub_x = subview(l1_x.result, [j], [VECTOR_SIZE], [1])
                    # gate is at [0..tile_n-1], up is at [tile_n..2*tile_n-1]
                    sub_gate = subview(l1_gate_up.result, [j], [VECTOR_SIZE], [1])
                    up_offset = arith.addi(j, cTileNIdx)
                    sub_up = subview(l1_gate_up.result, [up_offset], [VECTOR_SIZE], [1])
                    sub_out = subview(l1_out.result, [j], [VECTOR_SIZE], [1])

                    v_x = transfer_read(vecTy, sub_x, [c0], identity_map, cst0, [True])
                    v_gate = transfer_read(
                        vecTy, sub_gate, [c0], identity_map, cst0, [True]
                    )
                    v_up = transfer_read(
                        vecTy, sub_up, [c0], identity_map, cst0, [True]
                    )

                    # SwiGLU(x, gate, up) = SiLU(x * gate) * (x * up)
                    # SiLU(z) = z * 0.5 * (tanh(z/2) + 1)

                    # Compute x * gate
                    v_xg = arith.mulf(v_x, v_gate)

                    # SiLU(x * gate)
                    v_half_xg = arith.mulf(v_xg, v_half.result)
                    v_tanh = math_dialect.tanh(v_half_xg)
                    v_tanh_plus_one = arith.addf(v_tanh, v_one.result)
                    v_sigmoid = arith.mulf(v_tanh_plus_one, v_half.result)
                    v_silu_xg = arith.mulf(v_xg, v_sigmoid)

                    # Compute x * up
                    v_xu = arith.mulf(v_x, v_up)

                    # SwiGLU = SiLU(x * gate) * (x * up)
                    v_result = arith.mulf(v_silu_xg, v_xu)

                    transfer_write(None, v_result, sub_out, [c0], identity_map, [True])
                    yield_([])

                dma_memcpy_nd(
                    _l3_out,
                    l1_out,
                    dst_offsets=[_l_ivx],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_x)
                DeallocOp(l1_gate_up)
                DeallocOp(l1_out)
                yield_([])


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the SwiGLU example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size", type=int, default=16, help="Vector size for SIMD operations"
    )
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

    mlir_module = build_module(args.n, args.tile_n, INPUT_DATATYPE, args.vector_size)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    input_x = np.random.uniform(-2.0, 2.0, args.n).astype(INPUT_DATATYPE)
    input_gate = np.random.uniform(-2.0, 2.0, args.n).astype(INPUT_DATATYPE)
    input_up = np.random.uniform(-2.0, 2.0, args.n).astype(INPUT_DATATYPE)

    # Pack gate and up into [2, N]: row 0 = gate, row 1 = up
    input_gate_up = np.stack([input_gate, input_up]).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])

        # SwiGLU reference using tanh-based sigmoid (matches hardware computation)
        def swiglu_ref(x, gate, up):
            x_f32 = x.astype(np.float32)
            g_f32 = gate.astype(np.float32)
            u_f32 = up.astype(np.float32)
            xg = x_f32 * g_f32
            silu_xg = xg * 0.5 * (np.tanh(xg / 2.0) + 1.0)
            return silu_xg * (x_f32 * u_f32)

        sampled_values = np.array(
            [
                swiglu_ref(input_x[i], input_gate[i], input_up[i])
                for i in zip(*sampled_indices)
            ],
            dtype=INPUT_DATATYPE,
        )
        sampled_data = {
            "shape": (args.n,),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="swiglu",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_x, input_gate_up],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-1,
                atol=5e-2,
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
