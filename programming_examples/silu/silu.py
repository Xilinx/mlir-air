# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized SiLU (Sigmoid Linear Unit) Example

Implements element-wise SiLU on a 1D input [N]:
  SiLU(x) = x * sigmoid(x) = x * 0.5 * (tanh(x/2) + 1)

Uses the tanh-based sigmoid identity to avoid exp and division, which
have precision and correctness issues on AIE2P. The hardware tanh
intrinsic (__builtin_aie2p_tanh) is used directly.

Uses a 1x2 AIE herd with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
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
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0
    assert tile_n % vector_size == 0
    VECTOR_SIZE = vector_size
    index_type = IndexType.get()

    l3memrefTy = MemRefType.get([n], xrt_dtype_in)
    l1MemrefTy = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    vecTy = VectorType.get([VECTOR_SIZE], xrt_dtype_in)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def silu(arg0, arg1):

        @herd(name="herd_0", sizes=[1, num_tiles], operands=[arg0, arg1])
        def herd_body(_tx, _ty, _sx, _sy, _l3_in, _l3_out):
            l1_in = AllocOp(l1MemrefTy, [], [])
            l1_out = AllocOp(l1MemrefTy, [], [])

            for _l_ivx in range_(0, n, tile_n * num_tiles):
                offset_map = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(tile_n),
                            ),
                        )
                    ],
                )
                offset = affine_apply(offset_map, [_l_ivx, _ty])

                dma_memcpy_nd(
                    l1_in,
                    _l3_in,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )

                c0 = ConstantOp(index_type, 0)
                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)
                half_const = arith.ConstantOp(xrt_dtype_in, 0.5)
                one_const = arith.ConstantOp(xrt_dtype_in, 1.0)
                v_half = BroadcastOp(vecTy, half_const)
                v_one = BroadcastOp(vecTy, one_const)

                for j in range_(c0, cTileN, cVecSize):
                    sub_in = subview(l1_in.result, [j], [VECTOR_SIZE], [1])
                    sub_out = subview(l1_out.result, [j], [VECTOR_SIZE], [1])

                    v_x = transfer_read(vecTy, sub_in, [c0], identity_map, cst0, [True])

                    # SiLU(x) = x * sigmoid(x)
                    #         = x * 0.5 * (tanh(x/2) + 1)
                    # Uses hardware tanh intrinsic — no exp or division needed.
                    v_half_x = arith.mulf(v_x, v_half.result)
                    v_tanh = math_dialect.tanh(v_half_x)
                    v_tanh_plus_one = arith.addf(v_tanh, v_one.result)
                    v_sigmoid = arith.mulf(v_tanh_plus_one, v_half.result)
                    v_silu = arith.mulf(v_x, v_sigmoid)

                    transfer_write(None, v_silu, sub_out, [c0], identity_map, [True])
                    yield_([])

                dma_memcpy_nd(
                    _l3_out,
                    l1_out,
                    dst_offsets=[offset],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_in)
                DeallocOp(l1_out)
                yield_([])


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the SiLU example",
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
    input_a = np.random.uniform(-4.0, 4.0, args.n).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])

        # SiLU reference using tanh-based sigmoid (matches hardware computation)
        def silu_ref(x):
            x_f32 = x.astype(np.float32)
            return x_f32 * 0.5 * (np.tanh(x_f32 / 2.0) + 1.0)

        sampled_values = np.array(
            [silu_ref(input_a[i]) for i in zip(*sampled_indices)],
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
            instance_name="silu",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
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
