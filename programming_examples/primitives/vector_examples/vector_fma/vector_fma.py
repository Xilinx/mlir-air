# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write, BroadcastOp, fma
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in, alpha=2.0, vector_size=16):
    xrt_dtype_in = type_mapper(np_dtype_in)
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0
    VECTOR_SIZE = vector_size
    index_type = IndexType.get()

    # L3 MemRefTypes
    l3memrefTy = MemRefType.get([n], xrt_dtype_in)

    # L1 MemRefTypes
    l1MemrefTy = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    vecTy = VectorType.get([VECTOR_SIZE], xrt_dtype_in)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy, l3memrefTy)
    def vector_fma(arg0, arg1, arg2):
        # arg0 = b, arg1 = c, arg2 = output
        # Computes: output = alpha * b + c (via vector.fma)

        @herd(
            name="herd_0",
            sizes=[1, num_tiles],
            operands=[arg0, arg1, arg2],
        )
        def herd_body(
            _tx,
            _ty,
            _sx,
            _sy,
            _l3_b,
            _l3_c,
            _l3_out,
        ):
            l1_b_data = AllocOp(l1MemrefTy, [], [])
            l1_c_data = AllocOp(l1MemrefTy, [], [])
            l1_out_data = AllocOp(l1MemrefTy, [], [])

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
                    l1_b_data,
                    _l3_b,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                dma_memcpy_nd(
                    l1_c_data,
                    _l3_c,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )

                c0 = ConstantOp(index_type, 0)
                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)

                # Broadcast scalar alpha to vector
                a_const = arith.ConstantOp(xrt_dtype_in, alpha)
                v_a = BroadcastOp(vecTy, a_const)

                for j in range_(c0, cTileN, cVecSize):
                    sub_b = subview(l1_b_data.result, [j], [VECTOR_SIZE], [1])
                    sub_c = subview(l1_c_data.result, [j], [VECTOR_SIZE], [1])
                    sub_out = subview(l1_out_data.result, [j], [VECTOR_SIZE], [1])

                    v_b = transfer_read(vecTy, sub_b, [c0], identity_map, cst0, [True])
                    v_c = transfer_read(vecTy, sub_c, [c0], identity_map, cst0, [True])

                    # alpha * b + c via vector.fma
                    v_result = fma(v_a, v_b, v_c)
                    transfer_write(None, v_result, sub_out, [c0], identity_map, [True])
                    yield_([])

                dma_memcpy_nd(
                    _l3_out,
                    l1_out_data,
                    dst_offsets=[offset],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_b_data)
                DeallocOp(l1_c_data)
                DeallocOp(l1_out_data)

                yield_([])


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    VECTOR_SIZE = 16
    INPUT_DATATYPE = bfloat16
    ALPHA = 2.0

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vector_fma example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--alpha", type=float, default=ALPHA, help="Scalar multiplier a"
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector size for SIMD operations",
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

    mlir_module = build_module(
        args.n, args.tile_n, INPUT_DATATYPE, args.alpha, args.vector_size
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_b = np.random.uniform(-10.0, 10.0, args.n).astype(INPUT_DATATYPE)
    input_c = np.random.uniform(-10.0, 10.0, args.n).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])
        sampled_values = np.array(
            [args.alpha * input_b[i] + input_c[i] for i in zip(*sampled_indices)],
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
            instance_name="vector_fma",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_b, input_c],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-2,
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
