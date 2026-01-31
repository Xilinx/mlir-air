# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
from math import cos, sin, sqrt, exp

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from ml_dtypes import bfloat16

range_ = for_


@module_builder
def build_module(n, tile_n, herd_n, np_dtype_in):
    assert n % (tile_n * herd_n) == 0
    a_size = [n]
    out_size = a_size
    xrt_dtype_in = type_mapper(np_dtype_in)

    # L3 MemRefTypes
    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)

    # L1 MemRefTypes
    l1MemrefTy = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    # Function declaration
    softmax_func = FuncOp(
        "softmax_bf16",
        ([l1MemrefTy, T.i32(), l1MemrefTy], []),
        visibility="private",
    )
    for func in [softmax_func]:
        func.attributes["link_with"] = StringAttr.get("softmax.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def softmax(arg0, arg1):

        @herd(
            name="herd_0",
            sizes=[1, herd_n],
            operands=[arg0, arg1],
        )
        def herd_body(
            _tx,
            _ty,
            _sx,
            _sy,
            _l3_a,
            _l3_c,
        ):
            l1_a_data = AllocOp(l1MemrefTy, [], [])
            l1_out_data = AllocOp(l1MemrefTy, [], [])

            for t in range_(0, n, tile_n * herd_n):

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
                offset = affine_apply(offset_map, [t, _ty])

                dma_memcpy_nd(
                    l1_a_data,
                    _l3_a,
                    src_offsets=[
                        offset,
                    ],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                const_pos = ConstantOp(IntegerAttr.get(T.i32(), tile_n - 1), None)
                softmax_call = CallOp(
                    softmax_func,
                    [l1_a_data, const_pos, l1_out_data],
                )
                dma_memcpy_nd(
                    _l3_c,
                    l1_out_data,
                    dst_offsets=[
                        offset,
                    ],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_a_data)
                DeallocOp(l1_out_data)

                yield_([])

        herd_body.attributes["link_with"] = StringAttr.get("softmax.o")


if __name__ == "__main__":
    # Default values.
    N = 1024
    TILE_N = 256
    HERD_N = 4
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_dma example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Total number of elements",
    )
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--herd-n",
        type=int,
        default=HERD_N,
        help="Number of L1 tiles along the N dimension",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure to whether to run after compile",
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

    mlir_module = build_module(
        args.n,
        args.tile_n,
        args.herd_n,
        INPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Softmax
    num_tiles = args.n // args.tile_n
    inputs = np.random.randn(num_tiles, args.tile_n).astype(INPUT_DATATYPE)
    outputs = np.zeros(shape=(num_tiles, args.tile_n), dtype=INPUT_DATATYPE)

    max_val = np.max(inputs)
    for j in range(num_tiles):
        sum_val = 0.0
        for i in range(args.tile_n):
            outputs[j][i] = exp(inputs[j][i] - max_val)
            sum_val += outputs[j][i]
        for i in range(args.tile_n):
            outputs[j][i] = outputs[j][i] / sum_val

    if args.compile_mode == "compile-and-run":

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[inputs],
                expected_outputs=[outputs],
                rtol=1e-1,
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
