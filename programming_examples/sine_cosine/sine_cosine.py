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
def build_module(n, tile_n, herd_n, sin_or_cos, np_dtype_in):
    assert n % (tile_n * herd_n) == 0
    if sin_or_cos == "sin":
        isSine = True
        isCosine = False
    elif sin_or_cos == "cos":
        isSine = False
        isCosine = True
    else:
        raise AssertionError
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
    sinf_func = FuncOp(
        "sinf_bf16_24_8",
        ([l1MemrefTy, l1MemrefTy], []),
        visibility="private",
    )
    cosf_func = FuncOp(
        "cosf_bf16_24_8",
        ([l1MemrefTy, l1MemrefTy], []),
        visibility="private",
    )
    for func in [sinf_func, cosf_func]:
        func.attributes["link_with"] = StringAttr.get("sine_cosine.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def sine_cosine(arg0, arg1):

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

                if isSine:
                    sinf_call = CallOp(
                        sinf_func,
                        [l1_a_data, l1_out_data],
                    )
                elif isCosine:
                    cosf_call = CallOp(
                        cosf_func,
                        [l1_a_data, l1_out_data],
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

        herd_body.attributes["link_with"] = StringAttr.get("sine_cosine.o")


if __name__ == "__main__":
    # Default values.
    N = 48
    TILE_N = 24
    HERD_N = 2
    SIN_OR_COS = "sin"
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
        "--mode",
        type=str,
        default=SIN_OR_COS,
        choices=["sin", "cos"],
        help="Sine or cosine mode (must be one of [sin, cos])",
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
        args.mode,
        INPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    inputs = np.random.randn(
        args.n,
    ).astype(INPUT_DATATYPE)
    outputs = np.zeros(shape=(args.n), dtype=INPUT_DATATYPE)
    for n1 in range(args.n):
        if args.mode == "sin":
            outputs[n1] = INPUT_DATATYPE(sin(inputs[n1]))
        elif args.mode == "cos":
            outputs[n1] = INPUT_DATATYPE(cos(inputs[n1]))
        else:
            raise AssertionError

    if args.compile_mode == "compile-and-run":

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="sine_cosine",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[inputs],
                expected_outputs=[outputs],
                rtol=1e0,
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
