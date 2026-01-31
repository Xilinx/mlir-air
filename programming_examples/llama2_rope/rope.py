# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
import argparse
from math import cos, sin

from air.ir import *
from air.dialects.air import *
from air.dialects.affine import apply as affine_apply
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from ml_dtypes import bfloat16

range_ = for_


@module_builder
def build_module(num_heads, head_size, herd_n, np_dtype_in):
    assert (num_heads * head_size) % herd_n == 0
    inout_size = [3 * num_heads * head_size]  # 3 vectors: Q, K and V
    xrt_dtype_in = type_mapper(np_dtype_in)

    # L3 MemRefTypes
    memrefTyIn = MemRefType.get(inout_size, xrt_dtype_in)
    memrefTyOut = MemRefType.get(inout_size, xrt_dtype_in)

    # L1 MemRefTypes
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTyHeadSizeByTwo = MemRefType.get(
        shape=[head_size // 2],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyThreeByHeadSize = MemRefType.get(
        shape=[3, head_size],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    cosf_poly_func = FuncOp(
        "cosf_bf16_24_8",
        ([l1MemrefTyHeadSizeByTwo, l1MemrefTyHeadSizeByTwo], []),
        visibility="private",
    )
    sinf_poly_func = FuncOp(
        "sinf_bf16_24_8",
        ([l1MemrefTyHeadSizeByTwo, l1MemrefTyHeadSizeByTwo], []),
        visibility="private",
    )
    freq_pos_func = FuncOp(
        "freq_pos_bf16_24_8",
        ([T.i32(), l1MemrefTyHeadSizeByTwo], []),
        visibility="private",
    )
    shuffle_apply_rope_poly_func = FuncOp(
        "shuffle_apply_rope_bf16_48",
        (
            [
                T.i32(),
                l1MemrefTyHeadSizeByTwo,
                l1MemrefTyHeadSizeByTwo,
                l1MemrefTyThreeByHeadSize,
            ],
            [],
        ),
        visibility="private",
    )
    vector_copy_func = FuncOp(
        "vector_copy_bf16_192_16",
        ([l1MemrefTyThreeByHeadSize, l1MemrefTyThreeByHeadSize], []),
        visibility="private",
    )
    for func in [
        freq_pos_func,
        sinf_poly_func,
        cosf_poly_func,
        shuffle_apply_rope_poly_func,
        vector_copy_func,
    ]:
        func.attributes["link_with"] = StringAttr.get("rope.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyIn, memrefTyOut)
    def rope(arg0, arg1):

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

            zero_const = ConstantOp(FloatAttr.get(xrt_dtype_in, 0.0), None)
            one_const = ConstantOp(IntegerAttr.get(T.i32(), 1), None)

            for t in range_(0, num_heads * 3 * head_size, herd_n * 3 * head_size):
                l1_in_data = AllocOp(l1MemrefTyThreeByHeadSize, [], [])
                l1_out_data = AllocOp(l1MemrefTyThreeByHeadSize, [], [])
                l1_freq_pos_data = AllocOp(l1MemrefTyHeadSizeByTwo, [], [])

                offset_map = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(3 * head_size),
                            ),
                        )
                    ],
                )
                offset = affine_apply(offset_map, [t, _ty])

                dma_memcpy_nd(
                    l1_in_data,
                    _l3_a,
                    src_offsets=[
                        offset,
                    ],
                    src_sizes=[3 * head_size],
                    src_strides=[1],
                )

                vector_copy_call = CallOp(
                    vector_copy_func,
                    [l1_in_data, l1_out_data],
                )

                freq_pos_call = CallOp(
                    freq_pos_func,
                    [one_const, l1_freq_pos_data],
                )

                l1_sinf_vec = AllocOp(l1MemrefTyHeadSizeByTwo, [], [])
                l1_cosf_vec = AllocOp(l1MemrefTyHeadSizeByTwo, [], [])
                sinf_poly_call = CallOp(
                    sinf_poly_func,
                    [l1_freq_pos_data, l1_sinf_vec],
                )
                cosf_poly_call = CallOp(
                    cosf_poly_func,
                    [l1_freq_pos_data, l1_cosf_vec],
                )

                # sq
                rope_offset_0_const = ConstantOp(IntegerAttr.get(T.i32(), 0), None)
                shuffle_apply_rope_call = CallOp(
                    shuffle_apply_rope_poly_func,
                    [rope_offset_0_const, l1_cosf_vec, l1_sinf_vec, l1_out_data],
                )
                # sk
                rope_offset_1_const = ConstantOp(
                    IntegerAttr.get(T.i32(), head_size), None
                )
                shuffle_apply_rope_call = CallOp(
                    shuffle_apply_rope_poly_func,
                    [rope_offset_1_const, l1_cosf_vec, l1_sinf_vec, l1_out_data],
                )

                dma_memcpy_nd(
                    _l3_c,
                    l1_out_data,
                    dst_offsets=[
                        offset,
                    ],
                    dst_sizes=[3 * head_size],
                    dst_strides=[1],
                )
                DeallocOp(l1_sinf_vec)
                DeallocOp(l1_cosf_vec)
                DeallocOp(l1_freq_pos_data)
                DeallocOp(l1_in_data)
                DeallocOp(l1_out_data)
                DeallocOp(l1_freq_pos_data)

                yield_([])

        herd_body.attributes["link_with"] = StringAttr.get("rope.o")


if __name__ == "__main__":
    # Default values.
    HEAD_SIZE = 48
    NUM_HEADS = 8
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
        "--head_size",
        type=int,
        default=HEAD_SIZE,
        help="Head size",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=NUM_HEADS,
        help="Number of heads",
    )
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
        args.num_heads,
        args.head_size,
        args.herd_n,
        INPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    num_tiles = args.num_heads
    inputs = np.random.randn(
        num_tiles,
        3 * args.head_size,
    ).astype(INPUT_DATATYPE)
    outputs = inputs.copy()

    for i in range(0, num_tiles):
        for s in range(0, args.head_size, 2):
            freq = 1.0 / pow(10000.0, float(s) / float(args.head_size))
            val = 1 * freq

            fcr = cos(val)
            fci = sin(val)

            v0 = outputs[i][s]
            v1 = outputs[i][s + 1]

            outputs[i][s] = v0 * fcr - v1 * fci
            outputs[i][s + 1] = v0 * fci + v1 * fcr

            v0 = outputs[i][s + args.head_size]
            v1 = outputs[i][s + args.head_size + 1]
            outputs[i][s + args.head_size] = v0 * fcr - v1 * fci
            outputs[i][s + args.head_size + 1] = v0 * fci + v1 * fcr

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
                rtol=1e1,
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
