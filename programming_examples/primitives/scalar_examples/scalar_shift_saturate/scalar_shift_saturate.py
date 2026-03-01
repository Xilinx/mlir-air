# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scalar Shift + Saturate Example

Computes element-wise fixed-point quantization on a 1D i32 input [N]:
  output[i] = clip(input[i] >> shift_amount, -128, 127)

The scalar shrsi + maxsi + minsi + trunci chain is fused by the
LowerScalarShiftClampTruncToSRS pattern (mlir-aie PR #2894) into
a vectorized SRS (Shift-Round-Saturate) operation:
  broadcast_scalar -> cast(isResAcc) -> srs(narrowed) -> ext_elem

Input and output are both i32 memrefs. The trunci(i8) + extsi(i32)
round-trip preserves the pattern match while keeping DMA types uniform.

Uses a 1x2 AIE herd with DMA transfers between L3 and L1 memory.
"""

import argparse
import numpy as np

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype, shift_amount=4):
    xrt_dtype = type_mapper(np_dtype)
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0
    index_type = IndexType.get()

    # L3 MemRefTypes (i32 for both input and output)
    l3memrefTy = MemRefType.get([n], xrt_dtype)

    # L1 MemRefTypes
    l1MemrefTy = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def scalar_shift_saturate(arg0, arg1):

        @herd(
            name="herd_0",
            sizes=[1, num_tiles],
            operands=[arg0, arg1],
        )
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
                c1 = ConstantOp(index_type, 1)
                cTileN = ConstantOp(index_type, tile_n)

                # Constants for shift + signed saturation clamp
                shift_const = arith.ConstantOp(
                    xrt_dtype, IntegerAttr.get(xrt_dtype, shift_amount)
                )
                min_const = arith.ConstantOp(
                    xrt_dtype, IntegerAttr.get(xrt_dtype, -128)
                )
                max_const = arith.ConstantOp(xrt_dtype, IntegerAttr.get(xrt_dtype, 127))

                # Scalar loop: shift right, clamp to [-128, 127], truncate to i8
                for j in range_(c0, cTileN, c1):
                    scalar_val = load(l1_in.result, [j])

                    # Arithmetic right shift
                    shifted = arith.shrsi(scalar_val, shift_const)
                    # Signed saturation clamp
                    clamped_lo = arith.maxsi(shifted, min_const)
                    clamped_hi = arith.minsi(clamped_lo, max_const)
                    # Truncate to i8 (triggers SRS pattern from PR #2894)
                    truncated = arith.trunci(T.i8(), clamped_hi)
                    # Extend back to i32 for uniform output type
                    extended = arith.extsi(xrt_dtype, truncated)

                    store(extended, l1_out.result, [j])
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
    SHIFT_AMOUNT = 4
    INPUT_DATATYPE = np.int32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the scalar shift+saturate example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--shift-amount",
        type=int,
        default=SHIFT_AMOUNT,
        help="Right shift amount (quantization scale factor)",
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

    mlir_module = build_module(args.n, args.tile_n, INPUT_DATATYPE, args.shift_amount)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(42)
    # Use a range where shifted values span the i8 output range:
    # With shift=4, input range [-2048, 2047] produces shifted values in [-128, 127].
    # Extend slightly beyond to also exercise saturation clamping.
    max_val = (127 << args.shift_amount) + (1 << args.shift_amount)
    input_a = np.random.randint(-max_val, max_val, args.n, dtype=INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])

        # Reference: SRS (Shift-Round-Saturate) with truncation.
        # The AIE SRS intrinsic performs floor-toward-negative-infinity
        # (same as arithmetic right shift).
        def ref_shift_saturate(x, shift):
            shifted = x >> shift
            return np.clip(shifted, -128, 127).astype(np.int8).astype(np.int32)

        sampled_values = np.array(
            [
                ref_shift_saturate(input_a[i], args.shift_amount)
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
            instance_name="scalar_shift_saturate",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=0,
                atol=0,
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
