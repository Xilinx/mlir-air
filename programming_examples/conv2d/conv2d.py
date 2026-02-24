# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""2D Convolution Example

Implements a simple 2D convolution (NHWC layout, HWC filter) on a single
AIE tile using scalar load/store operations.

Input:  [1, H, W, Ci] i32
Filter: [Kh, Kw, Ci, Co] i32
Output: [1, Ho, Wo, Co] i32

Where Ho = H - Kh + 1, Wo = W - Kw + 1 (valid convolution, stride=1).

Data flows:
  1. DMA input tile and filter from L3 to L1
  2. Compute convolution on AIE tile
  3. DMA output tile from L1 to L3
"""

import argparse

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

# Small default sizes for a simple demonstration
H_DEFAULT = 8
W_DEFAULT = 8
CI_DEFAULT = 4
CO_DEFAULT = 4
KH = 3
KW = 3


@module_builder
def build_module(H, W, Ci, Co, Kh, Kw, np_dtype):
    xrt_dtype = type_mapper(np_dtype)
    Ho = H - Kh + 1
    Wo = W - Kw + 1

    # L3 types (flattened for DMA compatibility)
    l3InTy = MemRefType.get([1, H, W, Ci], xrt_dtype)
    l3FilterTy = MemRefType.get([Kh, Kw, Ci, Co], xrt_dtype)
    l3OutTy = MemRefType.get([1, Ho, Wo, Co], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1InTy = MemRefType.get([H, W, Ci], xrt_dtype, memory_space=l1_mem_space)
    l1FilterTy = MemRefType.get([Kh, Kw, Ci, Co], xrt_dtype, memory_space=l1_mem_space)
    l1OutTy = MemRefType.get([Ho, Wo, Co], xrt_dtype, memory_space=l1_mem_space)

    @FuncOp.from_py_func(l3InTy, l3FilterTy, l3OutTy)
    def conv2d(arg_in, arg_filter, arg_out):

        @herd(
            name="herd_0",
            sizes=[1, 1],
            operands=[arg_in, arg_filter, arg_out],
        )
        def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_filter, l3_out):
            l1_in = AllocOp(l1InTy, [], [])
            l1_filter = AllocOp(l1FilterTy, [], [])
            l1_out = AllocOp(l1OutTy, [], [])

            # DMA input and filter to L1
            dma_memcpy_nd(l1_in, l3_in)
            dma_memcpy_nd(l1_filter, l3_filter)

            zero = arith.ConstantOp(xrt_dtype, 0)

            # Initialize output to zero
            for oh in range_(Ho):
                for ow in range_(Wo):
                    for co in range_(Co):
                        store(zero, l1_out, [oh, ow, co])
                        yield_([])
                    yield_([])
                yield_([])

            # Convolution: output[oh, ow, co] += input[oh+kh, ow+kw, ci] * filter[kh, kw, ci, co]
            for oh in range_(Ho):
                for ow in range_(Wo):
                    for co in range_(Co):
                        for kh in range_(Kh):
                            for kw in range_(Kw):
                                for ci in range_(Ci):
                                    ih = arith.addi(oh, kh)
                                    iw = arith.addi(ow, kw)
                                    in_val = load(l1_in, [ih, iw, ci])
                                    f_val = load(l1_filter, [kh, kw, ci, co])
                                    prod = arith.muli(in_val, f_val)
                                    acc = load(l1_out, [oh, ow, co])
                                    new_acc = arith.addi(acc, prod)
                                    store(new_acc, l1_out, [oh, ow, co])
                                    yield_([])
                                yield_([])
                            yield_([])
                        yield_([])
                    yield_([])
                yield_([])

            # DMA output from L1 to L3
            dma_memcpy_nd(l3_out, l1_out)

            DeallocOp(l1_in)
            DeallocOp(l1_filter)
            DeallocOp(l1_out)


if __name__ == "__main__":
    INPUT_DATATYPE = np.int32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the 2D convolution example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--H", type=int, default=H_DEFAULT, help="Input height")
    parser.add_argument("--W", type=int, default=W_DEFAULT, help="Input width")
    parser.add_argument("--Ci", type=int, default=CI_DEFAULT, help="Input channels")
    parser.add_argument("--Co", type=int, default=CO_DEFAULT, help="Output channels")
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

    Ho = args.H - KH + 1
    Wo = args.W - KW + 1

    mlir_module = build_module(args.H, args.W, args.Ci, args.Co, KH, KW, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    input_data = np.random.randint(0, 4, size=(1, args.H, args.W, args.Ci)).astype(
        INPUT_DATATYPE
    )
    filter_data = np.random.randint(0, 4, size=(KH, KW, args.Ci, args.Co)).astype(
        INPUT_DATATYPE
    )

    # Reference convolution (NHWC layout)
    output_ref = np.zeros((1, Ho, Wo, args.Co), dtype=INPUT_DATATYPE)
    for oh in range(Ho):
        for ow in range(Wo):
            for co in range(args.Co):
                for kh in range(KH):
                    for kw in range(KW):
                        for ci in range(args.Ci):
                            output_ref[0, oh, ow, co] += (
                                input_data[0, oh + kh, ow + kw, ci]
                                * filter_data[kh, kw, ci, co]
                            )

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="conv2d",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_data, filter_data],
                expected_outputs=[output_ref],
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
