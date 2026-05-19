# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AWQ-style int4 to bfloat16 dequantization example.

Dequantizes int4 weights packed in uint8 pairs using per-group
scale and zero-point parameters:
  output[i] = (int4_weight[i] - zero_point[group]) * scale[group]

Scales and zero-points are interleaved into a single params buffer
to stay within the DMA channel limit (2 S2MM + 1 MM2S).

Uses a 1x1 AIE herd with an external C++ kernel compiled with Peano.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend


@module_builder
def build_module(n, group_size):
    bf16_type = type_mapper(bfloat16)
    i8_type = IntegerType.get_signless(8)
    n_packed = n // 2
    n_groups = n // group_size

    # L3 types: weights (i8), params (bf16, interleaved scale+zero), output (bf16)
    l3_w_ty = MemRefType.get([n_packed], i8_type)
    l3_p_ty = MemRefType.get([2 * n_groups], bf16_type)
    l3_out_ty = MemRefType.get([n], bf16_type)

    # L1 types
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_w_ty = MemRefType.get([n_packed], i8_type, memory_space=l1_space)
    l1_p_ty = MemRefType.get([2 * n_groups], bf16_type, memory_space=l1_space)
    l1_out_ty = MemRefType.get([n], bf16_type, memory_space=l1_space)

    # External kernel
    dequant_func = FuncOp(
        "dequant_int4_bf16",
        ([l1_w_ty, l1_p_ty, l1_out_ty], []),
        visibility="private",
    )
    dequant_func.attributes["link_with"] = StringAttr.get("dequant.o")
    dequant_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_w_ty, l3_p_ty, l3_out_ty)
    def dequant(arg_w, arg_p, arg_out):
        @launch(operands=[arg_w, arg_p, arg_out])
        def launch_body(lw, lp, lo):
            @segment(name="seg", operands=[lw, lp, lo])
            def segment_body(sw, sp, so):
                @herd(
                    name="dequant_herd",
                    sizes=[1, 1],
                    operands=[sw, sp, so],
                    link_with="dequant.o",
                )
                def herd_body(_tx, _ty, _sx, _sy, hw, hp, ho):
                    l1_w = AllocOp(l1_w_ty, [], [])
                    l1_p = AllocOp(l1_p_ty, [], [])
                    l1_out = AllocOp(l1_out_ty, [], [])

                    dma_memcpy_nd(l1_w, hw)
                    dma_memcpy_nd(l1_p, hp)

                    CallOp(dequant_func, [l1_w, l1_p, l1_out])

                    dma_memcpy_nd(ho, l1_out)

                    DeallocOp(l1_w)
                    DeallocOp(l1_p)
                    DeallocOp(l1_out)


if __name__ == "__main__":
    N = 1024
    GROUP_SIZE = 128

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="AWQ-style int4 to bf16 dequantization example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Number of elements")
    parser.add_argument(
        "--group-size", type=int, default=GROUP_SIZE, help="Quantization group size"
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

    if args.n % 2 != 0:
        parser.error("N must be even (2 int4 values per byte)")
    if args.n % args.group_size != 0:
        parser.error("N must be divisible by group_size")

    mlir_module = build_module(args.n, args.group_size)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    n_packed = args.n // 2
    n_groups = args.n // args.group_size

    # Generate random int4 weights (0..15) packed in uint8
    int4_vals = np.random.randint(0, 16, args.n).astype(np.uint8)
    packed_weights = np.zeros(n_packed, dtype=np.uint8)
    for i in range(n_packed):
        packed_weights[i] = (int4_vals[2 * i + 1] << 4) | (int4_vals[2 * i] & 0x0F)

    # Generate random scales and zero-points, interleave into params
    scales = np.random.uniform(0.01, 0.1, n_groups).astype(bfloat16)
    zeros = np.random.uniform(7.0, 9.0, n_groups).astype(bfloat16)
    params = np.zeros(2 * n_groups, dtype=bfloat16)
    for g in range(n_groups):
        params[2 * g] = scales[g]
        params[2 * g + 1] = zeros[g]

    # Reference dequantization
    ref_output = np.zeros(args.n, dtype=bfloat16)
    for i in range(args.n):
        g = i // args.group_size
        ref_output[i] = bfloat16(
            (float(int4_vals[i]) - float(zeros[g])) * float(scales[g])
        )

    packed_i8 = packed_weights.view(np.int8)

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_pingpong=True,
            output_format=args.output_format,
            instance_name="dequant",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[packed_i8, params],
                expected_outputs=[ref_output],
                rtol=1e-1,
                atol=5e-2,
            )
        )
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_pingpong=True,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
