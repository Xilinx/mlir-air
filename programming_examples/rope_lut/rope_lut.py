# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RoPE (Rotary Position Embeddings) with Precomputed LUT

Applies rotary position embeddings to a 2D input [seq_len, embed_dim]:
  output[r, 2i]   = input[r, 2i] * cos(r * freq_i) - input[r, 2i+1] * sin(r * freq_i)
  output[r, 2i+1] = input[r, 2i] * sin(r * freq_i) + input[r, 2i+1] * cos(r * freq_i)

where freq_i = 1 / (theta ^ (2i / embed_dim)), theta = 10000.

The cos/sin values are precomputed on the host and streamed in as a
look-up table (LUT) with interleaved [cos, sin, cos, sin, ...] layout.
Uses the external rope.cc kernel from mlir-aie (aie_kernels/aie2p).

Uses a single AIE tile with DMA transfers between L3 and L1 memory.
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


@module_builder
def build_module(seq_len, embed_dim, np_dtype_in):
    xrt_dtype = type_mapper(np_dtype_in)
    total = seq_len * embed_dim
    assert (
        embed_dim % 16 == 0
    ), "embed_dim must be divisible by 16 (kernel vector width)"
    index_type = IndexType.get()

    # L3 types
    l3DataTy = MemRefType.get([total], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get(
        shape=[embed_dim], element_type=xrt_dtype, memory_space=l1_mem_space
    )

    # External kernel: rope(input, lut, output, dims)
    rope_func = FuncOp(
        "rope", ([l1RowTy, l1RowTy, l1RowTy, T.i32()], []), visibility="private"
    )
    rope_func.attributes["link_with"] = StringAttr.get("rope.o")
    rope_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3DataTy, l3DataTy, l3DataTy)
    def rope_lut(arg0, arg1, arg2):
        # arg0 = input [total], arg1 = lut [total], arg2 = output [total]

        @herd(name="herd_0", sizes=[1, 1], operands=[arg0, arg1, arg2])
        def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_lut, l3_out):
            l1_in = AllocOp(l1RowTy, [], [])
            l1_lut = AllocOp(l1RowTy, [], [])
            l1_out = AllocOp(l1RowTy, [], [])

            dim_i32 = ConstantOp(T.i32(), embed_dim)

            for row_offset in range_(0, total, embed_dim):
                dma_memcpy_nd(
                    l1_in,
                    l3_in,
                    src_offsets=[row_offset],
                    src_sizes=[embed_dim],
                    src_strides=[1],
                )
                dma_memcpy_nd(
                    l1_lut,
                    l3_lut,
                    src_offsets=[row_offset],
                    src_sizes=[embed_dim],
                    src_strides=[1],
                )

                CallOp(rope_func, [l1_in, l1_lut, l1_out, dim_i32])

                dma_memcpy_nd(
                    l3_out,
                    l1_out,
                    dst_offsets=[row_offset],
                    dst_sizes=[embed_dim],
                    dst_strides=[1],
                )
                yield_([])

            DeallocOp(l1_in)
            DeallocOp(l1_lut)
            DeallocOp(l1_out)

        herd_body.attributes["link_with"] = StringAttr.get("rope.o")


if __name__ == "__main__":
    SEQ_LEN = 64
    EMBED_DIM = 64
    INPUT_DATATYPE = bfloat16
    THETA = 10000.0

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the RoPE (LUT-based) example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length")
    parser.add_argument(
        "--embed-dim", type=int, default=EMBED_DIM, help="Embedding dimension"
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

    mlir_module = build_module(args.seq_len, args.embed_dim, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    seq_len = args.seq_len
    embed_dim = args.embed_dim

    # Generate random input
    input_data = np.random.uniform(-4.0, 4.0, (seq_len, embed_dim)).astype(
        INPUT_DATATYPE
    )

    # Generate LUT: interleaved [cos, sin, cos, sin, ...] per row
    lut = np.zeros((seq_len, embed_dim), dtype=np.float32)
    for r in range(seq_len):
        for i in range(embed_dim // 2):
            freq = 1.0 / (THETA ** (2.0 * i / embed_dim))
            angle = r * freq
            lut[r, 2 * i] = np.cos(angle)
            lut[r, 2 * i + 1] = np.sin(angle)
    lut = lut.astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        # Compute reference output
        ref = np.copy(input_data).astype(np.float32)
        input_f32 = input_data.astype(np.float32)
        lut_f32 = lut.astype(np.float32)
        for r in range(seq_len):
            for i in range(embed_dim // 2):
                cos_v = lut_f32[r, 2 * i]
                sin_v = lut_f32[r, 2 * i + 1]
                x0 = input_f32[r, 2 * i]
                x1 = input_f32[r, 2 * i + 1]
                ref[r, 2 * i] = x0 * cos_v - x1 * sin_v
                ref[r, 2 * i + 1] = x0 * sin_v + x1 * cos_v
        ref_flat = ref.flatten().astype(INPUT_DATATYPE)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rope",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_data.flatten(), lut.flatten()],
                expected_outputs=[ref_flat],
                rtol=5e-2,
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
