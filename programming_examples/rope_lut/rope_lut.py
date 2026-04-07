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

Supports multi-tile herd (herd_x > 1) for row-parallel execution.
Each row's RoPE is independent — no cross-row dependencies.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
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
def build_module(seq_len, embed_dim, np_dtype_in, herd_x=1):
    xrt_dtype = type_mapper(np_dtype_in)
    total = seq_len * embed_dim
    assert (
        embed_dim % 16 == 0
    ), "embed_dim must be divisible by 16 (kernel vector width)"

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

    assert (
        seq_len % herd_x == 0
    ), f"seq_len ({seq_len}) must be divisible by herd_x ({herd_x})"
    rows_per_tile = seq_len // herd_x

    # Affine map: row_offset = (local_row + _tx * rows_per_tile) * embed_dim
    row_offset_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(1),
                        AffineConstantExpr.get(rows_per_tile),
                    ),
                ),
                AffineConstantExpr.get(embed_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3DataTy, l3DataTy, l3DataTy)
    def rope_lut(arg0, arg1, arg2):
        # arg0 = input [total], arg1 = lut [total], arg2 = output [total]

        @herd(name="herd_0", sizes=[herd_x, 1], operands=[arg0, arg1, arg2])
        def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_lut, l3_out):
            l1_in = AllocOp(l1RowTy, [], [])
            l1_lut = AllocOp(l1RowTy, [], [])
            l1_out = AllocOp(l1RowTy, [], [])

            dim_i32 = ConstantOp(T.i32(), embed_dim)

            for local_row in range_(rows_per_tile):
                row_offset = affine_apply(row_offset_map, [local_row, _tx])

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


def rope_reference(input_data, lut, embed_dim):
    """CPU F32 reference for RoPE with precomputed LUT (vectorized)."""
    x = input_data.astype(np.float32).reshape(-1, embed_dim)
    l = lut.astype(np.float32).reshape(-1, embed_dim)
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    cos_v = l[:, 0::2]
    sin_v = l[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos_v - x_odd * sin_v
    out[:, 1::2] = x_even * sin_v + x_odd * cos_v
    return out.astype(input_data.dtype)


def generate_lut(seq_len, embed_dim, dtype=bfloat16, theta=10000.0):
    """Generate interleaved [cos, sin, cos, sin, ...] RoPE LUT (vectorized)."""
    i_vals = np.arange(embed_dim // 2, dtype=np.float64)
    freqs = 1.0 / (theta ** (2.0 * i_vals / embed_dim))
    rows = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(rows, freqs)  # (seq_len, embed_dim//2)
    lut = np.empty((seq_len, embed_dim), dtype=np.float32)
    lut[:, 0::2] = np.cos(angles)
    lut[:, 1::2] = np.sin(angles)
    return lut.astype(dtype)


if __name__ == "__main__":
    THETA = 10000.0

    parser = argparse.ArgumentParser(
        description="RoPE (LUT-based) — build, run, profile",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=64, help="Number of rows")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--herd-x",
        type=int,
        default=1,
        help="Number of tiles (1=single, 8=multi-tile)",
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

    seq_len = args.seq_len
    embed_dim = args.embed_dim
    herd_x = args.herd_x
    print(f"RoPE LUT: seq_len={seq_len}, embed_dim={embed_dim}, herd=[{herd_x},1]")

    mlir_module = build_module(seq_len, embed_dim, bfloat16, herd_x=herd_x)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    if args.compile_mode == "compile-and-run":
        np.random.seed(0)
        input_data = np.random.uniform(-4.0, 4.0, (seq_len, embed_dim)).astype(bfloat16)
        lut = generate_lut(seq_len, embed_dim, bfloat16, THETA)
        y_expected = rope_reference(input_data, lut, embed_dim)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rope",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_data.flatten(), lut.flatten()],
                expected_outputs=[y_expected.flatten()],
                rtol=5e-2,
                atol=5e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
