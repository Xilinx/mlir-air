# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RoPE (Rotary Position Embedding) — half-split, matching HuggingFace Llama

Applies rotary position embeddings to a 2D input [rows, head_dim] using the
*half-split* convention (HuggingFace Llama `rotate_half`), pairing
(x[i], x[i + head_dim/2]):

    out[i]        = x[i] * cos[i] - x[i + half] * sin[i]
    out[i + half] = x[i] * sin[i] + x[i + half] * cos[i]

where half = head_dim/2 and the per-row cos/sin are streamed in as a
precomputed look-up table (LUT) in the concatenated half-split layout
    LUT[row] = [cos_0, ..., cos_{half-1}, sin_0, ..., sin_{half-1}]
matching `llama32_1b_weights.py:generate_rope_lut` and the kernel
`rope_halfsplit.cc` that llama-3.2-1B actually links (`external_kernels.py:
compile_rope`). This is **NOT** the interleaved variant in `rope_lut/` or
`rope_sincos/` (those are decoys; their math does not match llama).

Uses the external C++ kernel rope_halfsplit.cc -> rope.o (`@rope`).
Each row's RoPE is independent — no cross-row dependency — so rows are
spread across an `herd_x x herd_y` AIE grid. Each tile uses 3 independent
shim DMAs (input in, lut in, output out); NPU2 has 8 shim DMA channels, so
the herd is capped at herd_x * herd_y <= 8 tiles (8x1 / 2x4 place, 8x2 /
4x4 / 8x4 do not). The best config is herd_x=8, herd_y=1 (full chip width).

Each tile streams its rows one head_dim row per DMA / per kernel call
(matching rope_halfsplit.cc's single-row signature and the llama
prefill/decode builders). `herd_x` (AIE columns) is the scaling knob; the
rows are interleaved across the herd so tile t handles rows t, t+total_tiles,
... A batched-DMA variant (multiple rows per L3<->L1 transfer with per-row
subview kernel calls) was investigated but the air dependency pass
mis-schedules the per-row subview writes under a single bulk output DMA
(half the rows come back zero / NaN at rows_per_dma=2,4), so the faithful
one-row-per-DMA structure is used — see the performance notes.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(rows, head_dim, np_dtype_in, herd_x=8, herd_y=1):
    xrt_dtype = type_mapper(np_dtype_in)
    total = rows * head_dim
    assert (
        head_dim % 16 == 0
    ), "head_dim must be divisible by 16 (kernel vector width N=16)"
    total_tiles = herd_x * herd_y
    assert (
        rows % total_tiles == 0
    ), f"rows ({rows}) must be divisible by herd_x * herd_y ({total_tiles})"
    rows_per_tile = rows // total_tiles  # rows each tile handles, one row per DMA

    # L3 types (flat)
    l3DataTy = MemRefType.get([total], xrt_dtype)

    # L1 types: one head_dim row per DMA / per kernel call (matches the
    # rope_halfsplit.cc signature and the llama prefill/decode builders).
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get(
        shape=[head_dim], element_type=xrt_dtype, memory_space=l1_mem_space
    )

    # External kernel: rope(input_row, lut_row, output_row, dims)
    rope_func = FuncOp(
        "rope", ([l1RowTy, l1RowTy, l1RowTy, T.i32()], []), visibility="private"
    )
    rope_func.attributes["link_with"] = StringAttr.get("rope.o")
    rope_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # row_offset (in elements) = (local_row * total_tiles + tx*herd_y + ty) * head_dim
    # i.e. tiles are interleaved over rows: tile t handles rows t, t+total_tiles, ...
    row_offset_map = AffineMap.get(
        0,
        3,
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),  # loop_iv (already mult. of total_tiles)
                    AffineExpr.get_add(
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(1),  # _tx
                            AffineConstantExpr.get(herd_y),
                        ),
                        AffineSymbolExpr.get(2),  # _ty
                    ),
                ),
                AffineConstantExpr.get(head_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3DataTy, l3DataTy, l3DataTy)
    def rope_halfsplit(arg0, arg1, arg2):
        # arg0 = input [total], arg1 = lut [total], arg2 = output [total]

        @launch(operands=[arg0, arg1, arg2])
        def rope_launch(l_in, l_lut, l_out):

            @segment(name="rope_seg", operands=[l_in, l_lut, l_out])
            def rope_seg(s_in, s_lut, s_out):

                @herd(
                    name="herd_0",
                    sizes=[herd_x, herd_y],
                    operands=[s_in, s_lut, s_out],
                )
                def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_lut, l3_out):
                    l1_in = AllocOp(l1RowTy, [], [])
                    l1_lut = AllocOp(l1RowTy, [], [])
                    l1_out = AllocOp(l1RowTy, [], [])

                    dim_i32 = ConstantOp(T.i32(), head_dim)

                    # Outer loop strides by total_tiles rows; each tile picks its
                    # own row via (tx*herd_y+ty). One row per DMA per kernel call.
                    for loop_iv in range_(0, rows, total_tiles):
                        row_off = affine_apply(row_offset_map, [loop_iv, _tx, _ty])

                        dma_memcpy_nd(
                            l1_in,
                            l3_in,
                            src_offsets=[row_off],
                            src_sizes=[head_dim],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_lut,
                            l3_lut,
                            src_offsets=[row_off],
                            src_sizes=[head_dim],
                            src_strides=[1],
                        )

                        CallOp(rope_func, [l1_in, l1_lut, l1_out, dim_i32])

                        dma_memcpy_nd(
                            l3_out,
                            l1_out,
                            dst_offsets=[row_off],
                            dst_sizes=[head_dim],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_lut)
                    DeallocOp(l1_out)

                herd_body.attributes["link_with"] = StringAttr.get("rope.o")


def rope_halfsplit_reference(input_flat, lut_flat, head_dim):
    """Full-output FP32 half-split RoPE reference (HuggingFace Llama rotate_half /
    apply_rotary_pos_emb), matching rope_halfsplit.cc.

    LUT row layout: [cos_0..cos_{half-1}, sin_0..sin_{half-1}].
        out[i]        = x[i]*cos[i] - x[i+half]*sin[i]
        out[i+half]   = x[i]*sin[i] + x[i+half]*cos[i]
    bf16 inputs are upcast to f32, rotated, cast back to bf16 (the bf16-rounded
    reference a GPU/HF RoPE op is verified against). This is the half-split math,
    NOT the interleaved decoy in rope_lut/.
    """
    half = head_dim // 2
    x = input_flat.astype(np.float32).reshape(-1, head_dim)
    lut = lut_flat.astype(np.float32).reshape(-1, head_dim)
    cos_v = lut[:, :half]
    sin_v = lut[:, half:]
    x1 = x[:, :half]
    x2 = x[:, half:]
    out = np.empty_like(x)
    out[:, :half] = x1 * cos_v - x2 * sin_v
    out[:, half:] = x1 * sin_v + x2 * cos_v
    return out.astype(input_flat.dtype).flatten()


def generate_rope_lut(rows, head_dim, dtype=bfloat16, theta=500000.0):
    """Generate the half-split [cos..., sin...] RoPE LUT (concatenated layout),
    matching llama32_1b_weights.py:generate_rope_lut and rope_halfsplit.cc.

    For position pos and dimension index i (i < head_dim/2):
        freq_i = 1 / (theta ^ (2*i / head_dim));  angle = pos * freq_i
        LUT[pos, i]        = cos(angle)
        LUT[pos, i + half] = sin(angle)

    theta default 500000.0 = llama-3.2 rope_base.
    """
    half = head_dim // 2
    i_vals = np.arange(half, dtype=np.float64)
    freqs = 1.0 / (theta ** (2.0 * i_vals / head_dim))
    pos = np.arange(rows, dtype=np.float64)
    angles = np.outer(pos, freqs)  # (rows, half)
    lut = np.empty((rows, head_dim), dtype=np.float32)
    lut[:, :half] = np.cos(angles)
    lut[:, half:] = np.sin(angles)
    return lut.astype(dtype)


if __name__ == "__main__":
    ROWS = 65536  # prefill RoPE Q: n_heads(32) * seq(2048)
    HEAD_DIM = 64  # llama-3.2-1B head_dim
    THETA = 500000.0  # llama-3.2 rope_base
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="rope_halfsplit.py",
        description="Builds, runs, and tests the standalone half-split RoPE kernel "
        "(HuggingFace Llama convention, matching rope_halfsplit.cc)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--rows", type=int, default=ROWS, help="Number of rows")
    parser.add_argument(
        "--head-dim", type=int, default=HEAD_DIM, help="Head dimension (per-row width)"
    )
    parser.add_argument(
        "--herd-x",
        type=int,
        default=8,
        help="Herd x dimension (AIE columns, default: 8 — full chip width)",
    )
    parser.add_argument(
        "--herd-y",
        type=int,
        default=1,
        help="Herd y dimension (AIE rows, default: 1). NPU2 caps the herd at "
        "herd_x * herd_y <= 8 tiles (3 shim DMAs/tile, 8 shim channels)",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after warmup) and "
        "print Latency in addition to the correctness check",
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

    if args.perf_iters < 0:
        parser.error("--perf-iters must be >= 0")

    mlir_module = build_module(
        args.rows,
        args.head_dim,
        INPUT_DATATYPE,
        herd_x=args.herd_x,
        herd_y=args.herd_y,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # randn input (matching the GPU/HF reference and other registry kernels).
    rng = np.random.default_rng(0)
    input_data = rng.standard_normal(
        (args.rows, args.head_dim), dtype=np.float32
    ).astype(INPUT_DATATYPE)
    lut = generate_rope_lut(args.rows, args.head_dim, INPUT_DATATYPE, THETA)

    # Reference: full-output FP32 half-split rotate (HF rotate_half), cast bf16.
    expected = rope_halfsplit_reference(
        input_data.flatten(), lut.flatten(), args.head_dim
    )

    if args.compile_mode == "compile-and-run":
        # bf16 half-split RoPE: rtol = canonical bf16 1.6e-2; atol set from the
        # measured worst-case single-element error (see kernel_registry/details/
        # RoPE_bf16.md). RoPE is a handful of bf16 mul + one add/sub per element
        # with NO reduction, so it is near the cleanest tier (close to EltwiseAdd).
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rope_halfsplit",
            report_precision=True,
            n_perf_iters=args.perf_iters,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_data.flatten(), lut.flatten()],
                expected_outputs=[expected],
                rtol=1.6e-2,
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
