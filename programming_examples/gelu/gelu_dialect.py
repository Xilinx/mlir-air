# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized GELU (Tanh Approximation) Example

Implements element-wise GELU on a 1D input [N] using the standard
tanh approximation:
  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Uses the hardware tanh intrinsic (__builtin_aie2p_tanh) directly,
matching the IRON project's GELU implementation. No exp or division
needed.

Uses a 1x2 AIE herd with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write, BroadcastOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

SQRT_2_OVER_PI = 0.7978845608  # sqrt(2/pi)
GELU_BETA = 0.044715


@module_builder
def build_module(n, tile_n, np_dtype_in, vector_size=16, herd_x=1, herd_y=None):
    xrt_dtype_in = type_mapper(np_dtype_in)
    # herd_x / herd_y were previously hardcoded (num_tiles=2 -> 1x2 herd).
    # Exposed as tunable knobs (type-(a) change) so the herd can be swept.
    # Default herd_x=1, herd_y=2 preserves the original 1x2 herd (lit test).
    if herd_y is None:
        herd_y = 2
    total_tiles = herd_x * herd_y
    assert n % (tile_n * total_tiles) == 0
    assert tile_n % vector_size == 0
    VECTOR_SIZE = vector_size
    index_type = IndexType.get()

    l3memrefTy = MemRefType.get([n], xrt_dtype_in)
    l1MemrefTy = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    vecTy = VectorType.get([VECTOR_SIZE], xrt_dtype_in)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def gelu(arg0, arg1):

        @herd(name="herd_0", sizes=[herd_x, herd_y], operands=[arg0, arg1])
        def herd_body(_tx, _ty, _sx, _sy, _l3_in, _l3_out):
            l1_in = AllocOp(l1MemrefTy, [], [])
            l1_out = AllocOp(l1MemrefTy, [], [])

            for _l_ivx in range_(0, n, tile_n * total_tiles):
                # linear tile index = tx * herd_y + ty; offset = iv + lin*tile_n
                offset_map = AffineMap.get(
                    0,
                    3,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineExpr.get_add(
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(herd_y),
                                    ),
                                    AffineSymbolExpr.get(2),
                                ),
                                AffineConstantExpr.get(tile_n),
                            ),
                        )
                    ],
                )
                offset = affine_apply(offset_map, [_l_ivx, _tx, _ty])

                dma_memcpy_nd(
                    l1_in,
                    _l3_in,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )

                c0 = ConstantOp(index_type, 0)
                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)
                half_const = arith.ConstantOp(xrt_dtype_in, 0.5)
                one_const = arith.ConstantOp(xrt_dtype_in, 1.0)
                beta_const = arith.ConstantOp(xrt_dtype_in, GELU_BETA)
                s2opi_const = arith.ConstantOp(xrt_dtype_in, SQRT_2_OVER_PI)
                v_half = BroadcastOp(vecTy, half_const)
                v_one = BroadcastOp(vecTy, one_const)
                v_beta = BroadcastOp(vecTy, beta_const)
                v_s2opi = BroadcastOp(vecTy, s2opi_const)

                for j in range_(c0, cTileN, cVecSize):
                    sub_in = subview(l1_in.result, [j], [VECTOR_SIZE], [1])
                    sub_out = subview(l1_out.result, [j], [VECTOR_SIZE], [1])

                    v_x = transfer_read(vecTy, sub_in, [c0], identity_map, cst0, [True])

                    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    # Uses hardware tanh intrinsic — no exp or division needed.
                    v_x2 = arith.mulf(v_x, v_x)
                    v_x3 = arith.mulf(v_x, v_x2)
                    v_beta_x3 = arith.mulf(v_x3, v_beta.result)
                    v_inner = arith.addf(v_x, v_beta_x3)
                    v_scaled = arith.mulf(v_inner, v_s2opi.result)
                    v_tanh = math_dialect.tanh(v_scaled)
                    v_one_plus_tanh = arith.addf(v_tanh, v_one.result)
                    v_half_x = arith.mulf(v_x, v_half.result)
                    v_gelu = arith.mulf(v_half_x, v_one_plus_tanh)

                    transfer_write(None, v_gelu, sub_out, [c0], identity_map, [True])
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


def gelu_reference(x):
    """Full-output FP32 GELU (tanh approximation) reference.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This is the *tanh* approximation (GELUTanh), matching the NPU kernel and
    the SmolVLA SigLIP vision MLP activation — NOT the exact erf-GELU. bf16
    inputs are upcast to f32, the whole formula is evaluated in f32, and the
    result is cast back to bf16 — the standard way a GPU/HF bf16 elementwise
    op is verified (isolates the NPU quantization error, not bf16-vs-fp32).
    """
    xf = x.astype(np.float32)
    inner = SQRT_2_OVER_PI * (xf + GELU_BETA * xf * xf * xf)
    return (0.5 * xf * (1.0 + np.tanh(inner))).astype(x.dtype)


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the GELU example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size", type=int, default=16, help="Vector size for SIMD operations"
    )
    parser.add_argument(
        "--herd-x",
        type=int,
        default=1,
        help="Herd x dimension (AIE columns, default: 1)",
    )
    parser.add_argument(
        "--herd-y",
        type=int,
        default=None,
        help="Herd y dimension (AIE rows, default: 2 — the original 1x2 herd)",
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
        args.n,
        args.tile_n,
        INPUT_DATATYPE,
        args.vector_size,
        herd_x=args.herd_x,
        herd_y=args.herd_y,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Use N(0,1) (matching the GPU/HF activation test standard) so the check
    # sees a realistic signed distribution. Generate in float32 to avoid a
    # large f64 intermediate.
    rng = np.random.default_rng(0)
    input_a = rng.standard_normal(args.n, dtype=np.float32).astype(INPUT_DATATYPE)

    # Reference: full-output FP32 GELU-tanh (bf16 in upcast to f32, tanh
    # approximation, cast back to bf16) — the bf16-rounded reference a GPU/HF
    # GELUTanh op is verified against. The NPU kernel uses the hardware
    # __builtin_aie2p_tanh; the bf16-tanh-LUT deviation is exactly this measures.
    expected = gelu_reference(input_a)

    if args.compile_mode == "compile-and-run":
        # bf16 GELU-tanh: rtol = canonical bf16 1.6e-2; atol = 5e-2 sized to
        # the measured worst-case single-element error (NPU2: abs_err max
        # ~1.56e-2). GELU chains a hardware tanh LUT plus several bf16
        # roundings (the bf16 "one transcendental" tier, like SiLU-and-Mul),
        # but its worst element is much smaller than SiLU's (0.125) because the
        # tanh argument is scaled down; atol=5e-2 matches the RoPE/RMSNorm/
        # EltwiseAdd convention and clears the worst element with margin. The
        # mean error (mean_rel_L1 ~8.4e-3) sits inside rtol.
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="gelu",
            runtime_loop_tiling_sizes=[4, 4],
            report_precision=True,
            n_perf_iters=args.perf_iters,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
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
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
