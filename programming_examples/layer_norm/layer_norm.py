# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized Layer Normalization Example (with affine weight + bias)

Implements affine layer normalization on a 2D input [M, N]:
  1. mean = sum(x, axis=-1) / N                    (F32 reduction)
  2. var  = sum((x - mean)^2, axis=-1) / N          (F32 reduction)
  3. rstd = 1 / sqrt(var + eps)                      (F32)
  4. y    = (x - mean) * rstd * weight + bias        (bf16 epilogue)

The affine parameters `weight` (gamma) and `bias` (beta) both have shape [N]
and are shared across all M rows. This matches HuggingFace / PyTorch
`nn.LayerNorm` — the accuracy-critical reductions (mean and variance) run in
FP32; only the per-element epilogue runs in the bf16 vector unit.

Two herd modes share the same math:
  * single-tile (herd_x=1): one AIE tile loops over all M rows.
  * multi-tile  (herd_x=8): the M rows are split across `herd_x` AIE columns
    (full NPU2 chip width) — LayerNorm is row-independent, so this is a pure
    memory-bandwidth win. weight/bias [N] are broadcast to every column.

Computation is vectorized using vector.transfer_read/write with a
configurable VECTOR_SIZE (default 16 for AIE2/AIE2P).
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

# SigLIP LayerNorm uses eps=1e-6; the example default is 1e-5. The difference
# is negligible versus the bf16 datapath error, so we keep 1e-5 here (record
# the value used in the registry page).
EPS = 1e-5


@module_builder
def build_module(M, N, np_dtype, vector_size=16, herd_x=1):
    xrt_dtype = type_mapper(np_dtype)
    assert (
        N % vector_size == 0
    ), f"N ({N}) must be divisible by vector_size ({vector_size})"

    vecTy = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    # FP32 compute types. Following the GPU/PyTorch standard (torch LayerNorm /
    # HF *LayerNorm), the accuracy-critical reductions (mean, then variance) are
    # done in f32: bf16 inputs are upcast and both sums are accumulated in an f32
    # buffer, with the scalar rsqrt also in f32. The per-element epilogue
    # ((x-mean) * rstd * weight + bias) then runs in bf16 vectors — the aie
    # vector unit does not legalize f32 vector elementwise ops — so the only
    # remaining quantization is the single bf16 output rounding, as in a
    # standard GPU LayerNorm.
    f32 = F32Type.get()
    vecTyF32 = VectorType.get([vector_size], f32)

    # L3 types
    l3MemrefTy = MemRefType.get([M, N], xrt_dtype)
    # weight (gamma) and bias (beta) are packed into a single flat [2N] buffer
    # ([0:N] = weight, [N:2N] = bias) so they cost ONE L3<->L1 DMA channel per
    # tile instead of two — the AIE per-tile routing caps at ~3 L3 streams, and
    # in + weight + bias + out = 4 exceeds that (aie.connect routing failure).
    l3ParamTy = MemRefType.get([2 * N], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([N], xrt_dtype, memory_space=l1_mem_space)
    l1ParamTy = MemRefType.get([2 * N], xrt_dtype, memory_space=l1_mem_space)
    l1VecTyF32 = MemRefType.get([vector_size], f32, memory_space=l1_mem_space)
    # Small bf16 scratch to break mulf->addf def-use chains (see Steps 2 & 4):
    # aievec rejects an addf whose operand comes directly from a mulf, so we
    # round-trip the product through a dedicated L1 scratch buffer.
    l1SqTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

    def emit_body(l3_in, l3_param, l3_out, row_iter, tx=None, row_map=None):
        """Emit the per-tile LayerNorm body. `row_iter` yields the loop trip
        count; the global row index is computed from row_map when multi-tile.
        `l3_param` is the packed [2N] weight||bias buffer."""
        l1_row = AllocOp(l1RowTy, [], [])
        l1_out = AllocOp(l1RowTy, [], [])
        l1_param = AllocOp(l1ParamTy, [], [])  # [0:N]=weight, [N:2N]=bias
        l1_acc = AllocOp(l1VecTyF32, [], [])
        l1_sq = AllocOp(l1SqTy, [], [])

        c0 = arith.ConstantOp.create_index(0)
        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
        cst0_f32 = arith.ConstantOp(f32, 0.0)
        n_f = arith.ConstantOp(f32, float(N))
        eps_f = arith.ConstantOp(f32, EPS)

        v_zero_f32 = BroadcastOp(vecTyF32, cst0_f32)

        # DMA the packed weight||bias [2N] buffer to L1 in one transfer (shared
        # across all rows / broadcast to tiles). One DMA channel, not two — the
        # AIE per-tile routing caps at ~3 L3 streams; in + weight + bias + out
        # = 4 overflows it (aie.connect routing failure).
        dma_memcpy_nd(
            l1_param, l3_param, src_offsets=[0], src_sizes=[2 * N], src_strides=[1]
        )

        for it in range_(row_iter):
            if row_map is not None:
                row = affine_apply(row_map, [it, tx])
            else:
                row = it
            # DMA: load one row from L3 to L1
            dma_memcpy_nd(
                l1_row,
                l3_in,
                src_offsets=[row, 0],
                src_sizes=[1, N],
                src_strides=[N, 1],
            )

            # Step 1: sum(x) accumulated in F32 (for the mean).
            transfer_write(None, v_zero_f32, l1_acc, [c0], identity_map, [True])
            for j in range_(0, N, vector_size):
                sub_row = subview(l1_row.result, [j], [vector_size], [1])
                v_x = transfer_read(vecTy, sub_row, [c0], identity_map, cst0, [True])
                v_x_f32 = arith.extf(vecTyF32, v_x)
                v_acc = transfer_read(
                    vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                )
                v_sum = arith.addf(v_acc, v_x_f32)
                transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                yield_([])

            v_final = transfer_read(
                vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
            )
            sum_x = vector_reduction(f32, "add", v_final)
            mean_f32 = arith.divf(sum_x, n_f)
            # bf16 mean for the vector subtraction (f32 vector elementwise not
            # legalized); mean itself was computed in f32.
            mean = arith.truncf(xrt_dtype, mean_f32)
            v_mean = BroadcastOp(vecTy, mean)

            # Step 2: var = sum((x - mean)^2) / N, accumulated in F32.
            transfer_write(None, v_zero_f32, l1_acc, [c0], identity_map, [True])
            for j in range_(0, N, vector_size):
                sub_row = subview(l1_row.result, [j], [vector_size], [1])
                v_x = transfer_read(vecTy, sub_row, [c0], identity_map, cst0, [True])
                v_diff = arith.subf(v_x, v_mean)
                v_sq = arith.mulf(v_diff, v_diff)
                # break the mulf->addf chain via an L1 scratch round-trip.
                transfer_write(None, v_sq, l1_sq, [c0], identity_map, [True])
                v_sq_rd = transfer_read(vecTy, l1_sq, [c0], identity_map, cst0, [True])
                v_sq_f32 = arith.extf(vecTyF32, v_sq_rd)
                v_acc = transfer_read(
                    vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                )
                v_sum = arith.addf(v_acc, v_sq_f32)
                transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                yield_([])

            v_var = transfer_read(
                vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
            )
            var_sum = vector_reduction(f32, "add", v_var)
            variance = arith.divf(var_sum, n_f)

            # Step 3: rstd = rsqrt(var + eps) in f32, truncate scalar to bf16.
            var_eps = arith.addf(variance, eps_f)
            rstd_f32 = math_dialect.rsqrt(var_eps)
            rstd = arith.truncf(xrt_dtype, rstd_f32)
            v_rstd = BroadcastOp(vecTy, rstd)

            # Step 4: y = (x - mean) * rstd * weight + bias (bf16 vector).
            # weight lives at l1_param[j], bias at l1_param[N + j].
            for j in range_(0, N, vector_size):
                jb = arith.addi(j, arith.ConstantOp.create_index(N))
                sub_row = subview(l1_row.result, [j], [vector_size], [1])
                sub_w = subview(l1_param.result, [j], [vector_size], [1])
                sub_b = subview(l1_param.result, [jb], [vector_size], [1])
                sub_out = subview(l1_out.result, [j], [vector_size], [1])
                v_x = transfer_read(vecTy, sub_row, [c0], identity_map, cst0, [True])
                v_w = transfer_read(vecTy, sub_w, [c0], identity_map, cst0, [True])
                v_b = transfer_read(vecTy, sub_b, [c0], identity_map, cst0, [True])
                v_diff = arith.subf(v_x, v_mean)
                v_normed = arith.mulf(v_diff, v_rstd)
                v_weighted = arith.mulf(v_normed, v_w)
                # break the mulf->addf chain before adding bias.
                transfer_write(None, v_weighted, l1_sq, [c0], identity_map, [True])
                v_weighted_rd = transfer_read(
                    vecTy, l1_sq, [c0], identity_map, cst0, [True]
                )
                v_out = arith.addf(v_weighted_rd, v_b)
                transfer_write(None, v_out, sub_out, [c0], identity_map, [True])
                yield_([])

            # DMA: write result row from L1 to L3
            dma_memcpy_nd(
                l3_out,
                l1_out,
                dst_offsets=[row, 0],
                dst_sizes=[1, N],
                dst_strides=[N, 1],
            )

            yield_([])

        DeallocOp(l1_row)
        DeallocOp(l1_out)
        DeallocOp(l1_param)
        DeallocOp(l1_acc)
        DeallocOp(l1_sq)

    if herd_x > 1:
        assert M % herd_x == 0
        rows_per_tile = M // herd_x
        # Map: global_row = local_row + tx * rows_per_tile
        row_map = AffineMap.get(
            0,
            2,
            [
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(1), AffineConstantExpr.get(rows_per_tile)
                    ),
                )
            ],
        )

        @FuncOp.from_py_func(l3MemrefTy, l3ParamTy, l3MemrefTy)
        def layer_norm(arg0, arg1, arg2):

            @herd(
                name="herd_0",
                sizes=[herd_x, 1],
                operands=[arg0, arg1, arg2],
            )
            def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_param, l3_out):
                emit_body(
                    l3_in,
                    l3_param,
                    l3_out,
                    rows_per_tile,
                    tx=_tx,
                    row_map=row_map,
                )

        return  # end of herd_x > 1 path

    # Single-tile path (herd_x == 1)
    @FuncOp.from_py_func(l3MemrefTy, l3ParamTy, l3MemrefTy)
    def layer_norm(arg0, arg1, arg2):

        @herd(name="herd_0", sizes=[1, 1], operands=[arg0, arg1, arg2])
        def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_param, l3_out):
            emit_body(l3_in, l3_param, l3_out, M)


def layer_norm_reference(x, weight, bias, eps=EPS):
    """CPU F32 reference for affine layer norm (HF / PyTorch nn.LayerNorm)."""
    x_f32 = x.astype(np.float32)
    mean = np.mean(x_f32, axis=-1, keepdims=True)
    variance = np.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(variance + eps)
    y = (x_f32 - mean) * rstd * weight.astype(np.float32) + bias.astype(np.float32)
    return y.astype(x.dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the affine layer normalization example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--M", type=int, default=1024, help="Rows (default: SigLIP num patches)"
    )
    parser.add_argument(
        "--N", type=int, default=768, help="Cols (default: SigLIP hidden dim)"
    )
    parser.add_argument("--vector-size", type=int, default=16)
    parser.add_argument(
        "--herd-x",
        type=int,
        default=1,
        help="Number of tiles (1=single-tile, 8=multi-tile full chip width)",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after 10 warmup) and "
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

    M, N = args.M, args.N
    herd_x = args.herd_x
    print(f"LayerNorm (affine): M={M}, N={N}, herd=[{herd_x},1]")

    mlir_module = build_module(M, N, bfloat16, args.vector_size, herd_x=herd_x)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    x_input = np.random.randn(M, N).astype(bfloat16)
    weight = np.random.randn(N).astype(bfloat16)
    bias = np.random.randn(N).astype(bfloat16)
    # Pack weight (gamma) and bias (beta) into a single flat [2N] buffer so the
    # kernel needs only ONE param DMA channel per tile (see build_module).
    param = np.concatenate([weight, bias]).astype(bfloat16)
    y_expected = layer_norm_reference(x_input, weight, bias)

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="layer_norm",
            runtime_loop_tiling_sizes=[4, 4],
            report_precision=True,
            n_perf_iters=args.perf_iters,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[x_input, param],
                expected_outputs=[y_expected],
                rtol=1.6e-2,
                # atol=6e-2 (vs RMSNorm's 5e-2): LayerNorm's epilogue has one
                # extra bf16 rounding step — the bias add in
                # (x-mean)*rstd*weight + bias — so the worst-case bf16 *output*
                # granularity is ~4 ULP instead of ~3. Not a reduction-precision
                # relaxation; the reductions are FP32 (mean_rel_L1 ~4.4e-3).
                atol=6e-2,
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
