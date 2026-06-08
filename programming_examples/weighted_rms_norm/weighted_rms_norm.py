# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized Weighted RMS Normalization Example

Implements weighted RMS normalization on a 2D input [M, N]:
  1. rms  = sum(x^2, axis=-1) / N
  2. rstd = 1 / sqrt(rms + eps)
  3. y    = x * rstd * weight

The weight vector has shape [N] and is shared across all M rows.

Uses a single AIE tile with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write with
configurable VECTOR_SIZE (default 16 for AIE2).
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

EPS = 1e-5


@module_builder
def build_module(M, N, np_dtype, vector_size=16, herd_x=1):
    xrt_dtype = type_mapper(np_dtype)
    assert (
        N % vector_size == 0
    ), f"N ({N}) must be divisible by vector_size ({vector_size})"

    vecTy = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    # FP32 compute types. Following the GPU/PyTorch standard (torch
    # rms_norm_composite / HF LlamaRMSNorm), the accuracy-critical reduction is
    # done in f32: bf16 inputs are squared, upcast, and the sum-of-squares is
    # accumulated in f32, with the scalar rsqrt also in f32. The per-element
    # epilogue (x * rstd * weight) then runs in bf16 vectors — the aie vector
    # unit does not legalize f32 vector elementwise mul — so the only remaining
    # quantization is the single bf16 output rounding, as in a standard GPU
    # RMSNorm.
    f32 = F32Type.get()
    vecTyF32 = VectorType.get([vector_size], f32)

    # L3 types
    l3MemrefTy = MemRefType.get([M, N], xrt_dtype)
    l3WeightTy = MemRefType.get([N], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([N], xrt_dtype, memory_space=l1_mem_space)
    l1VecTyF32 = MemRefType.get([vector_size], f32, memory_space=l1_mem_space)
    # Small bf16 scratch for the square round-trip that breaks the mulf->addf
    # def-use chain (see Step 1). Dedicated buffer so the reduction phase does
    # not alias the output buffer.
    l1SqTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

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

        # Multi-tile mode: each tile DMAs the full weight vector from L3
        # (the compiler may merge these identical loads into a broadcast).
        @FuncOp.from_py_func(l3MemrefTy, l3WeightTy, l3MemrefTy)
        def weighted_rms_norm(arg0, arg1, arg2):

            @herd(name="herd_0", sizes=[herd_x, 1], operands=[arg0, arg1, arg2])
            def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_weight, l3_out):
                l1_row = AllocOp(l1RowTy, [], [])
                l1_out = AllocOp(l1RowTy, [], [])
                l1_weight = AllocOp(l1RowTy, [], [])
                l1_acc = AllocOp(l1VecTyF32, [], [])
                l1_sq = AllocOp(l1SqTy, [], [])

                c0 = arith.ConstantOp.create_index(0)
                cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                cst0_f32 = arith.ConstantOp(f32, 0.0)
                n_f = arith.ConstantOp(f32, float(N))
                eps_f = arith.ConstantOp(f32, EPS)

                v_zero_f32 = BroadcastOp(vecTyF32, cst0_f32)

                # Weight DMA: same data to all tiles (broadcast)
                dma_memcpy_nd(
                    l1_weight,
                    l3_weight,
                    src_offsets=[0],
                    src_sizes=[N],
                    src_strides=[1],
                )

                for local_row in range_(rows_per_tile):
                    row = affine_apply(row_map, [local_row, _tx])
                    # DMA: load one row (tile-dependent offset)
                    dma_memcpy_nd(
                        l1_row,
                        l3_in,
                        src_offsets=[row, 0],
                        src_sizes=[1, N],
                        src_strides=[N, 1],
                    )

                    # Step 1: sum of x^2, accumulated in F32 (GPU standard).
                    # Square in bf16 vector (aievec-legal), upcast to f32, and
                    # accumulate into an f32 buffer — the running sum stays f32 so
                    # the reduction over N does not lose low-order bits.
                    transfer_write(None, v_zero_f32, l1_acc, [c0], identity_map, [True])
                    for j in range_(0, N, vector_size):
                        sub_row = subview(l1_row.result, [j], [vector_size], [1])
                        v_x = transfer_read(
                            vecTy, sub_row, [c0], identity_map, cst0, [True]
                        )
                        v_sq = arith.mulf(v_x, v_x)
                        # break the mulf->addf chain (aievec rejects addf fed
                        # directly by mulf); round-trip through a dedicated L1
                        # scratch buffer.
                        transfer_write(None, v_sq, l1_sq, [c0], identity_map, [True])
                        v_sq_rd = transfer_read(
                            vecTy, l1_sq, [c0], identity_map, cst0, [True]
                        )
                        v_sq_f32 = arith.extf(vecTyF32, v_sq_rd)
                        v_acc = transfer_read(
                            vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                        )
                        v_sum = arith.addf(v_acc, v_sq_f32)
                        transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                        yield_([])

                    # Horizontal reduce in f32
                    v_final = transfer_read(
                        vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                    )
                    total_sum = vector_reduction(f32, "add", v_final)
                    rms = arith.divf(total_sum, n_f)

                    # Step 2: rstd = rsqrt(mean + eps) in f32, truncate the
                    # scalar to bf16 (f32 scalar ops are legal; f32 *vector*
                    # elementwise is not, so the per-element scaling below stays
                    # in bf16).
                    rms_eps = arith.addf(rms, eps_f)
                    rstd_f32 = math_dialect.rsqrt(rms_eps)
                    rstd = arith.truncf(xrt_dtype, rstd_f32)

                    # Step 3: y = x * rstd * weight (bf16 vector elementwise)
                    v_rstd = BroadcastOp(vecTy, rstd)
                    for j in range_(0, N, vector_size):
                        sub_row = subview(l1_row.result, [j], [vector_size], [1])
                        sub_w = subview(l1_weight.result, [j], [vector_size], [1])
                        sub_out = subview(l1_out.result, [j], [vector_size], [1])
                        v_x = transfer_read(
                            vecTy, sub_row, [c0], identity_map, cst0, [True]
                        )
                        v_w = transfer_read(
                            vecTy, sub_w, [c0], identity_map, cst0, [True]
                        )
                        v_normed = arith.mulf(v_x, v_rstd)
                        v_weighted = arith.mulf(v_normed, v_w)
                        transfer_write(
                            None,
                            v_weighted,
                            sub_out,
                            [c0],
                            identity_map,
                            [True],
                        )
                        yield_([])

                    # DMA: write result row (tile-dependent offset)
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
                DeallocOp(l1_weight)
                DeallocOp(l1_acc)
                DeallocOp(l1_sq)

        return  # end of herd_x > 1 path

    # Original single-tile path (herd_x == 1)
    @FuncOp.from_py_func(l3MemrefTy, l3WeightTy, l3MemrefTy)
    def weighted_rms_norm(arg0, arg1, arg2):

        @herd(name="herd_0", sizes=[1, 1], operands=[arg0, arg1, arg2])
        def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_weight, l3_out):
            l1_row = AllocOp(l1RowTy, [], [])
            l1_out = AllocOp(l1RowTy, [], [])
            l1_weight = AllocOp(l1RowTy, [], [])
            l1_acc = AllocOp(l1VecTyF32, [], [])
            l1_sq = AllocOp(l1SqTy, [], [])

            c0 = arith.ConstantOp.create_index(0)
            cst0 = arith.ConstantOp(xrt_dtype, 0.0)
            cst0_f32 = arith.ConstantOp(f32, 0.0)
            n_f = arith.ConstantOp(f32, float(N))
            eps_f = arith.ConstantOp(f32, EPS)

            v_zero_f32 = BroadcastOp(vecTyF32, cst0_f32)

            # DMA weight to L1 (shared across all rows)
            dma_memcpy_nd(l1_weight, l3_weight)

            for row in range_(M):
                # DMA: load one row from L3 to L1
                dma_memcpy_nd(
                    l1_row,
                    l3_in,
                    src_offsets=[row, 0],
                    src_sizes=[1, N],
                    src_strides=[N, 1],
                )

                # Step 1: sum of x^2, accumulated in F32 (GPU standard).
                # Square in bf16 vector (aievec-legal), upcast to f32, and
                # accumulate into an f32 buffer — the running sum stays f32 so
                # the reduction over N does not lose low-order bits.
                transfer_write(None, v_zero_f32, l1_acc, [c0], identity_map, [True])
                for j in range_(0, N, vector_size):
                    sub_row = subview(l1_row.result, [j], [vector_size], [1])
                    v_x = transfer_read(
                        vecTy, sub_row, [c0], identity_map, cst0, [True]
                    )
                    v_sq = arith.mulf(v_x, v_x)
                    # break the mulf->addf chain (aievec rejects addf fed
                    # directly by mulf); round-trip through a dedicated L1
                    # scratch buffer.
                    transfer_write(None, v_sq, l1_sq, [c0], identity_map, [True])
                    v_sq_rd = transfer_read(
                        vecTy, l1_sq, [c0], identity_map, cst0, [True]
                    )
                    v_sq_f32 = arith.extf(vecTyF32, v_sq_rd)
                    v_acc = transfer_read(
                        vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                    )
                    v_sum = arith.addf(v_acc, v_sq_f32)
                    transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                    yield_([])

                # Horizontal reduce in f32
                v_final = transfer_read(
                    vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                )
                total_sum = vector_reduction(f32, "add", v_final)
                rms = arith.divf(total_sum, n_f)

                # Step 2: rstd = rsqrt(mean + eps) in f32, truncate scalar to bf16.
                rms_eps = arith.addf(rms, eps_f)
                rstd_f32 = math_dialect.rsqrt(rms_eps)
                rstd = arith.truncf(xrt_dtype, rstd_f32)

                # Step 3: y = x * rstd * weight (bf16 vector elementwise)
                v_rstd = BroadcastOp(vecTy, rstd)
                for j in range_(0, N, vector_size):
                    sub_row = subview(l1_row.result, [j], [vector_size], [1])
                    sub_w = subview(l1_weight.result, [j], [vector_size], [1])
                    sub_out = subview(l1_out.result, [j], [vector_size], [1])
                    v_x = transfer_read(
                        vecTy, sub_row, [c0], identity_map, cst0, [True]
                    )
                    v_w = transfer_read(vecTy, sub_w, [c0], identity_map, cst0, [True])
                    v_normed = arith.mulf(v_x, v_rstd)
                    v_weighted = arith.mulf(v_normed, v_w)
                    transfer_write(
                        None, v_weighted, sub_out, [c0], identity_map, [True]
                    )
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
            DeallocOp(l1_weight)
            DeallocOp(l1_acc)
            DeallocOp(l1_sq)


def rms_norm_reference(x, weight, eps=1e-5):
    """CPU F32 reference for weighted RMS norm."""
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + eps)
    return ((x_f32 / rms) * weight.astype(np.float32)).astype(x.dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weighted RMS Normalization — multi-tile with profiling",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Profile kernel execution"
    )
    parser.add_argument(
        "--M", type=int, default=2048, help="Rows (default: LLAMA seq_len)"
    )
    parser.add_argument(
        "--N", type=int, default=2048, help="Cols (default: LLAMA emb_dim)"
    )
    parser.add_argument("--vector-size", type=int, default=16)
    parser.add_argument(
        "--herd-x",
        type=int,
        default=1,
        help="Number of tiles (1=original, 8=multi-tile)",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Profiling iterations"
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
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
    )
    args = parser.parse_args()

    M, N = args.M, args.N
    herd_x = args.herd_x if hasattr(args, "herd_x") else 1
    print(f"Weighted RMSNorm: M={M}, N={N}, herd=[{herd_x},1]")

    mlir_module = build_module(M, N, bfloat16, args.vector_size, herd_x=herd_x)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    x_input = np.random.randn(M, N).astype(bfloat16)
    weight = np.random.randn(N).astype(bfloat16)
    y_expected = rms_norm_reference(x_input, weight)

    # Function signature is (input, weight, output) for both single- and
    # multi-tile modes. Per-tile intermediate buffers in the multi-tile path
    # are allocated internally (in L1) and not exposed at the L3 boundary.

    if args.profile:
        import time
        import pyxrt as xrt
        import filelock

        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="weighted_rms_norm",
        )
        artifact = backend.compile(mlir_module)

        # Hold the NPU lock for the entire NPU-touching scope: load, BO
        # setup, warmup, and the timed loop. Keeping load() under a separate
        # lock would let other processes interleave on the device during the
        # measurement and pollute the timings.
        with filelock.FileLock("/tmp/npu.lock"):
            backend.load(artifact)

            out_buf = np.zeros((M, N), dtype=bfloat16)
            inputs = [x_input, weight, out_buf]
            sizes = [a.size * a.itemsize for a in inputs]
            bos = [
                xrt.bo(
                    backend.device, s, xrt.bo.host_only, backend.kernel.group_id(i + 3)
                )
                for i, s in enumerate(sizes)
            ]

            # Warmup
            for i, a in enumerate(inputs):
                bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            backend.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            h = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
            h.wait()

            times = []
            for _ in range(args.iterations):
                for i, a in enumerate(inputs):
                    bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
                    bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                t0 = time.perf_counter()
                h = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
                h.wait()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        backend.unload()

        # Profile mode reports timing/bandwidth only. Correctness is covered
        # by `make run` / the compile-and-run path's correlation check.
        data_mb = (M * N * 2 * 2 + N * 2) / 1e6  # 2 matrices + 1 weight vector
        print(
            f"\n  Kernel: avg={np.mean(times):.1f}ms  min={np.min(times):.1f}ms  max={np.max(times):.1f}ms"
        )
        print(f"  Bandwidth: {data_mb / (np.min(times)/1000) / 1000:.2f} GB/s")

    elif args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="weighted_rms_norm",
            runtime_loop_tiling_sizes=[4, 4],
            report_precision=True,
            n_perf_iters=args.perf_iters,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[x_input, weight],
                expected_outputs=[y_expected],
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
