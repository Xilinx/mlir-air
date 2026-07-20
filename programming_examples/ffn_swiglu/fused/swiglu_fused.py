# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fused SwiGLU for NPU2 (AIE2P) — Single-launch, single-BD-chain design.

Implements:  output = SiLU(x @ W_gate) * (x @ W_up)

Architecture:
  Single launch with 6 herds named "herd_0" chained into one while_true
  loop body. Gate and up GEMMs share the SAME DMA channels (A_L2L1, B_L2L1)
  via FIFO ordering — 2 S2MM channels at compute tile (within hardware limit).

  ONE B_L3L2 channel carries both gate and up weight data. ONE segment K-loop
  of 2*k_tiles iterations creates a SINGLE memtile BD chain. The first k_tiles
  iterations carry gate data, the next k_tiles carry up data. FIFO ordering
  delivers gate data before up data to the core.

  4 function arguments: x[M,K], w_gate[K,N], w_up[K,N], out[M,N].
  No host-side weight preprocessing required.

Uses 8x8x8 bf16 mmul intrinsic with BFP16 emulation on AIE2P.
"""

import argparse
import os
import sys
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.linalg import fill
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_ as range_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from air.extras import types as extrasT
from air.dialects.linalg.opdsl.lang import *
import air.dialects.linalg.opdsl.lang as linalg_lang


@linalg_structured_op()
def block_matmul(
    A=TensorDef(linalg_lang.TV.T1, S.a, S.c, S.f, S.d, S.g, S.i),
    B=TensorDef(linalg_lang.TV.T2, S.b, S.c, S.e, S.f, S.i, S.h),
    C=TensorDef(linalg_lang.TV.U, S.b, S.a, S.e, S.d, S.g, S.h, output=True),
):
    domain(D.a, D.b, D.c, D.d, D.e, D.f, D.g, D.h, D.i)
    C[D.b, D.a, D.e, D.d, D.g, D.h] += (
        TypeFn.cast_signed(linalg_lang.TV.U, A[D.a, D.c, D.f, D.d, D.g, D.i])
    ) * (TypeFn.cast_signed(linalg_lang.TV.U, B[D.b, D.c, D.e, D.f, D.i, D.h]))


@module_builder
def build_module(m, k, n, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n):
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0

    xrt_dtype = type_mapper(bfloat16)
    mmul_m, mmul_k, mmul_n = 8, 8, 8
    k_tiles = k // tile_k_l2
    k_l1_iters = tile_k_l2 // tile_k_l1
    m_blks = tile_m // mmul_m
    n_blks = tile_n // mmul_n
    k_blks_l1 = tile_k_l1 // mmul_k
    flat_tile_size = tile_m * tile_n
    total_k = 2 * k_tiles  # gate + up phases combined

    # L3 types — separate w_gate and w_up
    l3_x_ty = MemRefType.get([m, k], xrt_dtype)
    l3_wg_ty = MemRefType.get([k, n], xrt_dtype)  # w_gate[K, N]
    l3_wu_ty = MemRefType.get([k, n], xrt_dtype)  # w_up[K, N]
    l3_out_ty = MemRefType.get([m, n], xrt_dtype)

    # L2 types (shared between gate and up phases)
    l2s = IntegerAttr.get(extrasT.i32(), MemorySpace.L2)
    l2TyA = MemRefType.get([herd_m, 1, tile_m, tile_k_l2], xrt_dtype, memory_space=l2s)
    l2TyB = MemRefType.get([1, herd_n, tile_k_l2, tile_n], xrt_dtype, memory_space=l2s)
    l2TyC = MemRefType.get(
        [herd_m, herd_n, tile_m, tile_n], xrt_dtype, memory_space=l2s
    )

    # L1 types — 6D block layout
    l1s = IntegerAttr.get(extrasT.i32(), MemorySpace.L1)
    a_l1 = [1, 1, k_blks_l1, m_blks, mmul_m, mmul_k]
    b_l1 = [1, 1, n_blks, k_blks_l1, mmul_k, mmul_n]
    c_l1 = [1, 1, n_blks, m_blks, mmul_m, mmul_n]
    c_herd = [herd_m, herd_n, n_blks, m_blks, mmul_m, mmul_n]

    l1TyA = MemRefType.get(a_l1, xrt_dtype, memory_space=l1s)
    l1TyB = MemRefType.get(b_l1, xrt_dtype, memory_space=l1s)
    acc_layout = StridedLayoutAttr.get(
        ShapedType.get_dynamic_size(),
        [
            flat_tile_size * herd_n,
            flat_tile_size,
            m_blks * mmul_m * mmul_n,
            mmul_m * mmul_n,
            mmul_n,
            1,
        ],
    )
    l1TyC = MemRefType.get(c_l1, xrt_dtype, memory_space=l1s, layout=acc_layout)
    l1TyCHerd = MemRefType.get(c_herd, xrt_dtype, memory_space=l1s)

    # Channels — single B_L3L2 for both gate and up phases
    Channel("A_L3L2")  # x tiles (shared gate/up)
    Channel("B_L3L2")  # weight tiles (gate first, then up)
    # L2→L1: SHARED channels for both gate and up phases
    Channel("A_L2L1", size=[herd_m, 1], broadcast_shape=[herd_m, herd_n])
    Channel("B_L2L1", size=[1, herd_n], broadcast_shape=[herd_m, herd_n])

    # External kernel functions
    silu_func = FuncOp("silu_inplace_bf16", ([l1TyC], []), visibility="private")
    elemwise_mul_func = FuncOp(
        "elemwise_mul_bf16", ([l1TyC, l1TyC], []), visibility="private"
    )
    for f in [silu_func, elemwise_mul_func]:
        f.attributes["link_with"] = StringAttr.get("swiglu_fused.o")
        f.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # ================================================================
    # Main function: x[M,K], w_gate[K,N], w_up[K,N], out[M,N]
    # ================================================================
    @FuncOp.from_py_func(l3_x_ty, l3_wg_ty, l3_wu_ty, l3_out_ty)
    def swiglu_fused(x_arg, wg_arg, wu_arg, out_arg):
        launch_m_size = m // (tile_m * herd_m)
        launch_n_size = n // (tile_n * herd_n)

        @launch(
            operands=[x_arg, wg_arg, wu_arg, out_arg],
            sizes=[launch_m_size, launch_n_size],
        )
        def launch_body(livx, livy, lsx, lsy, l3_x, l3_wg, l3_wu, l3_out):
            ix_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_m * herd_m)
                    )
                ],
            )
            iy_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_n * herd_n)
                    )
                ],
            )
            off_x = affine_apply(ix_map, [livx])
            off_y = affine_apply(iy_map, [livy])

            # Gate phase L3→channel: x + w_gate
            for i in range_(0, k_tiles):
                rmap = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_k_l2)
                        )
                    ],
                )
                roff = affine_apply(rmap, [i])
                ChannelPut(
                    "A_L3L2",
                    l3_x,
                    offsets=[0, 0, off_x, roff],
                    sizes=[herd_m, 1, tile_m, tile_k_l2],
                    strides=[k * tile_m, tile_k_l2, k, 1],
                )
                ChannelPut(
                    "B_L3L2",
                    l3_wg,
                    offsets=[0, 0, roff, off_y],
                    sizes=[1, herd_n, tile_k_l2, tile_n],
                    strides=[n * tile_k_l2, tile_n, n, 1],
                )
                yield_([])

            # Up phase L3→channel: x + w_up (separate array, same offsets)
            for i in range_(0, k_tiles):
                rmap = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_k_l2)
                        )
                    ],
                )
                roff = affine_apply(rmap, [i])
                ChannelPut(
                    "A_L3L2",
                    l3_x,
                    offsets=[0, 0, off_x, roff],
                    sizes=[herd_m, 1, tile_m, tile_k_l2],
                    strides=[k * tile_m, tile_k_l2, k, 1],
                )
                ChannelPut(
                    "B_L3L2",
                    l3_wu,
                    offsets=[0, 0, roff, off_y],
                    sizes=[1, herd_n, tile_k_l2, tile_n],
                    strides=[n * tile_k_l2, tile_n, n, 1],
                )
                yield_([])

            # === SEGMENT ===
            @segment(name="swiglu_seg", operands=[livx, livy, l3_x, l3_wg, l3_out])
            def seg(livx_s, livy_s, l3_x_s, l3_wg_s, l3_out_s):
                seg_ix = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_m),
                        )
                    ],
                )
                seg_iy = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_n * herd_n),
                        )
                    ],
                )
                seg_off_x = affine_apply(seg_ix, [livx_s])
                seg_off_y = affine_apply(seg_iy, [livy_s])

                # Shared L2 buffers
                l2_a = AllocOp(l2TyA, [], [])
                l2_b = AllocOp(l2TyB, [], [])
                l2_c = AllocOp(l2TyC, [], [])
                # Shared L1 input buffers
                l1_a = AllocOp(l1TyA, [], [])
                l1_b = AllocOp(l1TyB, [], [])
                # Two L1 accumulators
                l1_gate = AllocOp(l1TyCHerd, [], [])
                l1_up = AllocOp(l1TyCHerd, [], [])

                # ONE combined K-loop (2*k_tiles): single BD chain
                # L3→L2 gets + L2→L1 puts for both gate and up phases
                for ik in range_(0, total_k):
                    ChannelGet("A_L3L2", l2_a.result)
                    ChannelGet("B_L3L2", l2_b.result)

                    # L2→L1 puts (explicit channels)
                    for j in range_(0, k_l1_iters):
                        kmap = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(tile_k_l1),
                                )
                            ],
                        )
                        koff = affine_apply(kmap, [j])
                        for row in range(herd_m):
                            ChannelPut(
                                "A_L2L1",
                                l2_a.result,
                                indices=[row, 0],
                                offsets=[row, 0, 0, 0, 0, koff],
                                sizes=[1, 1, k_blks_l1, m_blks, mmul_m, mmul_k],
                                strides=[
                                    tile_m * tile_k_l2,
                                    tile_m * tile_k_l2,
                                    mmul_k,
                                    tile_k_l2 * mmul_m,
                                    tile_k_l2,
                                    1,
                                ],
                            )
                        for col in range(herd_n):
                            ChannelPut(
                                "B_L2L1",
                                l2_b.result,
                                indices=[0, col],
                                offsets=[0, col, 0, 0, koff, 0],
                                sizes=[1, 1, n_blks, k_blks_l1, mmul_k, mmul_n],
                                strides=[
                                    herd_n * tile_n * tile_k_l2,
                                    tile_n * tile_k_l2,
                                    mmul_n,
                                    tile_n * mmul_k,
                                    tile_n,
                                    1,
                                ],
                            )
                        yield_([])
                    yield_([])

                # Phase 1: Zero gate accumulator
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_a, l1_b, l1_gate, l2_a, l2_b],
                )
                def herd_z1(_tx, _ty, _sx, _sy, _a, _b, _c, _la, _lb):
                    sub = subview(
                        _c,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[1, 1, n_blks, m_blks, mmul_m, mmul_n],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    z = ConstantOp(FloatAttr.get(xrt_dtype, 0.0), None)
                    fill(z, outs=[sub])

                # Phase 2: Gate matmul K-loop (k_tiles iterations)
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_a, l1_b, l1_gate, l2_a, l2_b],
                )
                def herd_gate(_tx, _ty, _sx, _sy, _a, _b, _c, _la, _lb):
                    for j in range_(0, k_tiles * k_l1_iters):
                        ChannelGet("A_L2L1", _a, indices=[_tx, _ty])
                        ChannelGet("B_L2L1", _b, indices=[_tx, _ty])
                        sub = subview(
                            _c,
                            offsets=[_tx, _ty, 0, 0, 0, 0],
                            sizes=[1, 1, n_blks, m_blks, mmul_m, mmul_n],
                            strides=[1, 1, 1, 1, 1, 1],
                        )
                        block_matmul(_a, _b, outs=[sub])
                        yield_([])

                # Phase 3: Zero up accumulator
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_a, l1_b, l1_up, l2_a, l2_b],
                )
                def herd_z2(_tx, _ty, _sx, _sy, _a, _b, _c, _la, _lb):
                    sub = subview(
                        _c,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[1, 1, n_blks, m_blks, mmul_m, mmul_n],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    z = ConstantOp(FloatAttr.get(xrt_dtype, 0.0), None)
                    fill(z, outs=[sub])

                # Phase 4: Up matmul K-loop (k_tiles iterations)
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_a, l1_b, l1_up, l2_a, l2_b],
                )
                def herd_up(_tx, _ty, _sx, _sy, _a, _b, _c, _la, _lb):
                    for j in range_(0, k_tiles * k_l1_iters):
                        ChannelGet("A_L2L1", _a, indices=[_tx, _ty])
                        ChannelGet("B_L2L1", _b, indices=[_tx, _ty])
                        sub = subview(
                            _c,
                            offsets=[_tx, _ty, 0, 0, 0, 0],
                            sizes=[1, 1, n_blks, m_blks, mmul_m, mmul_n],
                            strides=[1, 1, 1, 1, 1, 1],
                        )
                        block_matmul(_a, _b, outs=[sub])
                        yield_([])

                # Phase 5: Fuse — SiLU(gate) then gate *= up
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_a, l1_b, l1_gate, l1_up, l2_a, l2_b],
                )
                def herd_fuse(_tx, _ty, _sx, _sy, _a, _b, _gate, _up, _la, _lb):
                    gate_sub = subview(
                        _gate,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[1, 1, n_blks, m_blks, mmul_m, mmul_n],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    up_sub = subview(
                        _up,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[1, 1, n_blks, m_blks, mmul_m, mmul_n],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    CallOp(silu_func, [gate_sub])
                    CallOp(elemwise_mul_func, [gate_sub, up_sub])

                herd_fuse.attributes["link_with"] = StringAttr.get("swiglu_fused.o")

                # Phase 6: Writeback via dma_memcpy_nd
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_a, l1_b, l1_gate, l1_up, l2_a, l2_b, l2_c],
                )
                def herd_wb(_tx, _ty, _sx, _sy, _a, _b, _gate, _up, _la, _lb, _lc):
                    gate_sub = subview(
                        _gate,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[1, 1, n_blks, m_blks, mmul_m, mmul_n],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    dma_memcpy_nd(
                        _lc,
                        gate_sub,
                        dst_offsets=[_tx, _ty, 0, 0],
                        dst_sizes=[1, 1, tile_m, tile_n],
                        dst_strides=[
                            herd_n * tile_m * tile_n,
                            tile_m * tile_n,
                            tile_n,
                            1,
                        ],
                        src_offsets=[_tx, _ty, 0, 0, 0, 0],
                        src_sizes=[1, 1, m_blks, mmul_m, n_blks, mmul_n],
                        src_strides=[
                            herd_n * flat_tile_size,
                            flat_tile_size,
                            mmul_m * mmul_n,
                            mmul_n,
                            m_blks * mmul_m * mmul_n,
                            1,
                        ],
                    )

                # L2→L3
                dma_memcpy_nd(
                    l3_out_s,
                    l2_c,
                    dst_offsets=[seg_off_x, seg_off_y],
                    dst_sizes=[herd_m * tile_m, herd_n * tile_n],
                    dst_strides=[n, 1],
                    src_offsets=[0, 0, 0, 0],
                    src_sizes=[herd_m, tile_m, herd_n, tile_n],
                    src_strides=[tile_m * herd_n * tile_n, tile_n, tile_m * tile_n, 1],
                )

                DeallocOp(l2_a)
                DeallocOp(l2_b)
                DeallocOp(l2_c)
                DeallocOp(l1_a)
                DeallocOp(l1_b)
                DeallocOp(l1_gate)
                DeallocOp(l1_up)


if __name__ == "__main__":
    M = 512
    K = 512
    N = 512
    TILE_M = 64
    TILE_K_L2 = 256
    TILE_K_L1 = 32
    TILE_N = 64
    HERD_M = 4
    HERD_N = 4

    parser = argparse.ArgumentParser(
        prog="swiglu_fused.py",
        description="Fused SwiGLU: output = SiLU(x @ W_gate) * (x @ W_up)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--tile-m", type=int, default=TILE_M)
    parser.add_argument("--tile-k-l2", type=int, default=TILE_K_L2)
    parser.add_argument("--tile-k-l1", type=int, default=TILE_K_L1)
    parser.add_argument("--tile-n", type=int, default=TILE_N)
    parser.add_argument("--herd-m", type=int, default=HERD_M)
    parser.add_argument("--herd-n", type=int, default=HERD_N)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="compile-and-run",
        choices=["compile-only", "compile-and-run", "profile"],
        dest="compile_mode",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="xclbin",
        choices=["xclbin", "elf", "none"],
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module(
        args.m,
        args.k,
        args.n,
        args.tile_m,
        args.tile_k_l2,
        args.tile_k_l1,
        args.tile_n,
        args.herd_m,
        args.herd_n,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(42)
    input_x = (np.random.randn(args.m, args.k) * 0.5).astype(bfloat16)
    input_wgate = (np.random.randn(args.k, args.n) * 0.5).astype(bfloat16)
    input_wup = (np.random.randn(args.k, args.n) * 0.5).astype(bfloat16)

    if args.compile_mode == "compile-and-run":
        # Reference: SiLU(x @ W_gate) * (x @ W_up) in f32
        x_f32 = input_x.astype(np.float32)
        gate_f32 = x_f32 @ input_wgate.astype(np.float32)
        up_f32 = x_f32 @ input_wup.astype(np.float32)
        silu_gate = gate_f32 * 0.5 * (np.tanh(gate_f32 / 2.0) + 1.0)
        ref_out = (silu_gate * up_f32).astype(bfloat16)

        num_samples = 200
        sampled_indices = np.vstack(
            [
                np.random.randint(0, args.m, num_samples),
                np.random.randint(0, args.n, num_samples),
            ]
        )
        sampled_values = np.array(
            [ref_out[i, j] for i, j in zip(*sampled_indices)], dtype=bfloat16
        )
        sampled_data = {
            "shape": (args.m, args.n),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            lower_linalg_to_func="swiglu_fused.o",
            instance_name="swiglu_fused",
            runtime_loop_tiling_sizes=[1, 1],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_x, input_wgate, input_wup],
                stochastic_expected_outputs=[sampled_data],
                rtol=0.1,
                atol=4.0,
                max_mismatch_percentage=5,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            target_device="npu2",
            output_format=args.output_format,
            omit_while_true_loop=False,
            lower_linalg_to_func="swiglu_fused.o",
            runtime_loop_tiling_sizes=[1, 1],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
        print("Compilation completed successfully!")
        sys.exit(0)

    elif args.compile_mode == "profile":
        import time, filelock, tempfile

        warmup, iters = 5, 20
        out = np.zeros((args.m, args.n), dtype=bfloat16)
        backend = XRTBackend(
            verbose=args.verbose,
            target_device="npu2",
            output_format="xclbin",
            omit_while_true_loop=False,
            lower_linalg_to_func="swiglu_fused.o",
            runtime_loop_tiling_sizes=[1, 1],
            instance_name="swiglu_fused",
        )
        compiled = backend.compile(mlir_module)
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            fn = backend.load(compiled)
            for i in range(warmup):
                fn(input_x, input_wgate, input_wup, out)
            times = []
            for i in range(iters):
                t0 = time.perf_counter()
                fn(input_x, input_wgate, input_wup, out)
                times.append((time.perf_counter() - t0) * 1e6)
        backend.unload()
        avg_us = sum(times) / len(times)
        min_us = min(times)
        flops = 4.0 * args.m * args.k * args.n + 9.0 * args.m * args.n
        print(f"Fused SwiGLU Profile: M={args.m} K={args.k} N={args.n}")
        print(f"  Avg latency: {avg_us:.1f} us  ({flops / (avg_us * 1e3):.1f} GFLOPS)")
        print(f"  Min latency: {min_us:.1f} us  ({flops / (min_us * 1e3):.1f} GFLOPS)")
        sys.exit(0)
