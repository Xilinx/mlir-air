#!/usr/bin/env python3
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Cascade connection verification test for AIE2P.

Verifies that hardware cascade channels correctly pass data between
adjacent compute tiles. Isolates cascade behavior from the full flash
attention pipeline to diagnose potential cascade data integrity issues.

Tests:
  1. Passthrough: Stage 3 fills a buffer with a known constant (42.0),
     cascades through stages 2 -> 1 -> 0, stage 0 outputs to L3.
     Verifies data arrives unchanged.

  2. Accumulation: Stage 3 fills with 0 and adds 10, then cascades.
     Stage 2 adds 20, stage 1 adds 30, stage 0 adds 40 and outputs.
     Expected result: all elements = 100.0.
     Verifies each stage can read, modify, and re-send cascade data.

Usage:
    cd programming_examples/flash_attention/kernel_fusion_based/test_cascade
    make compile-kernel
    python3 cascade_test.py                 # Run all tests
    python3 cascade_test.py --test passthrough
    python3 cascade_test.py -p              # Print MLIR only
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16
import filelock

import air
from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.arith import ConstantOp
from air.dialects.affine import apply as affine_apply
from air.dialects import scf, affine, arith
from air.backend.xrt import XRTBackend
from math import exp, sqrt

# Buffer size: 1024 bf16 elements (32 cascade beats of 32 elements each)
BUF_SIZE = 1024
NUM_CASCADE_STAGES = 4

# Flash attention merge test dimensions
TILE_SIZE_Q = 64
DK = DV = 64
MMUL_M = MMUL_K = MMUL_N = 8
NUM_Q_TILES = 4
SQRT_DK = sqrt(DK)  # = 8.0


def external_func(name, inputs, outputs=None, link_with=None, visibility="private"):
    if outputs is None:
        outputs = []
    func_type = FunctionType.get(inputs, outputs)
    func = FuncOp(name=name, type=func_type, visibility=visibility)
    func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
    if link_with:
        func.attributes["link_with"] = StringAttr.get(link_with)
    return func


@module_builder
def build_passthrough_test():
    """Cascade passthrough: stage 3 fills buffer, cascade to stage 0, output.

    Stage 3 fills buffer with 42.0.
    Data cascades: stage 3 -> stage 2 -> stage 1 -> stage 0.
    Stage 0 outputs the buffer to L3.
    Expected output: all elements = 42.0.
    """
    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    memref_buf_l1 = MemRefType.get([BUF_SIZE], bf16, memory_space=l1_space)
    memref_buf_l2 = MemRefType.get([BUF_SIZE], bf16, memory_space=l2_space)
    memref_buf_l3 = MemRefType.get([BUF_SIZE], bf16)

    # Kernel function declarations
    external_func(
        "fill_pattern", [memref_buf_l1, i32], link_with="cascade_kernel.o"
    )
    external_func(
        "zero_fill_cascade", [memref_buf_l1], link_with="cascade_kernel.o"
    )

    # Output channels: L1 -> L2 -> L3
    Channel("L1ToL2", size=[1, 1])
    Channel("L2ToL3", size=[1])

    # Cascade channel: 3 connections for 4 stages (stage 3->2, 2->1, 1->0)
    chan_cascade = Channel("cascade", size=[1, NUM_CASCADE_STAGES - 1])
    chan_cascade.attributes["channel_type"] = StringAttr.get("cascade")

    @FuncOp.from_py_func(memref_buf_l3)
    def cascade_passthrough(output_buf):
        c1 = ConstantOp(index_type, 1)

        @launch(operands=[output_buf], sizes=[c1, c1])
        def launch_body(launch_x, launch_y, lsize_x, lsize_y, out_arg):
            # L2 -> L3 output DMA
            ChannelGet("L2ToL3", out_arg, indices=[0])

            c1_launch = ConstantOp(index_type, 1)

            @segment(name="cascade_seg", operands=[], sizes=[c1_launch, c1_launch])
            def segment_body(seg_x, seg_y, ssize_x, ssize_y):
                # L2 output buffer
                l2_out = AllocOp(memref_buf_l2, [], [])
                # L1 buffer (each herd tile gets its own copy)
                l1_buf = AllocOp(memref_buf_l1, [], [])

                c_nstages = ConstantOp(index_type, NUM_CASCADE_STAGES)
                c_1tile = ConstantOp(index_type, 1)

                # Segment-side output path: L1 -> L2, then L2 -> L3
                ChannelGet("L1ToL2", l2_out.result, indices=[0, 0])
                ChannelPut("L2ToL3", l2_out.result, indices=[0])

                @herd(
                    name="cascade_herd",
                    sizes=[c_1tile, c_nstages],
                    operands=[l1_buf],
                    link_with="cascade_kernel.o",
                )
                def herd_body(tx, ty, sx, sy, buf):
                    c0 = ConstantOp(index_type, 0)
                    c1_h = ConstantOp(index_type, 1)

                    # === Stage 3 (last): fill buffer and send via cascade ===
                    set_last = IntegerSet.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(-(NUM_CASCADE_STAGES - 1)),
                            )
                        ],
                        [True],  # s1 == 3
                    )
                    if_last = affine.AffineIfOp(
                        set_last, cond_operands=[tx, ty], has_else=True
                    )
                    with InsertionPoint(if_last.then_block):
                        fill_val = ConstantOp(i32, 42)
                        CallOp([], "fill_pattern", [buf, fill_val])
                        subi = arith.SubIOp(ty, c1_h)
                        ChannelPut("cascade", buf, indices=[tx, subi])
                        affine.AffineYieldOp([])

                    with InsertionPoint(if_last.else_block):
                        # === Middle stages (1, 2): receive and forward ===
                        set_middle = IntegerSet.get(
                            0,
                            2,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(1),
                                    AffineConstantExpr.get(-1),
                                ),  # s1 >= 1
                                AffineExpr.get_add(
                                    AffineConstantExpr.get(NUM_CASCADE_STAGES - 2),
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(-1),
                                    ),
                                ),  # s1 <= N-2
                            ],
                            [False, False],
                        )
                        if_middle = affine.AffineIfOp(
                            set_middle, cond_operands=[tx, ty], has_else=True
                        )

                        with InsertionPoint(if_middle.then_block):
                            ChannelGet("cascade", buf, indices=[tx, ty])
                            subi2 = arith.SubIOp(ty, c1_h)
                            ChannelPut("cascade", buf, indices=[tx, subi2])
                            affine.AffineYieldOp([])

                        with InsertionPoint(if_middle.else_block):
                            # === Stage 0 (first): receive and output ===
                            ChannelGet("cascade", buf, indices=[tx, ty])
                            ChannelPut("L1ToL2", buf, indices=[tx, c0])
                            affine.AffineYieldOp([])

                        affine.AffineYieldOp([])

                DeallocOp(l1_buf)
                DeallocOp(l2_out)


@module_builder
def build_accumulation_test():
    """Cascade accumulation: each stage adds a unique value.

    Stage 3: zero-fill, add 10, send via cascade.
    Stage 2: receive, add 20, send.
    Stage 1: receive, add 30, send.
    Stage 0: receive, add 40, output.
    Expected output: all elements = 10 + 20 + 30 + 40 = 100.0.
    """
    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    memref_buf_l1 = MemRefType.get([BUF_SIZE], bf16, memory_space=l1_space)
    memref_buf_l2 = MemRefType.get([BUF_SIZE], bf16, memory_space=l2_space)
    memref_buf_l3 = MemRefType.get([BUF_SIZE], bf16)

    external_func(
        "fill_pattern", [memref_buf_l1, i32], link_with="cascade_kernel.o"
    )
    external_func(
        "add_const_bf16", [memref_buf_l1, i32], link_with="cascade_kernel.o"
    )
    external_func(
        "zero_fill_cascade", [memref_buf_l1], link_with="cascade_kernel.o"
    )

    Channel("L1ToL2", size=[1, 1])
    Channel("L2ToL3", size=[1])
    chan_cascade = Channel("cascade", size=[1, NUM_CASCADE_STAGES - 1])
    chan_cascade.attributes["channel_type"] = StringAttr.get("cascade")

    @FuncOp.from_py_func(memref_buf_l3)
    def cascade_accumulation(output_buf):
        c1 = ConstantOp(index_type, 1)

        @launch(operands=[output_buf], sizes=[c1, c1])
        def launch_body(launch_x, launch_y, lsize_x, lsize_y, out_arg):
            ChannelGet("L2ToL3", out_arg, indices=[0])

            c1_launch = ConstantOp(index_type, 1)

            @segment(name="cascade_seg", operands=[], sizes=[c1_launch, c1_launch])
            def segment_body(seg_x, seg_y, ssize_x, ssize_y):
                l2_out = AllocOp(memref_buf_l2, [], [])
                l1_buf = AllocOp(memref_buf_l1, [], [])

                c_nstages = ConstantOp(index_type, NUM_CASCADE_STAGES)
                c_1tile = ConstantOp(index_type, 1)

                ChannelGet("L1ToL2", l2_out.result, indices=[0, 0])
                ChannelPut("L2ToL3", l2_out.result, indices=[0])

                @herd(
                    name="cascade_herd",
                    sizes=[c_1tile, c_nstages],
                    operands=[l1_buf],
                    link_with="cascade_kernel.o",
                )
                def herd_body(tx, ty, sx, sy, buf):
                    c0 = ConstantOp(index_type, 0)
                    c1_h = ConstantOp(index_type, 1)

                    # === Stage 3: zero, add 10, send ===
                    set_s3 = IntegerSet.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(-3),
                            )
                        ],
                        [True],
                    )
                    if_s3 = affine.AffineIfOp(
                        set_s3, cond_operands=[tx, ty], has_else=True
                    )
                    with InsertionPoint(if_s3.then_block):
                        CallOp([], "zero_fill_cascade", [buf])
                        CallOp([], "add_const_bf16", [buf, ConstantOp(i32, 10)])
                        subi = arith.SubIOp(ty, c1_h)
                        ChannelPut("cascade", buf, indices=[tx, subi])
                        affine.AffineYieldOp([])

                    with InsertionPoint(if_s3.else_block):
                        # === Stage 2: receive, add 20, send ===
                        set_s2 = IntegerSet.get(
                            0,
                            2,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(1),
                                    AffineConstantExpr.get(-2),
                                )
                            ],
                            [True],
                        )
                        if_s2 = affine.AffineIfOp(
                            set_s2, cond_operands=[tx, ty], has_else=True
                        )
                        with InsertionPoint(if_s2.then_block):
                            ChannelGet("cascade", buf, indices=[tx, ty])
                            CallOp(
                                [], "add_const_bf16", [buf, ConstantOp(i32, 20)]
                            )
                            subi2 = arith.SubIOp(ty, c1_h)
                            ChannelPut("cascade", buf, indices=[tx, subi2])
                            affine.AffineYieldOp([])

                        with InsertionPoint(if_s2.else_block):
                            # === Stage 1: receive, add 30, send ===
                            set_s1 = IntegerSet.get(
                                0,
                                2,
                                [
                                    AffineExpr.get_add(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(-1),
                                    )
                                ],
                                [True],
                            )
                            if_s1 = affine.AffineIfOp(
                                set_s1, cond_operands=[tx, ty], has_else=True
                            )
                            with InsertionPoint(if_s1.then_block):
                                ChannelGet("cascade", buf, indices=[tx, ty])
                                CallOp(
                                    [],
                                    "add_const_bf16",
                                    [buf, ConstantOp(i32, 30)],
                                )
                                subi3 = arith.SubIOp(ty, c1_h)
                                ChannelPut(
                                    "cascade", buf, indices=[tx, subi3]
                                )
                                affine.AffineYieldOp([])

                            with InsertionPoint(if_s1.else_block):
                                # === Stage 0: receive, add 40, output ===
                                ChannelGet("cascade", buf, indices=[tx, ty])
                                CallOp(
                                    [],
                                    "add_const_bf16",
                                    [buf, ConstantOp(i32, 40)],
                                )
                                ChannelPut("L1ToL2", buf, indices=[tx, c0])
                                affine.AffineYieldOp([])

                            affine.AffineYieldOp([])
                        affine.AffineYieldOp([])

                DeallocOp(l1_buf)
                DeallocOp(l2_out)


@module_builder
def build_merge_test():
    """Flash attention cascade merge test with 4x4 herd.

    Replicates the exact cascade merge logic from the flash attention kernel:
    - 4 q_tiles x 4 cascade stages (4x4 herd)
    - Each stage has its own Gp [64,64], up [64,1], sp [64,1]
    - Stage values: gp=(ty+1)*5, up=ty+8, sp=ty+1
    - Stage 3 sends via cascade, stages 2,1 merge and forward,
      stage 0 merges, divides by sp, and outputs
    - Uses the exact same merge functions as attn.cc
    """
    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    # Buffer types matching flash attention
    memref_gp_l1 = MemRefType.get(
        [TILE_SIZE_Q, DK], bf16, memory_space=l1_space
    )
    memref_up_l1 = MemRefType.get([TILE_SIZE_Q, 1], bf16, memory_space=l1_space)
    # L2 output: one full Q-chunk worth of output
    lqp_full = NUM_Q_TILES * TILE_SIZE_Q  # 256
    memref_out_l2 = MemRefType.get([lqp_full, DK], bf16, memory_space=l2_space)
    # L3 output
    memref_out_l3 = MemRefType.get([lqp_full, DK], bf16)

    kernel_o = "cascade_merge_kernel.o"

    # Declare all kernel functions (matching attn.cc signatures)
    external_func("fill_gp_uniform", [memref_gp_l1, i32], link_with=kernel_o)
    external_func("fill_sp_uniform", [memref_up_l1, i32], link_with=kernel_o)
    external_func("zero_fill_sp_bf16", [memref_up_l1], link_with=kernel_o)
    external_func("neg_inf_fill_up_bf16", [memref_up_l1], link_with=kernel_o)
    external_func(
        "maximum_up_u_bf16", [memref_up_l1, memref_up_l1], link_with=kernel_o
    )
    external_func(
        "exp_up_minus_u",
        [memref_up_l1, memref_up_l1, memref_up_l1],
        link_with=kernel_o,
    )
    external_func(
        "mul_r_gp", [memref_up_l1, memref_gp_l1], link_with=kernel_o
    )
    external_func(
        "add_gp_g", [memref_gp_l1, memref_gp_l1], link_with=kernel_o
    )
    external_func(
        "accum_sp_r_s",
        [memref_up_l1, memref_up_l1, memref_up_l1],
        link_with=kernel_o,
    )
    external_func(
        "vector_copy_32elems",
        [i32, memref_up_l1, memref_up_l1],
        link_with=kernel_o,
    )
    external_func(
        "div_gp_sp", [memref_up_l1, memref_gp_l1], link_with=kernel_o
    )

    # Channels
    Channel("L1ToL2", size=[NUM_Q_TILES, 1])
    Channel("L2ToL3", size=[1])
    chan_cascade = Channel(
        "cascade", size=[NUM_Q_TILES, NUM_CASCADE_STAGES - 1]
    )
    chan_cascade.attributes["channel_type"] = StringAttr.get("cascade")

    @FuncOp.from_py_func(memref_out_l3)
    def cascade_merge(output_buf):
        c1 = ConstantOp(index_type, 1)

        @launch(operands=[output_buf], sizes=[c1, c1])
        def launch_body(launch_x, launch_y, lsize_x, lsize_y, out_arg):
            # L2->L3: gather output from all q_tiles
            ChannelGet("L2ToL3", out_arg, indices=[0])

            c1_launch = ConstantOp(index_type, 1)

            @segment(
                name="merge_seg", operands=[], sizes=[c1_launch, c1_launch]
            )
            def segment_body(seg_x, seg_y, ssize_x, ssize_y):
                # L2 output buffer for gathered result
                l2_out = AllocOp(memref_out_l2, [], [])
                # L1 buffers (per-tile): Gp, up, sp
                gp_buf = AllocOp(memref_gp_l1, [], [])
                up_buf = AllocOp(memref_up_l1, [], [])
                sp_buf = AllocOp(memref_up_l1, [], [])

                c_nstages = ConstantOp(index_type, NUM_CASCADE_STAGES)
                c_qtiles = ConstantOp(index_type, NUM_Q_TILES)

                # Segment output: gather 4 q_tiles into l2_out
                affine_map_tileq = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(TILE_SIZE_Q),
                        )
                    ],
                )
                par_out = scf.ForallOp(
                    lower_bounds=[0], upper_bounds=[NUM_Q_TILES], steps=[1]
                )
                with InsertionPoint(par_out.body):
                    apply_off = affine_apply(
                        affine_map_tileq, [par_out.induction_variables[0]]
                    )
                    ChannelGet(
                        "L1ToL2",
                        l2_out.result,
                        indices=[par_out.induction_variables[0], 0],
                        offsets=[apply_off, 0],
                        sizes=[TILE_SIZE_Q, DV],
                        strides=[DV, 1],
                    )
                    scf.InParallelOp()

                ChannelPut("L2ToL3", l2_out.result, indices=[0])

                @herd(
                    name="merge_herd",
                    sizes=[c_qtiles, c_nstages],
                    operands=[gp_buf, up_buf, sp_buf],
                    link_with=kernel_o,
                )
                def herd_body(tx, ty, sx, sy, gp, up, sp):
                    c0 = ConstantOp(index_type, 0)
                    c1_h = ConstantOp(index_type, 1)

                    # === INIT: fill per-stage values ===
                    # gp_val = (ty + 1) * 5
                    ty_i32 = arith.IndexCastOp(i32, ty).result
                    c1_i32 = ConstantOp(i32, 1)
                    c5_i32 = ConstantOp(i32, 5)
                    c8_i32 = ConstantOp(i32, 8)
                    ty_plus1 = arith.AddIOp(ty_i32, c1_i32).result
                    gp_val = arith.MulIOp(ty_plus1, c5_i32).result
                    # up_val = ty + 8
                    up_val = arith.AddIOp(ty_i32, c8_i32).result
                    # sp_val = ty + 1
                    sp_val = ty_plus1

                    CallOp([], "fill_gp_uniform", [gp, gp_val])
                    CallOp([], "fill_sp_uniform", [up, up_val])
                    CallOp([], "fill_sp_uniform", [sp, sp_val])

                    # === CASCADE MERGE (from flash attention attn.py) ===
                    r_l1_c = AllocOp(memref_up_l1, [], [])

                    # Stage 3 (last): send via cascade
                    set_last = IntegerSet.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(
                                    -(NUM_CASCADE_STAGES - 1)
                                ),
                            ),
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_add(
                                AffineConstantExpr.get(NUM_Q_TILES - 1),
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(-1),
                                ),
                            ),
                        ],
                        [True, False, False],
                    )
                    if_last = affine.AffineIfOp(
                        set_last, cond_operands=[tx, ty], has_else=True
                    )
                    with InsertionPoint(if_last.then_block):
                        subi = arith.SubIOp(ty, c1_h)
                        ChannelPut("cascade", gp, indices=[tx, subi])
                        ChannelPut("cascade", up, indices=[tx, subi])
                        ChannelPut("cascade", sp, indices=[tx, subi])
                        affine.AffineYieldOp([])

                    with InsertionPoint(if_last.else_block):
                        # Middle stages (1, 2)
                        set_middle = IntegerSet.get(
                            0,
                            2,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(1),
                                    AffineConstantExpr.get(-1),
                                ),
                                AffineExpr.get_add(
                                    AffineConstantExpr.get(
                                        NUM_CASCADE_STAGES - 2
                                    ),
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(-1),
                                    ),
                                ),
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_add(
                                    AffineConstantExpr.get(NUM_Q_TILES - 1),
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(0),
                                        AffineConstantExpr.get(-1),
                                    ),
                                ),
                            ],
                            [False, False, False, False],
                        )
                        if_middle = affine.AffineIfOp(
                            set_middle,
                            cond_operands=[tx, ty],
                            has_else=True,
                        )
                        with InsertionPoint(if_middle.then_block):
                            Gp_cas = AllocOp(memref_gp_l1, [], [])
                            up_cas = AllocOp(memref_up_l1, [], [])
                            sp_cas = AllocOp(memref_up_l1, [], [])
                            ChannelGet(
                                "cascade",
                                Gp_cas.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade",
                                up_cas.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade",
                                sp_cas.result,
                                indices=[tx, ty],
                            )
                            # Merge: same logic as flash attention
                            up_B_saved = AllocOp(memref_up_l1, [], [])
                            c0_i32_m = ConstantOp(i32, 0)
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [c0_i32_m, up, up_B_saved.result],
                            )
                            CallOp(
                                [],
                                "maximum_up_u_bf16",
                                [up_cas.result, up],
                            )
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_cas.result, up, r_l1_c.result],
                            )
                            r_B = AllocOp(memref_up_l1, [], [])
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_B_saved.result, up, r_B.result],
                            )
                            CallOp(
                                [],
                                "mul_r_gp",
                                [r_l1_c.result, Gp_cas.result],
                            )
                            CallOp([], "mul_r_gp", [r_B.result, gp])
                            CallOp(
                                [], "add_gp_g", [gp, Gp_cas.result]
                            )
                            sp_temp = AllocOp(memref_up_l1, [], [])
                            CallOp(
                                [], "zero_fill_sp_bf16", [sp_temp.result]
                            )
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [
                                    sp_cas.result,
                                    r_l1_c.result,
                                    sp_temp.result,
                                ],
                            )
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [sp, r_B.result, sp_temp.result],
                            )
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [c0_i32_m, sp_temp.result, sp_cas.result],
                            )
                            subi2 = arith.SubIOp(ty, c1_h)
                            ChannelPut(
                                "cascade",
                                Gp_cas.result,
                                indices=[tx, subi2],
                            )
                            ChannelPut(
                                "cascade", up, indices=[tx, subi2]
                            )
                            ChannelPut(
                                "cascade",
                                sp_cas.result,
                                indices=[tx, subi2],
                            )
                            DeallocOp(up_B_saved)
                            DeallocOp(r_B)
                            DeallocOp(sp_temp)
                            affine.AffineYieldOp([])

                        with InsertionPoint(if_middle.else_block):
                            # Stage 0 (first): receive, merge, divide, output
                            Gp_cas2 = AllocOp(memref_gp_l1, [], [])
                            up_cas2 = AllocOp(memref_up_l1, [], [])
                            sp_cas2 = AllocOp(memref_up_l1, [], [])
                            ChannelGet(
                                "cascade",
                                Gp_cas2.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade",
                                up_cas2.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade",
                                sp_cas2.result,
                                indices=[tx, ty],
                            )
                            up_B_saved2 = AllocOp(memref_up_l1, [], [])
                            c0_i32_f = ConstantOp(i32, 0)
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [c0_i32_f, up, up_B_saved2.result],
                            )
                            CallOp(
                                [],
                                "maximum_up_u_bf16",
                                [up_cas2.result, up],
                            )
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_cas2.result, up, r_l1_c.result],
                            )
                            r_B2 = AllocOp(memref_up_l1, [], [])
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_B_saved2.result, up, r_B2.result],
                            )
                            CallOp(
                                [],
                                "mul_r_gp",
                                [r_l1_c.result, Gp_cas2.result],
                            )
                            CallOp([], "mul_r_gp", [r_B2.result, gp])
                            CallOp(
                                [], "add_gp_g", [gp, Gp_cas2.result]
                            )
                            sp_temp2 = AllocOp(memref_up_l1, [], [])
                            CallOp(
                                [], "zero_fill_sp_bf16", [sp_temp2.result]
                            )
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [
                                    sp_cas2.result,
                                    r_l1_c.result,
                                    sp_temp2.result,
                                ],
                            )
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [sp, r_B2.result, sp_temp2.result],
                            )
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [c0_i32_f, sp_temp2.result, sp_cas2.result],
                            )
                            # Divide: Gp /= sp
                            CallOp(
                                [],
                                "div_gp_sp",
                                [sp_cas2.result, Gp_cas2.result],
                            )
                            DeallocOp(up_B_saved2)
                            DeallocOp(r_B2)
                            DeallocOp(sp_temp2)
                            # Output with untiling (col-major 8x8 -> row-major)
                            ChannelPut(
                                "L1ToL2",
                                Gp_cas2.result,
                                indices=[tx, 0],
                                offsets=[0, 0, 0, 0],
                                sizes=[
                                    TILE_SIZE_Q // MMUL_N,
                                    MMUL_M,
                                    DV // MMUL_M,
                                    MMUL_N,
                                ],
                                strides=[
                                    MMUL_M * MMUL_N,
                                    MMUL_N,
                                    TILE_SIZE_Q * MMUL_N,
                                    1,
                                ],
                            )
                            affine.AffineYieldOp([])
                        affine.AffineYieldOp([])

                DeallocOp(gp_buf)
                DeallocOp(up_buf)
                DeallocOp(sp_buf)
                DeallocOp(l2_out)


def merge_golden():
    """Compute the expected output of the cascade merge test in f32."""
    sqrt_dk = SQRT_DK

    # Per-stage initial values: gp=(ty+1)*5, up=ty+8, sp=ty+1
    stages = {
        3: {"gp": 20.0, "up": 11.0, "sp": 4.0},
        2: {"gp": 15.0, "up": 10.0, "sp": 3.0},
        1: {"gp": 10.0, "up": 9.0, "sp": 2.0},
        0: {"gp": 5.0, "up": 8.0, "sp": 1.0},
    }

    # Cascade merge: stage 3 -> 2 -> 1 -> 0
    gp = stages[3]["gp"]
    up_max = stages[3]["up"]
    sp_sum = stages[3]["sp"]

    for s in [2, 1, 0]:
        up_local = stages[s]["up"]
        gp_local = stages[s]["gp"]
        sp_local = stages[s]["sp"]

        max_new = max(up_local, up_max)
        r_A = exp((up_max - max_new) / sqrt_dk)
        r_B = exp((up_local - max_new) / sqrt_dk)

        gp = gp * r_A + gp_local * r_B
        sp_sum = sp_sum * r_A + sp_local * r_B
        up_max = max_new

    output = gp / sp_sum
    return output


def run_merge_test():
    """Compile, run, and verify the cascade merge test."""
    print(f"\n{'='*60}")
    print(f"Running: cascade_merge (4x4 herd, flash attention merge)")
    print(f"{'='*60}")

    mlir_module = build_merge_test()

    backend = XRTBackend(
        omit_while_true_loop=True,
        omit_pingpong="all",
        output_format="elf",
        instance_name="cascade_merge",
    )
    artifact = backend.compile(mlir_module)

    lqp_full = NUM_Q_TILES * TILE_SIZE_Q
    output = np.zeros((lqp_full, DV), dtype=bfloat16)

    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(artifact)
        results = invoker(output)

    npu_output = results[0].astype(np.float32)
    backend.unload()

    expected_val = merge_golden()
    expected = np.full_like(npu_output, expected_val)

    diff = np.abs(npu_output - expected)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    # Use bf16 tolerance (0.15 abs + 4% rel)
    tol = np.maximum(0.15, 0.04 * (np.abs(npu_output) + np.abs(expected)))
    num_wrong = int(np.sum(diff > tol))
    total_elems = npu_output.size

    print(f"  Herd size   : {NUM_Q_TILES}x{NUM_CASCADE_STAGES}")
    print(f"  Buffer types: Gp[{TILE_SIZE_Q},{DK}], up[{TILE_SIZE_Q},1], sp[{TILE_SIZE_Q},1]")
    print(f"  Golden value: {expected_val:.6f}")
    print(f"  Output[0:8] : {npu_output.flatten()[:8]}")
    print(f"  max_err = {max_err:.6f}, mean_err = {mean_err:.6f}")
    print(f"  Wrong elements: {num_wrong}/{total_elems}")

    if num_wrong == 0:
        print(f"  PASS")
        return True
    else:
        fail_pct = num_wrong / total_elems * 100
        print(f"  FAIL ({fail_pct:.1f}% errors)")
        print(f"    min output : {npu_output.min():.6f}")
        print(f"    max output : {npu_output.max():.6f}")
        print(f"    mean output: {npu_output.mean():.6f}")
        # Check per-q_tile
        for q in range(NUM_Q_TILES):
            tile = npu_output[q * TILE_SIZE_Q : (q + 1) * TILE_SIZE_Q]
            tile_mean = tile.mean()
            tile_diff = np.abs(tile - expected_val).max()
            print(f"    q_tile {q}: mean={tile_mean:.6f}, max_diff={tile_diff:.6f}")
        return False


def run_test(test_name, build_fn, expected_value):
    """Compile, run on NPU, and verify a cascade test."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}")

    mlir_module = build_fn()

    backend = XRTBackend(
        omit_while_true_loop=True,
        omit_pingpong="all",
        output_format="elf",
        instance_name=test_name,
    )
    artifact = backend.compile(mlir_module)

    output = np.zeros(BUF_SIZE, dtype=bfloat16)

    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(artifact)
        results = invoker(output)

    npu_output = results[0].astype(np.float32)
    expected = np.full(BUF_SIZE, expected_value, dtype=np.float32)
    backend.unload()

    diff = np.abs(npu_output - expected)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    num_wrong = int(np.sum(diff > 0.01))

    print(f"  Buffer size : {BUF_SIZE} bf16 elements")
    print(f"  Expected    : all elements = {expected_value}")
    print(f"  Output[0:10]: {npu_output[:10]}")
    print(f"  max_err = {max_err:.4f}, mean_err = {mean_err:.4f}")
    print(f"  Wrong elements: {num_wrong}/{BUF_SIZE}")

    if num_wrong == 0:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL")
        print(f"    min output : {npu_output.min():.4f}")
        print(f"    max output : {npu_output.max():.4f}")
        print(f"    mean output: {npu_output.mean():.4f}")
        # Show first few mismatches
        mismatch_idx = np.where(diff > 0.01)[0][:5]
        for idx in mismatch_idx:
            print(
                f"    [{idx}]: got {npu_output[idx]:.4f}, "
                f"expected {expected[idx]:.4f}, "
                f"diff {diff[idx]:.4f}"
            )
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test",
        choices=["passthrough", "accumulation", "merge", "all"],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print generated MLIR and exit (no compilation/run)",
    )
    args = parser.parse_args()

    if args.print_module_only:
        if args.test in ["passthrough", "all"]:
            print("=== Passthrough Test Module ===")
            print(build_passthrough_test())
        if args.test in ["accumulation", "all"]:
            print("\n=== Accumulation Test Module ===")
            print(build_accumulation_test())
        if args.test in ["merge", "all"]:
            print("\n=== Merge Test Module ===")
            print(build_merge_test())
        return

    os.makedirs("build_cascade", exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir("build_cascade")

    try:
        passed = 0
        total = 0

        if args.test in ["passthrough", "all"]:
            total += 1
            if run_test("cascade_passthrough", build_passthrough_test, 42.0):
                passed += 1

        if args.test in ["accumulation", "all"]:
            total += 1
            if run_test("cascade_accumulation", build_accumulation_test, 100.0):
                passed += 1

        if args.test in ["merge", "all"]:
            total += 1
            if run_merge_test():
                passed += 1

        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} tests passed")
        if passed == total:
            print("All cascade tests PASSED")
        else:
            print(f"WARNING: {total - passed} cascade test(s) FAILED")
            print("This indicates potential cascade data integrity issues")
        print(f"{'='*60}")

    finally:
        os.chdir(orig_dir)


if __name__ == "__main__":
    main()
