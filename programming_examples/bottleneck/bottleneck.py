# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
AIR Bottleneck Block Example

This implements a ResNet-style bottleneck block with residual (skip) connection
using a combination of air.channels and SHARED L1 MEMREFS for core-to-core
dataflow communication.

Architecture:
    Input (32x32x256, int8)
         │
         ├───────────────────────────────────────┐ (skip connection)
         │                                       │
         ▼                                       │
    1x1 Conv (256→64)                            │
         │                                       │
         ├──────────────┐                        │
         ▼              ▼                        │
    3x3 Conv (half)  3x3 Conv (half)             │
         │              │                        │
         └──────┬───────┘                        │
                ▼                                │
    1x1 Conv (64→256) + Add ◄────────────────────┘
                │
                ▼
    Output (32x32x256, uint8)

Each convolution is mapped to a separate AIE core. Data communication:
- External data (L3↔L2↔L1): air.channels for DMA-based transfers
- 3x3 conv → 1x1_skip: SHARED L1 MEMREFS for direct memory access

SHARED L1 MEMREF PATTERN:
Instead of using air.channels (which require DMA) between the 3x3 conv herds
and the conv1x1_skip herd, we use shared L1 buffers:
1. Allocate L1 buffers at SEGMENT level (outside all herds)
2. Pass the same buffer to multiple herds via their operands
3. Producer herds (conv3x3) write to the shared buffer
4. Consumer herd (conv1x1_skip) reads from the shared buffer

This models the AIE architecture where adjacent tiles can share L1 memory,
enabling zero-copy data transfer between neighboring cores.
"""

import argparse
import numpy as np

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

# Bottleneck block dimensions (matching mlir-aie example)
TENSOR_IN_W = 32
TENSOR_IN_H = 32
TENSOR_IN_C = 256

# Layer 1: 1x1 conv reduces channels
TENSOR_L1_IN_C = TENSOR_IN_C
TENSOR_L1_OUT_C = TENSOR_L1_IN_C // 4  # 64

# Layer 2: 3x3 conv maintains channels
TENSOR_L2_IN_C = TENSOR_L1_OUT_C  # 64
TENSOR_L2_OUT_C = TENSOR_L2_IN_C  # 64

# Layer 3: 1x1 conv restores channels + skip addition
TENSOR_L3_IN_C = TENSOR_L2_OUT_C  # 64
TENSOR_L3_OUT_C = TENSOR_L3_IN_C * 4  # 256

# Total sizes
ACTIVATIONS_IN = TENSOR_IN_W * TENSOR_IN_H * TENSOR_IN_C
ACTIVATIONS_OUT = ACTIVATIONS_IN
WEIGHTS_L1_SZ = TENSOR_L1_IN_C * TENSOR_L1_OUT_C
WEIGHTS_L2_SZ = 3 * 3 * TENSOR_L2_IN_C * TENSOR_L2_OUT_C
WEIGHTS_L3_SZ = TENSOR_L3_IN_C * TENSOR_L3_OUT_C
TOTAL_WEIGHTS = WEIGHTS_L1_SZ + WEIGHTS_L2_SZ + WEIGHTS_L3_SZ


@module_builder
def build_module():
    """
    Build the AIR bottleneck dataflow module.

    Returns:
        The constructed MLIR module.
    """
    # Type definitions
    index_type = IndexType.get()
    i8 = IntegerType.get_signless(8)
    i32 = IntegerType.get_signless(32)

    # L3 (external memory) types
    # Note: Using signless i8 for all integer types because MLIR's arith.constant
    # requires signless integer return types. The actual signedness is handled
    # by the kernel functions.
    l3_act_in_ty = MemRefType.get((ACTIVATIONS_IN,), i8)
    l3_wts_ty = MemRefType.get((TOTAL_WEIGHTS,), i8)
    l3_act_out_ty = MemRefType.get((ACTIVATIONS_OUT,), i8)

    # L2 memory space
    l2_mem_space = IntegerAttr.get(i32, MemorySpace.L2)

    # L1 memory space
    l1_mem_space = IntegerAttr.get(i32, MemorySpace.L1)

    # Per-row tile types (processing one row at a time for depth-first dataflow)
    # Layer 1 input: one row of 32 pixels with 256 input channels
    l1_layer1_in_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L1_IN_C), i8, memory_space=l1_mem_space
    )
    l1_wts_layer1_ty = MemRefType.get((WEIGHTS_L1_SZ,), i8, memory_space=l1_mem_space)
    l1_layer1_out_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L1_OUT_C), i8, memory_space=l1_mem_space
    )

    # Layer 2 (3x3 conv) types
    l1_layer2_in_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L2_IN_C), i8, memory_space=l1_mem_space
    )
    # L1 weights for layer 2 (36KB fits in AIE2's 64KB L1)
    l1_wts_layer2_ty = MemRefType.get((WEIGHTS_L2_SZ,), i8, memory_space=l1_mem_space)
    # Each 3x3 core produces half the output channels
    l1_layer2_out_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L2_OUT_C // 2), i8, memory_space=l1_mem_space
    )

    # Layer 3 (1x1 conv + skip) types
    l1_layer3_in_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L3_IN_C // 2), i8, memory_space=l1_mem_space
    )
    l1_wts_layer3_ty = MemRefType.get((WEIGHTS_L3_SZ,), i8, memory_space=l1_mem_space)
    l1_layer3_out_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L3_OUT_C), i8, memory_space=l1_mem_space
    )

    # L2 buffer types for skip connection
    l2_skip_buf_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L1_IN_C), i8, memory_space=l2_mem_space
    )

    # L2 buffer type for output
    l2_out_buf_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L3_OUT_C), i8, memory_space=l2_mem_space
    )

    # L2 buffer types for weight staging
    l2_wts_layer1_ty = MemRefType.get((WEIGHTS_L1_SZ,), i8, memory_space=l2_mem_space)
    l2_wts_layer2_ty = MemRefType.get((WEIGHTS_L2_SZ,), i8, memory_space=l2_mem_space)
    l2_wts_layer3_ty = MemRefType.get((WEIGHTS_L3_SZ,), i8, memory_space=l2_mem_space)

    # Declare external convolution kernel functions
    # These would be linked from compiled convolution kernels
    conv2dk1_i8_func = FuncOp(
        "conv2dk1_i8",
        (
            [
                l1_layer1_in_ty,
                l1_wts_layer1_ty,
                l1_layer1_out_ty,
                i32,
                i32,
                i32,
                i32,
            ],  # width, in_c, out_c, scale
            [],
        ),
        visibility="private",
    )
    conv2dk1_i8_func.attributes["link_with"] = StringAttr.get("conv2dk1.o")
    conv2dk1_i8_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # conv2dk3 uses L1 weights (36KB fits in AIE2's 64KB L1)
    conv2dk3_ui8_func = FuncOp(
        "conv2dk3_ui8",
        (
            [
                l1_layer2_in_ty,
                l1_layer2_in_ty,
                l1_layer2_in_ty,
                l1_wts_layer2_ty,
                l1_layer2_out_ty,  # Weights in L1 (via broadcast channel)
                i32,
                i32,
                i32,
                i32,
                i32,
                i32,
                i32,
                i32,
            ],  # width, in_c, out_c, kernel_h, kernel_w, row_pos, scale, channel_offset
            [],
        ),
        visibility="private",
    )
    conv2dk3_ui8_func.attributes["link_with"] = StringAttr.get("conv2dk3.o")
    conv2dk3_ui8_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    conv2dk1_skip_i8_func = FuncOp(
        "conv2dk1_skip_i8",
        (
            [
                l1_layer3_in_ty,
                l1_layer3_in_ty,
                l1_wts_layer3_ty,
                l1_layer3_out_ty,
                l1_layer1_in_ty,  # skip input
                i32,
                i32,
                i32,
                i32,
                i32,
            ],  # width, in_c, out_c, scale, skip_scale
            [],
        ),
        visibility="private",
    )
    conv2dk1_skip_i8_func.attributes["link_with"] = StringAttr.get("conv2dk1_skip.o")
    conv2dk1_skip_i8_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # =========================================================================
    # AIR Channel Declarations
    # =========================================================================
    # Channels for dataflow between cores - these model hardware FIFOs

    # L3 → L2: Input activations and weights from host
    channel("L3ToL2_ActIn", size=[1, 1])
    channel("L3ToL2_WtsL1", size=[1, 1])  # Weights for layer 1
    channel("L3ToL2_WtsL2", size=[1, 1])  # Weights for layer 2
    channel("L3ToL2_WtsL3", size=[1, 1])  # Weights for layer 3

    # L2 → L1: Weights to conv herds
    channel("L2ToL1_WtsL1", size=[1, 1])  # To conv1x1_reduce
    channel("L2ToL1_WtsL3", size=[1, 1])  # To conv1x1_skip
    # Broadcast channel for L2 weights to both 3x3 conv cores
    Channel("L2ToL1_WtsL2", size=[1, 1], broadcast_shape=[2, 1])

    # L2 → L1: Input activation to first conv (also feeds skip buffer)
    channel("L2ToL1_ActIn", size=[1, 1])

    # L1 ↔ L1: Inter-core dataflow channels
    # Broadcast from 1x1 conv to both 3x3 conv cores
    Channel("L1ToL1_Conv1ToConv3x3", size=[1, 1], broadcast_shape=[2, 1])

    # Skip buffer → final conv (via L2)
    channel("L1ToL1_SkipBufToSkip", size=[1, 1])

    # L1 → L2: Output from final conv
    channel("L1ToL2_ActOut", size=[1, 1])

    # L2 → L3: Output to host
    channel("L2ToL3_ActOut", size=[1, 1])

    @FuncOp.from_py_func(l3_act_in_ty, l3_wts_ty, l3_act_out_ty)
    def bottleneck_block(arg0, arg1, arg2):
        """
        Bottleneck block top-level function.

        Args:
            arg0: Input activations (32x32x256, int8)
            arg1: Weights (all layers concatenated)
            arg2: Output activations (32x32x256, uint8)
        """
        c1 = ConstantOp(index_type, 1)

        @launch(operands=[arg0, arg1, arg2], sizes=[c1, c1])
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_act_in,
            l3_wts,
            l3_act_out,
        ):
            c0 = ConstantOp(index_type, 0)
            c1_idx = ConstantOp(index_type, 1)

            # DMA input data from L3 to L2 channels
            # Send activations - one row at a time for depth-first processing
            for row_idx in range_(0, TENSOR_IN_H):
                row_offset_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(TENSOR_IN_W * TENSOR_L1_IN_C),
                        )
                    ],
                )
                row_offset = affine_apply(row_offset_map, [row_idx])
                ChannelPut(
                    "L3ToL2_ActIn",
                    l3_act_in,
                    offsets=[row_offset],
                    sizes=[TENSOR_IN_W * TENSOR_L1_IN_C],
                    strides=[1],
                )
                yield_([])

            # Send weights - separate channels for each layer
            # Layer 1 weights
            ChannelPut(
                "L3ToL2_WtsL1",
                l3_wts,
                offsets=[0],
                sizes=[WEIGHTS_L1_SZ],
                strides=[1],
            )
            # Layer 2 weights
            ChannelPut(
                "L3ToL2_WtsL2",
                l3_wts,
                offsets=[WEIGHTS_L1_SZ],
                sizes=[WEIGHTS_L2_SZ],
                strides=[1],
            )
            # Layer 3 weights
            ChannelPut(
                "L3ToL2_WtsL3",
                l3_wts,
                offsets=[WEIGHTS_L1_SZ + WEIGHTS_L2_SZ],
                sizes=[WEIGHTS_L3_SZ],
                strides=[1],
            )

            @segment(
                name="bottleneck_seg",
                operands=[l3_act_in, l3_wts, l3_act_out],
            )
            def segment_body(seg_act_in, seg_wts, seg_act_out):
                # Allocate L2 buffers
                l2_skip_buf = AllocOp(l2_skip_buf_ty, [], [])

                # Allocate L2 weight buffers for staging
                l2_wts_l1 = AllocOp(l2_wts_layer1_ty, [], [])
                l2_wts_l2 = AllocOp(l2_wts_layer2_ty, [], [])
                l2_wts_l3 = AllocOp(l2_wts_layer3_ty, [], [])

                # Receive weights from L3 to L2 buffers
                ChannelGet("L3ToL2_WtsL1", l2_wts_l1)
                ChannelGet("L3ToL2_WtsL2", l2_wts_l2)
                ChannelGet("L3ToL2_WtsL3", l2_wts_l3)

                # Forward weights from L2 to L1 for all conv herds
                ChannelPut("L2ToL1_WtsL1", l2_wts_l1)
                ChannelPut(
                    "L2ToL1_WtsL2", l2_wts_l2
                )  # Broadcast to both 3x3 conv cores
                ChannelPut("L2ToL1_WtsL3", l2_wts_l3)

                # =============================================================
                # SHARED L1 BUFFERS for 3x3 conv → conv1x1_skip communication
                # Allocated at segment level and passed to multiple herds
                # This enables direct memory access between neighboring AIE tiles
                # =============================================================
                shared_l1_conv3x3_A_out = AllocOp(
                    l1_layer2_out_ty, [], []
                )  # Core A output
                shared_l1_conv3x3_B_out = AllocOp(
                    l1_layer2_out_ty, [], []
                )  # Core B output

                # =============================================================
                # Runtime Parameters (RTPs) - defined at segment level
                # These are passed to herds as operands and lowered as RTPs
                # NOTE: Must use arith.ConstantOp.create_index() for AIR RTP
                # =============================================================
                # Scale for 1x1 reduce conv (rtp2[0] in reference)
                rtp_scale_conv1 = arith.ConstantOp.create_index(1)
                # Scale for 1x1 skip conv (rtp4[0] in reference)
                rtp_scale_conv3 = arith.ConstantOp.create_index(1)
                # Skip scale (rtp4[1] in reference)
                rtp_skip_scale = arith.ConstantOp.create_index(0)

                # =============================================================
                # Herd 0: 1x1 Convolution (dimension reduction: 256→64)
                # RTP: scale parameter passed as operand
                # =============================================================
                @herd(name="conv1x1_reduce", sizes=[1, 1], operands=[rtp_scale_conv1])
                def herd_conv1_body(h0_x, h0_y, h0_sx, h0_sy, scale):
                    c0_h = ConstantOp(index_type, 0)
                    c1_h = ConstantOp(index_type, 1)

                    # Allocate L1 buffers
                    wts = AllocOp(l1_wts_layer1_ty, [], [])
                    act_in = AllocOp(l1_layer1_in_ty, [], [])
                    act_out = AllocOp(l1_layer1_out_ty, [], [])

                    # Receive weights from L2
                    ChannelGet("L2ToL1_WtsL1", wts)

                    # Convert RTP scale from index to i32 for function call
                    scale_i32 = arith.index_cast(i32, scale)

                    # Process each row
                    for _ in range_(0, TENSOR_IN_H):
                        # Get input activation row
                        ChannelGet("L2ToL1_ActIn", act_in)

                        # Call 1x1 convolution kernel
                        # Note: scale comes from RTP (passed as herd operand, cast to i32)
                        width = ConstantOp(i32, TENSOR_IN_W)
                        in_c = ConstantOp(i32, TENSOR_L1_IN_C)
                        out_c = ConstantOp(i32, TENSOR_L1_OUT_C)
                        CallOp(
                            conv2dk1_i8_func,
                            [act_in, wts, act_out, width, in_c, out_c, scale_i32],
                        )

                        # Send output to 3x3 conv cores (broadcast)
                        ChannelPut("L1ToL1_Conv1ToConv3x3", act_out)

                        yield_([])

                    DeallocOp(wts)
                    DeallocOp(act_in)
                    DeallocOp(act_out)

                herd_conv1_body.attributes["link_with"] = StringAttr.get("conv2dk1.o")

                # =============================================================
                # Herd 1: 3x3 Convolution - Core A (first half of output channels)
                # Receives L2 weights via broadcast channel to L1
                # Output written to SHARED L1 buffer (passed as operand)
                # Uses 4-buffer rotation pattern for proper sliding window
                # =============================================================
                @herd(
                    name="conv3x3_coreA",
                    sizes=[1, 1],
                    operands=[shared_l1_conv3x3_A_out.result],
                )
                def herd_conv3x3_A_body(h1_x, h1_y, h1_sx, h1_sy, shared_out_buf):
                    c0_h = ConstantOp(index_type, 0)
                    c1_h = ConstantOp(index_type, 1)

                    # Allocate L1 buffers - 4 rows for sliding window rotation
                    row_buf_0 = AllocOp(l1_layer2_in_ty, [], [])
                    row_buf_1 = AllocOp(l1_layer2_in_ty, [], [])
                    row_buf_2 = AllocOp(l1_layer2_in_ty, [], [])
                    row_buf_3 = AllocOp(l1_layer2_in_ty, [], [])
                    # NOTE: Output buffer is shared_out_buf (passed from segment level)

                    # Allocate L1 weight buffer and receive via broadcast channel (index 0)
                    wts = AllocOp(l1_wts_layer2_ty, [], [])
                    ChannelGet("L2ToL1_WtsL2", wts, indices=[0, 0])

                    # Pre-load first two rows (using broadcast index 0)
                    ChannelGet(
                        "L1ToL1_Conv1ToConv3x3", row_buf_0, indices=[0, 0]
                    )  # input row 0
                    ChannelGet(
                        "L1ToL1_Conv1ToConv3x3", row_buf_1, indices=[0, 0]
                    )  # input row 1

                    # Constants for conv2dk3 calls
                    width = ConstantOp(i32, TENSOR_IN_W)
                    in_c = ConstantOp(i32, TENSOR_L2_IN_C)
                    out_c = ConstantOp(i32, TENSOR_L2_OUT_C)
                    k_h = ConstantOp(i32, 3)
                    k_w = ConstantOp(i32, 3)
                    row_pos_top = ConstantOp(i32, 0)  # top edge (pad top)
                    row_pos_mid = ConstantOp(i32, 1)  # middle
                    row_pos_bot = ConstantOp(i32, 2)  # bottom edge (pad bottom)
                    scale = ConstantOp(i32, 11)
                    chan_offset = ConstantOp(i32, 0)  # first half of channels

                    # Output row 0 (top padding): uses [row0, row0, row1]
                    CallOp(
                        conv2dk3_ui8_func,
                        [
                            row_buf_0,
                            row_buf_0,
                            row_buf_1,
                            wts,
                            shared_out_buf,
                            width,
                            in_c,
                            out_c,
                            k_h,
                            k_w,
                            row_pos_top,
                            scale,
                            chan_offset,
                        ],
                    )

                    # Process middle rows 1-30 in groups of 4 (7 full groups = 28 rows)
                    # Pattern: buffers rotate [0,1,2] -> [1,2,3] -> [2,3,0] -> [3,0,1] -> repeat
                    for _ in range_(0, 7):  # 7 groups of 4
                        # Group iteration 0: sliding window [buf0, buf1, buf2]
                        ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_2, indices=[0, 0])
                        CallOp(
                            conv2dk3_ui8_func,
                            [
                                row_buf_0,
                                row_buf_1,
                                row_buf_2,
                                wts,
                                shared_out_buf,
                                width,
                                in_c,
                                out_c,
                                k_h,
                                k_w,
                                row_pos_mid,
                                scale,
                                chan_offset,
                            ],
                        )

                        # Group iteration 1: sliding window [buf1, buf2, buf3]
                        ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_3, indices=[0, 0])
                        CallOp(
                            conv2dk3_ui8_func,
                            [
                                row_buf_1,
                                row_buf_2,
                                row_buf_3,
                                wts,
                                shared_out_buf,
                                width,
                                in_c,
                                out_c,
                                k_h,
                                k_w,
                                row_pos_mid,
                                scale,
                                chan_offset,
                            ],
                        )

                        # Group iteration 2: sliding window [buf2, buf3, buf0] (buf0 gets new data)
                        ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_0, indices=[0, 0])
                        CallOp(
                            conv2dk3_ui8_func,
                            [
                                row_buf_2,
                                row_buf_3,
                                row_buf_0,
                                wts,
                                shared_out_buf,
                                width,
                                in_c,
                                out_c,
                                k_h,
                                k_w,
                                row_pos_mid,
                                scale,
                                chan_offset,
                            ],
                        )

                        # Group iteration 3: sliding window [buf3, buf0, buf1] (buf1 gets new data)
                        ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_1, indices=[0, 0])
                        CallOp(
                            conv2dk3_ui8_func,
                            [
                                row_buf_3,
                                row_buf_0,
                                row_buf_1,
                                wts,
                                shared_out_buf,
                                width,
                                in_c,
                                out_c,
                                k_h,
                                k_w,
                                row_pos_mid,
                                scale,
                                chan_offset,
                            ],
                        )
                        yield_([])

                    # Handle remaining 2 rows (rows 29 and 30 of middle section)
                    # After 7 groups (28 iterations), buffer state is: buf0=newest-1, buf1=newest
                    # Continue rotation pattern:

                    # Row 29: sliding window [buf0, buf1, buf2]
                    ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_2, indices=[0, 0])
                    CallOp(
                        conv2dk3_ui8_func,
                        [
                            row_buf_0,
                            row_buf_1,
                            row_buf_2,
                            wts,
                            shared_out_buf,
                            width,
                            in_c,
                            out_c,
                            k_h,
                            k_w,
                            row_pos_mid,
                            scale,
                            chan_offset,
                        ],
                    )

                    # Row 30: sliding window [buf1, buf2, buf3]
                    ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_3, indices=[0, 0])
                    CallOp(
                        conv2dk3_ui8_func,
                        [
                            row_buf_1,
                            row_buf_2,
                            row_buf_3,
                            wts,
                            shared_out_buf,
                            width,
                            in_c,
                            out_c,
                            k_h,
                            k_w,
                            row_pos_mid,
                            scale,
                            chan_offset,
                        ],
                    )

                    # Output row 31 (bottom padding): uses [row30, row31, row31]
                    # buf2=row30, buf3=row31
                    CallOp(
                        conv2dk3_ui8_func,
                        [
                            row_buf_2,
                            row_buf_3,
                            row_buf_3,
                            wts,
                            shared_out_buf,
                            width,
                            in_c,
                            out_c,
                            k_h,
                            k_w,
                            row_pos_bot,
                            scale,
                            chan_offset,
                        ],
                    )

                    # Deallocate LOCAL L1 buffers (NOT shared buffer)
                    DeallocOp(wts)
                    DeallocOp(row_buf_0)
                    DeallocOp(row_buf_1)
                    DeallocOp(row_buf_2)
                    DeallocOp(row_buf_3)

                herd_conv3x3_A_body.attributes["link_with"] = StringAttr.get(
                    "conv2dk3.o"
                )

                # =============================================================
                # Herd 2: 3x3 Convolution - Core B (second half of output channels)
                # Receives L2 weights via broadcast channel to L1
                # Output written to SHARED L1 buffer (passed as operand)
                # Uses 4-buffer rotation pattern for proper sliding window
                # =============================================================
                @herd(
                    name="conv3x3_coreB",
                    sizes=[1, 1],
                    operands=[shared_l1_conv3x3_B_out.result],
                )
                def herd_conv3x3_B_body(h2_x, h2_y, h2_sx, h2_sy, shared_out_buf):
                    c0_h = ConstantOp(index_type, 0)
                    c1_h = ConstantOp(index_type, 1)

                    # Allocate L1 buffers - 4 rows for sliding window rotation
                    row_buf_0 = AllocOp(l1_layer2_in_ty, [], [])
                    row_buf_1 = AllocOp(l1_layer2_in_ty, [], [])
                    row_buf_2 = AllocOp(l1_layer2_in_ty, [], [])
                    row_buf_3 = AllocOp(l1_layer2_in_ty, [], [])
                    # NOTE: Output buffer is shared_out_buf (passed from segment level)

                    # Allocate L1 weight buffer and receive via broadcast channel (index 1)
                    wts = AllocOp(l1_wts_layer2_ty, [], [])
                    ChannelGet("L2ToL1_WtsL2", wts, indices=[1, 0])

                    # Pre-load first two rows (using broadcast index 1)
                    ChannelGet(
                        "L1ToL1_Conv1ToConv3x3", row_buf_0, indices=[1, 0]
                    )  # input row 0
                    ChannelGet(
                        "L1ToL1_Conv1ToConv3x3", row_buf_1, indices=[1, 0]
                    )  # input row 1

                    # Constants for conv2dk3 calls
                    width = ConstantOp(i32, TENSOR_IN_W)
                    in_c = ConstantOp(i32, TENSOR_L2_IN_C)
                    out_c = ConstantOp(i32, TENSOR_L2_OUT_C)
                    k_h = ConstantOp(i32, 3)
                    k_w = ConstantOp(i32, 3)
                    row_pos_top = ConstantOp(i32, 0)  # top edge (pad top)
                    row_pos_mid = ConstantOp(i32, 1)  # middle
                    row_pos_bot = ConstantOp(i32, 2)  # bottom edge (pad bottom)
                    scale = ConstantOp(i32, 11)
                    chan_offset = ConstantOp(
                        i32, TENSOR_L2_OUT_C // 2
                    )  # second half of channels

                    # Output row 0 (top padding): uses [row0, row0, row1]
                    CallOp(
                        conv2dk3_ui8_func,
                        [
                            row_buf_0,
                            row_buf_0,
                            row_buf_1,
                            wts,
                            shared_out_buf,
                            width,
                            in_c,
                            out_c,
                            k_h,
                            k_w,
                            row_pos_top,
                            scale,
                            chan_offset,
                        ],
                    )

                    # Process middle rows 1-30 in groups of 4 (7 full groups = 28 rows)
                    # Pattern: buffers rotate [0,1,2] -> [1,2,3] -> [2,3,0] -> [3,0,1] -> repeat
                    for _ in range_(0, 7):  # 7 groups of 4
                        # Group iteration 0: sliding window [buf0, buf1, buf2]
                        ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_2, indices=[1, 0])
                        CallOp(
                            conv2dk3_ui8_func,
                            [
                                row_buf_0,
                                row_buf_1,
                                row_buf_2,
                                wts,
                                shared_out_buf,
                                width,
                                in_c,
                                out_c,
                                k_h,
                                k_w,
                                row_pos_mid,
                                scale,
                                chan_offset,
                            ],
                        )

                        # Group iteration 1: sliding window [buf1, buf2, buf3]
                        ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_3, indices=[1, 0])
                        CallOp(
                            conv2dk3_ui8_func,
                            [
                                row_buf_1,
                                row_buf_2,
                                row_buf_3,
                                wts,
                                shared_out_buf,
                                width,
                                in_c,
                                out_c,
                                k_h,
                                k_w,
                                row_pos_mid,
                                scale,
                                chan_offset,
                            ],
                        )

                        # Group iteration 2: sliding window [buf2, buf3, buf0] (buf0 gets new data)
                        ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_0, indices=[1, 0])
                        CallOp(
                            conv2dk3_ui8_func,
                            [
                                row_buf_2,
                                row_buf_3,
                                row_buf_0,
                                wts,
                                shared_out_buf,
                                width,
                                in_c,
                                out_c,
                                k_h,
                                k_w,
                                row_pos_mid,
                                scale,
                                chan_offset,
                            ],
                        )

                        # Group iteration 3: sliding window [buf3, buf0, buf1] (buf1 gets new data)
                        ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_1, indices=[1, 0])
                        CallOp(
                            conv2dk3_ui8_func,
                            [
                                row_buf_3,
                                row_buf_0,
                                row_buf_1,
                                wts,
                                shared_out_buf,
                                width,
                                in_c,
                                out_c,
                                k_h,
                                k_w,
                                row_pos_mid,
                                scale,
                                chan_offset,
                            ],
                        )
                        yield_([])

                    # Handle remaining 2 rows (rows 29 and 30 of middle section)
                    # After 7 groups (28 iterations), buffer state is: buf0=newest-1, buf1=newest
                    # Continue rotation pattern:

                    # Row 29: sliding window [buf0, buf1, buf2]
                    ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_2, indices=[1, 0])
                    CallOp(
                        conv2dk3_ui8_func,
                        [
                            row_buf_0,
                            row_buf_1,
                            row_buf_2,
                            wts,
                            shared_out_buf,
                            width,
                            in_c,
                            out_c,
                            k_h,
                            k_w,
                            row_pos_mid,
                            scale,
                            chan_offset,
                        ],
                    )

                    # Row 30: sliding window [buf1, buf2, buf3]
                    ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_3, indices=[1, 0])
                    CallOp(
                        conv2dk3_ui8_func,
                        [
                            row_buf_1,
                            row_buf_2,
                            row_buf_3,
                            wts,
                            shared_out_buf,
                            width,
                            in_c,
                            out_c,
                            k_h,
                            k_w,
                            row_pos_mid,
                            scale,
                            chan_offset,
                        ],
                    )

                    # Output row 31 (bottom padding): uses [row30, row31, row31]
                    # buf2=row30, buf3=row31
                    CallOp(
                        conv2dk3_ui8_func,
                        [
                            row_buf_2,
                            row_buf_3,
                            row_buf_3,
                            wts,
                            shared_out_buf,
                            width,
                            in_c,
                            out_c,
                            k_h,
                            k_w,
                            row_pos_bot,
                            scale,
                            chan_offset,
                        ],
                    )

                    # Deallocate LOCAL L1 buffers (NOT shared buffer)
                    DeallocOp(wts)
                    DeallocOp(row_buf_0)
                    DeallocOp(row_buf_1)
                    DeallocOp(row_buf_2)
                    DeallocOp(row_buf_3)

                herd_conv3x3_B_body.attributes["link_with"] = StringAttr.get(
                    "conv2dk3.o"
                )

                # =============================================================
                # Herd 3: 1x1 Convolution + Skip Addition (restore: 64→256)
                # RTP: scale and skip_scale parameters passed as operands
                # SHARED L1: reads from shared_l1_conv3x3_A_out and shared_l1_conv3x3_B_out
                # =============================================================
                @herd(
                    name="conv1x1_skip",
                    sizes=[1, 1],
                    operands=[
                        rtp_scale_conv3,
                        rtp_skip_scale,
                        shared_l1_conv3x3_A_out.result,  # Shared buffer from conv3x3_coreA
                        shared_l1_conv3x3_B_out.result,  # Shared buffer from conv3x3_coreB
                    ],
                )
                def herd_skip_body(
                    h3_x,
                    h3_y,
                    h3_sx,
                    h3_sy,
                    scale,
                    skip_scale,
                    shared_in_A,
                    shared_in_B,
                ):
                    c0_h = ConstantOp(index_type, 0)
                    c1_h = ConstantOp(index_type, 1)

                    # Allocate L1 buffers (NOT for 3x3 conv outputs - those are shared)
                    wts = AllocOp(l1_wts_layer3_ty, [], [])
                    skip_in = AllocOp(l1_layer1_in_ty, [], [])
                    act_out = AllocOp(l1_layer3_out_ty, [], [])

                    # Receive weights from L2
                    ChannelGet("L2ToL1_WtsL3", wts)

                    # Convert RTP scale and skip_scale from index to i32 for function call
                    scale_i32 = arith.index_cast(i32, scale)
                    skip_scale_i32 = arith.index_cast(i32, skip_scale)

                    for _ in range_(0, TENSOR_IN_H):
                        # NOTE: Input from 3x3 conv cores is read directly from SHARED L1 buffers
                        # No ChannelGet needed - just use shared_in_A and shared_in_B

                        # Get skip connection input (this still comes via channel)
                        ChannelGet("L1ToL1_SkipBufToSkip", skip_in)

                        # Call 1x1 conv + skip addition kernel
                        # Note: scale and skip_scale come from RTP (passed as herd operands, cast to i32)
                        # Note: shared_in_A and shared_in_B are SHARED L1 buffers written by 3x3 conv herds
                        width = ConstantOp(i32, TENSOR_IN_W)
                        in_c = ConstantOp(i32, TENSOR_L3_IN_C)
                        out_c = ConstantOp(i32, TENSOR_L3_OUT_C)
                        CallOp(
                            conv2dk1_skip_i8_func,
                            [
                                shared_in_A,
                                shared_in_B,
                                wts,
                                act_out,
                                skip_in,
                                width,
                                in_c,
                                out_c,
                                scale_i32,
                                skip_scale_i32,
                            ],
                        )

                        # Send output
                        ChannelPut("L1ToL2_ActOut", act_out)

                        yield_([])

                    DeallocOp(wts)
                    DeallocOp(skip_in)
                    DeallocOp(act_out)

                herd_skip_body.attributes["link_with"] = StringAttr.get(
                    "conv2dk1_skip.o"
                )

                # Allocate L2 output buffer
                l2_out_buf = AllocOp(l2_out_buf_ty, [], [])

                # L2 → L1 data movement for skip buffer
                # Input goes to both first conv and skip buffer in parallel
                for row_idx in range_(0, TENSOR_IN_H):
                    # Receive activation row from L3
                    ChannelGet("L3ToL2_ActIn", l2_skip_buf)

                    # Forward to first conv
                    ChannelPut("L2ToL1_ActIn", l2_skip_buf)

                    # Also forward to skip connection (delayed)
                    ChannelPut("L1ToL1_SkipBufToSkip", l2_skip_buf)

                    yield_([])

                # L1 → L2 → L3 output data movement
                # Receive output from final conv and forward to L3
                for row_idx in range_(0, TENSOR_IN_H):
                    ChannelGet("L1ToL2_ActOut", l2_out_buf)
                    ChannelPut("L2ToL3_ActOut", l2_out_buf)
                    yield_([])

                DeallocOp(l2_skip_buf)
                DeallocOp(l2_out_buf)

            # Receive output from segment and DMA to L3
            for row_idx in range_(0, TENSOR_IN_H):
                row_offset_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(TENSOR_IN_W * TENSOR_L3_OUT_C),
                        )
                    ],
                )
                row_offset = affine_apply(row_offset_map, [row_idx])
                ChannelGet(
                    "L2ToL3_ActOut",
                    l3_act_out,
                    offsets=[row_offset],
                    sizes=[TENSOR_IN_W * TENSOR_L3_OUT_C],
                    strides=[1],
                )
                yield_([])


# =============================================================================
# Testbench Helper Functions
# =============================================================================


def reorder_mat(in_tensor: np.ndarray, out_layout: str, in_layout: str) -> np.ndarray:
    """
    Reorder tensor data layout.

    Implements data layout transformation similar to mlir-aie DataShaper.
    """
    if in_layout == "CYX" and out_layout == "YCXC8":
        # Input: (C, Y, X) -> Output: (Y, C//8, X, 8)
        C, Y, X = in_tensor.shape
        assert C % 8 == 0, f"Channel dimension {C} must be divisible by 8"
        # Reshape to (C//8, 8, Y, X) then transpose to (Y, C//8, X, 8)
        reshaped = in_tensor.reshape(C // 8, 8, Y, X)
        return reshaped.transpose(2, 0, 3, 1).copy()

    elif in_layout == "OIYX" and out_layout == "OIYXI8O8":
        # Weight reorder: (O, I, Y, X) -> (O//8, I//8, Y, X, 8, 8)
        O, I, Y, X = in_tensor.shape
        assert O % 8 == 0 and I % 8 == 0, f"O={O}, I={I} must be divisible by 8"
        # Reshape to (O//8, 8, I//8, 8, Y, X)
        reshaped = in_tensor.reshape(O // 8, 8, I // 8, 8, Y, X)
        # Transpose to (O//8, I//8, Y, X, 8, 8) -> dim order: (0, 2, 4, 5, 3, 1)
        return reshaped.transpose(0, 2, 4, 5, 3, 1).copy()

    elif in_layout == "YCXD" and out_layout == "CDYX":
        # Output reorder: (Y, C//8, X, 8) -> (C//8, 8, Y, X)
        Y, C8, X, D = in_tensor.shape
        # Transpose (Y, C//8, X, 8) -> (C//8, 8, Y, X)
        return in_tensor.transpose(1, 3, 0, 2).copy()

    else:
        raise NotImplementedError(
            f"Layout conversion {in_layout} -> {out_layout} not implemented"
        )


def compute_golden_reference(
    input_act: np.ndarray,
    weight1: np.ndarray,
    weight2: np.ndarray,
    weight3: np.ndarray,
) -> np.ndarray:
    """
    Compute golden reference for bottleneck block using NumPy.

    Implements the same quantized bottleneck computation as the reference PyTorch model:
    - 1x1 conv (256->64) + ReLU
    - 3x3 conv (64->64) + ReLU
    - 1x1 conv (64->256) + skip connection + ReLU

    Args:
        input_act: Input activation (C, H, W), int8
        weight1: Layer 1 weights (64, 256, 1, 1), int8
        weight2: Layer 2 weights (64, 64, 3, 3), int8
        weight3: Layer 3 weights (256, 64, 1, 1), int8

    Returns:
        Output activation (C, H, W), uint8
    """
    # Scale factors (matching reference design)
    inp_scale1 = 0.5
    inp_scale2 = 0.5
    inp_scale3 = 0.5
    inp_scale4 = 0.5
    weight_scale1 = 0.5
    weight_scale2 = 0.5
    weight_scale3 = 0.5

    min_val = 0
    max_val = 255

    # Input shape: (C, H, W)
    C_in, H, W = input_act.shape

    # Store original input for skip connection
    skip_input = input_act.astype(np.float32)

    # Layer 1: 1x1 conv (256->64)
    # weight1 shape: (64, 256, 1, 1)
    O1, I1, _, _ = weight1.shape
    conv1_out = np.zeros((O1, H, W), dtype=np.float32)
    for h in range(H):
        for w in range(W):
            for o in range(O1):
                conv1_out[o, h, w] = np.sum(
                    input_act[:, h, w].astype(np.float32)
                    * weight1[o, :, 0, 0].astype(np.float32)
                )

    # Apply scaling and ReLU
    conv1_scaled = conv1_out * inp_scale1 * weight_scale1
    relu1_out = np.clip(
        np.round(np.maximum(conv1_scaled, 0) / inp_scale2), min_val, max_val
    )

    # Layer 2: 3x3 conv (64->64) with zero padding
    O2, I2, K2, _ = weight2.shape
    conv2_out = np.zeros((O2, H, W), dtype=np.float32)
    relu1_padded = np.pad(
        relu1_out, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0
    )

    for h in range(H):
        for w in range(W):
            for o in range(O2):
                val = 0.0
                for ky in range(3):
                    for kx in range(3):
                        for i in range(I2):
                            val += (
                                relu1_padded[i, h + ky, w + kx] * weight2[o, i, ky, kx]
                            )
                conv2_out[o, h, w] = val

    # Apply scaling and ReLU
    conv2_scaled = conv2_out * inp_scale2 * weight_scale2
    relu2_out = np.clip(
        np.round(np.maximum(conv2_scaled, 0) / inp_scale3), min_val, max_val
    )

    # Layer 3: 1x1 conv (64->256)
    O3, I3, _, _ = weight3.shape
    conv3_out = np.zeros((O3, H, W), dtype=np.float32)
    for h in range(H):
        for w in range(W):
            for o in range(O3):
                conv3_out[o, h, w] = np.sum(
                    relu2_out[:, h, w].astype(np.float32)
                    * weight3[o, :, 0, 0].astype(np.float32)
                )

    # Apply scaling
    conv3_scaled = conv3_out * inp_scale3 * weight_scale3
    same_scale_init = np.clip(np.round(conv3_scaled / inp_scale1), -128, 127)

    # Skip connection: add original input
    skip_add = inp_scale1 * (same_scale_init + skip_input)

    # Final output with ReLU
    # Note: hardware kernel outputs the integer quantized value directly (uint8),
    # so we should NOT multiply by inp_scale4 here. The mlir-aie reference test.py
    # applies the scale AFTER reading from hardware (line 184: out.numpy() * inp_scale4).
    final_out = np.clip(np.round(skip_add / inp_scale4), min_val, max_val)

    return final_out.astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="bottleneck.py",
        description="Builds, runs, and tests the bottleneck block example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print MLIR IR and exit",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure whether to run after compile",
    )
    parser.add_argument(
        "--debug-ir",
        action="store_true",
        dest="debug_ir",
        help="Enable debug mode to emit IR after each pass",
    )
    args = parser.parse_args()

    # Build the module
    mlir_module = build_module()

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # ==========================================================================
    # Run test
    # ==========================================================================

    if args.compile_mode == "compile-and-run":
        # ======================================================================
        # Prepare test data (matching reference design test.py)
        # Only computed for compile-and-run to avoid expensive golden reference
        # ======================================================================

        # Use deterministic random seed for reproducibility
        np.random.seed(0)

        # Scale factor for output comparison
        inp_scale4 = 0.5

        # Input activation: 32x32x256, int8 (values 1-99 to match reference)
        input_act_chw = np.random.randint(
            1, 100, size=(TENSOR_IN_C, TENSOR_IN_H, TENSOR_IN_W)
        ).astype(np.int8)

        # Weights for each layer (values 50-99 to match reference)
        # Layer 1: 1x1 conv (256->64)
        weight1_oiyx = np.random.randint(
            50, 100, size=(TENSOR_L1_OUT_C, TENSOR_L1_IN_C, 1, 1)
        ).astype(np.int8)
        # Layer 2: 3x3 conv (64->64)
        weight2_oiyx = np.random.randint(
            50, 100, size=(TENSOR_L2_OUT_C, TENSOR_L2_IN_C, 3, 3)
        ).astype(np.int8)
        # Layer 3: 1x1 conv (64->256)
        weight3_oiyx = np.random.randint(
            50, 100, size=(TENSOR_L3_OUT_C, TENSOR_L3_IN_C, 1, 1)
        ).astype(np.int8)

        # Compute golden reference output
        print("Computing golden reference...")
        golden_output = compute_golden_reference(
            input_act_chw, weight1_oiyx, weight2_oiyx, weight3_oiyx
        )
        print(
            f"Golden output shape: {golden_output.shape}, dtype: {golden_output.dtype}"
        )
        print(f"Golden output range: [{golden_output.min()}, {golden_output.max()}]")

        # ==================================================================
        # Reorder data layouts for AIE (YCXC8 for activations, OIYXI8O8 for weights)
        # ==================================================================

        # Reorder input activation: CYX -> YCXC8 -> flatten
        ifm_mem_fmt = reorder_mat(input_act_chw, "YCXC8", "CYX")
        input_act_flat = ifm_mem_fmt.flatten().astype(np.int8)
        print(f"Input activation flattened shape: {input_act_flat.shape}")

        # Reorder weights: OIYX -> OIYXI8O8 -> flatten and concatenate
        wts1_fmt = reorder_mat(weight1_oiyx, "OIYXI8O8", "OIYX")
        wts2_fmt = reorder_mat(weight2_oiyx, "OIYXI8O8", "OIYX")
        wts3_fmt = reorder_mat(weight3_oiyx, "OIYXI8O8", "OIYX")

        total_wts = np.concatenate(
            [wts1_fmt.flatten(), wts2_fmt.flatten(), wts3_fmt.flatten()]
        ).astype(np.int8)
        print(f"Total weights shape: {total_wts.shape} (expected: {TOTAL_WEIGHTS})")

        # Flatten golden output for comparison: CYX -> YCXC8 -> flatten
        golden_fmt = reorder_mat(golden_output, "YCXC8", "CYX")
        expected_out = golden_fmt.flatten().astype(np.uint8)
        print(f"Expected output shape: {expected_out.shape}")

        print("\nRunning AIR bottleneck design...")
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            debug_ir=args.debug_ir,
            omit_pingpong="all",  # Disable all ping-pong to avoid shared buffer sync issues
        )

        # Custom comparison with scale factor tolerance
        def compare_with_tolerance(actual, expected):
            """Compare outputs with tolerance based on quantization scale."""
            actual_scaled = actual.astype(np.float32) * inp_scale4
            expected_scaled = expected.astype(np.float32) * inp_scale4

            if np.allclose(actual_scaled, expected_scaled, rtol=0, atol=inp_scale4):
                print("\n✓ PASS: Output matches golden reference!")
                return True
            else:
                diff = np.abs(actual_scaled - expected_scaled)
                print(f"\n✗ FAIL: Output mismatch")
                print(f"  Max difference: {diff.max():.4f}")
                print(f"  Mean difference: {diff.mean():.4f}")
                print(
                    f"  Mismatched elements: {np.sum(diff > inp_scale4)} / {len(diff)}"
                )
                return False

        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_act_flat, total_wts],
                expected_outputs=[expected_out],
                rtol=0,
                atol=1,  # Allow 1 unit of quantization error
            )
        )

    elif args.compile_mode == "compile-only":
        print("\nCompiling AIR bottleneck design (no execution)...")
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            debug_ir=args.debug_ir,
            omit_pingpong="all",  # Disable all ping-pong to avoid shared buffer sync issues
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
        print("Compilation successful!")
