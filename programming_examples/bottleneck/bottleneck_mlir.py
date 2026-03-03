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
from air.dialects.linalg import fill
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects import memref
from air.dialects.func import FuncOp, CallOp
from air.dialects import scf
from air.dialects.scf import for_, yield_
from air.dialects import vector as vector_dialect
from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend

range_ = for_


# AIE micro-kernel dimensions for int8 matmul:
#   AIE2:  vector<4x8xi8> × vector<8x8xi8> → vector<4x8xi32>  (M=4, K=8, N=8)
#   AIE2P: vector<8x8xi8> × vector<8x8xi8> → vector<8x8xi32>  (M=8, K=8, N=8)
# Detect target device from xrt-smi to select M dimension.
import subprocess

AIE2_K = 8
AIE2_N = 8


def _detect_aie_m():
    """Detect AIE micro-kernel M dimension based on target device."""
    try:
        result = subprocess.run(
            ["/opt/xilinx/xrt/bin/xrt-smi", "examine"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "Strix" in result.stdout or "npu2" in result.stdout.lower():
            return 8  # AIE2P: 8x8x8
    except Exception:
        pass
    return 4  # AIE2 default: 4x8x8


AIE2_M = _detect_aie_m()


def vectorized_block_matmul(
    a_buf,
    b_buf,
    c_buf,
    K_tiles,
    M_tiles,
    N_grps,
    b_n_offset=None,
    unsigned_a=False,
    b_ky=None,
    b_kx=None,
):
    """
    Vectorized block matmul using vector.contract with leading-unit-dim pattern.

    Buffers use 6D micro-blocked layout:
      A: [1, 1, K_tiles, M_tiles, AIE2_M, AIE2_K]  i8
      B: [B_N_total, 1, 1, K_tiles, AIE2_K, AIE2_N]  i8  (B_N_total >= N_grps)
      C: [N_grps, 1, 1, M_tiles, AIE2_M, AIE2_N]   i32

    If b_n_offset is provided (index SSA value), B tiles are read at
    [b_n_offset + n_grp, ...] instead of [n_grp, ...]. This allows
    reading from a larger weight buffer without copying.
    """
    from air.dialects.scf import ForOp

    index_type = IndexType.get()
    i8 = IntegerType.get_signless(8)
    i32 = IntegerType.get_signless(32)

    # 6D vector types with leading unit dims (matching memref trailing dims)
    a_vec_ty = VectorType.get([1, 1, 1, 1, AIE2_M, AIE2_K], i8)
    b_vec_ty = VectorType.get([1, 1, 1, 1, AIE2_K, AIE2_N], i8)
    c_vec_ty = VectorType.get([1, 1, 1, 1, AIE2_M, AIE2_N], i32)
    a_ext_ty = VectorType.get([1, 1, 1, 1, AIE2_M, AIE2_K], i32)
    b_ext_ty = VectorType.get([1, 1, 1, 1, AIE2_K, AIE2_N], i32)

    pad_i8 = ConstantOp(IntegerAttr.get(i8, 0), None)
    pad_i32 = ConstantOp(IntegerAttr.get(i32, 0), None)
    in_bounds_6d = [True, True, True, True, True, True]

    # 9-dim contract indexing maps (from leading-unit-dim-contract.mlir):
    #   A: (d0,d1,d2,d3,d4,d5,d6,d7,d8) -> (d0, d2, d5, d4, d6, d8)
    #   B: (d0,d1,d2,d3,d4,d5,d6,d7,d8) -> (d2, d1, d3, d5, d8, d7)
    #   C: (d0,d1,d2,d3,d4,d5,d6,d7,d8) -> (d0, d1, d3, d4, d6, d7)
    D = [AffineExpr.get_dim(i) for i in range(9)]
    map_a = AffineMapAttr.get(AffineMap.get(9, 0, [D[0], D[2], D[5], D[4], D[6], D[8]]))
    map_b = AffineMapAttr.get(AffineMap.get(9, 0, [D[2], D[1], D[3], D[5], D[8], D[7]]))
    map_c = AffineMapAttr.get(AffineMap.get(9, 0, [D[0], D[1], D[3], D[4], D[6], D[7]]))
    indexing_maps = [map_a, map_b, map_c]
    # d0..d8: par, par, red, par, par, red, par, par, red
    par = Attribute.parse("#vector.iterator_type<parallel>")
    red = Attribute.parse("#vector.iterator_type<reduction>")
    iterator_types = ArrayAttr.get([par, par, red, par, par, red, par, par, red])

    c0 = ConstantOp(index_type, 0)
    c1 = ConstantOp(index_type, 1)
    cK = ConstantOp(index_type, K_tiles)

    for n_grp in range_(0, N_grps):
        for m_tile in range_(0, M_tiles):
            # Load C tile: vector<1x1x1x1x4x8xi32>
            c_init = vector_dialect.transfer_read(
                c_vec_ty,
                c_buf,
                [n_grp, c0, c0, m_tile, c0, c0],
                AffineMap.get_minor_identity(6, 6),
                pad_i32,
                in_bounds_6d,
            )
            # K reduction with iter_args
            k_loop = ForOp(c0, cK, c1, iter_args=[c_init])
            with InsertionPoint(k_loop.body):
                k_tile = k_loop.induction_variable
                c_iter = k_loop.inner_iter_args[0]

                # Load A tile: vector<1x1x1x1x4x8xi8>
                a_vec = vector_dialect.transfer_read(
                    a_vec_ty,
                    a_buf,
                    [c0, c0, k_tile, m_tile, c0, c0],
                    AffineMap.get_minor_identity(6, 6),
                    pad_i8,
                    in_bounds_6d,
                )
                # Load B tile: vector<1x1x1x1x8x8xi8>
                b_n_idx = (
                    arith.addi(b_n_offset, n_grp) if b_n_offset is not None else n_grp
                )
                # B index: [n, k, ky, kx, 0, 0] for OIYXI8O8,
                #       or [n, 0, 0, k, 0, 0] for 1x1 conv weights
                b_dim1 = k_tile if b_ky is not None else c0
                b_dim2 = b_ky if b_ky is not None else c0
                b_dim3 = b_kx if b_kx is not None else k_tile
                b_vec = vector_dialect.transfer_read(
                    b_vec_ty,
                    b_buf,
                    [b_n_idx, b_dim1, b_dim2, b_dim3, c0, c0],
                    AffineMap.get_minor_identity(6, 6),
                    pad_i8,
                    in_bounds_6d,
                )
                # A: extui for uint8 activations, extsi for signed
                # B: always extsi (signed int8 weights)
                a_ext = (
                    arith.extui(a_ext_ty, a_vec)
                    if unsigned_a
                    else arith.extsi(a_ext_ty, a_vec)
                )
                b_ext = arith.extsi(b_ext_ty, b_vec)
                c_new = vector_dialect.contract(
                    c_vec_ty,
                    a_ext,
                    b_ext,
                    c_iter,
                    indexing_maps,
                    iterator_types,
                )
                yield_([c_new])

            c_final = k_loop.results[0]
            # Store C tile
            vector_dialect.transfer_write(
                None,
                c_final,
                c_buf,
                [n_grp, c0, c0, m_tile, c0, c0],
                AffineMap.get_minor_identity(6, 6),
                in_bounds_6d,
            )
            yield_([])
        yield_([])


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
    # L1 weights for layer 2 in OIYXI8O8 layout [OC//8, IC//8, 3, 3, 8, 8]
    l1_wts_layer2_ty = MemRefType.get(
        (TENSOR_L2_OUT_C // 8, TENSOR_L2_IN_C // 8, 3, 3, 8, 8),
        i8,
        memory_space=l1_mem_space,
    )  # [8, 8, 3, 3, 8, 8] = 36864 bytes
    # Padded input row for spatial shift: flat [(W+2) * IC] bytes
    CONV3X3_PAD_W = TENSOR_IN_W + 2  # 34
    CONV3X3_PAD_BYTES = CONV3X3_PAD_W * TENSOR_L2_IN_C  # 2176
    # i32 accumulator for conv3x3: 6D [N_grps, 1, 1, M_tiles, AIE2_M, AIE2_N]
    OC_PER_CORE = TENSOR_L2_OUT_C // 2  # 32
    l1_conv3x3_acc_ty = MemRefType.get(
        (OC_PER_CORE // 8, 1, 1, TENSOR_IN_W // AIE2_M, AIE2_M, 8),
        i32,
        memory_space=l1_mem_space,
    )
    # Each 3x3 core produces half the output channels
    l1_layer2_out_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L2_OUT_C // 2), i8, memory_space=l1_mem_space
    )
    # Combined output buffer for both 3x3 conv cores (shared L1, 6D block format)
    # Core 0 writes first 1024 bytes, Core 1 writes next 1024 bytes
    # 6D shape [1, 1, IC//8, W//AIE2_M, AIE2_M, 8] same byte layout as YCXC8
    CONV3X3_OUT_HALF_SIZE = TENSOR_IN_W * 1 * (TENSOR_L2_OUT_C // 2)  # 1024
    l1_layer2_out_combined_ty = MemRefType.get(
        [1, 1, TENSOR_L2_OUT_C // 8, TENSOR_IN_W // AIE2_M, AIE2_M, 8],
        i8,
        memory_space=l1_mem_space,
    )

    # Layer 3 (1x1 conv + skip) types
    l1_layer3_in_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L3_IN_C // 2), i8, memory_space=l1_mem_space
    )
    # 6D block format for block_matmul: [N//8, 1, 1, K//8, 8, 8]
    l1_wts_layer3_ty = MemRefType.get(
        [TENSOR_L3_OUT_C // 8, 1, 1, TENSOR_L3_IN_C // 8, 8, 8],
        i8,
        memory_space=l1_mem_space,
    )
    l1_layer3_out_ty = MemRefType.get(
        (TENSOR_IN_W, 1, TENSOR_L3_OUT_C), i8, memory_space=l1_mem_space
    )

    # Output channel tiling for conv1x1_skip (64 channels per tile, 4 tiles total)
    N_TILE_SKIP = 64
    # Weight tile: [N_TILE//8, 1, 1, K//8, 8, 8] for block_matmul B input
    l1_skip_b_tile_ty = MemRefType.get(
        [N_TILE_SKIP // 8, 1, 1, TENSOR_L3_IN_C // 8, 8, 8],
        i8,
        memory_space=l1_mem_space,
    )
    # i32 accumulator tile: [N_TILE//8, 1, 1, M//AIE2_M, AIE2_M, 8]
    l1_skip_c_tile_ty = MemRefType.get(
        [N_TILE_SKIP // 8, 1, 1, TENSOR_IN_W // AIE2_M, AIE2_M, 8],
        i32,
        memory_space=l1_mem_space,
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
    l2_wts_layer2_ty = MemRefType.get(
        (TENSOR_L2_OUT_C // 8, TENSOR_L2_IN_C // 8, 3, 3, 8, 8),
        i8,
        memory_space=l2_mem_space,
    )
    l2_wts_layer3_ty = MemRefType.get(
        [TENSOR_L3_OUT_C // 8, 1, 1, TENSOR_L3_IN_C // 8, 8, 8],
        i8,
        memory_space=l2_mem_space,
    )

    # conv2dk1 kernel is implemented inline in MLIR (no external .cc needed)

    # conv2dk3 kernel is implemented inline in MLIR using vectorized_block_matmul
    # (vector.contract / aievec.matmul) with scalar SRS post-processing

    # conv2dk1_skip kernel is implemented inline in MLIR (no external .cc needed)

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
                shared_l1_conv3x3_out = AllocOp(
                    l1_layer2_out_combined_ty, [], []
                )  # 2048-byte 6D buffer for both 3x3 conv cores

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

                    # 6D micro-blocked L1 buffers for vectorized block_matmul
                    M1 = TENSOR_IN_W  # 32
                    K1 = TENSOR_L1_IN_C  # 256
                    N1 = TENSOR_L1_OUT_C  # 64
                    l1_conv1_a_ty = MemRefType.get(
                        [1, 1, K1 // 8, M1 // AIE2_M, AIE2_M, 8],
                        i8,
                        memory_space=l1_mem_space,
                    )
                    l1_conv1_b_ty = MemRefType.get(
                        [N1 // 8, 1, 1, K1 // 8, 8, 8],
                        i8,
                        memory_space=l1_mem_space,
                    )
                    l1_conv1_c_ty = MemRefType.get(
                        [N1 // 8, 1, 1, M1 // AIE2_M, AIE2_M, 8],
                        i32,
                        memory_space=l1_mem_space,
                    )
                    # Output type (same byte count as original, 4D YCXC8)
                    l1_conv1_out_ty = MemRefType.get(
                        [N1 // 8, M1 // AIE2_M, AIE2_M, 8],
                        i8,
                        memory_space=l1_mem_space,
                    )

                    wts = AllocOp(l1_conv1_b_ty, [], [])
                    act_in = AllocOp(l1_conv1_a_ty, [], [])
                    act_out_acc = AllocOp(l1_conv1_c_ty, [], [])
                    act_out = AllocOp(l1_conv1_out_ty, [], [])

                    # Receive weights from L2 (same bytes, 6D type)
                    ChannelGet("L2ToL1_WtsL1", wts)

                    # Convert RTP scale from index to i32
                    scale_i32 = arith.index_cast(i32, scale)
                    c0_idx = ConstantOp(index_type, 0)
                    c0_i32 = ConstantOp(i32, 0)
                    c255_i32 = ConstantOp(i32, 255)
                    # Process each row
                    for _ in range_(0, TENSOR_IN_H):
                        # Get input activation row (6D type, same bytes)
                        ChannelGet("L2ToL1_ActIn", act_in)

                        # ============================================
                        # VECTORIZED CONV2DK1 via vector.contract
                        # ============================================
                        # Zero i32 accumulator
                        zero_i32 = ConstantOp(IntegerAttr.get(i32, 0), None)
                        fill(zero_i32, outs=[act_out_acc])

                        # Vectorized MAC: vector.contract → aievec.matmul
                        vectorized_block_matmul(
                            act_in,
                            wts,
                            act_out_acc,
                            K_tiles=K1 // AIE2_K,  # 256//8=32
                            M_tiles=M1 // AIE2_M,  # 32//4=8
                            N_grps=N1 // AIE2_N,  # 64//8=8
                        )

                        # Post-matmul: SRS on i32 accumulator → i8 output
                        # acc: [N/8, 1, 1, M/AIE2_M, AIE2_M, 8] i32
                        # out: [N/8, M/AIE2_M, AIE2_M, 8] i8
                        for d0 in range_(0, N1 // 8):
                            for d3 in range_(0, M1 // AIE2_M):
                                for d4 in range_(0, AIE2_M):
                                    for d5 in range_(0, 8):
                                        val = load(
                                            act_out_acc,
                                            [d0, c0_idx, c0_idx, d3, d4, d5],
                                        )
                                        # Hardware rounds via positive_inf SRS
                                        shifted = arith.shrsi(val, scale_i32)
                                        clamped = arith.minsi(
                                            arith.maxsi(shifted, c0_i32),
                                            c255_i32,
                                        )
                                        # Two-step truncation (AIE2 only supports
                                        # 32→16 and 16→8, not 32→8 directly)
                                        result_i16 = arith.trunci(
                                            IntegerType.get_signless(16), clamped
                                        )
                                        result_i8 = arith.trunci(i8, result_i16)
                                        store(
                                            result_i8,
                                            act_out,
                                            [d0, d3, d4, d5],
                                        )
                                        yield_([])
                                    yield_([])
                                yield_([])
                            yield_([])

                        # Send output to 3x3 conv cores (broadcast)
                        ChannelPut("L1ToL1_Conv1ToConv3x3", act_out)

                        yield_([])

                    DeallocOp(wts)
                    DeallocOp(act_in)
                    DeallocOp(act_out_acc)
                    DeallocOp(act_out)

                # No link_with -- kernel is inline MLIR (vectorized block_matmul)

                # =============================================================
                # Herd: 3x3 Convolution (2x1 herd: two cores, each half channels)
                # Core (0,0) processes channels 0-31, Core (1,0) channels 32-63
                # Receives L2 weights via broadcast channel to L1
                # Output written to SHARED L1 buffer (passed as operand)
                # Uses 4-buffer rotation pattern for proper sliding window
                # =============================================================
                @herd(
                    name="conv3x3",
                    sizes=[2, 1],
                    operands=[shared_l1_conv3x3_out.result],
                )
                def herd_conv3x3_body(tx, ty, sx, sy, shared_out_full):
                    c0_h = ConstantOp(index_type, 0)
                    W = TENSOR_IN_W  # 32
                    IC = TENSOR_L2_IN_C  # 64
                    OC = TENSOR_L2_OUT_C  # 64
                    OC_HALF = OC // 2  # 32
                    ROW_BYTES = W * IC  # 2048

                    row_buf_0 = AllocOp(l1_layer2_in_ty, [], [])
                    row_buf_1 = AllocOp(l1_layer2_in_ty, [], [])
                    row_buf_2 = AllocOp(l1_layer2_in_ty, [], [])
                    row_buf_3 = AllocOp(l1_layer2_in_ty, [], [])
                    wts = AllocOp(l1_wts_layer2_ty, [], [])  # [8,8,3,3,8,8]
                    acc = AllocOp(l1_conv3x3_acc_ty, [], [])  # [4,32,8] i32
                    # Padded row for spatial shift: flat 2176 bytes
                    padded_row_ty = MemRefType.get(
                        (CONV3X3_PAD_BYTES,), i8, memory_space=l1_mem_space
                    )
                    padded_row = AllocOp(padded_row_ty, [], [])

                    ChannelGet("L2ToL1_WtsL2", wts, indices=[tx, c0_h])
                    ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_0, indices=[tx, c0_h])
                    ChannelGet("L1ToL1_Conv1ToConv3x3", row_buf_1, indices=[tx, c0_h])

                    # Shared output as flat
                    shared_flat_ty = MemRefType.get(
                        (CONV3X3_OUT_HALF_SIZE * 2,),
                        i8,
                        memory_space=l1_mem_space,
                    )
                    shared_flat = memref.ReinterpretCastOp(
                        shared_flat_ty,
                        shared_out_full,
                        static_offsets=[0],
                        static_sizes=[CONV3X3_OUT_HALF_SIZE * 2],
                        static_strides=[1],
                        offsets=[],
                        sizes=[],
                        strides=[],
                    ).result

                    # Constants
                    c0_idx = ConstantOp(index_type, 0)
                    c0_i8 = ConstantOp(IntegerAttr.get(i8, 0), None)
                    c0_i32 = ConstantOp(IntegerAttr.get(i32, 0), None)
                    c255_i32 = ConstantOp(i32, 255)
                    c_scale = ConstantOp(i32, 11)
                    c_8_idx = ConstantOp(index_type, 8)
                    c_ic = ConstantOp(index_type, IC)

                    tx_i32 = arith.index_cast(i32, tx)
                    out_byte_offset = arith.index_cast(
                        index_type,
                        arith.muli(tx_i32, ConstantOp(i32, CONV3X3_OUT_HALF_SIZE)),
                    )
                    # Per-core N offset: tx * (OC_HALF // 8)
                    n_offset = arith.muli(tx, ConstantOp(index_type, OC_HALF // 8))

                    # Flat view for row copy
                    flat_ty = MemRefType.get(
                        (ROW_BYTES,), i8, memory_space=l1_mem_space
                    )

                    # Reinterpret padded row as 6D for vector.transfer_read
                    # For kx offset: [1, 1, IC//8, (W+2)//AIE2_M, AIE2_M, 8]
                    # won't work because W+2 is not divisible by AIE2_M.
                    # Instead, for each kx, create a 6D view at byte offset
                    # kx*IC in the padded row.
                    padded_6d_ty = MemRefType.get(
                        [1, 1, IC // 8, W // AIE2_M, AIE2_M, 8],
                        i8,
                        layout=StridedLayoutAttr.get(
                            offset=-9223372036854775808,
                            strides=[
                                ROW_BYTES,
                                ROW_BYTES,
                                W * 8,
                                AIE2_M * 8,
                                8,
                                1,
                            ],
                        ),
                        memory_space=l1_mem_space,
                    )

                    # Additional constants for the row loop
                    c1_idx = ConstantOp(index_type, 1)
                    c2_idx = ConstantOp(index_type, 2)
                    c3_idx = ConstantOp(index_type, 3)
                    c4_idx = ConstantOp(index_type, 4)
                    c31_idx = ConstantOp(index_type, 31)
                    c30_idx = ConstantOp(index_type, 30)

                    # Flat view of padded_row at offset IC for memref.copy dst
                    padded_dst_ty = MemRefType.get(
                        (ROW_BYTES,),
                        i8,
                        layout=StridedLayoutAttr.get(
                            offset=-9223372036854775808,
                            strides=[1],
                        ),
                        memory_space=l1_mem_space,
                    )
                    padded_dst_view = memref.ReinterpretCastOp(
                        padded_dst_ty,
                        padded_row,
                        static_offsets=[-9223372036854775808],
                        static_sizes=[ROW_BYTES],
                        static_strides=[1],
                        offsets=[c_ic],
                        sizes=[],
                        strides=[],
                    ).result

                    # Pre-flatten all 4 row buffers for memref.copy
                    row_buf_0_flat = memref.ReinterpretCastOp(
                        flat_ty,
                        row_buf_0,
                        static_offsets=[0],
                        static_sizes=[ROW_BYTES],
                        static_strides=[1],
                        offsets=[],
                        sizes=[],
                        strides=[],
                    ).result
                    row_buf_1_flat = memref.ReinterpretCastOp(
                        flat_ty,
                        row_buf_1,
                        static_offsets=[0],
                        static_sizes=[ROW_BYTES],
                        static_strides=[1],
                        offsets=[],
                        sizes=[],
                        strides=[],
                    ).result
                    row_buf_2_flat = memref.ReinterpretCastOp(
                        flat_ty,
                        row_buf_2,
                        static_offsets=[0],
                        static_sizes=[ROW_BYTES],
                        static_strides=[1],
                        offsets=[],
                        sizes=[],
                        strides=[],
                    ).result
                    row_buf_3_flat = memref.ReinterpretCastOp(
                        flat_ty,
                        row_buf_3,
                        static_offsets=[0],
                        static_sizes=[ROW_BYTES],
                        static_strides=[1],
                        offsets=[],
                        sizes=[],
                        strides=[],
                    ).result

                    # ===== Single scf.for over 32 output rows =====
                    # Pre-load first 2 input rows into slots 0 and 1
                    # (already done above: ChannelGet into row_buf_0 and row_buf_1)

                    H = TENSOR_IN_H  # 32
                    for out_row in range_(0, H):
                        # --- Step A: Load next input row if needed ---
                        # For out_row in [1, 30], load row (out_row+1) into
                        # slot (out_row+1) % 4
                        load_row = arith.addi(out_row, c1_idx)
                        should_load = arith.andi(
                            arith.cmpi(arith.CmpIPredicate.sge, out_row, c1_idx),
                            arith.cmpi(arith.CmpIPredicate.sle, out_row, c30_idx),
                        )
                        if_load = scf.IfOp(should_load, hasElse=True)
                        with InsertionPoint(if_load.then_block):
                            next_slot = arith.remui(load_row, c4_idx)
                            # ChannelGet into correct buffer based on slot
                            is_s0 = arith.cmpi(
                                arith.CmpIPredicate.eq, next_slot, c0_idx
                            )
                            if_s0 = scf.IfOp(is_s0, hasElse=True)
                            with InsertionPoint(if_s0.then_block):
                                ChannelGet(
                                    "L1ToL1_Conv1ToConv3x3",
                                    row_buf_0,
                                    indices=[tx, c0_h],
                                )
                                yield_([])
                            with InsertionPoint(if_s0.else_block):
                                yield_([])

                            is_s1 = arith.cmpi(
                                arith.CmpIPredicate.eq, next_slot, c1_idx
                            )
                            if_s1 = scf.IfOp(is_s1, hasElse=True)
                            with InsertionPoint(if_s1.then_block):
                                ChannelGet(
                                    "L1ToL1_Conv1ToConv3x3",
                                    row_buf_1,
                                    indices=[tx, c0_h],
                                )
                                yield_([])
                            with InsertionPoint(if_s1.else_block):
                                yield_([])

                            is_s2 = arith.cmpi(
                                arith.CmpIPredicate.eq, next_slot, c2_idx
                            )
                            if_s2 = scf.IfOp(is_s2, hasElse=True)
                            with InsertionPoint(if_s2.then_block):
                                ChannelGet(
                                    "L1ToL1_Conv1ToConv3x3",
                                    row_buf_2,
                                    indices=[tx, c0_h],
                                )
                                yield_([])
                            with InsertionPoint(if_s2.else_block):
                                yield_([])

                            is_s3 = arith.cmpi(
                                arith.CmpIPredicate.eq, next_slot, c3_idx
                            )
                            if_s3 = scf.IfOp(is_s3, hasElse=True)
                            with InsertionPoint(if_s3.then_block):
                                ChannelGet(
                                    "L1ToL1_Conv1ToConv3x3",
                                    row_buf_3,
                                    indices=[tx, c0_h],
                                )
                                yield_([])
                            with InsertionPoint(if_s3.else_block):
                                yield_([])

                            yield_([])
                        with InsertionPoint(if_load.else_block):
                            yield_([])

                        # --- Step B: Compute slots for 3 sliding window rows ---
                        # line0 = max(0, out_row - 1), line1 = out_row,
                        # line2 = min(31, out_row + 1)
                        row_minus_1 = arith.subi(out_row, c1_idx)
                        line0_row = arith.select(
                            arith.cmpi(arith.CmpIPredicate.sge, row_minus_1, c0_idx),
                            row_minus_1,
                            c0_idx,
                        )
                        line1_row = out_row
                        row_plus_1 = arith.addi(out_row, c1_idx)
                        line2_row = arith.select(
                            arith.cmpi(arith.CmpIPredicate.sle, row_plus_1, c31_idx),
                            row_plus_1,
                            c31_idx,
                        )
                        line0_slot = arith.remui(line0_row, c4_idx)
                        line1_slot = arith.remui(line1_row, c4_idx)
                        line2_slot = arith.remui(line2_row, c4_idx)

                        # --- Step C: Conv3x3 body (emitted ONCE) ---
                        fill(c0_i32, outs=[acc])

                        for ky in range_(0, 3):
                            # Select slot for this ky
                            is_ky0 = arith.cmpi(arith.CmpIPredicate.eq, ky, c0_idx)
                            is_ky1 = arith.cmpi(arith.CmpIPredicate.eq, ky, c1_idx)
                            slot = arith.select(
                                is_ky0,
                                line0_slot,
                                arith.select(is_ky1, line1_slot, line2_slot),
                            )

                            # Zero padded row, then copy selected row buffer
                            fill(c0_i8, outs=[padded_row])

                            # Copy from row_buf_{slot} to padded_row[IC:]
                            # Using scf.for loop (single copy in IR, emitted
                            # once inside the ky scf.for loop)
                            for bi in range_(0, ROW_BYTES):
                                # Select source byte based on slot
                                v0 = load(row_buf_0_flat, [bi])
                                v1 = load(row_buf_1_flat, [bi])
                                v2 = load(row_buf_2_flat, [bi])
                                v3 = load(row_buf_3_flat, [bi])
                                is_s0 = arith.cmpi(arith.CmpIPredicate.eq, slot, c0_idx)
                                is_s1 = arith.cmpi(arith.CmpIPredicate.eq, slot, c1_idx)
                                is_s2 = arith.cmpi(arith.CmpIPredicate.eq, slot, c2_idx)
                                v = arith.select(
                                    is_s0,
                                    v0,
                                    arith.select(
                                        is_s1,
                                        v1,
                                        arith.select(is_s2, v2, v3),
                                    ),
                                )
                                di = arith.addi(c_ic, bi)
                                store(v, padded_row, [di])
                                yield_([])

                            # 3x3 kernel: kx loop (scf.for, emitted once)
                            for kx in range_(0, 3):
                                kx_offset = arith.muli(kx, c_ic)
                                shifted_a = memref.ReinterpretCastOp(
                                    padded_6d_ty,
                                    padded_row,
                                    static_offsets=[-9223372036854775808],
                                    static_sizes=[
                                        1,
                                        1,
                                        IC // 8,
                                        W // AIE2_M,
                                        AIE2_M,
                                        8,
                                    ],
                                    static_strides=[
                                        ROW_BYTES,
                                        ROW_BYTES,
                                        W * 8,
                                        AIE2_M * 8,
                                        8,
                                        1,
                                    ],
                                    offsets=[kx_offset],
                                    sizes=[],
                                    strides=[],
                                ).result

                                vectorized_block_matmul(
                                    shifted_a,
                                    wts,
                                    acc,
                                    K_tiles=IC // AIE2_K,  # 8
                                    M_tiles=W // AIE2_M,  # 8
                                    N_grps=OC_HALF // AIE2_N,  # 4
                                    b_n_offset=n_offset,
                                    unsigned_a=True,
                                    b_ky=ky,
                                    b_kx=kx,
                                )
                                yield_([])
                            yield_([])

                        # --- Step D: SRS + store to shared L1 ---
                        for oc_grp in range_(0, OC_HALF // 8):
                            for d3 in range_(0, W // AIE2_M):
                                for d4 in range_(0, AIE2_M):
                                    for oc8 in range_(0, 8):
                                        val = load(
                                            acc,
                                            [
                                                oc_grp,
                                                c0_idx,
                                                c0_idx,
                                                d3,
                                                d4,
                                                oc8,
                                            ],
                                        )
                                        shifted = arith.shrsi(val, c_scale)
                                        clamped = arith.minsi(
                                            arith.maxsi(shifted, c0_i32),
                                            c255_i32,
                                        )
                                        result = arith.trunci(i8, clamped)
                                        x_pos = arith.addi(
                                            arith.muli(
                                                d3,
                                                ConstantOp(index_type, AIE2_M),
                                            ),
                                            d4,
                                        )
                                        out_flat = arith.addi(
                                            out_byte_offset,
                                            arith.addi(
                                                arith.addi(
                                                    arith.muli(
                                                        oc_grp,
                                                        ConstantOp(index_type, W * 8),
                                                    ),
                                                    arith.muli(x_pos, c_8_idx),
                                                ),
                                                oc8,
                                            ),
                                        )
                                        store(result, shared_flat, [out_flat])
                                        yield_([])
                                    yield_([])
                                yield_([])
                            yield_([])

                        yield_([])

                    DeallocOp(wts)
                    DeallocOp(acc)
                    DeallocOp(padded_row)
                    DeallocOp(row_buf_0)
                    DeallocOp(row_buf_1)
                    DeallocOp(row_buf_2)
                    DeallocOp(row_buf_3)

                # No link_with -- kernel is inline MLIR (vectorized conv3x3)

                # =============================================================
                # Herd 3: 1x1 Convolution + Skip Addition (restore: 64→256)
                # RTP: scale and skip_scale parameters passed as operands
                # SHARED L1: reads from shared_l1_conv3x3_out (64-ch buffer)
                # =============================================================
                @herd(
                    name="conv1x1_skip",
                    sizes=[1, 1],
                    operands=[
                        rtp_scale_conv3,
                        rtp_skip_scale,
                        shared_l1_conv3x3_out.result,  # 64-ch shared buffer
                    ],
                )
                def herd_skip_body(
                    h3_x,
                    h3_y,
                    h3_sx,
                    h3_sy,
                    scale,
                    skip_scale,
                    shared_in_full,
                ):
                    # Full conv1x1_skip: matmul + two-stage SRS + skip add
                    M3 = TENSOR_IN_W
                    K3 = TENSOR_L3_IN_C
                    N3 = TENSOR_L3_OUT_C
                    N_TILES = N3 // N_TILE_SKIP

                    c0_idx = ConstantOp(index_type, 0)
                    c0_i32 = ConstantOp(i32, 0)
                    c127_i32 = ConstantOp(i32, 127)
                    cn128_i32 = ConstantOp(i32, -128)
                    c255_i32 = ConstantOp(i32, 255)
                    c_n_tile_grps = ConstantOp(index_type, N_TILE_SKIP // 8)
                    c_aie2_m = ConstantOp(index_type, AIE2_M)
                    c_8_idx = ConstantOp(index_type, 8)
                    i16 = IntegerType.get_signless(16)

                    wts = AllocOp(l1_wts_layer3_ty, [], [])
                    c_acc = AllocOp(l1_skip_c_tile_ty, [], [])
                    skip_in = AllocOp(l1_layer1_in_ty, [], [])
                    act_out = AllocOp(l1_layer3_out_ty, [], [])

                    ChannelGet("L2ToL1_WtsL3", wts)
                    scale_i32 = arith.index_cast(i32, scale)

                    for _ in range_(0, TENSOR_IN_H):
                        ChannelGet("L1ToL1_SkipBufToSkip", skip_in)

                        for oc_tile in range_(0, N_TILES):
                            # Zero accumulator
                            zero_i32 = ConstantOp(IntegerAttr.get(i32, 0), None)
                            fill(zero_i32, outs=[c_acc])

                            # Vectorized matmul
                            b_offset = arith.muli(oc_tile, c_n_tile_grps)
                            vectorized_block_matmul(
                                shared_in_full,
                                wts,
                                c_acc,
                                K_tiles=K3 // AIE2_K,
                                M_tiles=M3 // AIE2_M,
                                N_grps=N_TILE_SKIP // AIE2_N,
                                b_n_offset=b_offset,
                                unsigned_a=True,  # shared L1 input is uint8
                            )

                            # Two-stage SRS + skip add
                            for d0 in range_(0, N_TILE_SKIP // 8):
                                for d3 in range_(0, M3 // AIE2_M):
                                    for d4 in range_(0, AIE2_M):
                                        for d5 in range_(0, 8):
                                            val = load(
                                                c_acc,
                                                [d0, c0_idx, c0_idx, d3, d4, d5],
                                            )
                                            # Stage 1 SRS: signed clamp
                                            # Hardware rounds via positive_inf SRS
                                            shifted = arith.shrsi(val, scale_i32)
                                            clamped_s = arith.minsi(
                                                arith.maxsi(shifted, cn128_i32),
                                                c127_i32,
                                            )

                                            # Output indices
                                            oc_grp = arith.addi(
                                                arith.muli(oc_tile, c_n_tile_grps),
                                                d0,
                                            )
                                            x_pos = arith.addi(
                                                arith.muli(d3, c_aie2_m), d4
                                            )
                                            dim2 = arith.addi(
                                                arith.muli(x_pos, c_8_idx),
                                                d5,
                                            )

                                            # Skip connection add
                                            skip_val = load(
                                                skip_in, [oc_grp, c0_idx, dim2]
                                            )
                                            skip_i32 = arith.extsi(i32, skip_val)
                                            skip_sum = arith.addi(clamped_s, skip_i32)

                                            # Stage 2 SRS: unsigned clamp
                                            clamped_u = arith.minsi(
                                                arith.maxsi(skip_sum, c0_i32),
                                                c255_i32,
                                            )
                                            result_i16 = arith.trunci(i16, clamped_u)
                                            result_i8 = arith.trunci(i8, result_i16)

                                            store(
                                                result_i8,
                                                act_out,
                                                [oc_grp, c0_idx, dim2],
                                            )
                                            yield_([])
                                        yield_([])
                                    yield_([])
                                yield_([])
                            yield_([])

                        ChannelPut("L1ToL2_ActOut", act_out)
                        yield_([])

                    DeallocOp(wts)
                    DeallocOp(c_acc)
                    DeallocOp(skip_in)
                    DeallocOp(act_out)

                # No link_with -- kernel is inline MLIR

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

    elif in_layout == "OIYX" and out_layout == "HWCF":
        # Weight reorder for linalg.conv_2d_nhwc_hwcf:
        # (O, I, Y, X) -> (H=Y, W=X, C=I, F=O)
        return in_tensor.transpose(2, 3, 1, 0).copy()

    else:
        raise NotImplementedError(
            f"Layout conversion {in_layout} -> {out_layout} not implemented"
        )


def _srs_round(x):
    """SRS rounding: positive_inf mode = floor(x + 0.5), matching AIE hardware."""
    return np.floor(x + 0.5)


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

    Uses positive_inf rounding (floor(x+0.5)) to match AIE hardware SRS behavior.

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
        _srs_round(np.maximum(conv1_scaled, 0) / inp_scale2), min_val, max_val
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
        _srs_round(np.maximum(conv2_scaled, 0) / inp_scale3), min_val, max_val
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
    same_scale_init = np.clip(_srs_round(conv3_scaled / inp_scale1), -128, 127)

    # Skip connection: add original input
    skip_add = inp_scale1 * (same_scale_init + skip_input)

    # Final output with ReLU
    # Note: hardware kernel outputs the integer quantized value directly (uint8),
    # so we should NOT multiply by inp_scale4 here. The mlir-aie reference test.py
    # applies the scale AFTER reading from hardware (line 184: out.numpy() * inp_scale4).
    final_out = np.clip(_srs_round(skip_add / inp_scale4), min_val, max_val)

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

        # Custom comparison with tolerance for quantization rounding.
        # AIE2P SRS uses positive_inf rounding which can differ by 1 per
        # stage from Python's rounding. With 3 SRS stages in the bottleneck
        # pipeline, rounding differences can propagate through matmul
        # accumulation, causing larger differences in a small fraction of
        # elements. We require 99% of elements to match within atol=1.
        def compare_with_tolerance(actual, expected):
            """Compare outputs allowing quantization rounding tolerance."""
            diff = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
            n_close = np.sum(diff <= 1)
            total = len(diff)
            pct = 100.0 * n_close / total

            if pct >= 99.0:
                print(f"\nPASS: {n_close}/{total} elements within atol=1 ({pct:.2f}%)")
                return True
            else:
                print(
                    f"\nFAIL: Only {n_close}/{total} elements within atol=1 ({pct:.2f}%)"
                )
                print(f"  Max difference: {diff.max()}")
                print(f"  Mean difference: {diff.mean():.4f}")
                return False

        # Compile and run directly to get actual outputs for custom comparison
        # (XRTRunner._check_outputs uses exact match for integers, but AIE2P
        # SRS positive_inf rounding can differ by 1 from Python's rounding)
        import filelock

        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            debug_ir=args.debug_ir,
            omit_pingpong="all",
        )
        output_placeholder = np.zeros(expected_out.shape, expected_out.dtype)
        expanded_inputs = [input_act_flat, total_wts, output_placeholder]

        compiled_module = backend.compile(mlir_module)
        with filelock.FileLock("/tmp/npu.lock"):
            module_function = backend.load(compiled_module)
            actual_outputs = module_function(*expanded_inputs)
        backend.unload()

        actual_out = actual_outputs[len([input_act_flat, total_wts])]

        if compare_with_tolerance(actual_out, expected_out):
            print("PASS!")
            exit(0)
        else:
            exit(1)

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
