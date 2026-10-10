//===- dma_to_channel_auto_packet_broadcast.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Tests for broadcast-aware auto-packet-switching detection in
// air-dma-to-channel. Broadcast channels can distribute across their column
// span, reducing per-column shim DMA pressure.

// RUN: air-opt %s -air-dma-to-channel -split-input-file 2>/dev/null | FileCheck %s

// -----

// Test 1: 4 broadcast inputs spanning 8 columns -> no upgrade.
// Each broadcast channel has broadcast_shape=[8,1]. Pressure:
// ceil(4/8) = 1 <= 2 (per-column limit). No dma_packet expected.

// CHECK:       air.channel @channel_0 [1, 1] {broadcast_shape = [8, 1]}
// CHECK:       air.channel @channel_1 [1, 1] {broadcast_shape = [8, 1]}
// CHECK:       air.channel @channel_2 [1, 1] {broadcast_shape = [8, 1]}
// CHECK:       air.channel @channel_3 [1, 1] {broadcast_shape = [8, 1]}
// CHECK:       air.channel @channel_4 [8, 4]
// CHECK-NOT:   channel_type = "npu_dma_packet"
// CHECK-LABEL: func.func @broadcast_4x8_no_upgrade

#set_ty0 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 7 >= 0, s1 == 0)>
#set_ty1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 7 >= 0, s1 - 1 == 0)>
#set_ty2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 7 >= 0, s1 - 2 == 0)>
#set_ty3 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 7 >= 0, s1 - 3 == 0)>
module {
  func.func @broadcast_4x8_no_upgrade(
      %arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>,
      %arg2: memref<1024xbf16>, %arg3: memref<1024xbf16>,
      %arg4: memref<1024xbf16>) {
    air.launch () in () args(%b0=%arg0, %b1=%arg1, %b2=%arg2, %b3=%arg3,
                              %co=%arg4)
        : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sb0=%b0, %sb1=%b1, %sb2=%b2, %sb3=%b3,
                             %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16> {
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c8, %sy=%c4)
            args(%hb0=%sb0, %hb1=%sb1, %hb2=%sb2, %hb3=%sb3, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
              memref<1024xbf16>, memref<1024xbf16> {
          %buf = memref.alloc() : memref<256xbf16, 2>
          %buf_c = memref.alloc() : memref<256xbf16, 2>
          // 4 broadcast inputs, each spanning 8 columns.
          affine.if #set_ty0()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb0[] [] [])
                {broadcast_set = #set_ty0} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set_ty1()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb1[] [] [])
                {broadcast_set = #set_ty1} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set_ty2()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb2[] [] [])
                {broadcast_set = #set_ty2} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set_ty3()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb3[] [] [])
                {broadcast_set = #set_ty3} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          // 1 output (non-broadcast).
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<256xbf16, 2>)
          memref.dealloc %buf : memref<256xbf16, 2>
          memref.dealloc %buf_c : memref<256xbf16, 2>
        }
      }
    }
    return
  }
}

// -----

// Test 2: 6 broadcast inputs spanning 2 columns -> upgrade.
// Each broadcast channel has broadcast_shape=[2,1]. Pressure:
// ceil(6/2) = 3 > 2 (per-column limit). All 6 inputs upgraded to dma_packet.

// CHECK:       air.channel @channel_0 {{.*}} {broadcast_shape = [2, 1], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_1 {{.*}} {broadcast_shape = [2, 1], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_2 {{.*}} {broadcast_shape = [2, 1], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_3 {{.*}} {broadcast_shape = [2, 1], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_4 {{.*}} {broadcast_shape = [2, 1], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_5 {{.*}} {broadcast_shape = [2, 1], channel_type = "npu_dma_packet"}
// CHECK-LABEL: func.func @broadcast_6x2_upgrade

#set2_ty0 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
#set2_ty1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
#set2_ty2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 2 == 0)>
#set2_ty3 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 3 == 0)>
#set2_ty4 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 4 == 0)>
#set2_ty5 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 5 == 0)>
module {
  func.func @broadcast_6x2_upgrade(
      %arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>,
      %arg2: memref<1024xbf16>, %arg3: memref<1024xbf16>,
      %arg4: memref<1024xbf16>, %arg5: memref<1024xbf16>,
      %arg6: memref<1024xbf16>) {
    air.launch () in () args(%b0=%arg0, %b1=%arg1, %b2=%arg2,
                              %b3=%arg3, %b4=%arg4, %b5=%arg5,
                              %co=%arg6)
        : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16> {
      air.segment @seg args(%sb0=%b0, %sb1=%b1, %sb2=%b2,
                             %sb3=%b3, %sb4=%b4, %sb5=%b5,
                             %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16> {
        %c2 = arith.constant 2 : index
        %c6 = arith.constant 6 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c2, %sy=%c6)
            args(%hb0=%sb0, %hb1=%sb1, %hb2=%sb2,
                 %hb3=%sb3, %hb4=%sb4, %hb5=%sb5, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
              memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
              memref<1024xbf16> {
          %buf = memref.alloc() : memref<256xbf16, 2>
          %buf_c = memref.alloc() : memref<256xbf16, 2>
          affine.if #set2_ty0()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb0[] [] [])
                {broadcast_set = #set2_ty0} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set2_ty1()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb1[] [] [])
                {broadcast_set = #set2_ty1} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set2_ty2()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb2[] [] [])
                {broadcast_set = #set2_ty2} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set2_ty3()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb3[] [] [])
                {broadcast_set = #set2_ty3} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set2_ty4()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb4[] [] [])
                {broadcast_set = #set2_ty4} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set2_ty5()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb5[] [] [])
                {broadcast_set = #set2_ty5} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          // 1 output (non-broadcast).
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<256xbf16, 2>)
          memref.dealloc %buf : memref<256xbf16, 2>
          memref.dealloc %buf_c : memref<256xbf16, 2>
        }
      }
    }
    return
  }
}

// -----

// Test 3: Mixed non-broadcast + broadcast inputs -> upgrade.
// 2 non-broadcast inputs + 3 broadcast inputs spanning 4 columns.
// Pressure: 2 + ceil(3/4) = 2 + 1 = 3 > 2. All 5 inputs upgraded.

// CHECK:       air.channel @channel_0 {{.*}} {channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_1 {{.*}} {channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_2 {{.*}} {broadcast_shape = [4, 1], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_3 {{.*}} {broadcast_shape = [4, 1], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_4 {{.*}} {broadcast_shape = [4, 1], channel_type = "npu_dma_packet"}
// CHECK-LABEL: func.func @mixed_broadcast_upgrade

#set3_bcast = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set3_bcast2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set3_bcast3 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
module {
  func.func @mixed_broadcast_upgrade(
      %arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>,
      %arg2: memref<1024xbf16>, %arg3: memref<1024xbf16>,
      %arg4: memref<1024xbf16>, %arg5: memref<1024xbf16>) {
    air.launch () in () args(%a0=%arg0, %a1=%arg1,
                              %b0=%arg2, %b1=%arg3, %b2=%arg4,
                              %co=%arg5)
        : memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16> {
      air.segment @seg args(%sa0=%a0, %sa1=%a1,
                             %sb0=%b0, %sb1=%b1, %sb2=%b2,
                             %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16> {
        %c4 = arith.constant 4 : index
        %c3 = arith.constant 3 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c4, %sy=%c3)
            args(%ha0=%sa0, %ha1=%sa1,
                 %hb0=%sb0, %hb1=%sb1, %hb2=%sb2, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16>,
              memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
              memref<1024xbf16> {
          %buf = memref.alloc() : memref<256xbf16, 2>
          %buf2 = memref.alloc() : memref<256xbf16, 2>
          %buf_c = memref.alloc() : memref<256xbf16, 2>
          // 2 non-broadcast inputs.
          air.dma_memcpy_nd (%buf[] [] [], %ha0[] [] []) :
              (memref<256xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%buf2[] [] [], %ha1[] [] []) :
              (memref<256xbf16, 2>, memref<1024xbf16>)
          // 3 broadcast inputs spanning 4 columns.
          affine.if #set3_bcast()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb0[] [] [])
                {broadcast_set = #set3_bcast} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set3_bcast2()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb1[] [] [])
                {broadcast_set = #set3_bcast2} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set3_bcast3()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb2[] [] [])
                {broadcast_set = #set3_bcast3} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          // 1 output (non-broadcast).
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<256xbf16, 2>)
          memref.dealloc %buf : memref<256xbf16, 2>
          memref.dealloc %buf2 : memref<256xbf16, 2>
          memref.dealloc %buf_c : memref<256xbf16, 2>
        }
      }
    }
    return
  }
}

// -----

// Test 4: Mixed broadcast spans -> no upgrade.
// 2 broadcasts spanning 8 cols + 2 broadcasts spanning 2 cols.
// Pressure: ceil(2/8) + ceil(2/2) = 1 + 1 = 2 <= 2. No upgrade.

// CHECK:       air.channel @channel_0 [1, 1] {broadcast_shape = [8, 1]}
// CHECK:       air.channel @channel_1 [1, 1] {broadcast_shape = [8, 1]}
// CHECK:       air.channel @channel_2 [1, 1] {broadcast_shape = [2, 1]}
// CHECK:       air.channel @channel_3 [1, 1] {broadcast_shape = [2, 1]}
// CHECK:       air.channel @channel_4 [8, 4]
// CHECK-NOT:   channel_type = "npu_dma_packet"
// CHECK-LABEL: func.func @mixed_spans_no_upgrade

#set4_wide0 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 7 >= 0, s1 == 0)>
#set4_wide1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 7 >= 0, s1 - 1 == 0)>
#set4_narrow0 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 2 == 0)>
#set4_narrow1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 3 == 0)>
module {
  func.func @mixed_spans_no_upgrade(
      %arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>,
      %arg2: memref<1024xbf16>, %arg3: memref<1024xbf16>,
      %arg4: memref<1024xbf16>) {
    air.launch () in () args(%b0=%arg0, %b1=%arg1, %b2=%arg2, %b3=%arg3,
                              %co=%arg4)
        : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sb0=%b0, %sb1=%b1, %sb2=%b2, %sb3=%b3,
                             %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16> {
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c8, %sy=%c4)
            args(%hb0=%sb0, %hb1=%sb1, %hb2=%sb2, %hb3=%sb3, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
              memref<1024xbf16>, memref<1024xbf16> {
          %buf = memref.alloc() : memref<256xbf16, 2>
          %buf_c = memref.alloc() : memref<256xbf16, 2>
          // 2 wide broadcasts (span 8 columns).
          affine.if #set4_wide0()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb0[] [] [])
                {broadcast_set = #set4_wide0} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set4_wide1()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb1[] [] [])
                {broadcast_set = #set4_wide1} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          // 2 narrow broadcasts (span 2 columns).
          affine.if #set4_narrow0()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb2[] [] [])
                {broadcast_set = #set4_narrow0} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set4_narrow1()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb3[] [] [])
                {broadcast_set = #set4_narrow1} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          // 1 output (non-broadcast).
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<256xbf16, 2>)
          memref.dealloc %buf : memref<256xbf16, 2>
          memref.dealloc %buf_c : memref<256xbf16, 2>
        }
      }
    }
    return
  }
}

// -----

// Test 5: Row-only broadcasts (broadcast_shape=[1,4]), each fixed to its own
// herd column by its broadcast_set. Each feeds one column, so per-column
// pressure is 1 <= 2 -> no upgrade.

// CHECK:       air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
// CHECK:       air.channel @channel_1 [1, 1] {broadcast_shape = [1, 4]}
// CHECK:       air.channel @channel_2 [1, 1] {broadcast_shape = [1, 4]}
// CHECK-NOT:   channel_type = "npu_dma_packet"
// CHECK-LABEL: func.func @row_broadcast_distinct_columns

#set5_row0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set5_row1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set5_row2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  func.func @row_broadcast_distinct_columns(
      %arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>,
      %arg2: memref<1024xbf16>, %arg3: memref<1024xbf16>) {
    air.launch () in () args(%b0=%arg0, %b1=%arg1, %b2=%arg2,
                              %co=%arg3)
        : memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sb0=%b0, %sb1=%b1, %sb2=%b2,
                             %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16> {
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c3, %sy=%c4)
            args(%hb0=%sb0, %hb1=%sb1, %hb2=%sb2, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16>,
              memref<1024xbf16>, memref<1024xbf16> {
          %buf = memref.alloc() : memref<256xbf16, 2>
          %buf_c = memref.alloc() : memref<256xbf16, 2>
          // 3 row-only broadcasts, one per column.
          affine.if #set5_row0()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb0[] [] [])
                {broadcast_set = #set5_row0} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set5_row1()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb1[] [] [])
                {broadcast_set = #set5_row1} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set5_row2()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb2[] [] [])
                {broadcast_set = #set5_row2} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          // 1 output (non-broadcast).
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<256xbf16, 2>)
          memref.dealloc %buf : memref<256xbf16, 2>
          memref.dealloc %buf_c : memref<256xbf16, 2>
        }
      }
    }
    return
  }
}

// -----

// Test 6: Row-only broadcasts all fixed to herd column 0. They compete for
// that column: per-column pressure = 3 > 2 -> upgrade. Tests that row-only
// broadcasts aren't incorrectly discounted.

// CHECK:       air.channel @channel_0 {{.*}} {broadcast_shape = [1, 4], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_1 {{.*}} {broadcast_shape = [1, 4], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_2 {{.*}} {broadcast_shape = [1, 4], channel_type = "npu_dma_packet"}
// CHECK-LABEL: func.func @row_broadcast_same_column_upgrade

#set6_row0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set6_row1 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set6_row2 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  func.func @row_broadcast_same_column_upgrade(
      %arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>,
      %arg2: memref<1024xbf16>, %arg3: memref<1024xbf16>) {
    air.launch () in () args(%b0=%arg0, %b1=%arg1, %b2=%arg2,
                              %co=%arg3)
        : memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sb0=%b0, %sb1=%b1, %sb2=%b2,
                             %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16> {
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c3, %sy=%c4)
            args(%hb0=%sb0, %hb1=%sb1, %hb2=%sb2, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16>,
              memref<1024xbf16>, memref<1024xbf16> {
          %buf = memref.alloc() : memref<256xbf16, 2>
          %buf_c = memref.alloc() : memref<256xbf16, 2>
          // 3 row-only broadcasts into column 0.
          affine.if #set6_row0()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb0[] [] [])
                {broadcast_set = #set6_row0} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set6_row1()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb1[] [] [])
                {broadcast_set = #set6_row1} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set6_row2()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb2[] [] [])
                {broadcast_set = #set6_row2} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          // 1 output (non-broadcast).
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<256xbf16, 2>)
          memref.dealloc %buf : memref<256xbf16, 2>
          memref.dealloc %buf_c : memref<256xbf16, 2>
        }
      }
    }
    return
  }
}

// -----

// Test 7: Column-fixed broadcasts in two herds. Herd @a has two on its
// column 0, herd @b one on its column 1. Herd columns are local and the herds
// are not placed yet, so they may share a column: per-column pressure =
// 2 + 1 = 3 > 2 -> upgrade.

// CHECK:       air.channel @channel_0 {{.*}} {broadcast_shape = [1, 4], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_1 {{.*}} {broadcast_shape = [1, 4], channel_type = "npu_dma_packet"}
// CHECK:       air.channel @channel_2 [2, 4]
// CHECK:       air.channel @channel_3 {{.*}} {broadcast_shape = [1, 4], channel_type = "npu_dma_packet"}
// CHECK-LABEL: func.func @row_broadcast_two_herds_upgrade

#set7_col0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set7_col1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  func.func @row_broadcast_two_herds_upgrade(
      %arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>,
      %arg2: memref<1024xbf16>, %arg3: memref<1024xbf16>) {
    air.launch () in () args(%b0=%arg0, %b1=%arg1, %b2=%arg2,
                              %co=%arg3)
        : memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sb0=%b0, %sb1=%b1, %sb2=%b2,
                             %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16> {
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        air.herd @a tile (%tx, %ty) in (%sx=%c2, %sy=%c4)
            args(%hb0=%sb0, %hb1=%sb1, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16> {
          %buf = memref.alloc() : memref<256xbf16, 2>
          %buf_c = memref.alloc() : memref<256xbf16, 2>
          // 2 row-only broadcasts into column 0 of @a.
          affine.if #set7_col0()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb0[] [] [])
                {broadcast_set = #set7_col0} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          affine.if #set7_col0()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb1[] [] [])
                {broadcast_set = #set7_col0} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<256xbf16, 2>)
          memref.dealloc %buf : memref<256xbf16, 2>
          memref.dealloc %buf_c : memref<256xbf16, 2>
        }
        air.herd @b tile (%tx, %ty) in (%sx=%c2, %sy=%c4)
            args(%hb2=%sb2)
            : memref<1024xbf16> {
          %buf = memref.alloc() : memref<256xbf16, 2>
          // 1 row-only broadcast into column 1 of @b.
          affine.if #set7_col1()[%tx, %ty] {
            air.dma_memcpy_nd (%buf[] [] [], %hb2[] [] [])
                {broadcast_set = #set7_col1} :
                (memref<256xbf16, 2>, memref<1024xbf16>)
          }
          memref.dealloc %buf : memref<256xbf16, 2>
        }
      }
    }
    return
  }
}
