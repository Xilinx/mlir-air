//===- control_packet_overlay.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=insert-control-packet-flow' --split-input-file | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK: %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL_2:.*]] = aie.tile(0, 0)
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT:   aie.packet_source<%[[VAL_2]], DMA : 0>
// CHECK-NEXT:   aie.packet_dest<%[[VAL_1]], Ctrl : 0>
// CHECK-NEXT: }
// CHECK: aie.packet_flow(1) {
// CHECK-NEXT:   aie.packet_source<%[[VAL_2]], DMA : 0>
// CHECK-NEXT:   aie.packet_dest<%[[VAL_0]], Ctrl : 0>
// CHECK-NEXT: }

#map = affine_map<(d0) -> (d0)>
module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_0 = aie.tile(0, 0)
  }
}

// -----

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CHECK: %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK: %[[tile_0_3:.*]] = aie.tile(0, 3)
// CHECK: %[[tile_0_4:.*]] = aie.tile(0, 4)
// CHECK: %[[tile_0_5:.*]] = aie.tile(0, 5)
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CHECK-NEXT:   aie.packet_dest<%[[tile_0_5]], Ctrl : 0>
// CHECK-NEXT: }
// CHECK: aie.packet_flow(1) {
// CHECK-NEXT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CHECK-NEXT:   aie.packet_dest<%[[tile_0_4]], Ctrl : 0>
// CHECK-NEXT: }
// CHECK: aie.packet_flow(2) {
// CHECK-NEXT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CHECK-NEXT:   aie.packet_dest<%[[tile_0_3]], Ctrl : 0>
// CHECK-NEXT: }
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT:   aie.packet_source<%[[tile_0_0]], DMA : 1>
// CHECK-NEXT:   aie.packet_dest<%[[tile_0_2]], Ctrl : 0>
// CHECK-NEXT: }
// CHECK: aie.packet_flow(1) {
// CHECK-NEXT:   aie.packet_source<%[[tile_0_0]], DMA : 1>
// CHECK-NEXT:   aie.packet_dest<%[[tile_0_1]], Ctrl : 0>
// CHECK-NEXT: }

#map = affine_map<(d0) -> (d0)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
  } {sym_name = "segment0"}
}
