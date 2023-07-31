//===- specialize_channel_broadcast.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=specialize-channel-bundle' | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(2, 1)
// CHECK:   %[[VAL_2:.*]] = AIE.tile(1, 2)
// CHECK:   %[[VAL_3:.*]] = AIE.tile(2, 2)
// CHECK-COUNT-6:    air.channel @{{.*}}[1, 3] {broadcast_shape = [2, 1]}
// CHECK: }
   
AIE.device(xcvc1902) {
    %0 = AIE.tile(1, 1)
    %1 = AIE.tile(2, 1)
    %2 = AIE.tile(1, 2)
    %3 = AIE.tile(2, 2)
    air.channel @channel_0 [2, 3] {broadcast_shape = [4, 3]}
}
