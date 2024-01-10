//===- specialize_channel_broadcast.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=specialize-channel-bundle' | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK:   %[[VAL_2:.*]] = aie.tile(1, 2)
// CHECK:   %[[VAL_3:.*]] = aie.tile(2, 2)
// CHECK-COUNT-3:    air.channel @{{.*}}[1, 1] {broadcast_shape = [4, 1]}
// CHECK: }
   
aie.device(xcvc1902) {
    %0 = aie.tile(1, 1)
    %1 = aie.tile(2, 1)
    %2 = aie.tile(1, 2)
    %3 = aie.tile(2, 2)
    air.channel @channel_0 [1, 3] {broadcast_shape = [4, 3]}
}
