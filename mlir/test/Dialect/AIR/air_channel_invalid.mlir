//===- air_channel_invalid.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --split-input-file --verify-diagnostics %s

// -----

// Test: rank mismatch between size and broadcast_shape.
// expected-error @+1 {{'air.channel' op bundle rank should match broadcast_shape rank}}
air.channel @rank_mismatch [2, 2] {broadcast_shape = [4, 4, 4]}

// -----

// Test: broadcast_shape smaller than size (shrinking is invalid).
// expected-error @+1 {{'air.channel' op broadcast_shape[0] (1) must be >= size[0] (2): broadcasting cannot shrink a dimension}}
air.channel @shrink [2, 2] {broadcast_shape = [1, 2]}

// -----

// Test: size is not 1 and not equal to broadcast_shape (NumPy rule violation).
// expected-error @+1 {{'air.channel' op size[0] (2) is not compatible with broadcast_shape[0] (8): size must be 1 or equal to broadcast_shape (NumPy broadcasting rules)}}
air.channel @bad_broadcast [2, 2] {broadcast_shape = [8, 2]}

// -----

// Test: violation in second dimension.
// expected-error @+1 {{'air.channel' op size[1] (3) is not compatible with broadcast_shape[1] (4): size must be 1 or equal to broadcast_shape (NumPy broadcasting rules)}}
air.channel @bad_dim1 [1, 3] {broadcast_shape = [1, 4]}

// -----

// Test: 3D channel with broadcasting violation.
// expected-error @+1 {{'air.channel' op size[2] (3) is not compatible with broadcast_shape[2] (4): size must be 1 or equal to broadcast_shape (NumPy broadcasting rules)}}
air.channel @bad_3d [2, 1, 3] {broadcast_shape = [2, 4, 4]}

// -----

// Test: valid broadcasting (positive test - should pass with no errors).
air.channel @valid_broadcast [1, 4] {broadcast_shape = [4, 4]}

// -----

// Test: valid 3D broadcasting (positive test).
air.channel @valid_3d [2, 1, 4] {broadcast_shape = [2, 4, 4]}

// -----

// Test: valid all-ones size broadcasting (positive test).
air.channel @valid_all_ones [1, 1] {broadcast_shape = [4, 4]}

// -----

// Test: channel.put with scf.for induction variable as channel bundle index.
air.channel @temporal_put [2, 1]
func.func @channel_put_temporal_for_iv(%m: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c2 step %c1 {
    // expected-error @+1 {{'air.channel.put' op channel index 0 is an scf.for induction variable; channel bundle indices must not be temporal scf.for induction variables}}
    air.channel.put @temporal_put[%iv, %c0] (%m[] [] []) : (memref<64xi32>)
  }
  return
}

// -----

// Test: channel.get with scf.for induction variable as channel bundle index.
air.channel @temporal_get [2, 1]
func.func @channel_get_temporal_for_iv(%m: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c2 step %c1 {
    // expected-error @+1 {{'air.channel.get' op channel index 0 is an scf.for induction variable; channel bundle indices must not be temporal scf.for induction variables}}
    air.channel.get @temporal_get[%iv, %c0] (%m[] [] []) : (memref<64xi32>)
  }
  return
}
