//===- opt_shim_dma_bds_stride_limit.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" -split-input-file | FileCheck %s

// Shim transfers whose natural descriptor has a stride above the hardware
// limit, from a K loop walking a wide weight: each step reads the next 64 rows
// of a 37888-column matrix, so one step is 64 * 37888 elements.

// The form a Triton K loop over a weight lowers to: an i32 loop whose offset
// is the iteration times one step plus the column block the launch picks, on
// an unranked memref, one put at that offset and one 64 columns further. The
// K loop and the rows are one walk over 4096 rows. Both loops fold, the walk
// is split with a legal stride, and each put stays a single transfer.

// CHECK-LABEL: func.func @triton_k_loop_weight
// CHECK: air.channel.put {{.*}} @channel_0[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c148{{.*}}, %c512{{.*}}, %c8{{.*}}, %c64{{.*}}] [%c256{{.*}}, %c303104{{.*}}, %c37888{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put {{.*}} @channel_1[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c64{{.*}}] [%c148{{.*}}, %c512{{.*}}, %c8{{.*}}, %c64{{.*}}] [%c256{{.*}}, %c303104{{.*}}, %c37888{{.*}}, %c1{{.*}}])
// CHECK-NOT: air.channel.put
#plus64 = affine_map<()[s0] -> (s0 + 64)>
func.func @triton_k_loop_weight(%arg0: memref<*xbf16>) {
  %c1 = arith.constant 1 : index
  %c148 = arith.constant 148 : index
  %0 = air.launch async (%x, %y, %z) in (%sx=%c1, %sy=%c148, %sz=%c1) args(%b=%arg0) : memref<*xbf16> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c37888 = arith.constant 37888 : index
    %c2424832 = arith.constant 2424832 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %col = arith.muli %y, %c256 : index
    %col_i32 = arith.index_cast %col : index to i32
    %base = arith.index_cast %col_i32 : i32 to index
    %t0 = air.wait_all async
    %1 = scf.for %k = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%t = %t0) -> (!air.async.token) : i32 {
      %i = arith.index_cast %k : i32 to index
      %step = arith.muli %i, %c2424832 : index
      %off = arith.addi %step, %base : index
      %put = air.channel.put async [%t] @channel_0[] (%b[0, %off] [64, 64] [37888, 1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
      scf.yield %put : !air.async.token
    }
    %2 = scf.for %k = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%t = %t0) -> (!air.async.token) : i32 {
      %i = arith.index_cast %k : i32 to index
      %step = arith.muli %i, %c2424832 : index
      %off = arith.addi %step, %base : index
      %off64 = affine.apply #plus64()[%off]
      %put = air.channel.put async [%t] @channel_1[] (%b[0, %off64] [64, 64] [37888, 1]) {metadata = @airMemcpyId2} : (memref<*xbf16>)
      scf.yield %put : !air.async.token
    }
  }
  return
}
air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]

// -----

// The same walk with a ranked memref and nothing else around it. The loop's
// step is the outer dimension's whole extent, so the loop and that dimension
// are one walk over 4096 rows. It folds, and the walk is split with the row
// stride inside a legal one.

// CHECK-LABEL: func.func @k_loop_continues_rows
// CHECK: air.channel.put {{.*}}(%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c512{{.*}}, %c8{{.*}}, %c64{{.*}}] [%c303104{{.*}}, %c37888{{.*}}, %c1{{.*}}])
// CHECK-NOT: air.channel.put
func.func @k_loop_continues_rows(%arg0: memref<4096x37888xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c37888 = arith.constant 37888 : index
  %c2424832 = arith.constant 2424832 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %0 = air.wait_all async
  %1 = scf.for %iv = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%t = %0) -> (!air.async.token) : i32 {
    %i = arith.index_cast %iv : i32 to index
    %off = arith.muli %i, %c2424832 : index
    %put = air.channel.put async [%t] @channel_0[] (%arg0[%c0, %off] [%c64, %c64] [%c37888, %c1]) {metadata = @airMemcpyId1} : (memref<4096x37888xbf16>)
    scf.yield %put : !air.async.token
  }
  return
}
air.channel @channel_0 [1, 1]

// -----

// Each step skips 64 rows but reads only 32, so the loop is not one walk with
// the rows and there is no legal stride for it: each step keeps its transfer.

// CHECK-LABEL: func.func @k_loop_skips_rows
// CHECK-COUNT-4: air.channel.put {{.*}} [%c32{{.*}}, %c64{{.*}}] [%c37888{{.*}}, %c1{{.*}}]
// CHECK-NOT: air.channel.put
func.func @k_loop_skips_rows(%arg0: memref<4096x37888xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c37888 = arith.constant 37888 : index
  %c2424832 = arith.constant 2424832 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = air.wait_all async
  %1 = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%t = %0) -> (!air.async.token) : i32 {
    %i = arith.index_cast %iv : i32 to index
    %off = arith.muli %i, %c2424832 : index
    %put = air.channel.put async [%t] @channel_0[] (%arg0[%c0, %off] [%c32, %c64] [%c37888, %c1]) {metadata = @airMemcpyId1} : (memref<4096x37888xbf16>)
    scf.yield %put : !air.async.token
  }
  return
}
air.channel @channel_0 [1, 1]

// -----

// The same walk already folded, inside a dimension over the column tiles.
// The over-limit dimension is merged into the rows it continues, and the
// 4096 rows are split with a legal stride.

// CHECK-LABEL: func.func @over_limit_inner_dim
// CHECK: air.channel.put {{.*}} [%c37{{.*}}, %c512{{.*}}, %c8{{.*}}, %c64{{.*}}] [%c256{{.*}}, %c303104{{.*}}, %c37888{{.*}}, %c1{{.*}}]
func.func @over_limit_inner_dim(%arg0: memref<4096x37888xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c37 = arith.constant 37 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %c37888 = arith.constant 37888 : index
  %c2424832 = arith.constant 2424832 : index
  %0 = air.channel.put async @channel_0[] (%arg0[%c0, %c0, %c0, %c0] [%c37, %c64, %c64, %c64] [%c256, %c2424832, %c37888, %c1]) {metadata = @airMemcpyId1} : (memref<4096x37888xbf16>)
  return
}
air.channel @channel_0 [1, 1]

// -----

// The over-limit dimension continues the rows inside it, but the 4087 rows it
// would make have no split with legal strides, so the two are left as they
// are for the lowering to unroll. Merging them anyway would leave a split that
// is still over the limit and contiguous, to be merged again.

// CHECK-LABEL: func.func @walk_without_legal_split
// CHECK: air.channel.put {{.*}} [%c67{{.*}}, %c61{{.*}}, %c64{{.*}}] [%c2311168{{.*}}, %c37888{{.*}}, %c1{{.*}}]
func.func @walk_without_legal_split(%arg0: memref<*xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c61 = arith.constant 61 : index
  %c64 = arith.constant 64 : index
  %c67 = arith.constant 67 : index
  %c37888 = arith.constant 37888 : index
  %c2311168 = arith.constant 2311168 : index
  %0 = air.channel.put async @channel_0[] (%arg0[%c0, %c0, %c0] [%c67, %c61, %c64] [%c2311168, %c37888, %c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
  return
}
air.channel @channel_0 [1, 1]

// -----

// The loop's span is the outer wrap times its stride, but the stride it would
// insert is reduced modulo the buffer and no longer continues the rows, so the
// loop is not folded and each step keeps its transfer.

// CHECK-LABEL: func.func @wrapped_stride
// CHECK-COUNT-2: air.channel.put {{.*}} [%c128{{.*}}, %c64{{.*}}] [%c37888{{.*}}, %c1{{.*}}]
// CHECK-NOT: air.channel.put
func.func @wrapped_stride(%arg0: memref<70x37888xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c37888 = arith.constant 37888 : index
  %c4849664 = arith.constant 4849664 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = air.wait_all async
  %1 = scf.for %iv = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%t = %0) -> (!air.async.token) : i32 {
    %i = arith.index_cast %iv : i32 to index
    %off = arith.muli %i, %c4849664 : index
    %put = air.channel.put async [%t] @channel_0[] (%arg0[%c0, %off] [%c128, %c64] [%c37888, %c1]) {metadata = @airMemcpyId1} : (memref<70x37888xbf16>)
    scf.yield %put : !air.async.token
  }
  return
}
air.channel @channel_0 [1, 1]
