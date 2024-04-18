//===- scf_parallel_to_herd_launch.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd -cse %s | FileCheck %s

// CHECK-LABEL: func.func @scf0() {
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: air.herd @herd_0  tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C2]], {{.*}}=%[[C2]])
func.func @scf0()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

func.func @scferror0(%c0 : index)  {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: lower bound is not a constant}}
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

func.func @scferror1(%c1 : index)  {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: step is not a constant}}
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

func.func @scferror2(%c2 : index)  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: upper bound is not a constant}}
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

func.func @scferror3()  {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: step '2' does not evenly divide range '7'}}
  scf.parallel (%x,%y) = (%c2, %c2) to (%c9, %c9) step (%c2, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

// CHECK: #[[M0:.*]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-LABEL: func.func @scf1() {
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: air.herd @herd_0  tile (%[[A0:.*]], {{.*}}) in ({{.*}}=%[[C4]], {{.*}}=%[[C1]])
// CHECK: affine.apply #[[M0]](%[[A0]])
func.func @scf1()  {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%x) = (%c0) to (%c128) step (%c32) {
    %2 = arith.muli %x, %x : index
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf2() {
// CHECK: %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK: scf.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (%[[VAL_1]], %[[VAL_1]]) to (%[[VAL_0]], %[[VAL_2]]) step (%[[VAL_0]], %[[VAL_0]]) {
// CHECK:   %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:   %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:   air.herd @herd_0  tile (%[[VAL_7:.*]], %[[VAL_8:.*]]) in (%[[VAL_9:.*]]=%[[VAL_5]], %[[VAL_10:.*]]=%[[VAL_6]]) args(%[[VAL_11:.*]]=%[[VAL_3]], %[[VAL_12:.*]]=%[[VAL_4]]) : index, index
func.func @scf2()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%a,%b,%x,%y) = (%c0,%c0,%c0,%c0) to (%c1,%c2,%c3,%c4) step (%c1,%c1,%c1,%c1) {
    %2 = arith.muli %x, %y : index
  }
  return
}

// -----

// This test demonstrates that while forming air.herd we look through func.call ops, fetch
// the corresponding function declaration's 'link_with' attribute and attach it to the newly
// formed air.herd op.

// CHECK-LABEL: module {
//       CHECK:  func.func private @matmul_i32_i32
//  CHECK-SAME:        attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
//       CHECK:  func.func @matmul_small_dispatch_0_matmul_8x32x16_i32(
//       CHECK:    air.herd @herd_0
//  CHECK-SAME:        attributes {link_with = "/path/to/mm_microkernel.o"} {
//       CHECK:       func.call @matmul_i32_i32
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
module {
  func.func private @matmul_i32_i32(memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
  func.func @matmul_small_dispatch_0_matmul_8x32x16_i32(%base_buffer: memref<i32, 2 : i32>, %base_buffer_14: memref<i32, 2 : i32>, %base_buffer_18: memref<i32, 2 : i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%x,%y) = (%c0,%c0) to (%c1,%c1) step (%c1, %c1) {
      %2 = arith.addi %x, %y : index
      func.call @matmul_i32_i32(%base_buffer, %c0, %base_buffer_14, %c0, %base_buffer_18, %c0) : (memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) -> ()
      scf.reduce
    }
    return
  }
}

// -----

// This test demonstrates the relaying of `link_with` construct to air.herd op even if the
// func.call op is not an immediate child of scf.parallel.

// CHECK-LABEL: module {
//       CHECK:  func.func private @matmul_scalar_i32_i32
//  CHECK-SAME:        attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
//       CHECK:  func.func @matmul_small_nested_scf_dispatch_0_matmul_8x32x16_i32(
//       CHECK:    air.herd @herd_0
//  CHECK-SAME:        attributes {link_with = "/path/to/mm_microkernel.o"} {
//       CHECK:       scf.for
//  CHECK-SAME:       {
//       CHECK:           func.call @matmul_scalar_i32_i32
//       CHECK:       }
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
module {
  func.func private @matmul_scalar_i32_i32(memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
  func.func @matmul_small_nested_scf_dispatch_0_matmul_8x32x16_i32(%base_buffer: memref<i32, 2 : i32>, %base_buffer_14: memref<i32, 2 : i32>, %base_buffer_18: memref<i32, 2 : i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    scf.parallel (%x,%y) = (%c0,%c0) to (%c1,%c1) step (%c1, %c1) {
      %2 = arith.addi %x, %y : index
      scf.for %arg0 = %c0 to %c32 step %c4 {
        func.call @matmul_scalar_i32_i32(%base_buffer, %c0, %base_buffer_14, %c0, %base_buffer_18, %c0) : (memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) -> ()
      }
      scf.reduce
    }
    return
  }
}

// -----

// This test demonstrates how to assign multiple air.herds, which use some shared L1 memrefs, with the same symbolic name, in order to represent that they shall get mapped to the same set of compute resources.

// CHECK-LABEL: module {
//       CHECK:  func.func @shared_herd_name(
//       CHECK:    air.herd @herd_0
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    air.herd @herd_0
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    air.herd @herd_0
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
module {
  func.func @shared_herd_name(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
      %alloc_3 = memref.alloc() : memref<1x1x32x32x4x4xbf16, 2 : i32>
      %alloc_4 = memref.alloc() : memref<1x1x128x128xbf16, 1 : i32>
      scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
        %subview_16 = memref.subview %alloc_3[0, 0, %arg6, %arg5, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
        linalg.fill ins(%cst : bf16) outs(%subview_16 : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>)
        scf.reduce 
      }
      scf.for %arg5 = %c1 to %c16 step %c1 {
        scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
          %subview_18 = memref.subview %alloc_3[0, 0, %arg7, %arg6, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
          linalg.fill ins(%cst : bf16) outs(%subview_18 : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>)
          scf.reduce 
        }
      }
      %transpose = memref.transpose %alloc_3 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x32x4x32x4xbf16, strided<[16384, 16384, 16, 4, 512, 1]>, 2 : i32>
      air.dma_memcpy_nd (%alloc_4[] [] [], %transpose[] [] []) : (memref<1x1x128x128xbf16, 1 : i32>, memref<1x1x32x4x32x4xbf16, strided<[16384, 16384, 16, 4, 512, 1]>, 2 : i32>)
      memref.dealloc %alloc_4 : memref<1x1x128x128xbf16, 1 : i32>
      memref.dealloc %alloc_3 : memref<1x1x32x32x4x4xbf16, 2 : i32>
      scf.reduce 
    }
    return
  }
}

// -----

// Contrary to the "shared_herd_name" test above, when multiple herds do not share any L1 memref, then they must be assigned with unique symbolic names, and subsequently mapped to distinct hw compute resources.

// CHECK-LABEL: module {
//       CHECK:  func.func @unique_herd_name(
//       CHECK:    air.herd @herd_0
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    air.herd @herd_1
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
module {
  func.func @unique_herd_name(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
      %alloc_1 = memref.alloc() : memref<1x1x32x32x4x4xbf16, 2 : i32>
      %alloc_2 = memref.alloc() : memref<1x1x32x32x4x4xbf16, 2 : i32>
      scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
        %subview_16 = memref.subview %alloc_1[0, 0, %arg6, %arg5, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
        linalg.fill ins(%cst : bf16) outs(%subview_16 : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>)
        scf.reduce 
      }
      scf.for %arg5 = %c1 to %c16 step %c1 {
        scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
          %subview_18 = memref.subview %alloc_2[0, 0, %arg7, %arg6, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
          linalg.fill ins(%cst : bf16) outs(%subview_18 : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>)
          scf.reduce 
        }
      }
      memref.dealloc %alloc_2 : memref<1x1x32x32x4x4xbf16, 2 : i32>
      memref.dealloc %alloc_1 : memref<1x1x32x32x4x4xbf16, 2 : i32>
      scf.reduce 
    }
    return
  }
}

// -----

// This test demonstrates how to infer an air.dma_memcpy_nd op between L2 and L1, not within two scf.parallel loop nests, gets inferred with a herd around it. 

// CHECK-LABEL: module {
//       CHECK:  func.func @l2_to_l1_dma_infer_herd(
//       CHECK:    air.herd @herd_0
//       CHECK:       %[[VAL_0:.*]] = affine.apply
//       CHECK:       %[[VAL_1:.*]] = affine.apply
//       CHECK:       memref.subview %{{.*}}[0, 0, %[[VAL_1]], %[[VAL_0]], 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    air.herd @herd_0
//       CHECK:       %[[VAL_0:.*]] = affine.apply
//       CHECK:       %[[VAL_1:.*]] = affine.apply
//       CHECK:       memref.subview %{{.*}}[0, 0, %[[VAL_1]], %[[VAL_0]], 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
//       CHECK:       %[[VAL_2:.*]] = affine.apply
//       CHECK:       %[[VAL_3:.*]] = affine.apply
//       CHECK:       memref.subview %{{.*}}[0, 0, %[[VAL_2]], %[[VAL_3]]] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x128x128xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[16384, 16384, 128, 1], offset: ?>, 1 : i32>
//       CHECK:       air.dma_memcpy_nd {{.*}} : (memref<1x1x64x64xbf16, strided<[16384, 16384, 128, 1], offset: ?>, 1 : i32>, memref<1x1x16x4x16x4xbf16, strided<[16384, 16384, 16, 4, 512, 1], offset: ?>, 2 : i32>)
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
module {
  func.func @l2_to_l1_dma_infer_herd() {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
      %alloc_3 = memref.alloc() : memref<1x1x32x32x4x4xbf16, 2 : i32>
      %alloc_4 = memref.alloc() : memref<1x1x128x128xbf16, 1 : i32>
      scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
        %subview_16 = memref.subview %alloc_3[0, 0, %arg6, %arg5, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
        linalg.fill ins(%cst : bf16) outs(%subview_16 : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>)
        scf.reduce 
      }
      %transpose = memref.transpose %alloc_3 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x32x4x32x4xbf16, strided<[16384, 16384, 16, 4, 512, 1]>, 2 : i32>
      air.dma_memcpy_nd (%alloc_4[] [] [], %transpose[] [] []) : (memref<1x1x128x128xbf16, 1 : i32>, memref<1x1x32x4x32x4xbf16, strided<[16384, 16384, 16, 4, 512, 1]>, 2 : i32>)
      memref.dealloc %alloc_4 : memref<1x1x128x128xbf16, 1 : i32>
      memref.dealloc %alloc_3 : memref<1x1x32x32x4x4xbf16, 2 : i32>
      scf.reduce 
    }
    return
  }
}
