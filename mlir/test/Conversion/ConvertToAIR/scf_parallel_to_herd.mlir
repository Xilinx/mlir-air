//===- scf_parallel_to_herd.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd %s | FileCheck %s
// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd="depth=-1" %s | FileCheck %s --check-prefix=DEPTHM1
// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd="depth=0" %s | FileCheck %s --check-prefix=DEPTH0

// CHECK-LABEL: func.func @scf0() {
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: air.herd @herd_0  tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C2]], {{.*}}=%[[C2]])
func.func @scf0()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %src = memref.alloc() : memref<2x2xi32, 2 : i32>
  %dst = memref.alloc() : memref<2x2xi32, 2 : i32>
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %0 = memref.load %src[%x, %y] : memref<2x2xi32, 2 : i32>
    memref.store %0, %dst[%x, %y] : memref<2x2xi32, 2 : i32>
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
// CHECK: air.herd @herd_0  tile (%[[A0:.*]], %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c1{{.*}})
// CHECK: affine.apply #[[M0]](%[[A0]])
func.func @scf1()  {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %src = memref.alloc() : memref<128xi32, 2 : i32>
  %dst = memref.alloc() : memref<128xi32, 2 : i32>
  scf.parallel (%x) = (%c0) to (%c128) step (%c32) {
    %0 = memref.load %src[%x] : memref<128xi32, 2 : i32>
    memref.store %0, %dst[%x] : memref<128xi32, 2 : i32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf2() {
// CHECK: scf.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (%c0{{.*}}, %c0{{.*}}) to (%c1{{.*}}, %c2{{.*}}) step (%c1{{.*}}, %c1{{.*}}) {
// CHECK:   air.herd @herd_0  tile (%[[VAL_7:.*]], %[[VAL_8:.*]]) in (%{{.*}}=%c3{{.*}}, %{{.*}}=%c4{{.*}}) args(%{{.*}}=%[[VAL_4]], %{{.*}}=%[[VAL_3]]) : index, index {
// CHECK:     memref.alloc() : memref<1x2x3x4xi32, 2 : i32>
// CHECK:     memref.alloc() : memref<1x2x3x4xi32, 2 : i32>
func.func @scf2()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %src = memref.alloc() : memref<1x2x3x4xi32, 2 : i32>
  %dst = memref.alloc() : memref<1x2x3x4xi32, 2 : i32>
  scf.parallel (%a,%b,%x,%y) = (%c0,%c0,%c0,%c0) to (%c1,%c2,%c3,%c4) step (%c1,%c1,%c1,%c1) {
    %0 = memref.load %src[%a,%b,%x,%y] : memref<1x2x3x4xi32, 2 : i32>
    memref.store %0, %dst[%a,%b,%x,%y] : memref<1x2x3x4xi32, 2 : i32>
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
//       CHECK:    }
//       CHECK:    air.herd @herd_0
//       CHECK:    }
//       CHECK:    air.herd @herd_0
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
// DEPTHM1-LABEL: @shared_herd_name
//       DEPTHM1:    scf.parallel {{.*}} {
//       DEPTHM1:      air.herd @herd_0
//       DEPTHM1:      }
//       DEPTHM1:      air.herd @herd_0
//       DEPTHM1:      }
//       DEPTHM1:      air.herd @herd_0
//       DEPTHM1:      }
//       DEPTHM1:      scf.reduce
//       DEPTHM1:    }
//       DEPTHM1:    return
//       DEPTHM1:  }
//       DEPTHM1: }
// DEPTH0-LABEL: @shared_herd_name
//       DEPTH0:    air.herd @herd_0
//       DEPTH0:      scf.parallel {{.*}}
//       DEPTH0:        scf.reduce
//       DEPTH0:      }
//       DEPTH0:      scf.parallel {{.*}}
//       DEPTH0:        scf.reduce
//       DEPTH0:      }
//       DEPTH0:      scf.parallel {{.*}}
//       DEPTH0:        scf.reduce
//       DEPTH0:      }
//       DEPTH0:    }
//       DEPTH0:    return
//       DEPTH0:  }
//       DEPTH0: }
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
//       CHECK:    }
//       CHECK:    air.herd @herd_1
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
// DEPTHM1-LABEL: @unique_herd_name
//       DEPTHM1:    scf.parallel {{.*}} {
//       DEPTHM1:      air.herd @herd_0
//       DEPTHM1:      }
//       DEPTHM1:      air.herd @herd_1
//       DEPTHM1:      }
//       DEPTHM1:      scf.reduce
//       DEPTHM1:    }
//       DEPTHM1:    return
//       DEPTHM1:  }
//       DEPTHM1: }
// DEPTH0-LABEL: @unique_herd_name
//       DEPTH0:    air.herd @herd_0
//       DEPTH0:      scf.parallel {{.*}} {
//       DEPTH0:        scf.reduce
//       DEPTH0:      }
//       DEPTH0:      scf.parallel {{.*}} {
//       DEPTH0:        scf.reduce
//       DEPTH0:      }
//       DEPTH0:    }
//       DEPTH0:    return
//       DEPTH0:  }
//       DEPTH0: }
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
//       CHECK:       memref.subview %{{.*}}[0, 0, %[[VAL_0]], %[[VAL_1]], 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
//       CHECK:    }
//       CHECK:    air.herd @herd_0
//       CHECK:       %[[VAL_0:.*]] = affine.apply
//       CHECK:       %[[VAL_1:.*]] = affine.apply
//       CHECK:       memref.subview %{{.*}}[0, 0, %[[VAL_0]], %[[VAL_1]], 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
//       CHECK:       %[[VAL_2:.*]] = affine.apply
//       CHECK:       %[[VAL_3:.*]] = affine.apply
//       CHECK:       memref.subview %{{.*}}[0, 0, %[[VAL_2]], %[[VAL_3]]] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x128x128xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[16384, 16384, 128, 1], offset: ?>, 1 : i32>
//       CHECK:       air.dma_memcpy_nd {{.*}} : (memref<1x1x64x64xbf16, strided<[16384, 16384, 128, 1], offset: ?>, 1 : i32>, memref<1x1x16x4x16x4xbf16, strided<[16384, 16384, 16, 4, 512, 1], offset: ?>, 2 : i32>)
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
// DEPTHM1-LABEL: @l2_to_l1_dma_infer_herd
//       DEPTHM1:    scf.parallel {{.*}} {
//       DEPTHM1:      air.herd @herd_0
//       DEPTHM1:      }
//       DEPTHM1:      air.herd @herd_0
//       DEPTHM1:      }
//       DEPTHM1:      scf.reduce
//       DEPTHM1:    }
//       DEPTHM1:    return
//       DEPTHM1:  }
//       DEPTHM1: }
// DEPTH0-LABEL: @l2_to_l1_dma_infer_herd
//       DEPTH0:    air.herd @herd_0
//       DEPTH0:      scf.parallel {{.*}} {
//       DEPTH0:        scf.reduce
//       DEPTH0:      }
//       DEPTH0:      scf.parallel {{.*}} {
//       DEPTH0:        scf.reduce
//       DEPTH0:      }
//       DEPTH0:    }
//       DEPTH0:    return
//       DEPTH0:  }
//       DEPTH0: }
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

// -----

// Lowering scf.parallel with an scf.reduce region to an air.herd representing a hardware reduction pipeline.

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0] : (s0 - 3 == 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>
// CHECK: air.channel @channel_0 [3] {channel_type = "cascade"}
// CHECK-LABEL: scf_reduce
// CHECK: air.herd @herd_0  tile (%[[arg0:.*]], %[[arg1:.*]]) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c1{{.*}})
// CHECK: %[[alloc_4:.*]] = memref.alloc() : memref<32xi32, 2 : i32>
// CHECK: linalg.fill{{.*}}outs(%[[alloc_4]]
// CHECK: scf.for
// CHECK: air.dma_memcpy_nd
// CHECK: air.dma_memcpy_nd
// CHECK: linalg.vecmat
// CHECK: }
// CHECK: affine.if [[$SET0]]()
// CHECK: %[[alloc_5:.*]] = memref.alloc()
// CHECK: linalg.fill{{.*}}outs(%[[alloc_5]]
// CHECK: linalg.add ins(%[[alloc_4]], %[[alloc_5]]{{.*}}outs(%[[alloc_4]]
// CHECK: %[[idx:.*]] = arith.subi %[[arg0]], %c1{{.*}}
// CHECK: air.channel.put  @channel_0[%[[idx]]] (%[[alloc_4]][] [] [])
// CHECK: } else {
// CHECK: affine.if [[$SET1]]()
// CHECK: %[[alloc_5:.*]] = memref.alloc()
// CHECK: linalg.fill{{.*}}outs(%[[alloc_5]]
// CHECK: air.channel.get  @channel_0[%[[arg0]]] (%alloc_5[] [] [])
// CHECK: linalg.add ins(%[[alloc_4]], %[[alloc_5]]{{.*}}outs(%[[alloc_4]]
// CHECK: %[[idx:.*]] = arith.subi %[[arg0]], %c1{{.*}}
// CHECK: air.channel.put  @channel_0[%[[idx]]] (%[[alloc_4]][] [] [])
// CHECK: } else {
// CHECK: %[[alloc_5:.*]] = memref.alloc()
// CHECK: linalg.fill{{.*}}outs(%[[alloc_5]]
// CHECK: air.channel.get  @channel_0[%[[arg0]]] (%[[alloc_5]][] [] [])
// CHECK: linalg.add ins(%[[alloc_4]], %[[alloc_5]]{{.*}}outs(%[[alloc_4]]
// CHECK: }
// CHECK: }

#map = affine_map<(d0, d1) -> (d0 * 32 + d1 * 128)>
module {
  func.func @scf_reduce() -> memref<256xi32> {
    %c0_i32 = arith.constant 0 : i32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<256xi32>
    %alloc_0 = memref.alloc() : memref<512xi32, 1 : i32>
    %alloc_1 = memref.alloc() : memref<512x64xi32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<32xi32, 2 : i32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_2 : memref<32xi32, 2 : i32>)
    %0 = scf.parallel (%arg2) = (%c0) to (%c4) step (%c1) init (%alloc_2) -> memref<32xi32, 2 : i32> {
      %alloc_3 = memref.alloc() : memref<32xi32, 2 : i32>
      linalg.fill ins(%c0_i32 : i32) outs(%alloc_3 : memref<32xi32, 2 : i32>)
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %1 = affine.apply #map(%arg3, %arg2)
        %alloc_4 = memref.alloc() : memref<32xi32, 2 : i32>
        %alloc_5 = memref.alloc() : memref<32x32xi32, 2 : i32>
        air.dma_memcpy_nd (%alloc_4[] [] [], %alloc_0[%1] [%c32] [%c1]) {id = 3 : i32} : (memref<32xi32, 2 : i32>, memref<512xi32, 1 : i32>)
        air.dma_memcpy_nd (%alloc_5[] [] [], %alloc_1[%1, %c0] [%c32, %c32] [%c64, %c1]) {id = 4 : i32} : (memref<32x32xi32, 2 : i32>, memref<512x64xi32, 1 : i32>)
        linalg.vecmat ins(%alloc_4, %alloc_5 : memref<32xi32, 2 : i32>, memref<32x32xi32, 2 : i32>) outs(%alloc_3 : memref<32xi32, 2 : i32>)
        memref.dealloc %alloc_4 : memref<32xi32, 2 : i32>
        memref.dealloc %alloc_5 : memref<32x32xi32, 2 : i32>
      }
      scf.reduce(%alloc_3 : memref<32xi32, 2 : i32>) {
      ^bb0(%arg3: memref<32xi32, 2 : i32>, %arg4: memref<32xi32, 2 : i32>):
        linalg.add ins(%arg3, %arg4 : memref<32xi32, 2 : i32>, memref<32xi32, 2 : i32>) outs(%arg3 : memref<32xi32, 2 : i32>)
        scf.reduce.return %arg3 : memref<32xi32, 2 : i32>
      }
    }
    return %alloc : memref<256xi32>
  }
}

// -----

// Lowering scf.parallel with an scf.reduce: maintain air.herd's IsolateFromAbove trait.

// CHECK-LABEL: scf_reduce_1
// CHECK: air.herd @herd_0  tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c1{{.*}}) args(%{{.*}}=%{{.*}}) : index

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  func.func @scf_reduce_1(%arg0: memref<512xi32>, %arg1: memref<512x256xi32>, %arg2: memref<256xi32>, %arg3: index) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.apply #map()[%arg3]
    %alloc = memref.alloc() : memref<64xi32, 1>
    %alloc_0 = memref.alloc() : memref<32xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<32xi32, 2>)
    %1 = scf.parallel (%arg4) = (%c0) to (%c4) step (%c1) init (%alloc_0) -> memref<32xi32, 2> {
      %alloc_1 = memref.alloc() : memref<32x1xi32, 2>
      %subview = memref.subview %alloc_1[0, 0] [32, 1] [1, 1] : memref<32x1xi32, 2> to memref<32xi32, strided<[1]>, 2>
      %cast = memref.cast %subview : memref<32xi32, strided<[1]>, 2> to memref<32xi32, 2>
      memref.dealloc %alloc_1 : memref<32x1xi32, 2>
      scf.reduce(%cast : memref<32xi32, 2>) {
      ^bb0(%arg5: memref<32xi32, 2>, %arg6: memref<32xi32, 2>):
        linalg.add ins(%arg5, %arg6 : memref<32xi32, 2>, memref<32xi32, 2>) outs(%arg5 : memref<32xi32, 2>)
        scf.reduce.return %arg5 : memref<32xi32, 2>
      }
    }
    air.dma_memcpy_nd (%alloc[%0] [%c32] [%c1], %1[] [] []) {id = 5 : i32} : (memref<64xi32, 1>, memref<32xi32, 2>)
    memref.dealloc %alloc_0 : memref<32xi32, 2>
    memref.dealloc %alloc : memref<64xi32, 1>
    return
  }
}
