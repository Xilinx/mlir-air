//===- scf_parallel_to_segment.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-par-to-segment %s | FileCheck %s
// RUN: air-opt -air-par-to-segment="depth=1" %s | FileCheck %s --check-prefix=DEPTH1
// CHECK-LABEL: func.func @f0
// CHECK air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
// DEPTH1-LABEL: func.func @f0
// DEPTH1 scf.parallel (%{{.*}}, %{{.*}}) = (%c0{{.*}}, %c0{{.*}}) to (%c2{{.*}}, %c2{{.*}}) step (%c1{{.*}}, %c1{{.*}})
func.func @f0()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%x,%y) = (%c0,%c0) to (%c2, %c2) step (%c1,%c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// CHECK-LABEL: func.func @f1
// CHECK air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c4{{.*}})
// CHECK air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
// CHECK air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
// DEPTH1-LABEL: func.func @f1
// DEPTH1 scf.parallel (%{{.*}}, %{{.*}}) = (%c0{{.*}}, %c0{{.*}}) to (%c512{{.*}}, %c512{{.*}}) step (%c128{{.*}}, %c128{{.*}})
// DEPTH1 air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
// DEPTH1 scf.parallel (%{{.*}}, %{{.*}}) = (%c0{{.*}}, %c0{{.*}}) to (%c32{{.*}}, %c32{{.*}}) step (%c16{{.*}}, %c16{{.*}})
func.func @f1() {
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
    scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
      scf.parallel (%arg7, %arg8) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
        %2 = arith.addi %arg3, %arg4 : index
        %3 = arith.addi %arg5, %arg6 : index
        %4 = arith.addi %arg7, %arg8 : index
        scf.reduce 
      }
      scf.reduce 
    }
    scf.reduce 
  }
  return
}

// CHECK-LABEL: func.func @f2
// CHECK air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c4{{.*}})
// CHECK air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
// CHECK air.herd {{.*}} tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
// DEPTH1-LABEL: func.func @f2
// DEPTH1 scf.parallel (%{{.*}}, %{{.*}}) = (%c0{{.*}}, %c0{{.*}}) to (%c512{{.*}}, %c512{{.*}}) step (%c128{{.*}}, %c128{{.*}})
// DEPTH1 air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
// DEPTH1 air.herd {{.*}} tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
#map = affine_map<(d0) -> (d0 * 16)>
func.func @f2() {
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
      %c2 = arith.constant 2 : index
      air.herd @herd_0  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : index, index, index, index {
        %0 = affine.apply #map(%arg5)
        %1 = affine.apply #map(%arg4)
        %2 = arith.addi %arg8, %arg9 : index
        %3 = arith.addi %arg10, %arg11 : index
        %4 = arith.addi %1, %0 : index
      }
      scf.reduce 
    }
    scf.reduce 
  }
  return
}

// CHECK-LABEL: func.func @f3
// CHECK air.launch {{.*}} (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c4{{.*}})
// CHECK air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
// CHECK air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
// DEPTH1-LABEL: func.func @f3
// DEPTH1 air.launch {{.*}} (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c4{{.*}})
// DEPTH1 scf.parallel (%{{.*}}, %{{.*}}) = (%c0{{.*}}, %c0{{.*}}) to (%c32{{.*}}, %c32{{.*}}) step (%c16{{.*}}, %c16{{.*}})
// DEPTH1 air.segment {{.*}} unroll (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
#map1 = affine_map<(d0) -> (d0 * 128)>
func.func @f3() {
  %cst = arith.constant 0.000000e+00 : bf16
  %c4 = arith.constant 4 : index
  air.launch (%arg0, %arg1) in (%arg2=%c4, %arg3=%c4) {
    %c0_7 = arith.constant 0 : index
    %c32_8 = arith.constant 32 : index
    %c16_9 = arith.constant 16 : index
    %0 = affine.apply #map1(%arg1)
    %1 = affine.apply #map1(%arg0)
    scf.parallel (%arg4, %arg5) = (%c0_7, %c0_7) to (%c32_8, %c32_8) step (%c16_9, %c16_9) {
      scf.parallel (%arg6, %arg7) = (%c0_7, %c0_7) to (%c32_8, %c32_8) step (%c16_9, %c16_9) {
        %2 = arith.addi %1, %0 : index
        %3 = arith.addi %arg4, %arg5 : index
        %4 = arith.addi %arg6, %arg7 : index
        scf.reduce 
      }
      scf.reduce 
    }
  }
  return
}

