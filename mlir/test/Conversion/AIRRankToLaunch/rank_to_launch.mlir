// RUN: air-opt %s -air-rank-to-launch | FileCheck %s

// CHECK-LABEL: func.func @test_rank_1d_serialization
// CHECK-NOT: air.rank
// CHECK: scf.for
// CHECK:   air.launch
func.func @test_rank_1d_serialization(%arg0 : memref<16x16xf32>) {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) args(%a=%arg0) : memref<16x16xf32> {
    %c1 = arith.constant 1 : index
    air.launch (%lx) in (%ls = %c1) args(%la=%a) : memref<16x16xf32> {
    }
  }
  return
}

// CHECK-LABEL: func.func @test_rank_2d_serialization
// CHECK-NOT: air.rank
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     air.launch
func.func @test_rank_2d_serialization(%arg0 : memref<16x16xf32>) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  air.rank (%rx, %ry) in (%sx = %c2, %sy = %c4) args(%a=%arg0) : memref<16x16xf32> {
    %c1 = arith.constant 1 : index
    air.launch (%lx) in (%ls = %c1) args(%la=%a) : memref<16x16xf32> {
    }
  }
  return
}

// CHECK-LABEL: func.func @test_rank_async_serialization
// CHECK-NOT: air.rank
// CHECK: scf.for
// CHECK: air.wait_all async
func.func @test_rank_async_serialization(%arg0 : memref<16x16xf32>) {
  %c2 = arith.constant 2 : index
  %t0 = air.rank async (%rx) in (%sx = %c2) args(%a=%arg0) : memref<16x16xf32> {
    %c1 = arith.constant 1 : index
    air.launch (%lx) in (%ls = %c1) args(%la=%a) : memref<16x16xf32> {
    }
  }
  air.wait_all [%t0]
  return
}
