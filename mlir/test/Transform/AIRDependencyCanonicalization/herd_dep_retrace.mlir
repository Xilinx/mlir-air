// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// Verify that air-dependency-canonicalize correctly establishes deps between
// herds that access the same memref. When herd_1 (fill) and herd_2 (compute)
// both write to %alloc_L1, and herd_3 (output) reads from %alloc_L1, herd_3
// must depend on herd_2 (not just herd_1).

// CHECK-LABEL: func.func @herd_dep_retrace
// CHECK: %[[ALLOC_L1:.*]], %{{.*}} = air.execute
// CHECK: %[[FILL_HERD:.*]] = air.herd @herd_0 async [%[[ALLOC_L1]]]
// CHECK: %[[COMPUTE_HERD:.*]] = air.herd @herd_0 async [%[[FILL_HERD]]]
// The output herd must depend on the compute herd (not just fill).
// CHECK: air.herd @herd_0 async [%[[COMPUTE_HERD]]]

module {
  func.func @herd_dep_retrace(%arg0: memref<32x32xi16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<32x32xi16> attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32} {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        // Allocate shared L1 output buffer
        %async_token_0, %alloc_L1 = air.execute -> (memref<1x1x4x8x4x8xi16, 2 : i32>) {
          %a = memref.alloc() : memref<1x1x4x8x4x8xi16, 2 : i32>
          air.execute_terminator %a : memref<1x1x4x8x4x8xi16, 2 : i32>
        } {id = 1 : i32}
        // Herd 1: fill the L1 buffer
        %2 = air.herd @herd_0 async [%async_token_0] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%buf=%alloc_L1) : memref<1x1x4x8x4x8xi16, 2 : i32> attributes {id = 3 : i32} {
          %subview = memref.subview %buf[%tx, %ty, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi16, 2 : i32> to memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
          %cst = arith.constant 0 : i16
          %async_token_fill = air.execute {
            linalg.fill ins(%cst : i16) outs(%subview : memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>)
          } {id = 4 : i32}
        }
        // Herd 2: compute (writes to same L1 buffer)
        %3 = air.herd @herd_0 async [%2] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%buf=%alloc_L1) : memref<1x1x4x8x4x8xi16, 2 : i32> attributes {id = 5 : i32} {
          %subview = memref.subview %buf[%tx, %ty, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi16, 2 : i32> to memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
          %cst = arith.constant 0 : i16
          %async_token_compute = air.execute {
            linalg.fill ins(%cst : i16) outs(%subview : memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>)
          } {id = 6 : i32}
        }
        // Herd 3: output (reads from same L1 buffer) — should depend on herd 2
        %4 = air.herd @herd_0 async [%2] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%buf=%alloc_L1) : memref<1x1x4x8x4x8xi16, 2 : i32> attributes {id = 7 : i32} {
          %subview = memref.subview %buf[%tx, %ty, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi16, 2 : i32> to memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
          %cst = arith.constant 0 : i16
          %async_token_output = air.execute {
            linalg.fill ins(%cst : i16) outs(%subview : memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>)
          } {id = 8 : i32}
        }
      }
    }
    return
  }
}
