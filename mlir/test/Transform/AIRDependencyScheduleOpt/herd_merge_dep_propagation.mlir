// RUN: air-opt %s -air-isolate-async-dma-loop-nests="scope=func" | FileCheck %s

// Verify that when multiple same-named herds with inter-herd dependencies are
// merged into one, the inter-herd ordering is preserved as intra-herd deps.
// Herd 2 (compute) depends on herd 1 (fill), and herd 3 (output) depends on
// herd 2. After merge, the output group's ops must depend on the compute
// group's completion.

// CHECK-LABEL: func.func @herd_merge_dep_propagation
// CHECK: air.segment
// After merge, there should be a single herd with barrier deps between groups.
// After merge, single herd with correct intra-herd dependency chain:
// CHECK: air.herd @herd_0 async
// The fill group:
// CHECK: %[[FILL_TOK:.*]] = air.execute
// CHECK-NEXT: linalg.fill
// The compute group's execute must depend on fill (barrier from fill_herd):
// CHECK: %[[COMPUTE_TOK:.*]] = air.execute [%[[FILL_TOK]]
// CHECK-NEXT: linalg.fill
// The output channel.put must depend on compute (barrier from compute_herd):
// CHECK: air.channel.put async [%[[COMPUTE_TOK]]

module {
  air.channel @channel_out [1, 1]
  func.func @herd_merge_dep_propagation(%arg0: memref<32x32xi16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<32x32xi16> attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32} {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c256 = arith.constant 256 : index
        %c1024 = arith.constant 1024 : index
        %async_token_0, %alloc_L1 = air.execute -> (memref<1x1x4x8x4x8xi16, 2 : i32>) {
          %a = memref.alloc() : memref<1x1x4x8x4x8xi16, 2 : i32>
          air.execute_terminator %a : memref<1x1x4x8x4x8xi16, 2 : i32>
        } {id = 1 : i32}
        // Herd 1: fill
        %2 = air.herd @herd_0 async [%async_token_0] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%buf=%alloc_L1) : memref<1x1x4x8x4x8xi16, 2 : i32> attributes {id = 3 : i32} {
          %cst = arith.constant 0 : i16
          %subview = memref.subview %buf[%tx, %ty, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi16, 2 : i32> to memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%subview : memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>)
          } {id = 4 : i32}
        }
        // Herd 2: compute (depends on herd 1)
        %3 = air.herd @herd_0 async [%2] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%buf=%alloc_L1) : memref<1x1x4x8x4x8xi16, 2 : i32> attributes {id = 5 : i32} {
          %cst = arith.constant 0 : i16
          %subview = memref.subview %buf[%tx, %ty, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi16, 2 : i32> to memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%subview : memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>)
          } {id = 6 : i32}
        }
        // Herd 3: output channel.put (depends on herd 2)
        %4 = air.herd @herd_0 async [%3] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%buf=%alloc_L1) : memref<1x1x4x8x4x8xi16, 2 : i32> attributes {id = 7 : i32} {
          %ci1 = arith.constant 1 : index
          %ci8 = arith.constant 8 : index
          %ci32 = arith.constant 32 : index
          %ci256 = arith.constant 256 : index
          %ci1024 = arith.constant 1024 : index
          %5 = air.channel.put async @channel_out[%tx, %ty] (%buf[%tx, %ty, %tx, %tx, %tx, %tx] [%ci1, %ci1, %ci8, %ci8, %ci8, %ci8] [%ci1024, %ci1024, %ci32, %ci8, %ci256, %ci1]) {id = 8 : i32} : (memref<1x1x4x8x4x8xi16, 2 : i32>)
        }
      }
    }
    return
  }
}
