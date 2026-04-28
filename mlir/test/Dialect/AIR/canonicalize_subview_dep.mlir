// RUN: air-opt -canonicalize %s | FileCheck %s

// Verify that canonicalize does not remove an async dependency between ops
// that access the same memory through a subview alias. The channel.put reads
// the base memref %buf, while the execute writes to %subview (a subview of
// %buf). The dependency must be preserved.

// CHECK-LABEL: func.func @subview_dep_preserved
// CHECK: air.execute -> (memref
// CHECK: memref.subview
// CHECK: %[[FILL_TOK:.*]] = air.execute [
// CHECK: linalg.fill
// The channel.put must still depend on the fill token (not removed
// despite fill writing to %subview and channel.put reading from %buf,
// because they alias through memref.subview):
// CHECK: air.channel.put async [%[[FILL_TOK]]]

module {
  air.channel @channel_out [1, 1]
  func.func @subview_dep_preserved() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32} {
        %c1_s = arith.constant 1 : index
        %2 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) attributes {id = 3 : i32} {
          %c0_i16 = arith.constant 0 : i16
          %c0_0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c8_0 = arith.constant 8 : index
          %c32_0 = arith.constant 32 : index
          %c256_0 = arith.constant 256 : index
          %c1024_0 = arith.constant 1024 : index
          // Alloc the output buffer
          %async_token_alloc, %buf = air.execute -> (memref<1x1x4x8x4x8xi16, 2 : i32>) {
            %alloc = memref.alloc() : memref<1x1x4x8x4x8xi16, 2 : i32>
            air.execute_terminator %alloc : memref<1x1x4x8x4x8xi16, 2 : i32>
          } {id = 4 : i32}
          // Create subview
          %subview = memref.subview %buf[%tx, %ty, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi16, 2 : i32> to memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
          // Write to subview
          %async_token_fill = air.execute [%async_token_alloc] {
            linalg.fill ins(%c0_i16 : i16) outs(%subview : memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>)
          } {id = 5 : i32}
          // Read from base memref — should preserve dep on fill
          %3 = air.channel.put async [%async_token_alloc, %async_token_fill] @channel_out[%tx, %ty] (%buf[%c0_0, %c0_0, %c0_0, %c0_0, %c0_0, %c0_0] [%c1_0, %c1_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c1024_0, %c1024_0, %c32_0, %c8_0, %c256_0, %c1_0]) {id = 6 : i32} : (memref<1x1x4x8x4x8xi16, 2 : i32>)
          %async_token_dealloc = air.execute [%3] {
            memref.dealloc %buf : memref<1x1x4x8x4x8xi16, 2 : i32>
          } {id = 7 : i32}
        }
      }
    }
    return
  }
}
