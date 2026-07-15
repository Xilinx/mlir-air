// RUN: air-opt -canonicalize -split-input-file %s | FileCheck %s

// Reproducer for #1559, lifted from the herd body of
// programming_examples/matrix_multiplication/i8 build_peano/air_project_1x1/
// placed.air_input.mlir (--herd-m 1 --herd-n 1 --m 32 --n 32 --k 32
// --tile-m 32 --tile-k-l2 32 --tile-k-l1 16 --tile-n 32).
//
// The fill writes %subview = memref.subview %results_17[...]; the
// channel.put reads the base %results_17. Without alias-aware
// CanonicalizeAsyncOpDeps, the dep on the fill token (%async_token_18)
// is incorrectly dropped because %subview and %results_17 are different
// SSA values.

// CHECK-LABEL: func.func @reproducer_subview_alias
// CHECK: %{{.*}}, %[[BUF:[a-zA-Z0-9_]+]] = air.execute -> (memref<1x1x4x8x4x8xi16, 2 : i32>)
// CHECK: memref.subview %[[BUF]]
// CHECK: %[[FILL_TOK:[a-zA-Z0-9_]+]] = air.execute [
// CHECK:   func.call @linalg_fill_i16_view1x1x4x8x4x8xi16as2
// The fill→channel.put RAW edge must survive canonicalize:
// CHECK: air.channel.put async [%[[FILL_TOK]]] @channel_4{{.*}} (%[[BUF]]

module {
  air.channel @channel_4 [1, 1]
  func.func private @linalg_fill_i16_view1x1x4x8x4x8xi16as2(i16, memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>) attributes {link_with = "mm.o", llvm.emit_c_interface}
  func.func @reproducer_subview_alias() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @matmul_seg async attributes {id = 2 : i32} {
        %c1_3 = arith.constant 1 : index
        %2 = air.herd @herd_0 async tile (%arg10, %arg11) in (%arg12=%c1_3, %arg13=%c1_3) attributes {id = 5 : i32} {
          %c0_i16 = arith.constant 0 : i16
          %c0_13 = arith.constant 0 : index
          %c1_12 = arith.constant 1 : index
          %c4_15 = arith.constant 4 : index
          %c8_14 = arith.constant 8 : index
          %c32_11 = arith.constant 32 : index
          %c256 = arith.constant 256 : index
          %async_token_16, %results_17 = air.execute -> (memref<1x1x4x8x4x8xi16, 2 : i32>) {
            %alloc = memref.alloc() : memref<1x1x4x8x4x8xi16, 2 : i32>
            air.execute_terminator %alloc : memref<1x1x4x8x4x8xi16, 2 : i32>
          }
          %subview = memref.subview %results_17[%arg10, %arg11, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi16, 2 : i32> to memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
          %async_token_18 = air.execute [%async_token_16] {
            func.call @linalg_fill_i16_view1x1x4x8x4x8xi16as2(%c0_i16, %subview) : (i16, memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>) -> ()
          }
          %20 = air.channel.put async [%async_token_16, %async_token_18] @channel_4[%arg10, %arg11] (%results_17[%c0_13, %c0_13, %c0_13] [%c32_11, %c4_15, %c8_14] [%c8_14, %c256, %c1_12]) {id = 13 : i32} : (memref<1x1x4x8x4x8xi16, 2 : i32>)
          %async_token_33 = air.execute [%20] {
            memref.dealloc %results_17 : memref<1x1x4x8x4x8xi16, 2 : i32>
          }
        }
      }
    }
    return
  }
}

// -----

// Negative test: confirms canonicalize STILL removes a genuinely dead
// async dep. Two independent allocs %A and %B; the channel.put reads
// only %A but lists both fill tokens as deps. The dep on the fill of
// %B has no RAW/WAR/WAW with the put and must be dropped — otherwise
// a buggy mayAlias that always returns true would also pass the
// reproducer test above.

// CHECK-LABEL: func.func @dead_dep_removed
// CHECK: %{{.*}}, %[[A:[a-zA-Z0-9_]+]] = air.execute -> (memref<8x8xi16
// CHECK: %{{.*}}, %[[B:[a-zA-Z0-9_]+]] = air.execute -> (memref<8x8xi16
// CHECK: %[[FILL_A:[a-zA-Z0-9_]+]] = air.execute [%{{.*}}] {
// CHECK-NEXT:   linalg.fill {{.*}} outs(%[[A]]
// CHECK: %[[FILL_B:[a-zA-Z0-9_]+]] = air.execute [%{{.*}}] {
// CHECK-NEXT:   linalg.fill {{.*}} outs(%[[B]]
// The put reads %A only; the dep on %FILL_B (which writes %B) must be
// dropped. CHECK requires the remaining dep list to be exactly
// [%FILL_A] — a buggy mayAlias that always returns true would leave
// both deps and fail this check.
// CHECK: air.channel.put async [%[[FILL_A]]] @channel_x[] (%[[A]]

module {
  air.channel @channel_x []
  func.func @dead_dep_removed() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32} {
        %c0_i16 = arith.constant 0 : i16
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c8_s = arith.constant 8 : index
        %tok_A, %A = air.execute -> (memref<8x8xi16, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x8xi16, 1 : i32>
          air.execute_terminator %alloc : memref<8x8xi16, 1 : i32>
        }
        %tok_B, %B = air.execute -> (memref<8x8xi16, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x8xi16, 1 : i32>
          air.execute_terminator %alloc : memref<8x8xi16, 1 : i32>
        }
        %fill_A = air.execute [%tok_A] {
          linalg.fill ins(%c0_i16 : i16) outs(%A : memref<8x8xi16, 1 : i32>)
        }
        %fill_B = air.execute [%tok_B] {
          linalg.fill ins(%c0_i16 : i16) outs(%B : memref<8x8xi16, 1 : i32>)
        }
        %put = air.channel.put async [%fill_A, %fill_B] @channel_x[] (%A[%c0_s, %c0_s] [%c8_s, %c8_s] [%c8_s, %c1_s]) {id = 1 : i32} : (memref<8x8xi16, 1 : i32>)
        %dA = air.execute [%put] {
          memref.dealloc %A : memref<8x8xi16, 1 : i32>
        }
        %dB = air.execute [%fill_B] {
          memref.dealloc %B : memref<8x8xi16, 1 : i32>
        }
      }
    }
    return
  }
}
