//===- cascade_placement.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=6 num-cols=4 row-anchor=2 col-anchor=0' | FileCheck %s

// Test that cascade-connected herds are placed as neighbors.
// The cascade connections are:
// - conv3x3_coreA -> conv1x1_skip via L1ToL1_Conv3x3AToSkip (cascade channel)
// - conv3x3_coreB -> conv1x1_skip via L1ToL1_Conv3x3BToSkip (cascade channel)
//
// The cascade-aware placement coordinates multiple producers for the same consumer:
// - conv3x3_coreA at (1, 4): NORTH of conv1x1_skip (north-to-south cascade)
// - conv3x3_coreB at (0, 3): WEST of conv1x1_skip (west-to-east cascade)
// - conv1x1_skip at (1, 3): adjacent to BOTH producers
//
// This satisfies the AIE cascade constraint where cascade connections must be
// between immediate neighbors (west-to-east or north-to-south).

// CHECK: air.herd @conv1x1_reduce {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 2 : i64}
// CHECK: air.herd @conv3x3_coreA {{.*}} attributes {{{.*}}x_loc = 1 : i64, y_loc = 4 : i64}
// CHECK: air.herd @conv3x3_coreB {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 3 : i64}
// CHECK: air.herd @conv1x1_skip {{.*}} attributes {{{.*}}x_loc = 1 : i64, y_loc = 3 : i64}

module {
  air.channel @L1ToL1_Conv3x3AToSkip [1] {channel_type = "cascade"}
  air.channel @L1ToL1_Conv3x3BToSkip [1] {channel_type = "cascade"}
  air.channel @L1ToL1_Conv1ToConv3x3 [1, 1] {broadcast_shape = [2 : index, 1 : index]}
  air.channel @L2ToL1_ActIn [1, 1]
  air.channel @L1ToL2_ActOut [1, 1]
  
  func.func @bottleneck_cascade_placement(%arg0: memref<262144xi8>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<262144xi8> attributes {id = 1 : i32} {
      %1 = air.segment @bottleneck_seg async attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        
        // conv1x1_reduce herd - no cascade connections
        %2 = air.herd @conv1x1_reduce async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_0) attributes {id = 3 : i32} {
          %alloc = memref.alloc() : memref<32x1x64xui8, 2 : i32>
          air.channel.put @L1ToL1_Conv1ToConv3x3[] (%alloc[] [] []) : (memref<32x1x64xui8, 2 : i32>)
          memref.dealloc %alloc : memref<32x1x64xui8, 2 : i32>
        }
        
        // conv3x3_coreA herd - produces to cascade channel L1ToL1_Conv3x3AToSkip
        %3 = air.herd @conv3x3_coreA async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_0) attributes {id = 4 : i32} {
          %c1_1 = arith.constant 1 : index
          %alloc = memref.alloc() : memref<32x1x32xui8, 2 : i32>
          air.channel.get @L1ToL1_Conv1ToConv3x3[%c1_1, %c1_1] (%alloc[] [] []) : (memref<32x1x32xui8, 2 : i32>)
          air.channel.put @L1ToL1_Conv3x3AToSkip[] (%alloc[] [] []) : (memref<32x1x32xui8, 2 : i32>)
          memref.dealloc %alloc : memref<32x1x32xui8, 2 : i32>
        }
        
        // conv3x3_coreB herd - produces to cascade channel L1ToL1_Conv3x3BToSkip
        %4 = air.herd @conv3x3_coreB async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_0) attributes {id = 5 : i32} {
          %c1_1 = arith.constant 1 : index
          %alloc = memref.alloc() : memref<32x1x32xui8, 2 : i32>
          air.channel.get @L1ToL1_Conv1ToConv3x3[%c1_1, %c1_1] (%alloc[] [] []) : (memref<32x1x32xui8, 2 : i32>)
          air.channel.put @L1ToL1_Conv3x3BToSkip[] (%alloc[] [] []) : (memref<32x1x32xui8, 2 : i32>)
          memref.dealloc %alloc : memref<32x1x32xui8, 2 : i32>
        }
        
        // conv1x1_skip herd - consumes from both cascade channels
        %5 = air.herd @conv1x1_skip async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_0) attributes {id = 6 : i32} {
          %alloc_a = memref.alloc() : memref<32x1x32xui8, 2 : i32>
          %alloc_b = memref.alloc() : memref<32x1x32xui8, 2 : i32>
          air.channel.get @L1ToL1_Conv3x3AToSkip[] (%alloc_a[] [] []) : (memref<32x1x32xui8, 2 : i32>)
          air.channel.get @L1ToL1_Conv3x3BToSkip[] (%alloc_b[] [] []) : (memref<32x1x32xui8, 2 : i32>)
          air.channel.put @L1ToL2_ActOut[] (%alloc_a[] [] []) : (memref<32x1x32xui8, 2 : i32>)
          memref.dealloc %alloc_a : memref<32x1x32xui8, 2 : i32>
          memref.dealloc %alloc_b : memref<32x1x32xui8, 2 : i32>
        }
        
      }
    }
    return
  }
}
