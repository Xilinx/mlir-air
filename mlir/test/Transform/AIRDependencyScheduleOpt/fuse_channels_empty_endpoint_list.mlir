//===- fuse_channels_empty_endpoint_list.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels | FileCheck %s

// A channel reaches the fusion helpers with no puts, or no gets, once an
// earlier fusion in the same run has moved them onto another symbol. The
// helpers read a_vec[0] as the pattern to compare against, and
// a_puts[0]/b_puts[0]/a_gets[0]/b_gets[0] as the ops to merge, so an empty list
// aborted the pass on
//
//   std::vector<xilinx::air::ChannelGetOp>::operator[]:
//   Assertion '__n < this->size()' failed
//
// which without assertions is an out-of-bounds read instead. Both sites are hit
// in turn: guarding only the first moves the abort to the second.
//
// This is the segment of herd_dataflow with its per-column staging written out
// rather than left in an scf.forall -- three herds chained by channels, and a
// memtile buffer fanned out to a row of cores with a constant bundle index per
// column. The check is only that the pass completes and the pipeline is intact.

// CHECK-LABEL: func.func @func1
// CHECK: air.segment
// CHECK: air.herd
// CHECK: air.herd
// CHECK: air.herd
// CHECK: return

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 256)>
#map2 = affine_map<()[s0] -> (s0 * 256 + 64)>
#map3 = affine_map<()[s0] -> (s0 * 256 + 128)>
#map4 = affine_map<()[s0] -> (s0 * 256 + 192)>
module {
  air.channel @channel_0 []
  air.channel @channel_1 []
  air.channel @channel_2 []
  air.channel @channel_3 []
  air.channel @channel_4 []
  air.channel @channel_5 []
  air.channel @channel_6 []
  air.channel @channel_7 []
  air.channel @channel_8 []
  air.channel @channel_9 []
  air.channel @channel_10 []
  air.channel @channel_11 []
  func.func private @add_3_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "extern_func.o", llvm.emit_c_interface}
  air.channel @L2ToL1Chan1 [4, 1]
  air.channel @L2ToL1Chan2 [4, 1]
  air.channel @L1ToL1Chan1 [4, 1]
  air.channel @L1ToL1Chan2 [4, 1] {channel_type = "npu_cascade"}
  air.channel @L1ToL2Chan1 [4, 1]
  func.func @func1(%arg0: memref<256x256xbf16>, %arg1: memref<256x256xbf16>, %arg2: memref<256x256xbf16>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c4, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> attributes {id = 1 : i32} {
      %1 = affine.apply #map()[%arg3]
      %2 = affine.apply #map1()[%arg4]
      %3 = air.channel.put async  @channel_0[] (%arg7[%1, %2] [64, 64] [256, 1]) {id = 1 : i32} : (memref<256x256xbf16>)
      %4 = air.channel.put async  @channel_1[] (%arg8[%1, %2] [64, 64] [256, 1]) {id = 2 : i32} : (memref<256x256xbf16>)
      %5 = affine.apply #map2()[%arg4]
      %6 = air.channel.put async  @channel_2[] (%arg7[%1, %5] [64, 64] [256, 1]) {id = 3 : i32} : (memref<256x256xbf16>)
      %7 = air.channel.put async  @channel_3[] (%arg8[%1, %5] [64, 64] [256, 1]) {id = 4 : i32} : (memref<256x256xbf16>)
      %8 = affine.apply #map3()[%arg4]
      %9 = air.channel.put async  @channel_4[] (%arg7[%1, %8] [64, 64] [256, 1]) {id = 5 : i32} : (memref<256x256xbf16>)
      %10 = air.channel.put async  @channel_5[] (%arg8[%1, %8] [64, 64] [256, 1]) {id = 6 : i32} : (memref<256x256xbf16>)
      %11 = affine.apply #map4()[%arg4]
      %12 = air.channel.put async  @channel_6[] (%arg7[%1, %11] [64, 64] [256, 1]) {id = 7 : i32} : (memref<256x256xbf16>)
      %13 = air.channel.put async  @channel_7[] (%arg8[%1, %11] [64, 64] [256, 1]) {id = 8 : i32} : (memref<256x256xbf16>)
      %14 = air.channel.get async  @channel_8[] (%arg9[%1, %2] [64, 64] [256, 1]) {id = 9 : i32} : (memref<256x256xbf16>)
      %15 = air.channel.get async [%14]  @channel_9[] (%arg9[%1, %5] [64, 64] [256, 1]) {id = 10 : i32} : (memref<256x256xbf16>)
      %16 = air.channel.get async [%15]  @channel_10[] (%arg9[%1, %8] [64, 64] [256, 1]) {id = 11 : i32} : (memref<256x256xbf16>)
      %17 = air.channel.get async [%16]  @channel_11[] (%arg9[%1, %11] [64, 64] [256, 1]) {id = 12 : i32} : (memref<256x256xbf16>)
      %18 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c4_0 = arith.constant 4 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %async_token, %results = air.execute -> (memref<64x256xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x256xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x256xbf16, 1 : i32>
        }
        %async_token_2, %results_3 = air.execute -> (memref<64x256xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x256xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x256xbf16, 1 : i32>
        }
        %async_token_4, %results_5 = air.execute -> (memref<64x256xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x256xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x256xbf16, 1 : i32>
        }
        %19 = air.channel.get async [%async_token]  @channel_0[] (%results[0, 0] [64, 64] [256, 1]) {id = 13 : i32} : (memref<64x256xbf16, 1 : i32>)
        %20 = air.channel.get async [%async_token_2]  @channel_1[] (%results_3[0, 0] [64, 64] [256, 1]) {id = 14 : i32} : (memref<64x256xbf16, 1 : i32>)
        %21 = air.channel.get async [%19]  @channel_2[] (%results[0, 64] [64, 64] [256, 1]) {id = 15 : i32} : (memref<64x256xbf16, 1 : i32>)
        %22 = air.channel.get async [%20]  @channel_3[] (%results_3[0, 64] [64, 64] [256, 1]) {id = 16 : i32} : (memref<64x256xbf16, 1 : i32>)
        %23 = air.channel.get async [%21]  @channel_4[] (%results[0, 128] [64, 64] [256, 1]) {id = 17 : i32} : (memref<64x256xbf16, 1 : i32>)
        %24 = air.channel.get async [%22]  @channel_5[] (%results_3[0, 128] [64, 64] [256, 1]) {id = 18 : i32} : (memref<64x256xbf16, 1 : i32>)
        %25 = air.channel.get async [%23]  @channel_6[] (%results[0, 192] [64, 64] [256, 1]) {id = 19 : i32} : (memref<64x256xbf16, 1 : i32>)
        %26 = air.channel.get async [%24]  @channel_7[] (%results_3[0, 192] [64, 64] [256, 1]) {id = 20 : i32} : (memref<64x256xbf16, 1 : i32>)
        %27 = air.channel.put async [%19]  @L2ToL1Chan1[%c0, %c0] (%results[0, 0] [64, 64] [256, 1]) {id = 21 : i32} : (memref<64x256xbf16, 1 : i32>)
        %28 = air.channel.put async [%20]  @L2ToL1Chan2[%c0, %c0] (%results_3[0, 0] [64, 64] [256, 1]) {id = 22 : i32} : (memref<64x256xbf16, 1 : i32>)
        %29 = air.channel.put async [%21]  @L2ToL1Chan1[%c1_1, %c0] (%results[0, 64] [64, 64] [256, 1]) {id = 23 : i32} : (memref<64x256xbf16, 1 : i32>)
        %30 = air.channel.put async [%22]  @L2ToL1Chan2[%c1_1, %c0] (%results_3[0, 64] [64, 64] [256, 1]) {id = 24 : i32} : (memref<64x256xbf16, 1 : i32>)
        %31 = air.channel.put async [%23]  @L2ToL1Chan1[%c2, %c0] (%results[0, 128] [64, 64] [256, 1]) {id = 25 : i32} : (memref<64x256xbf16, 1 : i32>)
        %32 = air.channel.put async [%24]  @L2ToL1Chan2[%c2, %c0] (%results_3[0, 128] [64, 64] [256, 1]) {id = 26 : i32} : (memref<64x256xbf16, 1 : i32>)
        %33 = air.channel.put async [%25]  @L2ToL1Chan1[%c3, %c0] (%results[0, 192] [64, 64] [256, 1]) {id = 27 : i32} : (memref<64x256xbf16, 1 : i32>)
        %34 = air.channel.put async [%26]  @L2ToL1Chan2[%c3, %c0] (%results_3[0, 192] [64, 64] [256, 1]) {id = 28 : i32} : (memref<64x256xbf16, 1 : i32>)
        %35 = air.herd @herd_0 async [%async_token_4]  tile (%arg10, %arg11) in (%arg12=%c4_0, %arg13=%c1_1) attributes {id = 3 : i32} {
          %c16 = arith.constant 16 : index
          %c1_9 = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %cst = arith.constant 0.000000e+00 : bf16
          %c0_10 = arith.constant 0 : index
          %async_token_11, %results_12 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_15, %results_16 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %46 = air.channel.get async [%async_token_11]  @L2ToL1Chan1[%arg10, %c0_10] (%results_12[] [] []) {id = 29 : i32} : (memref<64x64xbf16, 2 : i32>)
          %47 = air.channel.get async [%async_token_13]  @L2ToL1Chan2[%arg10, %c0_10] (%results_14[] [] []) {id = 30 : i32} : (memref<64x64xbf16, 2 : i32>)
          %48 = air.wait_all async [%async_token_15, %46, %47] 
          %49 = scf.for %arg14 = %c0_10 to %c64 step %c1_9 iter_args(%arg15 = %48) -> (!air.async.token) {
            %51 = scf.for %arg16 = %c0_10 to %c64 step %c16 iter_args(%arg17 = %arg15) -> (!air.async.token) {
              %async_token_20, %results_21 = air.execute [%arg17] -> (vector<16xbf16>) {
                %54 = vector.transfer_read %results_12[%arg14, %arg16], %cst {in_bounds = [true]} : memref<64x64xbf16, 2 : i32>, vector<16xbf16>
                air.execute_terminator %54 : vector<16xbf16>
              }
              %async_token_22, %results_23 = air.execute [%arg17] -> (vector<16xbf16>) {
                %54 = vector.transfer_read %results_14[%arg14, %arg16], %cst {in_bounds = [true]} : memref<64x64xbf16, 2 : i32>, vector<16xbf16>
                air.execute_terminator %54 : vector<16xbf16>
              }
              %52 = arith.addf %results_21, %results_23 : vector<16xbf16>
              %async_token_24 = air.execute [%arg17] {
                vector.transfer_write %52, %results_16[%arg14, %arg16] {in_bounds = [true]} : vector<16xbf16>, memref<64x64xbf16, 2 : i32>
              }
              %53 = air.wait_all async [%async_token_20, %async_token_22, %async_token_24] 
              scf.yield %53 : !air.async.token
            }
            scf.yield %51 : !air.async.token
          }
          %async_token_17 = air.execute [%49] {
            memref.dealloc %results_14 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_18 = air.execute [%49] {
            memref.dealloc %results_12 : memref<64x64xbf16, 2 : i32>
          }
          %50 = air.channel.put async [%49]  @L1ToL1Chan1[%arg10, %c0_10] (%results_16[] [] []) {id = 31 : i32} : (memref<64x64xbf16, 2 : i32>)
          %async_token_19 = air.execute [%50] {
            memref.dealloc %results_16 : memref<64x64xbf16, 2 : i32>
          }
        }
        %36 = air.herd @herd_1 async [%35]  tile (%arg10, %arg11) in (%arg12=%c4_0, %arg13=%c1_1) attributes {id = 4 : i32} {
          %c16 = arith.constant 16 : index
          %c1_9 = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %cst = arith.constant 0.000000e+00 : bf16
          %c0_10 = arith.constant 0 : index
          %async_token_11, %results_12 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %46 = air.channel.get async [%async_token_11]  @L1ToL1Chan1[%arg10, %c0_10] (%results_12[] [] []) {id = 32 : i32} : (memref<64x64xbf16, 2 : i32>)
          %47 = air.wait_all async [%async_token_13, %46] 
          %48 = scf.for %arg14 = %c0_10 to %c64 step %c1_9 iter_args(%arg15 = %47) -> (!air.async.token) {
            %50 = scf.for %arg16 = %c0_10 to %c64 step %c16 iter_args(%arg17 = %arg15) -> (!air.async.token) {
              %async_token_17, %results_18 = air.execute [%arg17] -> (vector<16xbf16>) {
                %52 = vector.transfer_read %results_12[%arg14, %arg16], %cst {in_bounds = [true]} : memref<64x64xbf16, 2 : i32>, vector<16xbf16>
                air.execute_terminator %52 : vector<16xbf16>
              }
              %async_token_19 = air.execute [%arg17] {
                vector.transfer_write %results_18, %results_14[%arg14, %arg16] {in_bounds = [true]} : vector<16xbf16>, memref<64x64xbf16, 2 : i32>
              }
              %51 = air.wait_all async [%async_token_17, %async_token_19] 
              scf.yield %51 : !air.async.token
            }
            scf.yield %50 : !air.async.token
          }
          %async_token_15 = air.execute [%48] {
            memref.dealloc %results_12 : memref<64x64xbf16, 2 : i32>
          }
          %49 = air.channel.put async [%48]  @L1ToL1Chan2[%arg10, %c0_10] (%results_14[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
          %async_token_16 = air.execute [%49] {
            memref.dealloc %results_14 : memref<64x64xbf16, 2 : i32>
          }
        }
        %37 = air.herd @herd_2 async [%36]  tile (%arg10, %arg11) in (%arg12=%c4_0, %arg13=%c1_1) attributes {id = 5 : i32, link_with = "extern_func.o"} {
          %c0_9 = arith.constant 0 : index
          %async_token_10, %results_11 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_12, %results_13 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %46 = air.channel.get async [%async_token_10]  @L1ToL1Chan2[%arg10, %c0_9] (%results_11[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
          %async_token_14 = air.execute [%46, %async_token_12] {
            func.call @add_3_bf16(%results_11, %results_13) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_15 = air.execute [%async_token_14] {
            memref.dealloc %results_11 : memref<64x64xbf16, 2 : i32>
          }
          %47 = air.channel.put async [%async_token_14]  @L1ToL2Chan1[%arg10, %c0_9] (%results_13[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
          %async_token_16 = air.execute [%47] {
            memref.dealloc %results_13 : memref<64x64xbf16, 2 : i32>
          }
        }
        %async_token_6 = air.execute [%34, %32, %30, %28] {
          memref.dealloc %results_3 : memref<64x256xbf16, 1 : i32>
        }
        %async_token_7 = air.execute [%33, %31, %29, %27] {
          memref.dealloc %results : memref<64x256xbf16, 1 : i32>
        }
        %38 = air.channel.get async [%37]  @L1ToL2Chan1[%c0, %c0] (%results_5[0, 0] [64, 64] [256, 1]) {id = 36 : i32} : (memref<64x256xbf16, 1 : i32>)
        %39 = air.channel.put async [%38]  @channel_8[] (%results_5[0, 0] [64, 64] [256, 1]) {id = 37 : i32} : (memref<64x256xbf16, 1 : i32>)
        %40 = air.channel.get async [%37]  @L1ToL2Chan1[%c1_1, %c0] (%results_5[0, 64] [64, 64] [256, 1]) {id = 38 : i32} : (memref<64x256xbf16, 1 : i32>)
        %41 = air.channel.put async [%38, %40]  @channel_9[] (%results_5[0, 64] [64, 64] [256, 1]) {id = 39 : i32} : (memref<64x256xbf16, 1 : i32>)
        %42 = air.channel.get async [%37]  @L1ToL2Chan1[%c2, %c0] (%results_5[0, 128] [64, 64] [256, 1]) {id = 40 : i32} : (memref<64x256xbf16, 1 : i32>)
        %43 = air.channel.put async [%38, %40, %42]  @channel_10[] (%results_5[0, 128] [64, 64] [256, 1]) {id = 41 : i32} : (memref<64x256xbf16, 1 : i32>)
        %44 = air.channel.get async [%37]  @L1ToL2Chan1[%c3, %c0] (%results_5[0, 192] [64, 64] [256, 1]) {id = 42 : i32} : (memref<64x256xbf16, 1 : i32>)
        %45 = air.channel.put async [%38, %40, %42, %44]  @channel_11[] (%results_5[0, 192] [64, 64] [256, 1]) {id = 43 : i32} : (memref<64x256xbf16, 1 : i32>)
        %async_token_8 = air.execute [%45, %43, %41, %39] {
          memref.dealloc %results_5 : memref<64x256xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
