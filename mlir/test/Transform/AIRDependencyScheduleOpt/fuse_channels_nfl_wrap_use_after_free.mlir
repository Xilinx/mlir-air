//===- fuse_channels_nfl_wrap_use_after_free.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels | FileCheck %s

// AIRFuseChannels segfaulted on this input.
//
// The NFL ("new for loop") path collected the channel ops it was going to erase
// *before* calling wrapRegionsWithForLoops, which clones each region's parent
// op into a fresh scf.for and then erases the original -- destroying every op
// inside it, including the ones already collected. The loop that followed then
// called air::isAsyncOp on freed memory. The erase list is re-derived from the
// channels after the wrap now; the channels themselves survive it.
//
// From flash_attention/kernel_fusion_based on NPU2 at num_heads_per_unroll=2,
// which is what puts the fusable channel ops inside an scf.if whose enclosing
// region gets wrapped. This is a crash test first: what the CHECKs pin is only
// that the pass ran to completion and still produced the hierarchy.
//
// CHECK-LABEL: @attention_bf16
// CHECK: air.launch
// CHECK: air.segment
// CHECK: air.herd
#map = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 65536)>
#map2 = affine_map<()[s0] -> (s0 * 65536 + 8192)>
#map3 = affine_map<()[s0] -> (s0 * 65536 + 16384)>
#map4 = affine_map<()[s0] -> (s0 * 65536 + 24576)>
#map5 = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 16384 + 32768)>
#map6 = affine_map<()[s0] -> (s0 * 65536 + 32768)>
#map7 = affine_map<()[s0] -> (s0 * 65536 + 40960)>
#map8 = affine_map<()[s0] -> (s0 * 65536 + 49152)>
#map9 = affine_map<()[s0] -> (s0 * 65536 + 57344)>
#map10 = affine_map<()[s0] -> (s0 - 1)>
#map11 = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @add_gp_g(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @exp_up_minus_u(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @maximum_up_u_bf16(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  air.channel @QKIn_0 [2]
  air.channel @QKIn_1 [2]
  air.channel @QKIn_2 [2]
  air.channel @QKIn_3 [2]
  air.channel @VIn_0 [2]
  air.channel @VIn_1 [2]
  air.channel @VIn_2 [2]
  air.channel @VIn_3 [2]
  air.channel @QK2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @cascade_gp [4, 3] {channel_type = "npu_cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "npu_cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "npu_cascade"}
  air.channel @Gp2L2 [4, 1]
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<2x512x64xbf16>, %arg1: memref<2x512x64xbf16>, %arg2: memref<2x512x64xbf16>, %arg3: memref<2x512x64xbf16>) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16> attributes {id = 1 : i32} {
      %c2_0 = arith.constant 2 : index
      %c1_1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QKIn_0[%c0] (%arg8[0, 0, 0, %1] [4, 1, 64, 64] [4096, 64, 64, 1]) {id = 1 : i32} : (memref<2x512x64xbf16>)
      %3 = air.channel.put async  @QKIn_1[%c0] (%arg8[0, 0, 0, %1] [4, 1, 64, 64] [4096, 64, 64, 1]) {id = 2 : i32} : (memref<2x512x64xbf16>)
      %4 = air.channel.put async  @QKIn_2[%c0] (%arg8[0, 0, 0, %1] [4, 1, 64, 64] [4096, 64, 64, 1]) {id = 3 : i32} : (memref<2x512x64xbf16>)
      %5 = air.channel.put async  @QKIn_3[%c0] (%arg8[0, 0, 0, %1] [4, 1, 64, 64] [4096, 64, 64, 1]) {id = 4 : i32} : (memref<2x512x64xbf16>)
      %6 = affine.apply #map1()[%arg5]
      %7 = air.channel.put async  @QKIn_0[%c0] (%arg9[0, 0, 0, %6] [2, 1, 64, 64] [4096, 64, 64, 1]) {id = 5 : i32} : (memref<2x512x64xbf16>)
      %8 = affine.apply #map2()[%arg5]
      %9 = air.channel.put async  @QKIn_1[%c0] (%arg9[0, 0, 0, %8] [2, 1, 64, 64] [4096, 64, 64, 1]) {id = 6 : i32} : (memref<2x512x64xbf16>)
      %10 = affine.apply #map3()[%arg5]
      %11 = air.channel.put async  @QKIn_2[%c0] (%arg9[0, 0, 0, %10] [2, 1, 64, 64] [4096, 64, 64, 1]) {id = 7 : i32} : (memref<2x512x64xbf16>)
      %12 = affine.apply #map4()[%arg5]
      %13 = air.channel.put async  @QKIn_3[%c0] (%arg9[0, 0, 0, %12] [2, 1, 64, 64] [4096, 64, 64, 1]) {id = 8 : i32} : (memref<2x512x64xbf16>)
      %14 = air.channel.put async  @VIn_0[%c0] (%arg10[0, 0, %6] [2, 64, 64] [4096, 64, 1]) {id = 9 : i32} : (memref<2x512x64xbf16>)
      %15 = air.channel.put async  @VIn_1[%c0] (%arg10[0, 0, %8] [2, 64, 64] [4096, 64, 1]) {id = 10 : i32} : (memref<2x512x64xbf16>)
      %16 = air.channel.put async  @VIn_2[%c0] (%arg10[0, 0, %10] [2, 64, 64] [4096, 64, 1]) {id = 11 : i32} : (memref<2x512x64xbf16>)
      %17 = air.channel.put async  @VIn_3[%c0] (%arg10[0, 0, %12] [2, 64, 64] [4096, 64, 1]) {id = 12 : i32} : (memref<2x512x64xbf16>)
      %18 = affine.apply #map5()[%arg5, %arg4]
      %19 = air.channel.put async  @QKIn_0[%c1_1] (%arg8[0, 0, 0, %18] [4, 1, 64, 64] [4096, 64, 64, 1]) {id = 13 : i32} : (memref<2x512x64xbf16>)
      %20 = air.channel.put async  @QKIn_1[%c1_1] (%arg8[0, 0, 0, %18] [4, 1, 64, 64] [4096, 64, 64, 1]) {id = 14 : i32} : (memref<2x512x64xbf16>)
      %21 = air.channel.put async  @QKIn_2[%c1_1] (%arg8[0, 0, 0, %18] [4, 1, 64, 64] [4096, 64, 64, 1]) {id = 15 : i32} : (memref<2x512x64xbf16>)
      %22 = air.channel.put async  @QKIn_3[%c1_1] (%arg8[0, 0, 0, %18] [4, 1, 64, 64] [4096, 64, 64, 1]) {id = 16 : i32} : (memref<2x512x64xbf16>)
      %23 = affine.apply #map6()[%arg5]
      %24 = air.channel.put async  @QKIn_0[%c1_1] (%arg9[0, 0, 0, %23] [2, 1, 64, 64] [4096, 64, 64, 1]) {id = 17 : i32} : (memref<2x512x64xbf16>)
      %25 = affine.apply #map7()[%arg5]
      %26 = air.channel.put async  @QKIn_1[%c1_1] (%arg9[0, 0, 0, %25] [2, 1, 64, 64] [4096, 64, 64, 1]) {id = 18 : i32} : (memref<2x512x64xbf16>)
      %27 = affine.apply #map8()[%arg5]
      %28 = air.channel.put async  @QKIn_2[%c1_1] (%arg9[0, 0, 0, %27] [2, 1, 64, 64] [4096, 64, 64, 1]) {id = 19 : i32} : (memref<2x512x64xbf16>)
      %29 = affine.apply #map9()[%arg5]
      %30 = air.channel.put async  @QKIn_3[%c1_1] (%arg9[0, 0, 0, %29] [2, 1, 64, 64] [4096, 64, 64, 1]) {id = 20 : i32} : (memref<2x512x64xbf16>)
      %31 = air.channel.put async  @VIn_0[%c1_1] (%arg10[0, 0, %23] [2, 64, 64] [4096, 64, 1]) {id = 21 : i32} : (memref<2x512x64xbf16>)
      %32 = air.channel.put async  @VIn_1[%c1_1] (%arg10[0, 0, %25] [2, 64, 64] [4096, 64, 1]) {id = 22 : i32} : (memref<2x512x64xbf16>)
      %33 = air.channel.put async  @VIn_2[%c1_1] (%arg10[0, 0, %27] [2, 64, 64] [4096, 64, 1]) {id = 23 : i32} : (memref<2x512x64xbf16>)
      %34 = air.channel.put async  @VIn_3[%c1_1] (%arg10[0, 0, %29] [2, 64, 64] [4096, 64, 1]) {id = 24 : i32} : (memref<2x512x64xbf16>)
      %35 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2_0, %arg15=%c1_1) attributes {id = 2 : i32} {
        %c2_2 = arith.constant 2 : index
        %c1_3 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0_4 = arith.constant 0 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_5, %results_6 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_7, %results_8 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_9, %results_10 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_11, %results_12 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_15, %results_16 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17, %results_18 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_19, %results_20 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        }
        %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_23, %results_24 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_25, %results_26 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_27, %results_28 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_31, %results_32 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_33, %results_34 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %38 = scf.for %arg16 = %c0_4 to %c4 step %c1_3 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 25 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @QK2L1_0_0[%c0_4, %c0_4, %c0_4] (%results[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 26 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @QK2L1_0_1[%c0_4, %c0_4, %c0_4] (%results[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %39 = scf.for %arg16 = %c0_4 to %c2_2 step %c1_3 iter_args(%arg17 = %38) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @QK2L1_0_0[%c0_4, %c0_4, %c0_4] (%results[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @QK2L1_0_1[%c0_4, %c0_4, %c0_4] (%results[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %async_token_35 = air.execute [%39] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %40 = scf.for %arg16 = %c0_4 to %c4 step %c1_3 iter_args(%arg17 = %async_token_5) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_6[] [] []) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @QK2L1_1_0[%c0_4, %c0_4, %c0_4] (%results_6[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @QK2L1_1_1[%c0_4, %c0_4, %c0_4] (%results_6[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %41 = scf.for %arg16 = %c0_4 to %c2_2 step %c1_3 iter_args(%arg17 = %40) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_6[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @QK2L1_1_0[%c0_4, %c0_4, %c0_4] (%results_6[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @QK2L1_1_1[%c0_4, %c0_4, %c0_4] (%results_6[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %async_token_36 = air.execute [%41] {
          memref.dealloc %results_6 : memref<64x64xbf16, 1 : i32>
        }
        %42 = scf.for %arg16 = %c0_4 to %c4 step %c1_3 iter_args(%arg17 = %async_token_7) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_8[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @QK2L1_2_0[%c0_4, %c0_4, %c0_4] (%results_8[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @QK2L1_2_1[%c0_4, %c0_4, %c0_4] (%results_8[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %43 = scf.for %arg16 = %c0_4 to %c2_2 step %c1_3 iter_args(%arg17 = %42) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_8[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @QK2L1_2_0[%c0_4, %c0_4, %c0_4] (%results_8[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @QK2L1_2_1[%c0_4, %c0_4, %c0_4] (%results_8[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %async_token_37 = air.execute [%43] {
          memref.dealloc %results_8 : memref<64x64xbf16, 1 : i32>
        }
        %44 = scf.for %arg16 = %c0_4 to %c4 step %c1_3 iter_args(%arg17 = %async_token_9) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_10[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @QK2L1_3_0[%c0_4, %c0_4, %c0_4] (%results_10[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @QK2L1_3_1[%c0_4, %c0_4, %c0_4] (%results_10[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %45 = scf.for %arg16 = %c0_4 to %c2_2 step %c1_3 iter_args(%arg17 = %44) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_10[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @QK2L1_3_0[%c0_4, %c0_4, %c0_4] (%results_10[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @QK2L1_3_1[%c0_4, %c0_4, %c0_4] (%results_10[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %async_token_38 = air.execute [%45] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %46 = scf.for %arg16 = %c0_4 to %c2_2 step %c1_3 iter_args(%arg17 = %async_token_11) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @VIn_0[%arg12] (%results_12[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @V2L1_0_0[%c0_4, %c0_4, %c0_4] (%results_12[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @V2L1_0_1[%c0_4, %c0_4, %c0_4] (%results_12[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %async_token_39 = air.execute [%46] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %47 = scf.for %arg16 = %c0_4 to %c2_2 step %c1_3 iter_args(%arg17 = %async_token_13) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @VIn_1[%arg12] (%results_14[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @V2L1_1_0[%c0_4, %c0_4, %c0_4] (%results_14[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 53 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @V2L1_1_1[%c0_4, %c0_4, %c0_4] (%results_14[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 54 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %async_token_40 = air.execute [%47] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %48 = scf.for %arg16 = %c0_4 to %c2_2 step %c1_3 iter_args(%arg17 = %async_token_15) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @VIn_2[%arg12] (%results_16[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @V2L1_2_0[%c0_4, %c0_4, %c0_4] (%results_16[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 56 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @V2L1_2_1[%c0_4, %c0_4, %c0_4] (%results_16[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 57 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %async_token_41 = air.execute [%48] {
          memref.dealloc %results_16 : memref<64x64xbf16, 1 : i32>
        }
        %49 = scf.for %arg16 = %c0_4 to %c2_2 step %c1_3 iter_args(%arg17 = %async_token_17) -> (!air.async.token) {
          %53 = air.channel.get async [%arg17]  @VIn_3[%arg12] (%results_18[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 1 : i32>)
          %54 = arith.cmpi eq, %arg12, %c0_4 : index
          %55 = scf.if %54 -> (!air.async.token) {
            %56 = air.channel.put async [%53]  @V2L1_3_0[%c0_4, %c0_4, %c0_4] (%results_18[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 59 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          } else {
            %56 = air.channel.put async [%53]  @V2L1_3_1[%c0_4, %c0_4, %c0_4] (%results_18[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) {id = 60 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %56 : !air.async.token
          }
          scf.yield %55 : !air.async.token
        }
        %async_token_42 = air.execute [%49] {
          memref.dealloc %results_18 : memref<64x64xbf16, 1 : i32>
        }
        %50 = air.herd @herd_0 async [%async_token_21, %async_token_23, %async_token_25, %async_token_27, %async_token_29, %async_token_31, %async_token_33]  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%arg12, %arg21=%results_22, %arg22=%results_24, %arg23=%results_26, %arg24=%results_28, %arg25=%results_30, %arg26=%results_32, %arg27=%results_34) : index, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32> attributes {id = 3 : i32, link_with = "attn_npu2.o"} {
          %c0_i32 = arith.constant 0 : i32
          %c3 = arith.constant 3 : index
          %c2_51 = arith.constant 2 : index
          %c1_52 = arith.constant 1 : index
          %c0_53 = arith.constant 0 : index
          %async_token_54 = air.execute {
            func.call @zero_fill_gp_bf16(%arg25) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_55 = air.execute {
            func.call @zero_fill_sp_bf16(%arg27) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_56 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %53 = arith.cmpi eq, %arg17, %c0_53 : index
          scf.if %53 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          %54 = arith.cmpi eq, %arg17, %c1_52 : index
          scf.if %54 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          %55 = arith.cmpi eq, %arg17, %c2_51 : index
          scf.if %55 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          %56 = arith.cmpi eq, %arg17, %c3 : index
          scf.if %56 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          %57 = arith.cmpi eq, %arg16, %c0_53 : index
          scf.if %57 {
            %async_token_57 = air.execute {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %53 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          scf.if %54 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          scf.if %55 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          scf.if %56 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          %58 = arith.cmpi eq, %arg16, %c1_52 : index
          scf.if %58 {
            %async_token_57 = air.execute {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %53 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          scf.if %54 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          scf.if %55 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          scf.if %56 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          %59 = arith.cmpi eq, %arg16, %c2_51 : index
          scf.if %59 {
            %async_token_57 = air.execute {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %53 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          scf.if %54 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          scf.if %55 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          scf.if %56 {
            %63 = arith.cmpi eq, %arg20, %c0_53 : index
            scf.if %63 {
              %64 = air.channel.get async  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
            } else {
              %64 = air.channel.get async  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
            }
          }
          %60 = arith.cmpi eq, %arg16, %c3 : index
          scf.if %60 {
            %async_token_57 = air.execute {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %61 = air.wait_all async [%async_token_54, %async_token_55, %async_token_56] 
          %62 = scf.for %arg28 = %c0_53 to %c2_51 step %c1_52 iter_args(%arg29 = %61) -> (!air.async.token) {
            %async_token_57 = air.execute [%arg29] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %53 {
              %64 = arith.cmpi eq, %arg20, %c0_53 : index
              scf.if %64 {
                %65 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              } else {
                %65 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              }
            }
            scf.if %54 {
              %64 = arith.cmpi eq, %arg20, %c0_53 : index
              scf.if %64 {
                %65 = air.channel.get async  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              } else {
                %65 = air.channel.get async  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              }
            }
            scf.if %55 {
              %64 = arith.cmpi eq, %arg20, %c0_53 : index
              scf.if %64 {
                %65 = air.channel.get async  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              } else {
                %65 = air.channel.get async  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              }
            }
            scf.if %56 {
              %64 = arith.cmpi eq, %arg20, %c0_53 : index
              scf.if %64 {
                %65 = air.channel.get async  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              } else {
                %65 = air.channel.get async  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              }
            }
            %async_token_58 = air.execute [%async_token_57] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg21, %arg22, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %53 {
              %64 = arith.cmpi eq, %arg20, %c0_53 : index
              scf.if %64 {
                %65 = air.channel.get async  @V2L1_0_0[%c0_53, %arg17, %arg16] (%arg23[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
              } else {
                %65 = air.channel.get async  @V2L1_0_1[%c0_53, %arg17, %arg16] (%arg23[] [] []) {id = 102 : i32} : (memref<64x64xbf16, 2 : i32>)
              }
            }
            scf.if %54 {
              %64 = arith.cmpi eq, %arg20, %c0_53 : index
              scf.if %64 {
                %65 = air.channel.get async  @V2L1_1_0[%c0_53, %arg17, %arg16] (%arg23[] [] []) {id = 103 : i32} : (memref<64x64xbf16, 2 : i32>)
              } else {
                %65 = air.channel.get async  @V2L1_1_1[%c0_53, %arg17, %arg16] (%arg23[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              }
            }
            scf.if %55 {
              %64 = arith.cmpi eq, %arg20, %c0_53 : index
              scf.if %64 {
                %65 = air.channel.get async  @V2L1_2_0[%c0_53, %arg17, %arg16] (%arg23[] [] []) {id = 105 : i32} : (memref<64x64xbf16, 2 : i32>)
              } else {
                %65 = air.channel.get async  @V2L1_2_1[%c0_53, %arg17, %arg16] (%arg23[] [] []) {id = 106 : i32} : (memref<64x64xbf16, 2 : i32>)
              }
            }
            scf.if %56 {
              %64 = arith.cmpi eq, %arg20, %c0_53 : index
              scf.if %64 {
                %65 = air.channel.get async  @V2L1_3_0[%c0_53, %arg17, %arg16] (%arg23[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              } else {
                %65 = air.channel.get async  @V2L1_3_1[%c0_53, %arg17, %arg16] (%arg23[] [] []) {id = 108 : i32} : (memref<64x64xbf16, 2 : i32>)
              }
            }
            %async_token_59, %results_60 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_61, %results_62 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_63 = air.execute [%async_token_61, %async_token_59, %async_token_58] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg26, %results_60, %results_62) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_64 = air.execute [%async_token_63] {
              func.call @mul_r_gp(%results_62, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_65 = air.execute [%async_token_64] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg23, %arg25) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_66 = air.execute [%async_token_64] {
              func.call @accum_sp_r_s(%arg27, %results_62, %results_60) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_67 = air.execute [%async_token_66] {
              memref.dealloc %results_62 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_68 = air.execute [%async_token_66] {
              func.call @vector_copy_32elems(%c0_i32, %results_60, %arg27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_69 = air.execute [%async_token_68] {
              memref.dealloc %results_60 : memref<64x1xbf16, 2 : i32>
            }
            %63 = air.wait_all async [%async_token_65, %async_token_68] 
            scf.yield %63 : !air.async.token
          }
          scf.if %56 {
            %63 = affine.apply #map10()[%arg17]
            %64 = air.channel.put async [%62]  @cascade_gp[%arg16, %63] (%arg25[] [] []) {id = 109 : i32} : (memref<64x64xbf16, 2 : i32>)
            %65 = air.channel.put async [%62]  @cascade_up[%arg16, %63] (%arg26[] [] []) {id = 110 : i32} : (memref<64x1xbf16, 2 : i32>)
            %66 = air.channel.put async [%62]  @cascade_sp[%arg16, %63] (%arg27[] [] []) {id = 111 : i32} : (memref<64x1xbf16, 2 : i32>)
          } else {
            scf.if %53 {
              %async_token_57, %results_58 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_59, %results_60 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_61, %results_62 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %63 = air.channel.get async [%async_token_57]  @cascade_gp[%arg16, %arg17] (%results_58[] [] []) {id = 112 : i32} : (memref<64x64xbf16, 2 : i32>)
              %64 = air.channel.get async [%async_token_59]  @cascade_up[%arg16, %arg17] (%results_60[] [] []) {id = 113 : i32} : (memref<64x1xbf16, 2 : i32>)
              %65 = air.channel.get async [%async_token_61]  @cascade_sp[%arg16, %arg17] (%results_62[] [] []) {id = 114 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_65 = air.execute [%async_token_63, %62] {
                func.call @vector_copy_32elems(%c0_i32, %arg26, %results_64) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_65, %64] {
                func.call @maximum_up_u_bf16(%results_60, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_67, %async_token_66] {
                func.call @exp_up_minus_u(%results_60, %arg26, %results_68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_69] {
                memref.dealloc %results_60 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_71, %async_token_69] {
                func.call @exp_up_minus_u(%results_64, %arg26, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_74 = air.execute [%async_token_73] {
                memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_69, %63] {
                func.call @mul_r_gp(%results_68, %results_58) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_73] {
                func.call @mul_r_gp(%results_72, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_76, %async_token_75] {
                func.call @add_gp_g(%arg25, %results_58) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_78, %results_79 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_80 = air.execute [%async_token_78] {
                func.call @zero_fill_sp_bf16(%results_79) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_81 = air.execute [%async_token_80, %async_token_75, %65] {
                func.call @accum_sp_r_s(%results_62, %results_68, %results_79) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_82 = air.execute [%async_token_81] {
                memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_81, %async_token_76] {
                func.call @accum_sp_r_s(%arg27, %results_72, %results_79) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_84 = air.execute [%async_token_83] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_85 = air.execute [%async_token_83] {
                func.call @vector_copy_32elems(%c0_i32, %results_79, %results_62) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_86 = air.execute [%async_token_85] {
                memref.dealloc %results_79 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_87 = air.execute [%async_token_85, %async_token_77] {
                func.call @div_gp_sp(%results_62, %results_58) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_88 = air.execute [%async_token_87] {
                memref.dealloc %results_62 : memref<64x1xbf16, 2 : i32>
              }
              %66 = air.channel.put async [%async_token_87]  @Gp2L2[%arg16, %c0_53] (%results_58[0, 0, 0, 0] [8, 8, 8, 8] [64, 8, 512, 1]) {id = 115 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_89 = air.execute [%66] {
                memref.dealloc %results_58 : memref<64x64xbf16, 2 : i32>
              }
            } else {
              %async_token_57, %results_58 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_59, %results_60 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_61, %results_62 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %63 = air.channel.get async [%async_token_57]  @cascade_gp[%arg16, %arg17] (%results_58[] [] []) {id = 116 : i32} : (memref<64x64xbf16, 2 : i32>)
              %64 = air.channel.get async [%async_token_59]  @cascade_up[%arg16, %arg17] (%results_60[] [] []) {id = 117 : i32} : (memref<64x1xbf16, 2 : i32>)
              %65 = air.channel.get async [%async_token_61]  @cascade_sp[%arg16, %arg17] (%results_62[] [] []) {id = 118 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_65 = air.execute [%async_token_63, %62] {
                func.call @vector_copy_32elems(%c0_i32, %arg26, %results_64) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_65, %64] {
                func.call @maximum_up_u_bf16(%results_60, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_67, %async_token_66] {
                func.call @exp_up_minus_u(%results_60, %arg26, %results_68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_69] {
                memref.dealloc %results_60 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_71, %async_token_69] {
                func.call @exp_up_minus_u(%results_64, %arg26, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_74 = air.execute [%async_token_73] {
                memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_69, %63] {
                func.call @mul_r_gp(%results_68, %results_58) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_73] {
                func.call @mul_r_gp(%results_72, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_76, %async_token_75] {
                func.call @add_gp_g(%arg25, %results_58) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_78, %results_79 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_80 = air.execute [%async_token_78] {
                func.call @zero_fill_sp_bf16(%results_79) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_81 = air.execute [%async_token_80, %async_token_75, %65] {
                func.call @accum_sp_r_s(%results_62, %results_68, %results_79) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_82 = air.execute [%async_token_81] {
                memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_81, %async_token_76] {
                func.call @accum_sp_r_s(%arg27, %results_72, %results_79) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_84 = air.execute [%async_token_83] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_85 = air.execute [%async_token_83] {
                func.call @vector_copy_32elems(%c0_i32, %results_79, %results_62) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_86 = air.execute [%async_token_85] {
                memref.dealloc %results_79 : memref<64x1xbf16, 2 : i32>
              }
              %66 = affine.apply #map10()[%arg17]
              %67 = air.channel.put async [%async_token_77]  @cascade_gp[%arg16, %66] (%results_58[] [] []) {id = 119 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_87 = air.execute [%67] {
                memref.dealloc %results_58 : memref<64x64xbf16, 2 : i32>
              }
              %68 = air.channel.put async [%async_token_73]  @cascade_up[%arg16, %66] (%arg26[] [] []) {id = 120 : i32} : (memref<64x1xbf16, 2 : i32>)
              %69 = air.channel.put async [%async_token_85]  @cascade_sp[%arg16, %66] (%results_62[] [] []) {id = 121 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_88 = air.execute [%69] {
                memref.dealloc %results_62 : memref<64x1xbf16, 2 : i32>
              }
            }
          }
        }
        %async_token_43 = air.execute [%50] {
          memref.dealloc %results_34 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_44 = air.execute [%50] {
          memref.dealloc %results_32 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_45 = air.execute [%50] {
          memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_46 = air.execute [%50] {
          memref.dealloc %results_28 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_47 = air.execute [%50] {
          memref.dealloc %results_26 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_48 = air.execute [%50] {
          memref.dealloc %results_24 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_49 = air.execute [%50] {
          memref.dealloc %results_22 : memref<64x64xbf16, 2 : i32>
        }
        %51 = scf.parallel (%arg16) = (%c0_4) to (%c4) step (%c1_3) init (%async_token_19) -> !air.async.token {
          %53 = affine.apply #map11()[%arg16]
          %54 = air.channel.get async [%async_token_19]  @Gp2L2[%arg16, %c0_4] (%results_20[%53, 0] [64, 64] [64, 1]) {id = 122 : i32} : (memref<256x64xbf16, 1 : i32>)
          scf.reduce(%54 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %55 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %55 : !air.async.token
          }
        }
        %52 = air.channel.put async [%51]  @GpOut[%arg12] (%results_20[] [] []) {id = 123 : i32} : (memref<256x64xbf16, 1 : i32>)
        %async_token_50 = air.execute [%52] {
          memref.dealloc %results_20 : memref<256x64xbf16, 1 : i32>
        }
      }
      %36 = air.channel.get async [%35]  @GpOut[%c0] (%arg11[%1] [16384] [1]) {id = 124 : i32} : (memref<2x512x64xbf16>)
      %37 = air.channel.get async [%35]  @GpOut[%c1_1] (%arg11[%18] [16384] [1]) {id = 125 : i32} : (memref<2x512x64xbf16>)
    }
    return
  }
}
