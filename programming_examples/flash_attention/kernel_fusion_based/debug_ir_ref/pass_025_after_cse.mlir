#map = affine_map<()[s0, s1] -> (s0 * 32768 + s1 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 32768)>
#map2 = affine_map<()[s0] -> (s0 * 32768 + 4096)>
#map3 = affine_map<()[s0] -> (s0 * 32768 + 8192)>
#map4 = affine_map<()[s0] -> (s0 * 32768 + 12288)>
#map5 = affine_map<()[s0, s1] -> (s0 * 32768 + s1 * 16384 + 16384)>
#map6 = affine_map<()[s0] -> (s0 * 32768 + 16384)>
#map7 = affine_map<()[s0] -> (s0 * 32768 + 20480)>
#map8 = affine_map<()[s0] -> (s0 * 32768 + 24576)>
#map9 = affine_map<()[s0] -> (s0 * 32768 + 28672)>
#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set4 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
  air.channel @channel_0 [4, 2]
  func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @maximum_up_u_bf16(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @exp_up_minus_u(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @add_gp_g(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  air.channel @QK2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QKIn_0 [2]
  air.channel @QK2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QKIn_1 [2]
  air.channel @QK2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QKIn_2 [2]
  air.channel @QK2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QKIn_3 [2]
  air.channel @V2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @VIn_0 [2]
  air.channel @V2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @VIn_1 [2]
  air.channel @V2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @VIn_2 [2]
  air.channel @V2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @VIn_3 [2]
  air.channel @cascade_gp [4, 3] {channel_type = "cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  func.func @attention_bf16(%arg0: memref<2x256x64xbf16>, %arg1: memref<2x256x64xbf16>, %arg2: memref<2x256x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_0]) {id = 1 : i32} : (memref<2x256x64xbf16>)
      %3 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_0]) {id = 2 : i32} : (memref<2x256x64xbf16>)
      %4 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_0]) {id = 3 : i32} : (memref<2x256x64xbf16>)
      %5 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_0]) {id = 4 : i32} : (memref<2x256x64xbf16>)
      %6 = affine.apply #map1()[%arg5]
      %7 = air.channel.put async  @QKIn_0[%c0] (%arg9[%c0, %6] [%c64, %c64] [%c64, %c1_0]) {id = 5 : i32} : (memref<2x256x64xbf16>)
      %8 = affine.apply #map2()[%arg5]
      %9 = air.channel.put async  @QKIn_1[%c0] (%arg9[%c0, %8] [%c64, %c64] [%c64, %c1_0]) {id = 6 : i32} : (memref<2x256x64xbf16>)
      %10 = affine.apply #map3()[%arg5]
      %11 = air.channel.put async  @QKIn_2[%c0] (%arg9[%c0, %10] [%c64, %c64] [%c64, %c1_0]) {id = 7 : i32} : (memref<2x256x64xbf16>)
      %12 = affine.apply #map4()[%arg5]
      %13 = air.channel.put async  @QKIn_3[%c0] (%arg9[%c0, %12] [%c64, %c64] [%c64, %c1_0]) {id = 8 : i32} : (memref<2x256x64xbf16>)
      %14 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %6] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 9 : i32} : (memref<2x256x64xbf16>)
      %15 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %8] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 10 : i32} : (memref<2x256x64xbf16>)
      %16 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %10] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 11 : i32} : (memref<2x256x64xbf16>)
      %17 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %12] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 12 : i32} : (memref<2x256x64xbf16>)
      %18 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %1] [%c64, %c64] [%c64, %c1_0]) {id = 13 : i32} : (memref<2x256x64xbf16>)
      %19 = air.channel.get async  @channel_0[%c1_0, %c0] (%arg11[%c64, %1] [%c64, %c64] [%c64, %c1_0]) {id = 14 : i32} : (memref<2x256x64xbf16>)
      %20 = air.channel.get async  @channel_0[%c2, %c0] (%arg11[%c128, %1] [%c64, %c64] [%c64, %c1_0]) {id = 15 : i32} : (memref<2x256x64xbf16>)
      %21 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %1] [%c64, %c64] [%c64, %c1_0]) {id = 16 : i32} : (memref<2x256x64xbf16>)
      %22 = affine.apply #map5()[%arg5, %arg4]
      %23 = air.channel.put async  @QKIn_0[%c1_0] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_0]) {id = 17 : i32} : (memref<2x256x64xbf16>)
      %24 = air.channel.put async  @QKIn_1[%c1_0] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_0]) {id = 18 : i32} : (memref<2x256x64xbf16>)
      %25 = air.channel.put async  @QKIn_2[%c1_0] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_0]) {id = 19 : i32} : (memref<2x256x64xbf16>)
      %26 = air.channel.put async  @QKIn_3[%c1_0] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_0]) {id = 20 : i32} : (memref<2x256x64xbf16>)
      %27 = affine.apply #map6()[%arg5]
      %28 = air.channel.put async  @QKIn_0[%c1_0] (%arg9[%c0, %27] [%c64, %c64] [%c64, %c1_0]) {id = 21 : i32} : (memref<2x256x64xbf16>)
      %29 = affine.apply #map7()[%arg5]
      %30 = air.channel.put async  @QKIn_1[%c1_0] (%arg9[%c0, %29] [%c64, %c64] [%c64, %c1_0]) {id = 22 : i32} : (memref<2x256x64xbf16>)
      %31 = affine.apply #map8()[%arg5]
      %32 = air.channel.put async  @QKIn_2[%c1_0] (%arg9[%c0, %31] [%c64, %c64] [%c64, %c1_0]) {id = 23 : i32} : (memref<2x256x64xbf16>)
      %33 = affine.apply #map9()[%arg5]
      %34 = air.channel.put async  @QKIn_3[%c1_0] (%arg9[%c0, %33] [%c64, %c64] [%c64, %c1_0]) {id = 24 : i32} : (memref<2x256x64xbf16>)
      %35 = air.channel.put async  @VIn_0[%c1_0] (%arg10[%c0, %c0, %27] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 25 : i32} : (memref<2x256x64xbf16>)
      %36 = air.channel.put async  @VIn_1[%c1_0] (%arg10[%c0, %c0, %29] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 26 : i32} : (memref<2x256x64xbf16>)
      %37 = air.channel.put async  @VIn_2[%c1_0] (%arg10[%c0, %c0, %31] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 27 : i32} : (memref<2x256x64xbf16>)
      %38 = air.channel.put async  @VIn_3[%c1_0] (%arg10[%c0, %c0, %33] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 28 : i32} : (memref<2x256x64xbf16>)
      %39 = air.channel.get async  @channel_0[%c0, %c1_0] (%arg11[%c0, %22] [%c64, %c64] [%c64, %c1_0]) {id = 29 : i32} : (memref<2x256x64xbf16>)
      %40 = air.channel.get async  @channel_0[%c1_0, %c1_0] (%arg11[%c64, %22] [%c64, %c64] [%c64, %c1_0]) {id = 30 : i32} : (memref<2x256x64xbf16>)
      %41 = air.channel.get async  @channel_0[%c2, %c1_0] (%arg11[%c128, %22] [%c64, %c64] [%c64, %c1_0]) {id = 31 : i32} : (memref<2x256x64xbf16>)
      %42 = air.channel.get async  @channel_0[%c3, %c1_0] (%arg11[%c192, %22] [%c64, %c64] [%c64, %c1_0]) {id = 32 : i32} : (memref<2x256x64xbf16>)
      %43 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) attributes {id = 2 : i32} {
        %c3_1 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c64_3 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c8 = arith.constant 8 : index
        %c1_4 = arith.constant 1 : index
        %c0_5 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_6, %results_7 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_8, %results_9 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_10, %results_11 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_12, %results_13 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_14, %results_15 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_16, %results_17 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_22, %results_23 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_24, %results_25 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_26, %results_27 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_28, %results_29 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_30, %results_31 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_32, %results_33 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_34, %results_35 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_36, %results_37 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_38, %results_39 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_40, %results_41 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %44 = scf.for %arg16 = %c0_5 to %c4 step %c1_4 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %74 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %75 = arith.cmpi eq, %arg12, %c0_5 : index
          %76 = scf.if %75 -> (!air.async.token) {
            %77 = air.channel.put async [%74]  @QK2L1_0_0[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %77 : !air.async.token
          } else {
            %77 = air.channel.put async [%74]  @QK2L1_0_1[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %77 : !air.async.token
          }
          scf.yield %76 : !air.async.token
        }
        %45 = air.channel.get async [%44]  @QKIn_0[%arg12] (%results[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
        %46 = arith.cmpi eq, %arg12, %c0_5 : index
        %47 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%45]  @QK2L1_0_0[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%45]  @QK2L1_0_1[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %48 = scf.for %arg16 = %c0_5 to %c4 step %c1_4 iter_args(%arg17 = %async_token_6) -> (!air.async.token) {
          %74 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %75 = scf.if %46 -> (!air.async.token) {
            %76 = air.channel.put async [%74]  @QK2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          } else {
            %76 = air.channel.put async [%74]  @QK2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          }
          scf.yield %75 : !air.async.token
        }
        %49 = air.channel.get async [%48]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
        %50 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%49]  @QK2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%49]  @QK2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %51 = scf.for %arg16 = %c0_5 to %c4 step %c1_4 iter_args(%arg17 = %async_token_8) -> (!air.async.token) {
          %74 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
          %75 = scf.if %46 -> (!air.async.token) {
            %76 = air.channel.put async [%74]  @QK2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          } else {
            %76 = air.channel.put async [%74]  @QK2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          }
          scf.yield %75 : !air.async.token
        }
        %52 = air.channel.get async [%51]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
        %53 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%52]  @QK2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%52]  @QK2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %54 = scf.for %arg16 = %c0_5 to %c4 step %c1_4 iter_args(%arg17 = %async_token_10) -> (!air.async.token) {
          %74 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
          %75 = scf.if %46 -> (!air.async.token) {
            %76 = air.channel.put async [%74]  @QK2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          } else {
            %76 = air.channel.put async [%74]  @QK2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 53 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          }
          scf.yield %75 : !air.async.token
        }
        %55 = air.channel.get async [%54]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 1 : i32>)
        %56 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%55]  @QK2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 55 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%55]  @QK2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 56 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %57 = air.channel.get async [%async_token_12]  @VIn_0[%arg12] (%results_13[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 1 : i32>)
        %58 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%57]  @V2L1_0_0[%c0_5, %c0_5, %c0_5] (%results_13[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 58 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%57]  @V2L1_0_1[%c0_5, %c0_5, %c0_5] (%results_13[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 59 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %59 = air.channel.get async [%async_token_14]  @VIn_1[%arg12] (%results_15[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 1 : i32>)
        %60 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%59]  @V2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_15[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 61 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%59]  @V2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_15[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 62 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %61 = air.channel.get async [%async_token_16]  @VIn_2[%arg12] (%results_17[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 1 : i32>)
        %62 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%61]  @V2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_17[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 64 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%61]  @V2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_17[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 65 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %63 = air.channel.get async [%async_token_18]  @VIn_3[%arg12] (%results_19[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 1 : i32>)
        %64 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%63]  @V2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_19[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 67 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%63]  @V2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_19[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_4]) {id = 68 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %65 = air.channel.get async [%async_token_20]  @Gp2L2[%c0_5, %c0_5] (%results_21[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %66 = air.channel.get async [%async_token_22]  @Gp2L2[%c1_4, %c0_5] (%results_23[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %67 = air.channel.get async [%async_token_24]  @Gp2L2[%c2_2, %c0_5] (%results_25[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %68 = air.channel.get async [%async_token_26]  @Gp2L2[%c3_1, %c0_5] (%results_27[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %69 = air.channel.put async [%65]  @channel_0[%c0_5, %arg12] (%results_21[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %70 = air.channel.put async [%66]  @channel_0[%c1_4, %arg12] (%results_23[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %71 = air.channel.put async [%67]  @channel_0[%c2_2, %arg12] (%results_25[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %72 = air.channel.put async [%68]  @channel_0[%c3_1, %arg12] (%results_27[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %73 = air.herd @herd_0 async [%async_token_28, %async_token_30, %async_token_32, %async_token_34, %async_token_36, %async_token_38, %async_token_40]  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%results_29, %arg21=%results_31, %arg22=%results_33, %arg23=%results_35, %arg24=%results_37, %arg25=%results_39, %arg26=%results_41, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {id = 3 : i32, link_with = "attn.o"} {
          %c512_61 = arith.constant 512 : index
          %c64_62 = arith.constant 64 : index
          %c8_63 = arith.constant 8 : index
          %c0_64 = arith.constant 0 : index
          %c1_65 = arith.constant 1 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_66 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_67 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_68 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async  @QK2L1_0_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async  @QK2L1_0_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%74]  @QK2L1_1_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%74]  @QK2L1_1_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%75]  @QK2L1_2_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%75]  @QK2L1_2_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %77 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%76]  @QK2L1_3_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%76]  @QK2L1_3_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %78 = arith.index_cast %arg16 : index to i32
          %79 = arith.cmpi eq, %78, %c0_i32 : i32
          scf.if %79 {
            %async_token_82 = air.execute [%77] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %80 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async  @QK2L1_0_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async  @QK2L1_0_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %81 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%80]  @QK2L1_1_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%80]  @QK2L1_1_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %82 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%81]  @QK2L1_2_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%81]  @QK2L1_2_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %83 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%82]  @QK2L1_3_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%82]  @QK2L1_3_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %84 = arith.cmpi eq, %78, %c1_i32 : i32
          scf.if %84 {
            %async_token_82 = air.execute [%83] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %85 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async  @QK2L1_0_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async  @QK2L1_0_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %86 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%85]  @QK2L1_1_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%85]  @QK2L1_1_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %87 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%86]  @QK2L1_2_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%86]  @QK2L1_2_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %88 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%87]  @QK2L1_3_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%87]  @QK2L1_3_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %89 = arith.cmpi eq, %78, %c2_i32 : i32
          scf.if %89 {
            %async_token_82 = air.execute [%88] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %90 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async  @QK2L1_0_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async  @QK2L1_0_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 102 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %91 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%90]  @QK2L1_1_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 103 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%90]  @QK2L1_1_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %92 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%91]  @QK2L1_2_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 105 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%91]  @QK2L1_2_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 106 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %93 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%92]  @QK2L1_3_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%92]  @QK2L1_3_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 108 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %94 = arith.cmpi eq, %78, %c3_i32 : i32
          scf.if %94 {
            %async_token_82 = air.execute [%93] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %async_token_69 = air.execute {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          %95 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async  @QK2L1_0_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 109 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async  @QK2L1_0_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %96 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%95]  @QK2L1_1_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 111 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%95]  @QK2L1_1_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 112 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %97 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%96]  @QK2L1_2_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%96]  @QK2L1_2_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 114 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %98 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%97]  @QK2L1_3_0[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 115 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%97]  @QK2L1_3_1[%c0_64, %arg17, %arg16] (%arg21[] [] []) {id = 116 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %async_token_70 = air.execute [%98, %async_token_69] {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %99 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async  @V2L1_0_0[%c0_64, %arg17, %arg16] (%arg22[] [] []) {id = 117 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async  @V2L1_0_1[%c0_64, %arg17, %arg16] (%arg22[] [] []) {id = 118 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %100 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%99]  @V2L1_1_0[%c0_64, %arg17, %arg16] (%arg22[] [] []) {id = 119 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%99]  @V2L1_1_1[%c0_64, %arg17, %arg16] (%arg22[] [] []) {id = 120 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %101 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%100]  @V2L1_2_0[%c0_64, %arg17, %arg16] (%arg22[] [] []) {id = 121 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%100]  @V2L1_2_1[%c0_64, %arg17, %arg16] (%arg22[] [] []) {id = 122 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %102 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg27, %c0_64 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%101]  @V2L1_3_0[%c0_64, %arg17, %arg16] (%arg22[] [] []) {id = 123 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%101]  @V2L1_3_1[%c0_64, %arg17, %arg16] (%arg22[] [] []) {id = 124 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_75 = air.execute [%async_token_73, %async_token_71, %async_token_70, %async_token_68] {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %results_72, %results_74) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_76 = air.execute [%async_token_75, %async_token_66] {
            func.call @mul_r_gp(%results_74, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_77 = air.execute [%async_token_76, %102] {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_78 = air.execute [%async_token_76, %async_token_67] {
            func.call @accum_sp_r_s(%arg26, %results_74, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_79 = air.execute [%async_token_78] {
            func.call @vector_copy_32elems(%c0_i32, %results_72, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_80 = air.execute [%async_token_79] {
            memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_81 = air.execute [%async_token_78] {
            memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
          }
          %103 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.subi %arg17, %c1_65 : index
            %105 = air.channel.put async [%async_token_77]  @cascade_gp[%arg16, %104] (%arg24[] [] []) {id = 125 : i32} : (memref<64x64xbf16, 2 : i32>)
            %106 = air.channel.put async [%async_token_68]  @cascade_up[%arg16, %104] (%arg25[] [] []) {id = 126 : i32} : (memref<64x1xbf16, 2 : i32>)
            %107 = air.channel.put async [%async_token_79]  @cascade_sp[%arg16, %104] (%arg26[] [] []) {id = 127 : i32} : (memref<64x1xbf16, 2 : i32>)
            %108 = air.wait_all async [%105, %106, %107] 
            affine.yield %108 : !air.async.token
          } else {
            %104 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_82, %results_83 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_84, %results_85 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_86, %results_87 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %106 = air.channel.get async [%async_token_82]  @cascade_gp[%arg16, %arg17] (%results_83[] [] []) {id = 128 : i32} : (memref<64x64xbf16, 2 : i32>)
              %107 = air.channel.get async [%async_token_84]  @cascade_up[%arg16, %arg17] (%results_85[] [] []) {id = 129 : i32} : (memref<64x1xbf16, 2 : i32>)
              %108 = air.channel.get async [%async_token_86]  @cascade_sp[%arg16, %arg17] (%results_87[] [] []) {id = 130 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_88, %results_89 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_90 = air.execute [%async_token_88, %async_token_68] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_89) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_91 = air.execute [%async_token_90, %107] {
                func.call @maximum_up_u_bf16(%results_85, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_92, %results_93 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_94 = air.execute [%async_token_92, %async_token_91] {
                func.call @exp_up_minus_u(%results_85, %arg25, %results_93) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_95, %results_96 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_97 = air.execute [%async_token_95, %async_token_94] {
                func.call @exp_up_minus_u(%results_89, %arg25, %results_96) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_98 = air.execute [%async_token_94, %106] {
                func.call @mul_r_gp(%results_93, %results_83) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_99 = air.execute [%async_token_97, %async_token_77] {
                func.call @mul_r_gp(%results_96, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_100 = air.execute [%async_token_99, %async_token_98] {
                func.call @add_gp_g(%arg24, %results_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_101, %results_102 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_103 = air.execute [%async_token_101] {
                func.call @zero_fill_sp_bf16(%results_102) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_104 = air.execute [%async_token_103, %async_token_98, %108] {
                func.call @accum_sp_r_s(%results_87, %results_93, %results_102) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_105 = air.execute [%async_token_104, %async_token_99, %async_token_79] {
                func.call @accum_sp_r_s(%arg26, %results_96, %results_102) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_106 = air.execute [%async_token_105] {
                func.call @vector_copy_32elems(%c0_i32, %results_102, %results_87) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %109 = arith.subi %arg17, %c1_65 : index
              %110 = air.channel.put async [%async_token_100]  @cascade_gp[%arg16, %109] (%results_83[] [] []) {id = 131 : i32} : (memref<64x64xbf16, 2 : i32>)
              %111 = air.channel.put async [%async_token_97]  @cascade_up[%arg16, %109] (%arg25[] [] []) {id = 132 : i32} : (memref<64x1xbf16, 2 : i32>)
              %112 = air.channel.put async [%async_token_106]  @cascade_sp[%arg16, %109] (%results_87[] [] []) {id = 133 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_107 = air.execute [%110] {
                memref.dealloc %results_83 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_108 = air.execute [%async_token_94] {
                memref.dealloc %results_85 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_109 = air.execute [%112] {
                memref.dealloc %results_87 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_110 = air.execute [%async_token_97] {
                memref.dealloc %results_89 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_111 = air.execute [%async_token_104] {
                memref.dealloc %results_93 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_105] {
                memref.dealloc %results_96 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_113 = air.execute [%async_token_106] {
                memref.dealloc %results_102 : memref<64x1xbf16, 2 : i32>
              }
              %113 = air.wait_all async [%110, %111, %112] 
              affine.yield %113 : !air.async.token
            } else {
              %async_token_82, %results_83 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_84, %results_85 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_86, %results_87 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %106 = air.channel.get async [%async_token_82]  @cascade_gp[%arg16, %arg17] (%results_83[] [] []) {id = 134 : i32} : (memref<64x64xbf16, 2 : i32>)
              %107 = air.channel.get async [%async_token_84]  @cascade_up[%arg16, %arg17] (%results_85[] [] []) {id = 135 : i32} : (memref<64x1xbf16, 2 : i32>)
              %108 = air.channel.get async [%async_token_86]  @cascade_sp[%arg16, %arg17] (%results_87[] [] []) {id = 136 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_88, %results_89 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_90 = air.execute [%async_token_88, %async_token_68] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_89) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_91 = air.execute [%async_token_90, %107] {
                func.call @maximum_up_u_bf16(%results_85, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_92, %results_93 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_94 = air.execute [%async_token_92, %async_token_91] {
                func.call @exp_up_minus_u(%results_85, %arg25, %results_93) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_95, %results_96 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_97 = air.execute [%async_token_95, %async_token_94] {
                func.call @exp_up_minus_u(%results_89, %arg25, %results_96) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_98 = air.execute [%async_token_94, %106] {
                func.call @mul_r_gp(%results_93, %results_83) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_99 = air.execute [%async_token_97, %async_token_77] {
                func.call @mul_r_gp(%results_96, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_100 = air.execute [%async_token_99, %async_token_98] {
                func.call @add_gp_g(%arg24, %results_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_101, %results_102 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_103 = air.execute [%async_token_101] {
                func.call @zero_fill_sp_bf16(%results_102) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_104 = air.execute [%async_token_103, %async_token_98, %108] {
                func.call @accum_sp_r_s(%results_87, %results_93, %results_102) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_105 = air.execute [%async_token_104, %async_token_99, %async_token_79] {
                func.call @accum_sp_r_s(%arg26, %results_96, %results_102) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_106 = air.execute [%async_token_105] {
                func.call @vector_copy_32elems(%c0_i32, %results_102, %results_87) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_107 = air.execute [%async_token_106, %async_token_100] {
                func.call @div_gp_sp(%results_87, %results_83) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %109 = air.channel.put async [%async_token_107]  @Gp2L2[%arg16, %c0_64] (%results_83[%c0_64, %c0_64, %c0_64, %c0_64] [%c8_63, %c8_63, %c8_63, %c8_63] [%c64_62, %c8_63, %c512_61, %c1_65]) {id = 137 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_108 = air.execute [%109] {
                memref.dealloc %results_83 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_109 = air.execute [%async_token_94] {
                memref.dealloc %results_85 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_110 = air.execute [%async_token_107] {
                memref.dealloc %results_87 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_111 = air.execute [%async_token_97] {
                memref.dealloc %results_89 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_104] {
                memref.dealloc %results_93 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_113 = air.execute [%async_token_105] {
                memref.dealloc %results_96 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_114 = air.execute [%async_token_106] {
                memref.dealloc %results_102 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %109 : !air.async.token
            }
            %105 = air.wait_all async [%95, %96, %97, %99, %100, %101, %async_token_77, %async_token_79] 
            affine.yield %105 : !air.async.token
          }
        }
        %async_token_42 = air.execute [%73] {
          memref.dealloc %results_29 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_43 = air.execute [%73] {
          memref.dealloc %results_31 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_44 = air.execute [%73] {
          memref.dealloc %results_33 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_45 = air.execute [%73] {
          memref.dealloc %results_35 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_46 = air.execute [%73] {
          memref.dealloc %results_37 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_47 = air.execute [%73] {
          memref.dealloc %results_39 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_48 = air.execute [%73] {
          memref.dealloc %results_41 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_49 = air.execute [%47] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_50 = air.execute [%58] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_51 = air.execute [%50] {
          memref.dealloc %results_7 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_52 = air.execute [%60] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_53 = air.execute [%53] {
          memref.dealloc %results_9 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_54 = air.execute [%62] {
          memref.dealloc %results_17 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_55 = air.execute [%56] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_56 = air.execute [%64] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_57 = air.execute [%72] {
          memref.dealloc %results_27 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_58 = air.execute [%71] {
          memref.dealloc %results_25 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_59 = air.execute [%70] {
          memref.dealloc %results_23 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_60 = air.execute [%69] {
          memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
