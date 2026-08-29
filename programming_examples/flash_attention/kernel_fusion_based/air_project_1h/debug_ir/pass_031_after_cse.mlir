#map = affine_map<()[s0] -> (s0 * 16384)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set6 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set7 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
  air.channel @channel_0 [4, 1]
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
  air.channel @QK2L1_0 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @QK2L1_1 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @QK2L1_2 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @QK2L1_3 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @V2L1_0 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_0 [1]
  air.channel @V2L1_1 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_1 [1]
  air.channel @V2L1_2 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_2 [1]
  air.channel @V2L1_3 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_3 [1]
  air.channel @cascade_gp [4, 3] {channel_type = "cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  func.func @attention_bf16(%arg0: memref<256x64xbf16>, %arg1: memref<512x64xbf16>, %arg2: memref<512x64xbf16>, %arg3: memref<256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<256x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16>, memref<256x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c24576 = arith.constant 24576 : index
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = air.channel.put async  @QK2L1_0[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 1 : i32} : (memref<256x64xbf16>)
      %3 = air.channel.put async  @QK2L1_1[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 2 : i32} : (memref<256x64xbf16>)
      %4 = air.channel.put async  @QK2L1_2[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 3 : i32} : (memref<256x64xbf16>)
      %5 = air.channel.put async  @QK2L1_3[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 4 : i32} : (memref<256x64xbf16>)
      %6 = air.channel.put async  @QK2L1_0[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c0] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 5 : i32} : (memref<512x64xbf16>)
      %7 = air.channel.put async  @QK2L1_1[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c8192] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 6 : i32} : (memref<512x64xbf16>)
      %8 = air.channel.put async  @QK2L1_2[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c16384] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 7 : i32} : (memref<512x64xbf16>)
      %9 = air.channel.put async  @QK2L1_3[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c24576] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 8 : i32} : (memref<512x64xbf16>)
      %10 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %c0] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 9 : i32} : (memref<512x64xbf16>)
      %11 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %c8192] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 10 : i32} : (memref<512x64xbf16>)
      %12 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %c16384] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 11 : i32} : (memref<512x64xbf16>)
      %13 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %c24576] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 12 : i32} : (memref<512x64xbf16>)
      %14 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 13 : i32} : (memref<256x64xbf16>)
      %15 = air.channel.get async  @channel_0[%c1_0, %c0] (%arg11[%c64, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 14 : i32} : (memref<256x64xbf16>)
      %16 = air.channel.get async  @channel_0[%c2, %c0] (%arg11[%c128, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 15 : i32} : (memref<256x64xbf16>)
      %17 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 16 : i32} : (memref<256x64xbf16>)
      %18 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c1_0, %arg15=%c1_0) attributes {id = 2 : i32} {
        %c3_1 = arith.constant 3 : index
        %c64_2 = arith.constant 64 : index
        %c512_3 = arith.constant 512 : index
        %c8_4 = arith.constant 8 : index
        %c1_5 = arith.constant 1 : index
        %c2_6 = arith.constant 2 : index
        %c0_7 = arith.constant 0 : index
        %c4_8 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
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
        %19 = air.channel.get async [%async_token_15]  @VIn_0[%c0_7] (%results_16[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
        %20 = air.channel.put async [%19]  @V2L1_0[%c0_7, %c0_7] (%results_16[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_17 = air.execute [%20] {
          memref.dealloc %results_16 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %21 = air.channel.get async [%async_token_18]  @VIn_0[%c0_7] (%results_19[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
        %22 = air.channel.put async [%21]  @V2L1_0[%c0_7, %c0_7] (%results_19[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_20 = air.execute [%22] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %23 = air.channel.get async [%async_token_21]  @VIn_1[%c0_7] (%results_22[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
        %24 = air.channel.put async [%23]  @V2L1_1[%c0_7, %c0_7] (%results_22[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_23 = air.execute [%24] {
          memref.dealloc %results_22 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_24, %results_25 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %25 = air.channel.get async [%async_token_24]  @VIn_1[%c0_7] (%results_25[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
        %26 = air.channel.put async [%25]  @V2L1_1[%c0_7, %c0_7] (%results_25[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_26 = air.execute [%26] {
          memref.dealloc %results_25 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_27, %results_28 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %27 = air.channel.get async [%async_token_27]  @VIn_2[%c0_7] (%results_28[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
        %28 = air.channel.put async [%27]  @V2L1_2[%c0_7, %c0_7] (%results_28[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_29 = air.execute [%28] {
          memref.dealloc %results_28 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_30, %results_31 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %29 = air.channel.get async [%async_token_30]  @VIn_2[%c0_7] (%results_31[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
        %30 = air.channel.put async [%29]  @V2L1_2[%c0_7, %c0_7] (%results_31[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_32 = air.execute [%30] {
          memref.dealloc %results_31 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_33, %results_34 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %31 = air.channel.get async [%async_token_33]  @VIn_3[%c0_7] (%results_34[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
        %32 = air.channel.put async [%31]  @V2L1_3[%c0_7, %c0_7] (%results_34[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_35 = air.execute [%32] {
          memref.dealloc %results_34 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_36, %results_37 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %33 = air.channel.get async [%async_token_36]  @VIn_3[%c0_7] (%results_37[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
        %34 = air.channel.put async [%33]  @V2L1_3[%c0_7, %c0_7] (%results_37[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_38 = air.execute [%34] {
          memref.dealloc %results_37 : memref<64x64xbf16, 1 : i32>
        }
        %35 = air.channel.get async [%async_token]  @Gp2L2[%c0_7, %c0_7] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %36 = air.channel.get async [%async_token_9]  @Gp2L2[%c1_5, %c0_7] (%results_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %37 = air.channel.get async [%async_token_11]  @Gp2L2[%c2_6, %c0_7] (%results_12[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %38 = air.channel.get async [%async_token_13]  @Gp2L2[%c3_1, %c0_7] (%results_14[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %39 = air.channel.put async [%35]  @channel_0[%c0_7, %c0_7] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %40 = air.channel.put async [%36]  @channel_0[%c1_5, %c0_7] (%results_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %41 = air.channel.put async [%37]  @channel_0[%c2_6, %c0_7] (%results_12[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %42 = air.channel.put async [%38]  @channel_0[%c3_1, %c0_7] (%results_14[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %43 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) attributes {id = 3 : i32, link_with = "attn.o"} {
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c0_43 = arith.constant 0 : index
          %c1_44 = arith.constant 1 : index
          %c8_45 = arith.constant 8 : index
          %c64_46 = arith.constant 64 : index
          %c512_47 = arith.constant 512 : index
          %async_token_48, %results_49 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_50, %results_51 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_52, %results_53 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_54, %results_55 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_56, %results_57 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_58 = air.execute [%async_token_52] {
            func.call @zero_fill_gp_bf16(%results_53) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_59 = air.execute [%async_token_48] {
            func.call @zero_fill_sp_bf16(%results_49) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_60 = air.execute [%async_token_50] {
            func.call @neg_inf_fill_up_bf16(%results_51) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %44 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_54]  @QK2L1_0[%arg16, %arg17] (%results_55[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_54]  @QK2L1_1[%arg16, %arg17] (%results_55[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_2[%arg16, %arg17] (%results_55[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_3[%arg16, %arg17] (%results_55[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %45 = arith.index_cast %arg16 : index to i32
          %46 = arith.cmpi eq, %45, %c0_i32 : i32
          scf.if %46 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56, %44] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %47 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_54]  @QK2L1_0[%arg16, %arg17] (%results_55[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_54]  @QK2L1_1[%arg16, %arg17] (%results_55[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_2[%arg16, %arg17] (%results_55[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_3[%arg16, %arg17] (%results_55[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %48 = arith.cmpi eq, %45, %c1_i32 : i32
          scf.if %48 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56, %47] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %49 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_54]  @QK2L1_0[%arg16, %arg17] (%results_55[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_54]  @QK2L1_1[%arg16, %arg17] (%results_55[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_2[%arg16, %arg17] (%results_55[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_3[%arg16, %arg17] (%results_55[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %50 = arith.cmpi eq, %45, %c2_i32 : i32
          scf.if %50 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56, %49] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %51 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_54]  @QK2L1_0[%arg16, %arg17] (%results_55[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_54]  @QK2L1_1[%arg16, %arg17] (%results_55[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_2[%arg16, %arg17] (%results_55[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_3[%arg16, %arg17] (%results_55[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %52 = arith.cmpi eq, %45, %c3_i32 : i32
          scf.if %52 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56, %51] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %async_token_61, %results_62 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_63, %results_64 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_65 = air.execute [%async_token_63] {
            %collapse_shape = memref.collapse_shape %results_64 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          %53 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_54]  @QK2L1_0[%arg16, %arg17] (%results_55[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_54]  @QK2L1_1[%arg16, %arg17] (%results_55[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_2[%arg16, %arg17] (%results_55[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_3[%arg16, %arg17] (%results_55[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %54 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_61]  @V2L1_0[%arg16, %arg17] (%results_62[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %55 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_61, %54]  @V2L1_1[%arg16, %arg17] (%results_62[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %56 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_61, %55]  @V2L1_2[%arg16, %arg17] (%results_62[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %57 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_61, %56]  @V2L1_3[%arg16, %arg17] (%results_62[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %async_token_66 = air.execute [%53, %async_token_65, %async_token_56, %async_token_54] {
            %collapse_shape = memref.collapse_shape %results_64 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%results_57, %results_55, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_69, %results_70 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_71 = air.execute [%async_token_50, %async_token_63, %async_token_69, %async_token_67, %async_token_66] {
            %collapse_shape = memref.collapse_shape %results_64 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %results_51, %results_68, %results_70) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_72 = air.execute [%async_token_52, %async_token_71] {
            func.call @mul_r_gp(%results_70, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_73 = air.execute [%async_token_72, %57, %async_token_61, %async_token_63] {
            %collapse_shape = memref.collapse_shape %results_64 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %results_62, %results_53) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_74 = air.execute [%async_token_48, %async_token_72] {
            func.call @accum_sp_r_s(%results_49, %results_70, %results_68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_75 = air.execute [%async_token_74] {
            func.call @vector_copy_32elems(%c0_i32, %results_68, %results_49) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_76 = air.execute [%async_token_75] {
            memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_77 = air.execute [%async_token_74] {
            memref.dealloc %results_70 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_78 = air.execute [%async_token_71, %async_token_73] {
            memref.dealloc %results_64 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_79 = air.execute [%54, %55, %56, %async_token_73] {
            memref.dealloc %results_62 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_80, %results_81 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_82, %results_83 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_84 = air.execute [%async_token_82] {
            %collapse_shape = memref.collapse_shape %results_83 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_54]  @QK2L1_0[%arg16, %arg17] (%results_55[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_54]  @QK2L1_1[%arg16, %arg17] (%results_55[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_2[%arg16, %arg17] (%results_55[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_54]  @QK2L1_3[%arg16, %arg17] (%results_55[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %59 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_80]  @V2L1_0[%arg16, %arg17] (%results_81[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %60 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_80, %59]  @V2L1_1[%arg16, %arg17] (%results_81[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %61 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_80, %60]  @V2L1_2[%arg16, %arg17] (%results_81[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %62 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_80, %61]  @V2L1_3[%arg16, %arg17] (%results_81[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %async_token_85 = air.execute [%58, %async_token_84, %async_token_56, %async_token_54] {
            %collapse_shape = memref.collapse_shape %results_83 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%results_57, %results_55, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %async_token_86, %results_87 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_88, %results_89 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_90 = air.execute [%async_token_50, %async_token_82, %async_token_88, %async_token_86, %async_token_85] {
            %collapse_shape = memref.collapse_shape %results_83 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %results_51, %results_87, %results_89) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_91 = air.execute [%async_token_52, %async_token_90] {
            func.call @mul_r_gp(%results_89, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_92 = air.execute [%async_token_91, %62, %async_token_80, %async_token_82] {
            %collapse_shape = memref.collapse_shape %results_83 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %results_81, %results_53) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_93 = air.execute [%async_token_48, %async_token_91] {
            func.call @accum_sp_r_s(%results_49, %results_89, %results_87) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_94 = air.execute [%async_token_93] {
            func.call @vector_copy_32elems(%c0_i32, %results_87, %results_49) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_95 = air.execute [%async_token_94] {
            memref.dealloc %results_87 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_96 = air.execute [%async_token_93] {
            memref.dealloc %results_89 : memref<64x1xbf16, 2 : i32>
          }
          %63 = air.wait_all async [%async_token_92, %async_token_94] 
          %async_token_97 = air.execute [%async_token_90, %async_token_92] {
            memref.dealloc %results_83 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_98 = air.execute [%59, %60, %61, %async_token_92] {
            memref.dealloc %results_81 : memref<64x64xbf16, 2 : i32>
          }
          %64 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %65 = arith.subi %arg17, %c1_44 : index
            %66 = air.channel.put async [%async_token_52, %63]  @cascade_gp[%arg16, %65] (%results_53[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %67 = air.channel.put async [%async_token_50]  @cascade_up[%arg16, %65] (%results_51[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
            %68 = air.channel.put async [%async_token_48, %63]  @cascade_sp[%arg16, %65] (%results_49[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
            %69 = air.wait_all async [%66, %67, %68] 
            affine.yield %69 : !air.async.token
          } else {
            %65 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_104, %results_105 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_106, %results_107 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_108, %results_109 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %66 = air.channel.get async [%async_token_104]  @cascade_gp[%arg16, %arg17] (%results_105[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %67 = air.channel.get async [%async_token_106]  @cascade_up[%arg16, %arg17] (%results_107[] [] []) {id = 61 : i32} : (memref<64x1xbf16, 2 : i32>)
              %68 = air.channel.get async [%async_token_108]  @cascade_sp[%arg16, %arg17] (%results_109[] [] []) {id = 62 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_110, %results_111 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_110, %async_token_50] {
                func.call @vector_copy_32elems(%c0_i32, %results_51, %results_111) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_113 = air.execute [%67, %async_token_112] {
                func.call @maximum_up_u_bf16(%results_107, %results_51) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_114, %results_115 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_116 = air.execute [%async_token_50, %async_token_114, %async_token_113] {
                func.call @exp_up_minus_u(%results_107, %results_51, %results_115) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_117, %results_118 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_119 = air.execute [%async_token_116, %async_token_117] {
                func.call @exp_up_minus_u(%results_111, %results_51, %results_118) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_120 = air.execute [%async_token_116, %66] {
                func.call @mul_r_gp(%results_115, %results_105) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_121 = air.execute [%async_token_52, %async_token_119] {
                func.call @mul_r_gp(%results_118, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_122 = air.execute [%async_token_120, %async_token_121] {
                func.call @add_gp_g(%results_53, %results_105) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_123, %results_124 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_125 = air.execute [%async_token_123] {
                func.call @zero_fill_sp_bf16(%results_124) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_126 = air.execute [%async_token_125, %async_token_120, %68] {
                func.call @accum_sp_r_s(%results_109, %results_115, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127 = air.execute [%async_token_48, %async_token_126, %async_token_121] {
                func.call @accum_sp_r_s(%results_49, %results_118, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_128 = air.execute [%async_token_127] {
                func.call @vector_copy_32elems(%c0_i32, %results_124, %results_109) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %69 = arith.subi %arg17, %c1_44 : index
              %70 = air.channel.put async [%async_token_122]  @cascade_gp[%arg16, %69] (%results_105[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              %71 = air.channel.put async [%async_token_50, %async_token_119]  @cascade_up[%arg16, %69] (%results_51[] [] []) {id = 64 : i32} : (memref<64x1xbf16, 2 : i32>)
              %72 = air.channel.put async [%async_token_128]  @cascade_sp[%arg16, %69] (%results_109[] [] []) {id = 65 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_129 = air.execute [%70] {
                memref.dealloc %results_105 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_130 = air.execute [%async_token_116] {
                memref.dealloc %results_107 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_131 = air.execute [%72] {
                memref.dealloc %results_109 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_132 = air.execute [%async_token_119] {
                memref.dealloc %results_111 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_133 = air.execute [%async_token_126] {
                memref.dealloc %results_115 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_134 = air.execute [%async_token_127] {
                memref.dealloc %results_118 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_135 = air.execute [%async_token_128] {
                memref.dealloc %results_124 : memref<64x1xbf16, 2 : i32>
              }
              %73 = air.wait_all async [%70, %71, %72] 
              affine.yield %73 : !air.async.token
            } else {
              %async_token_104, %results_105 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_106, %results_107 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_108, %results_109 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %66 = air.channel.get async [%async_token_104]  @cascade_gp[%arg16, %arg17] (%results_105[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              %67 = air.channel.get async [%async_token_106]  @cascade_up[%arg16, %arg17] (%results_107[] [] []) {id = 67 : i32} : (memref<64x1xbf16, 2 : i32>)
              %68 = air.channel.get async [%async_token_108]  @cascade_sp[%arg16, %arg17] (%results_109[] [] []) {id = 68 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_110, %results_111 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_110, %async_token_50] {
                func.call @vector_copy_32elems(%c0_i32, %results_51, %results_111) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_113 = air.execute [%67, %async_token_112] {
                func.call @maximum_up_u_bf16(%results_107, %results_51) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_114, %results_115 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_116 = air.execute [%async_token_50, %async_token_114, %async_token_113] {
                func.call @exp_up_minus_u(%results_107, %results_51, %results_115) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_117, %results_118 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_119 = air.execute [%async_token_116, %async_token_117] {
                func.call @exp_up_minus_u(%results_111, %results_51, %results_118) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_120 = air.execute [%async_token_116, %66] {
                func.call @mul_r_gp(%results_115, %results_105) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_121 = air.execute [%async_token_52, %async_token_119] {
                func.call @mul_r_gp(%results_118, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_122 = air.execute [%async_token_120, %async_token_121] {
                func.call @add_gp_g(%results_53, %results_105) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_123, %results_124 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_125 = air.execute [%async_token_123] {
                func.call @zero_fill_sp_bf16(%results_124) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_126 = air.execute [%async_token_125, %async_token_120, %68] {
                func.call @accum_sp_r_s(%results_109, %results_115, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127 = air.execute [%async_token_48, %async_token_126, %async_token_121] {
                func.call @accum_sp_r_s(%results_49, %results_118, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_128 = air.execute [%async_token_127] {
                func.call @vector_copy_32elems(%c0_i32, %results_124, %results_109) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_129 = air.execute [%async_token_128, %async_token_122] {
                func.call @div_gp_sp(%results_109, %results_105) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %69 = air.channel.put async [%async_token_129]  @Gp2L2[%arg16, %c0_43] (%results_105[%c0_43, %c0_43, %c0_43, %c0_43] [%c8_45, %c8_45, %c8_45, %c8_45] [%c64_46, %c8_45, %c512_47, %c1_44]) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_130 = air.execute [%69] {
                memref.dealloc %results_105 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_131 = air.execute [%async_token_116] {
                memref.dealloc %results_107 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_132 = air.execute [%async_token_129] {
                memref.dealloc %results_109 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_133 = air.execute [%async_token_119] {
                memref.dealloc %results_111 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_134 = air.execute [%async_token_126] {
                memref.dealloc %results_115 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_135 = air.execute [%async_token_127] {
                memref.dealloc %results_118 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_136 = air.execute [%async_token_128] {
                memref.dealloc %results_124 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %69 : !air.async.token
            }
            affine.yield %63 : !air.async.token
          }
          %async_token_99 = air.execute {
            memref.dealloc %results_57 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_100 = air.execute [%44, %47, %49, %51] {
            memref.dealloc %results_55 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_101 = air.execute [%64, %63, %async_token_58] {
            memref.dealloc %results_53 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_102 = air.execute [%async_token_60, %64] {
            memref.dealloc %results_51 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_103 = air.execute [%64, %63, %async_token_59] {
            memref.dealloc %results_49 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_39 = air.execute [%42] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_40 = air.execute [%41] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_41 = air.execute [%40] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_42 = air.execute [%39] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
