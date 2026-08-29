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
        %c8_3 = arith.constant 8 : index
        %c1_4 = arith.constant 1 : index
        %c2_5 = arith.constant 2 : index
        %c0_6 = arith.constant 0 : index
        %c4_7 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
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
        %19 = air.channel.get async [%async_token_14]  @VIn_0[%c0_6] (%results_15[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
        %20 = air.channel.put async [%19]  @V2L1_0[%c0_6, %c0_6] (%results_15[%c0_6, %c0_6, %c0_6] [%c8_3, %c64_2, %c8_3] [%c8_3, %c64_2, %c1_4]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_16 = air.execute [%20] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17, %results_18 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %21 = air.channel.get async [%async_token_17]  @VIn_0[%c0_6] (%results_18[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
        %22 = air.channel.put async [%21]  @V2L1_0[%c0_6, %c0_6] (%results_18[%c0_6, %c0_6, %c0_6] [%c8_3, %c64_2, %c8_3] [%c8_3, %c64_2, %c1_4]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_19 = air.execute [%22] {
          memref.dealloc %results_18 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %23 = air.channel.get async [%async_token_20]  @VIn_1[%c0_6] (%results_21[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
        %24 = air.channel.put async [%23]  @V2L1_1[%c0_6, %c0_6] (%results_21[%c0_6, %c0_6, %c0_6] [%c8_3, %c64_2, %c8_3] [%c8_3, %c64_2, %c1_4]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_22 = air.execute [%24] {
          memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_23, %results_24 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %25 = air.channel.get async [%async_token_23]  @VIn_1[%c0_6] (%results_24[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
        %26 = air.channel.put async [%25]  @V2L1_1[%c0_6, %c0_6] (%results_24[%c0_6, %c0_6, %c0_6] [%c8_3, %c64_2, %c8_3] [%c8_3, %c64_2, %c1_4]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_25 = air.execute [%26] {
          memref.dealloc %results_24 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_26, %results_27 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %27 = air.channel.get async [%async_token_26]  @VIn_2[%c0_6] (%results_27[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
        %28 = air.channel.put async [%27]  @V2L1_2[%c0_6, %c0_6] (%results_27[%c0_6, %c0_6, %c0_6] [%c8_3, %c64_2, %c8_3] [%c8_3, %c64_2, %c1_4]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_28 = air.execute [%28] {
          memref.dealloc %results_27 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %29 = air.channel.get async [%async_token_29]  @VIn_2[%c0_6] (%results_30[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
        %30 = air.channel.put async [%29]  @V2L1_2[%c0_6, %c0_6] (%results_30[%c0_6, %c0_6, %c0_6] [%c8_3, %c64_2, %c8_3] [%c8_3, %c64_2, %c1_4]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_31 = air.execute [%30] {
          memref.dealloc %results_30 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_32, %results_33 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %31 = air.channel.get async [%async_token_32]  @VIn_3[%c0_6] (%results_33[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
        %32 = air.channel.put async [%31]  @V2L1_3[%c0_6, %c0_6] (%results_33[%c0_6, %c0_6, %c0_6] [%c8_3, %c64_2, %c8_3] [%c8_3, %c64_2, %c1_4]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_34 = air.execute [%32] {
          memref.dealloc %results_33 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_35, %results_36 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %33 = air.channel.get async [%async_token_35]  @VIn_3[%c0_6] (%results_36[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
        %34 = air.channel.put async [%33]  @V2L1_3[%c0_6, %c0_6] (%results_36[%c0_6, %c0_6, %c0_6] [%c8_3, %c64_2, %c8_3] [%c8_3, %c64_2, %c1_4]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
        %async_token_37 = air.execute [%34] {
          memref.dealloc %results_36 : memref<64x64xbf16, 1 : i32>
        }
        %35 = air.channel.get async [%async_token]  @Gp2L2[%c0_6, %c0_6] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %36 = air.channel.get async [%async_token_8]  @Gp2L2[%c1_4, %c0_6] (%results_9[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %37 = air.channel.get async [%async_token_10]  @Gp2L2[%c2_5, %c0_6] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %38 = air.channel.get async [%async_token_12]  @Gp2L2[%c3_1, %c0_6] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %39 = air.channel.put async [%35]  @channel_0[%c0_6, %c0_6] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %40 = air.channel.put async [%36]  @channel_0[%c1_4, %c0_6] (%results_9[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %41 = air.channel.put async [%37]  @channel_0[%c2_5, %c0_6] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %42 = air.channel.put async [%38]  @channel_0[%c3_1, %c0_6] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %43 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_7, %arg19=%c4_7) attributes {id = 3 : i32, link_with = "attn.o"} {
          %c64_42 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c0_43 = arith.constant 0 : index
          %c1_44 = arith.constant 1 : index
          %c8_45 = arith.constant 8 : index
          %c512_46 = arith.constant 512 : index
          %async_token_47, %results_48 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_49, %results_50 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_51, %results_52 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_53, %results_54 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_55, %results_56 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_57 = air.execute [%async_token_51] {
            func.call @zero_fill_gp_bf16(%results_52) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_58 = air.execute [%async_token_47] {
            func.call @zero_fill_sp_bf16(%results_48) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_59 = air.execute [%async_token_49] {
            func.call @neg_inf_fill_up_bf16(%results_50) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %44 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_53]  @QK2L1_0[%arg16, %arg17] (%results_54[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_53]  @QK2L1_1[%arg16, %arg17] (%results_54[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_2[%arg16, %arg17] (%results_54[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_3[%arg16, %arg17] (%results_54[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %45 = arith.index_cast %arg16 : index to i32
          %46 = arith.cmpi eq, %45, %c0_i32 : i32
          scf.if %46 {
            %async_token_103 = air.execute [%async_token_53, %async_token_55, %44] {
              func.call @copy_tile(%results_54, %results_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %47 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_53]  @QK2L1_0[%arg16, %arg17] (%results_54[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_53]  @QK2L1_1[%arg16, %arg17] (%results_54[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_2[%arg16, %arg17] (%results_54[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_3[%arg16, %arg17] (%results_54[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %48 = arith.cmpi eq, %45, %c1_i32 : i32
          scf.if %48 {
            %async_token_103 = air.execute [%async_token_53, %async_token_55, %47] {
              func.call @copy_tile(%results_54, %results_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %49 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_53]  @QK2L1_0[%arg16, %arg17] (%results_54[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_53]  @QK2L1_1[%arg16, %arg17] (%results_54[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_2[%arg16, %arg17] (%results_54[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_3[%arg16, %arg17] (%results_54[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %50 = arith.cmpi eq, %45, %c2_i32 : i32
          scf.if %50 {
            %async_token_103 = air.execute [%async_token_53, %async_token_55, %49] {
              func.call @copy_tile(%results_54, %results_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %51 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_53]  @QK2L1_0[%arg16, %arg17] (%results_54[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_53]  @QK2L1_1[%arg16, %arg17] (%results_54[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_2[%arg16, %arg17] (%results_54[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_3[%arg16, %arg17] (%results_54[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %52 = arith.cmpi eq, %45, %c3_i32 : i32
          scf.if %52 {
            %async_token_103 = air.execute [%async_token_53, %async_token_55, %51] {
              func.call @copy_tile(%results_54, %results_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %async_token_60, %results_61 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_62, %results_63 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_64 = air.execute [%async_token_62] {
            %collapse_shape = memref.collapse_shape %results_63 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          %53 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_53]  @QK2L1_0[%arg16, %arg17] (%results_54[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_53]  @QK2L1_1[%arg16, %arg17] (%results_54[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_2[%arg16, %arg17] (%results_54[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_3[%arg16, %arg17] (%results_54[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %54 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_60]  @V2L1_0[%arg16, %arg17] (%results_61[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %55 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_60, %54]  @V2L1_1[%arg16, %arg17] (%results_61[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %56 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_60, %55]  @V2L1_2[%arg16, %arg17] (%results_61[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %57 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_60, %56]  @V2L1_3[%arg16, %arg17] (%results_61[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %async_token_65 = air.execute [%53, %async_token_64, %async_token_55, %async_token_53] {
            %collapse_shape = memref.collapse_shape %results_63 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%results_56, %results_54, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %async_token_66, %results_67 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_70 = air.execute [%async_token_49, %async_token_62, %async_token_68, %async_token_66, %async_token_65] {
            %collapse_shape = memref.collapse_shape %results_63 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %results_50, %results_67, %results_69) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_71 = air.execute [%async_token_51, %async_token_70] {
            func.call @mul_r_gp(%results_69, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_72 = air.execute [%async_token_71, %57, %async_token_60, %async_token_62] {
            %collapse_shape = memref.collapse_shape %results_63 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %results_61, %results_52) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_73 = air.execute [%async_token_47, %async_token_71] {
            func.call @accum_sp_r_s(%results_48, %results_69, %results_67) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_74 = air.execute [%async_token_73] {
            func.call @vector_copy_32elems(%c0_i32, %results_67, %results_48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_75 = air.execute [%async_token_74] {
            memref.dealloc %results_67 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_76 = air.execute [%async_token_73] {
            memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_77 = air.execute [%async_token_70, %async_token_72] {
            memref.dealloc %results_63 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_78 = air.execute [%54, %55, %56, %async_token_72] {
            memref.dealloc %results_61 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_79, %results_80 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_81, %results_82 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_83 = air.execute [%async_token_81] {
            %collapse_shape = memref.collapse_shape %results_82 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_53]  @QK2L1_0[%arg16, %arg17] (%results_54[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async [%async_token_53]  @QK2L1_1[%arg16, %arg17] (%results_54[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_2[%arg16, %arg17] (%results_54[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = air.channel.get async [%async_token_53]  @QK2L1_3[%arg16, %arg17] (%results_54[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            affine.yield %65 : !air.async.token
          }
          %59 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_79]  @V2L1_0[%arg16, %arg17] (%results_80[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %60 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_79, %59]  @V2L1_1[%arg16, %arg17] (%results_80[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %61 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_79, %60]  @V2L1_2[%arg16, %arg17] (%results_80[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %62 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %65 = air.channel.get async [%async_token_79, %61]  @V2L1_3[%arg16, %arg17] (%results_80[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %65 : !air.async.token
          } else {
            %65 = air.wait_all async 
            affine.yield %65 : !air.async.token
          }
          %async_token_84 = air.execute [%58, %async_token_83, %async_token_55, %async_token_53] {
            %collapse_shape = memref.collapse_shape %results_82 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%results_56, %results_54, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %async_token_85, %results_86 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_87, %results_88 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_89 = air.execute [%async_token_49, %async_token_81, %async_token_87, %async_token_85, %async_token_84] {
            %collapse_shape = memref.collapse_shape %results_82 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %results_50, %results_86, %results_88) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_90 = air.execute [%async_token_51, %async_token_89] {
            func.call @mul_r_gp(%results_88, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_91 = air.execute [%async_token_90, %62, %async_token_79, %async_token_81] {
            %collapse_shape = memref.collapse_shape %results_82 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %results_80, %results_52) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_92 = air.execute [%async_token_47, %async_token_90] {
            func.call @accum_sp_r_s(%results_48, %results_88, %results_86) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_93 = air.execute [%async_token_92] {
            func.call @vector_copy_32elems(%c0_i32, %results_86, %results_48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_94 = air.execute [%async_token_93] {
            memref.dealloc %results_86 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_95 = air.execute [%async_token_92] {
            memref.dealloc %results_88 : memref<64x1xbf16, 2 : i32>
          }
          %63 = air.wait_all async [%async_token_91, %async_token_93] 
          %async_token_96 = air.execute [%async_token_89, %async_token_91] {
            memref.dealloc %results_82 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_97 = air.execute [%59, %60, %61, %async_token_91] {
            memref.dealloc %results_80 : memref<64x64xbf16, 2 : i32>
          }
          %64 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %65 = arith.subi %arg17, %c1_44 : index
            %66 = air.channel.put async [%async_token_51, %63]  @cascade_gp[%arg16, %65] (%results_52[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %67 = air.channel.put async [%async_token_49]  @cascade_up[%arg16, %65] (%results_50[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
            %68 = air.channel.put async [%async_token_47, %63]  @cascade_sp[%arg16, %65] (%results_48[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
            %69 = air.wait_all async [%66, %67, %68] 
            affine.yield %69 : !air.async.token
          } else {
            %65 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_103, %results_104 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_105, %results_106 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_107, %results_108 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %66 = air.channel.get async [%async_token_103]  @cascade_gp[%arg16, %arg17] (%results_104[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %67 = air.channel.get async [%async_token_105]  @cascade_up[%arg16, %arg17] (%results_106[] [] []) {id = 61 : i32} : (memref<64x1xbf16, 2 : i32>)
              %68 = air.channel.get async [%async_token_107]  @cascade_sp[%arg16, %arg17] (%results_108[] [] []) {id = 62 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_109, %results_110 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_111 = air.execute [%async_token_109, %async_token_49] {
                func.call @vector_copy_32elems(%c0_i32, %results_50, %results_110) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_112 = air.execute [%67, %async_token_111] {
                func.call @maximum_up_u_bf16(%results_106, %results_50) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_113, %results_114 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_115 = air.execute [%async_token_49, %async_token_113, %async_token_112] {
                func.call @exp_up_minus_u(%results_106, %results_50, %results_114) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_116, %results_117 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_118 = air.execute [%async_token_115, %async_token_116] {
                func.call @exp_up_minus_u(%results_110, %results_50, %results_117) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_119 = air.execute [%async_token_115, %66] {
                func.call @mul_r_gp(%results_114, %results_104) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_120 = air.execute [%async_token_51, %async_token_118] {
                func.call @mul_r_gp(%results_117, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_121 = air.execute [%async_token_119, %async_token_120] {
                func.call @add_gp_g(%results_52, %results_104) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_122, %results_123 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_124 = air.execute [%async_token_122] {
                func.call @zero_fill_sp_bf16(%results_123) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_125 = air.execute [%async_token_124, %async_token_119, %68] {
                func.call @accum_sp_r_s(%results_108, %results_114, %results_123) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_126 = air.execute [%async_token_47, %async_token_125, %async_token_120] {
                func.call @accum_sp_r_s(%results_48, %results_117, %results_123) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127 = air.execute [%async_token_126] {
                func.call @vector_copy_32elems(%c0_i32, %results_123, %results_108) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %69 = arith.subi %arg17, %c1_44 : index
              %70 = air.channel.put async [%async_token_121]  @cascade_gp[%arg16, %69] (%results_104[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              %71 = air.channel.put async [%async_token_49, %async_token_118]  @cascade_up[%arg16, %69] (%results_50[] [] []) {id = 64 : i32} : (memref<64x1xbf16, 2 : i32>)
              %72 = air.channel.put async [%async_token_127]  @cascade_sp[%arg16, %69] (%results_108[] [] []) {id = 65 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_128 = air.execute [%70] {
                memref.dealloc %results_104 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_129 = air.execute [%async_token_115] {
                memref.dealloc %results_106 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_130 = air.execute [%72] {
                memref.dealloc %results_108 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_131 = air.execute [%async_token_118] {
                memref.dealloc %results_110 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_132 = air.execute [%async_token_125] {
                memref.dealloc %results_114 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_133 = air.execute [%async_token_126] {
                memref.dealloc %results_117 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_134 = air.execute [%async_token_127] {
                memref.dealloc %results_123 : memref<64x1xbf16, 2 : i32>
              }
              %73 = air.wait_all async [%70, %71, %72] 
              affine.yield %73 : !air.async.token
            } else {
              %async_token_103, %results_104 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_105, %results_106 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_107, %results_108 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %66 = air.channel.get async [%async_token_103]  @cascade_gp[%arg16, %arg17] (%results_104[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              %67 = air.channel.get async [%async_token_105]  @cascade_up[%arg16, %arg17] (%results_106[] [] []) {id = 67 : i32} : (memref<64x1xbf16, 2 : i32>)
              %68 = air.channel.get async [%async_token_107]  @cascade_sp[%arg16, %arg17] (%results_108[] [] []) {id = 68 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_109, %results_110 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_111 = air.execute [%async_token_109, %async_token_49] {
                func.call @vector_copy_32elems(%c0_i32, %results_50, %results_110) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_112 = air.execute [%67, %async_token_111] {
                func.call @maximum_up_u_bf16(%results_106, %results_50) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_113, %results_114 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_115 = air.execute [%async_token_49, %async_token_113, %async_token_112] {
                func.call @exp_up_minus_u(%results_106, %results_50, %results_114) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_116, %results_117 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_118 = air.execute [%async_token_115, %async_token_116] {
                func.call @exp_up_minus_u(%results_110, %results_50, %results_117) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_119 = air.execute [%async_token_115, %66] {
                func.call @mul_r_gp(%results_114, %results_104) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_120 = air.execute [%async_token_51, %async_token_118] {
                func.call @mul_r_gp(%results_117, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_121 = air.execute [%async_token_119, %async_token_120] {
                func.call @add_gp_g(%results_52, %results_104) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_122, %results_123 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_124 = air.execute [%async_token_122] {
                func.call @zero_fill_sp_bf16(%results_123) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_125 = air.execute [%async_token_124, %async_token_119, %68] {
                func.call @accum_sp_r_s(%results_108, %results_114, %results_123) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_126 = air.execute [%async_token_47, %async_token_125, %async_token_120] {
                func.call @accum_sp_r_s(%results_48, %results_117, %results_123) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127 = air.execute [%async_token_126] {
                func.call @vector_copy_32elems(%c0_i32, %results_123, %results_108) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_128 = air.execute [%async_token_127, %async_token_121] {
                func.call @div_gp_sp(%results_108, %results_104) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %69 = air.channel.put async [%async_token_128]  @Gp2L2[%arg16, %c0_43] (%results_104[%c0_43, %c0_43, %c0_43] [%c64_42, %c8_45, %c8_45] [%c8_45, %c512_46, %c1_44]) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_129 = air.execute [%69] {
                memref.dealloc %results_104 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_130 = air.execute [%async_token_115] {
                memref.dealloc %results_106 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_131 = air.execute [%async_token_128] {
                memref.dealloc %results_108 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_132 = air.execute [%async_token_118] {
                memref.dealloc %results_110 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_133 = air.execute [%async_token_125] {
                memref.dealloc %results_114 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_134 = air.execute [%async_token_126] {
                memref.dealloc %results_117 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_135 = air.execute [%async_token_127] {
                memref.dealloc %results_123 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %69 : !air.async.token
            }
            affine.yield %63 : !air.async.token
          }
          %async_token_98 = air.execute {
            memref.dealloc %results_56 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_99 = air.execute [%44, %47, %49, %51] {
            memref.dealloc %results_54 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_100 = air.execute [%64, %63, %async_token_57] {
            memref.dealloc %results_52 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_101 = air.execute [%async_token_59, %64] {
            memref.dealloc %results_50 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_102 = air.execute [%64, %63, %async_token_58] {
            memref.dealloc %results_48 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_38 = air.execute [%42] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_39 = air.execute [%41] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_40 = air.execute [%40] {
          memref.dealloc %results_9 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_41 = air.execute [%39] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        air.wait_all [%async_token_16, %async_token_19, %async_token_22, %async_token_25, %async_token_28, %async_token_31, %async_token_34, %async_token_37, %43, %async_token_38, %async_token_39, %async_token_40, %async_token_41]  {air.segment_end}
      }
    }
    return
  }
}
