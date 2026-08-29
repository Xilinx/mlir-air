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
  air.channel @GpOut [1]
  func.func @attention_bf16(%arg0: memref<256x64xbf16>, %arg1: memref<512x64xbf16>, %arg2: memref<512x64xbf16>, %arg3: memref<256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<256x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16>, memref<256x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c256 = arith.constant 256 : index
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
      %18 = air.wait_all async [%14, %15, %16, %17] 
      %19 = air.wait_all async 
      %20 = air.wait_all async 
      %21 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c1_0, %arg15=%c1_0) attributes {id = 2 : i32} {
        %c192_1 = arith.constant 192 : index
        %c128_2 = arith.constant 128 : index
        %c3_3 = arith.constant 3 : index
        %c64_4 = arith.constant 64 : index
        %c512_5 = arith.constant 512 : index
        %c8_6 = arith.constant 8 : index
        %c1_7 = arith.constant 1 : index
        %c2_8 = arith.constant 2 : index
        %c0_9 = arith.constant 0 : index
        %c4_10 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
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
        %c0_17 = arith.constant 0 : index
        %c64_18 = arith.constant 64 : index
        %c1_19 = arith.constant 1 : index
        %c0_20 = arith.constant 0 : index
        %c64_21 = arith.constant 64 : index
        %c1_22 = arith.constant 1 : index
        %c0_23 = arith.constant 0 : index
        %c64_24 = arith.constant 64 : index
        %c1_25 = arith.constant 1 : index
        %c0_26 = arith.constant 0 : index
        %c64_27 = arith.constant 64 : index
        %c1_28 = arith.constant 1 : index
        %c0_29 = arith.constant 0 : index
        %c64_30 = arith.constant 64 : index
        %c1_31 = arith.constant 1 : index
        %c0_32 = arith.constant 0 : index
        %c64_33 = arith.constant 64 : index
        %c1_34 = arith.constant 1 : index
        %c0_35 = arith.constant 0 : index
        %c64_36 = arith.constant 64 : index
        %c1_37 = arith.constant 1 : index
        %c0_38 = arith.constant 0 : index
        %c64_39 = arith.constant 64 : index
        %c1_40 = arith.constant 1 : index
        %async_token_41, %results_42 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_43, %results_44 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_45, %results_46 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_47, %results_48 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %22 = air.wait_all async 
        %async_token_49, %results_50 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
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
        %23 = scf.for %arg16 = %c0_9 to %c2_8 step %c1_7 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %44 = air.channel.get async [%arg17]  @VIn_0[%c0_9] (%results[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
          %45 = air.channel.put async [%44]  @V2L1_0[%c0_9, %c0_9] (%results[%c0_9, %c0_9, %c0_9, %c0_9] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_5, %c64_4, %c1_7]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %45 : !air.async.token
        }
        %24 = scf.for %arg16 = %c0_9 to %c2_8 step %c1_7 iter_args(%arg17 = %async_token_11) -> (!air.async.token) {
          %44 = air.channel.get async [%arg17]  @VIn_1[%c0_9] (%results_12[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %45 = air.channel.put async [%44]  @V2L1_1[%c0_9, %c0_9] (%results_12[%c0_9, %c0_9, %c0_9, %c0_9] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_5, %c64_4, %c1_7]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %45 : !air.async.token
        }
        %25 = scf.for %arg16 = %c0_9 to %c2_8 step %c1_7 iter_args(%arg17 = %async_token_13) -> (!air.async.token) {
          %44 = air.channel.get async [%arg17]  @VIn_2[%c0_9] (%results_14[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
          %45 = air.channel.put async [%44]  @V2L1_2[%c0_9, %c0_9] (%results_14[%c0_9, %c0_9, %c0_9, %c0_9] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_5, %c64_4, %c1_7]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %45 : !air.async.token
        }
        %26 = scf.for %arg16 = %c0_9 to %c2_8 step %c1_7 iter_args(%arg17 = %async_token_15) -> (!air.async.token) {
          %44 = air.channel.get async [%arg17]  @VIn_3[%c0_9] (%results_16[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
          %45 = air.channel.put async [%44]  @V2L1_3[%c0_9, %c0_9] (%results_16[%c0_9, %c0_9, %c0_9, %c0_9] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_5, %c64_4, %c1_7]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %45 : !air.async.token
        }
        %27 = air.channel.get async [%async_token_41]  @Gp2L2[%c0_9, %c0_9] (%results_42[%c0_38, %c0_9] [%c64_4, %c64_4] [%c64_39, %c1_40]) {id = 25 : i32} : (memref<64x64xbf16, 1 : i32>)
        %28 = air.wait_all async [%27] 
        %29 = air.channel.get async [%async_token_43]  @Gp2L2[%c1_7, %c0_9] (%results_44[%c0_32, %c0_9] [%c64_4, %c64_4] [%c64_33, %c1_34]) {id = 26 : i32} : (memref<64x64xbf16, 1 : i32>)
        %30 = air.wait_all async [%29] 
        %31 = air.channel.get async [%async_token_45]  @Gp2L2[%c2_8, %c0_9] (%results_46[%c0_26, %c0_9] [%c64_4, %c64_4] [%c64_27, %c1_28]) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
        %32 = air.wait_all async [%31] 
        %33 = air.channel.get async [%async_token_47]  @Gp2L2[%c3_3, %c0_9] (%results_48[%c0_20, %c0_9] [%c64_4, %c64_4] [%c64_21, %c1_22]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
        %34 = air.wait_all async [%33] 
        %35 = air.wait_all async [%28, %30, %32, %34] 
        %36 = air.wait_all async 
        %37 = air.channel.put async [%35]  @channel_0[%c0_9, %c0_9] (%results_42[%c0_35, %c0_9] [%c64_4, %c64_4] [%c64_36, %c1_37]) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
        %38 = air.channel.put async [%35]  @channel_0[%c1_7, %c0_9] (%results_44[%c0_29, %c0_9] [%c64_4, %c64_4] [%c64_30, %c1_31]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
        %39 = air.channel.put async [%35]  @channel_0[%c2_8, %c0_9] (%results_46[%c0_23, %c0_9] [%c64_4, %c64_4] [%c64_24, %c1_25]) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
        %40 = air.channel.put async [%35]  @channel_0[%c3_3, %c0_9] (%results_48[%c0_17, %c0_9] [%c64_4, %c64_4] [%c64_18, %c1_19]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
        %41 = air.wait_all async [%37, %38, %39, %40] 
        %42 = air.wait_all async 
        %43 = air.herd @herd_0 async [%async_token_49, %async_token_51, %async_token_53, %async_token_55, %async_token_57, %async_token_59, %async_token_61]  tile (%arg16, %arg17) in (%arg18=%c4_10, %arg19=%c4_10) args(%arg20=%results_50, %arg21=%results_52, %arg22=%results_54, %arg23=%results_56, %arg24=%results_58, %arg25=%results_60, %arg26=%results_62) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32> attributes {id = 3 : i32, link_with = "attn.o"} {
          %c512_78 = arith.constant 512 : index
          %c64_79 = arith.constant 64 : index
          %c8_80 = arith.constant 8 : index
          %c1_81 = arith.constant 1 : index
          %c0_82 = arith.constant 0 : index
          %c2_83 = arith.constant 2 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_84 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_85 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_86 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %44 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %56 = air.channel.get async  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %56 : !air.async.token
          } else {
            %56 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %57 = air.channel.get async  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %57 : !air.async.token
            } else {
              %57 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %58 = air.channel.get async  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              } else {
                %58 = air.channel.get async  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              }
              affine.yield %57 : !air.async.token
            }
            affine.yield %56 : !air.async.token
          }
          %45 = arith.index_cast %arg16 : index to i32
          %46 = arith.cmpi eq, %45, %c0_i32 : i32
          scf.if %46 {
            %async_token_87 = air.execute [%44] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %47 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %56 = air.channel.get async  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %56 : !air.async.token
          } else {
            %56 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %57 = air.channel.get async  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %57 : !air.async.token
            } else {
              %57 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %58 = air.channel.get async  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              } else {
                %58 = air.channel.get async  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              }
              affine.yield %57 : !air.async.token
            }
            affine.yield %56 : !air.async.token
          }
          %48 = arith.cmpi eq, %45, %c1_i32 : i32
          scf.if %48 {
            %async_token_87 = air.execute [%47] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %49 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %56 = air.channel.get async  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %56 : !air.async.token
          } else {
            %56 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %57 = air.channel.get async  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %57 : !air.async.token
            } else {
              %57 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %58 = air.channel.get async  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              } else {
                %58 = air.channel.get async  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              }
              affine.yield %57 : !air.async.token
            }
            affine.yield %56 : !air.async.token
          }
          %50 = arith.cmpi eq, %45, %c2_i32 : i32
          scf.if %50 {
            %async_token_87 = air.execute [%49] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %51 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %56 = air.channel.get async  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %56 : !air.async.token
          } else {
            %56 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %57 = air.channel.get async  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %57 : !air.async.token
            } else {
              %57 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %58 = air.channel.get async  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              } else {
                %58 = air.channel.get async  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              }
              affine.yield %57 : !air.async.token
            }
            affine.yield %56 : !air.async.token
          }
          %52 = arith.cmpi eq, %45, %c3_i32 : i32
          scf.if %52 {
            %async_token_87 = air.execute [%51] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %53 = air.wait_all async [%async_token_84, %async_token_85, %async_token_86] 
          %54 = scf.for %arg27 = %c0_82 to %c2_83 step %c1_81 iter_args(%arg28 = %53) -> (!air.async.token) {
            %async_token_87 = air.execute [%arg28] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %56 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %62 = air.channel.get async [%arg28]  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %62 : !air.async.token
            } else {
              %62 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %63 = air.channel.get async [%arg28]  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %63 : !air.async.token
              } else {
                %63 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %64 = air.channel.get async [%arg28]  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %64 : !air.async.token
                } else {
                  %64 = air.channel.get async [%arg28]  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %64 : !air.async.token
                }
                affine.yield %63 : !air.async.token
              }
              affine.yield %62 : !air.async.token
            }
            %57 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %62 = air.channel.get async  @V2L1_0[%arg16, %arg17] (%arg22[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %62 : !air.async.token
            } else {
              %62 = air.wait_all async 
              affine.yield %62 : !air.async.token
            }
            %58 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %62 = air.channel.get async [%57, %arg28]  @V2L1_1[%arg16, %arg17] (%arg22[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %62 : !air.async.token
            } else {
              %62 = air.wait_all async 
              affine.yield %62 : !air.async.token
            }
            %59 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %62 = air.channel.get async [%58]  @V2L1_2[%arg16, %arg17] (%arg22[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %62 : !air.async.token
            } else {
              %62 = air.wait_all async 
              affine.yield %62 : !air.async.token
            }
            %60 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %62 = air.channel.get async [%59]  @V2L1_3[%arg16, %arg17] (%arg22[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %62 : !air.async.token
            } else {
              %62 = air.wait_all async 
              affine.yield %62 : !air.async.token
            }
            %async_token_88 = air.execute [%async_token_87, %56] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_89, %results_90 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_91, %results_92 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_93 = air.execute [%async_token_91, %async_token_89, %async_token_88] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_90, %results_92) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_94 = air.execute [%async_token_93] {
              func.call @mul_r_gp(%results_92, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_95 = air.execute [%60, %async_token_94] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_96 = air.execute [%async_token_94] {
              func.call @accum_sp_r_s(%arg26, %results_92, %results_90) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_97 = air.execute [%async_token_96] {
              func.call @vector_copy_32elems(%c0_i32, %results_90, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_98 = air.execute [%async_token_97] {
              memref.dealloc %results_90 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_99 = air.execute [%async_token_96] {
              memref.dealloc %results_92 : memref<64x1xbf16, 2 : i32>
            }
            %61 = air.wait_all async [%async_token_95, %async_token_97] 
            scf.yield %61 : !air.async.token
          }
          %55 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %56 = arith.subi %arg17, %c1_81 : index
            %57 = air.channel.put async [%54]  @cascade_gp[%arg16, %56] (%arg24[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %58 = air.channel.put async [%54]  @cascade_up[%arg16, %56] (%arg25[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
            %59 = air.channel.put async [%54]  @cascade_sp[%arg16, %56] (%arg26[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
            %60 = air.wait_all async [%57, %58, %59] 
            affine.yield %60 : !air.async.token
          } else {
            %56 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_87, %results_88 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_89, %results_90 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_91, %results_92 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %57 = air.channel.get async [%async_token_87]  @cascade_gp[%arg16, %arg17] (%results_88[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %58 = air.channel.get async [%async_token_89]  @cascade_up[%arg16, %arg17] (%results_90[] [] []) {id = 61 : i32} : (memref<64x1xbf16, 2 : i32>)
              %59 = air.channel.get async [%async_token_91]  @cascade_sp[%arg16, %arg17] (%results_92[] [] []) {id = 62 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_93, %results_94 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_95 = air.execute [%async_token_93, %54] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_94) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_96 = air.execute [%async_token_95, %58] {
                func.call @maximum_up_u_bf16(%results_90, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_97, %results_98 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_99 = air.execute [%async_token_97, %async_token_96] {
                func.call @exp_up_minus_u(%results_90, %arg25, %results_98) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_100, %results_101 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_102 = air.execute [%async_token_100, %async_token_99] {
                func.call @exp_up_minus_u(%results_94, %arg25, %results_101) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_103 = air.execute [%async_token_99, %57] {
                func.call @mul_r_gp(%results_98, %results_88) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_104 = air.execute [%async_token_102] {
                func.call @mul_r_gp(%results_101, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_105 = air.execute [%async_token_104, %async_token_103] {
                func.call @add_gp_g(%arg24, %results_88) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_106, %results_107 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_108 = air.execute [%async_token_106] {
                func.call @zero_fill_sp_bf16(%results_107) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_109 = air.execute [%async_token_108, %async_token_103, %59] {
                func.call @accum_sp_r_s(%results_92, %results_98, %results_107) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_110 = air.execute [%async_token_109, %async_token_104] {
                func.call @accum_sp_r_s(%arg26, %results_101, %results_107) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_111 = air.execute [%async_token_110] {
                func.call @vector_copy_32elems(%c0_i32, %results_107, %results_92) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %60 = arith.subi %arg17, %c1_81 : index
              %61 = air.channel.put async [%async_token_105]  @cascade_gp[%arg16, %60] (%results_88[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              %62 = air.channel.put async [%async_token_102]  @cascade_up[%arg16, %60] (%arg25[] [] []) {id = 64 : i32} : (memref<64x1xbf16, 2 : i32>)
              %63 = air.channel.put async [%async_token_111]  @cascade_sp[%arg16, %60] (%results_92[] [] []) {id = 65 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_112 = air.execute [%61] {
                memref.dealloc %results_88 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_113 = air.execute [%async_token_99] {
                memref.dealloc %results_90 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_114 = air.execute [%63] {
                memref.dealloc %results_92 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_115 = air.execute [%async_token_102] {
                memref.dealloc %results_94 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_116 = air.execute [%async_token_109] {
                memref.dealloc %results_98 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_117 = air.execute [%async_token_110] {
                memref.dealloc %results_101 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_118 = air.execute [%async_token_111] {
                memref.dealloc %results_107 : memref<64x1xbf16, 2 : i32>
              }
              %64 = air.wait_all async [%61, %62, %63] 
              affine.yield %64 : !air.async.token
            } else {
              %async_token_87, %results_88 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_89, %results_90 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_91, %results_92 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %57 = air.channel.get async [%async_token_87]  @cascade_gp[%arg16, %arg17] (%results_88[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              %58 = air.channel.get async [%async_token_89]  @cascade_up[%arg16, %arg17] (%results_90[] [] []) {id = 67 : i32} : (memref<64x1xbf16, 2 : i32>)
              %59 = air.channel.get async [%async_token_91]  @cascade_sp[%arg16, %arg17] (%results_92[] [] []) {id = 68 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_93, %results_94 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_95 = air.execute [%async_token_93, %54] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_94) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_96 = air.execute [%async_token_95, %58] {
                func.call @maximum_up_u_bf16(%results_90, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_97, %results_98 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_99 = air.execute [%async_token_97, %async_token_96] {
                func.call @exp_up_minus_u(%results_90, %arg25, %results_98) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_100, %results_101 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_102 = air.execute [%async_token_100, %async_token_99] {
                func.call @exp_up_minus_u(%results_94, %arg25, %results_101) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_103 = air.execute [%async_token_99, %57] {
                func.call @mul_r_gp(%results_98, %results_88) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_104 = air.execute [%async_token_102] {
                func.call @mul_r_gp(%results_101, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_105 = air.execute [%async_token_104, %async_token_103] {
                func.call @add_gp_g(%arg24, %results_88) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_106, %results_107 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_108 = air.execute [%async_token_106] {
                func.call @zero_fill_sp_bf16(%results_107) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_109 = air.execute [%async_token_108, %async_token_103, %59] {
                func.call @accum_sp_r_s(%results_92, %results_98, %results_107) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_110 = air.execute [%async_token_109, %async_token_104] {
                func.call @accum_sp_r_s(%arg26, %results_101, %results_107) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_111 = air.execute [%async_token_110] {
                func.call @vector_copy_32elems(%c0_i32, %results_107, %results_92) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_112 = air.execute [%async_token_111, %async_token_105] {
                func.call @div_gp_sp(%results_92, %results_88) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %60 = air.channel.put async [%async_token_112]  @Gp2L2[%arg16, %c0_82] (%results_88[%c0_82, %c0_82, %c0_82, %c0_82] [%c8_80, %c8_80, %c8_80, %c8_80] [%c64_79, %c8_80, %c512_78, %c1_81]) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_113 = air.execute [%60] {
                memref.dealloc %results_88 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_114 = air.execute [%async_token_99] {
                memref.dealloc %results_90 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_115 = air.execute [%async_token_112] {
                memref.dealloc %results_92 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_116 = air.execute [%async_token_102] {
                memref.dealloc %results_94 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_117 = air.execute [%async_token_109] {
                memref.dealloc %results_98 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_118 = air.execute [%async_token_110] {
                memref.dealloc %results_101 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_119 = air.execute [%async_token_111] {
                memref.dealloc %results_107 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %60 : !air.async.token
            }
            affine.yield %54 : !air.async.token
          }
        }
        %async_token_63 = air.execute [%43] {
          memref.dealloc %results_50 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_64 = air.execute [%43] {
          memref.dealloc %results_52 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_65 = air.execute [%43] {
          memref.dealloc %results_54 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_66 = air.execute [%43] {
          memref.dealloc %results_56 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_67 = air.execute [%43] {
          memref.dealloc %results_58 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_68 = air.execute [%43] {
          memref.dealloc %results_60 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_69 = air.execute [%43] {
          memref.dealloc %results_62 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_70 = air.execute [%23] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_71 = air.execute [%24] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_72 = air.execute [%25] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_73 = air.execute [%26] {
          memref.dealloc %results_16 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_74 = air.execute [%40, %39, %38, %37] {
          memref.dealloc %results_48 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_75 = air.execute [%40, %39, %38, %37] {
          memref.dealloc %results_46 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_76 = air.execute [%40, %39, %38, %37] {
          memref.dealloc %results_44 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_77 = air.execute [%40, %39, %38, %37] {
          memref.dealloc %results_42 : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
