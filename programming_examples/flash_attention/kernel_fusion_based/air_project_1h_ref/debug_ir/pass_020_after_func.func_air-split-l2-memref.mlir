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
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c64 = arith.constant 64 : index
  %c0_0 = arith.constant 0 : index
  %c64_1 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
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
  func.func @attention_bf16(%arg0: memref<512x64xbf16>, %arg1: memref<512x64xbf16>, %arg2: memref<512x64xbf16>, %arg3: memref<512x64xbf16>) {
    %c1_2 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1_2) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<512x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c256_3 = arith.constant 256 : index
      %c24576 = arith.constant 24576 : index
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c2_4 = arith.constant 2 : index
      %c1_5 = arith.constant 1 : index
      %c64_6 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0_7 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = air.channel.put async  @QK2L1_0[%c0_7, %c0_7] (%arg8[%c0_7, %c0_7, %c0_7, %c0_7, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64_6, %c1_5]) {id = 1 : i32} : (memref<512x64xbf16>)
      %3 = air.channel.put async  @QK2L1_1[%c0_7, %c0_7] (%arg8[%c0_7, %c0_7, %c0_7, %c0_7, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64_6, %c1_5]) {id = 2 : i32} : (memref<512x64xbf16>)
      %4 = air.channel.put async  @QK2L1_2[%c0_7, %c0_7] (%arg8[%c0_7, %c0_7, %c0_7, %c0_7, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64_6, %c1_5]) {id = 3 : i32} : (memref<512x64xbf16>)
      %5 = air.channel.put async  @QK2L1_3[%c0_7, %c0_7] (%arg8[%c0_7, %c0_7, %c0_7, %c0_7, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64_6, %c1_5]) {id = 4 : i32} : (memref<512x64xbf16>)
      %6 = air.channel.put async  @QK2L1_0[%c0_7, %c0_7] (%arg9[%c0_7, %c0_7, %c0_7, %c0_7, %c0_7] [%c2_4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64_6, %c1_5]) {id = 5 : i32} : (memref<512x64xbf16>)
      %7 = air.channel.put async  @QK2L1_1[%c0_7, %c0_7] (%arg9[%c0_7, %c0_7, %c0_7, %c0_7, %c8192] [%c2_4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64_6, %c1_5]) {id = 6 : i32} : (memref<512x64xbf16>)
      %8 = air.channel.put async  @QK2L1_2[%c0_7, %c0_7] (%arg9[%c0_7, %c0_7, %c0_7, %c0_7, %c16384] [%c2_4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64_6, %c1_5]) {id = 7 : i32} : (memref<512x64xbf16>)
      %9 = air.channel.put async  @QK2L1_3[%c0_7, %c0_7] (%arg9[%c0_7, %c0_7, %c0_7, %c0_7, %c24576] [%c2_4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64_6, %c1_5]) {id = 8 : i32} : (memref<512x64xbf16>)
      %10 = air.channel.put async  @VIn_0[%c0_7] (%arg10[%c0_7, %c0_7, %c0_7] [%c2_4, %c64_6, %c64_6] [%c4096, %c64_6, %c1_5]) {id = 9 : i32} : (memref<512x64xbf16>)
      %11 = air.channel.put async  @VIn_1[%c0_7] (%arg10[%c0_7, %c0_7, %c8192] [%c2_4, %c64_6, %c64_6] [%c4096, %c64_6, %c1_5]) {id = 10 : i32} : (memref<512x64xbf16>)
      %12 = air.channel.put async  @VIn_2[%c0_7] (%arg10[%c0_7, %c0_7, %c16384] [%c2_4, %c64_6, %c64_6] [%c4096, %c64_6, %c1_5]) {id = 11 : i32} : (memref<512x64xbf16>)
      %13 = air.channel.put async  @VIn_3[%c0_7] (%arg10[%c0_7, %c0_7, %c24576] [%c2_4, %c64_6, %c64_6] [%c4096, %c64_6, %c1_5]) {id = 12 : i32} : (memref<512x64xbf16>)
      %14 = air.channel.get async  @channel_0[%c0_7, %c0_7] (%arg11[%c0_7, %1] [%c64_6, %c64_6] [%c64_6, %c1_5]) {id = 13 : i32} : (memref<512x64xbf16>)
      %15 = air.channel.get async  @channel_0[%c1_5, %c0_7] (%arg11[%c64_6, %1] [%c64_6, %c64_6] [%c64_6, %c1_5]) {id = 14 : i32} : (memref<512x64xbf16>)
      %16 = air.channel.get async  @channel_0[%c2_4, %c0_7] (%arg11[%c128, %1] [%c64_6, %c64_6] [%c64_6, %c1_5]) {id = 15 : i32} : (memref<512x64xbf16>)
      %17 = air.channel.get async  @channel_0[%c3, %c0_7] (%arg11[%c192, %1] [%c64_6, %c64_6] [%c64_6, %c1_5]) {id = 16 : i32} : (memref<512x64xbf16>)
      %18 = air.wait_all async [%14, %15, %16, %17] 
      %19 = air.wait_all async 
      %20 = air.wait_all async 
      %21 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c1_5, %arg15=%c1_5) attributes {id = 2 : i32} {
        %c192_8 = arith.constant 192 : index
        %c128_9 = arith.constant 128 : index
        %c3_10 = arith.constant 3 : index
        %c64_11 = arith.constant 64 : index
        %c512_12 = arith.constant 512 : index
        %c8_13 = arith.constant 8 : index
        %c1_14 = arith.constant 1 : index
        %c2_15 = arith.constant 2 : index
        %c0_16 = arith.constant 0 : index
        %c4_17 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
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
        %c0_24 = arith.constant 0 : index
        %c64_25 = arith.constant 64 : index
        %c1_26 = arith.constant 1 : index
        %c0_27 = arith.constant 0 : index
        %c64_28 = arith.constant 64 : index
        %c1_29 = arith.constant 1 : index
        %c0_30 = arith.constant 0 : index
        %c64_31 = arith.constant 64 : index
        %c1_32 = arith.constant 1 : index
        %c0_33 = arith.constant 0 : index
        %c64_34 = arith.constant 64 : index
        %c1_35 = arith.constant 1 : index
        %c0_36 = arith.constant 0 : index
        %c64_37 = arith.constant 64 : index
        %c1_38 = arith.constant 1 : index
        %c0_39 = arith.constant 0 : index
        %c64_40 = arith.constant 64 : index
        %c1_41 = arith.constant 1 : index
        %c0_42 = arith.constant 0 : index
        %c64_43 = arith.constant 64 : index
        %c1_44 = arith.constant 1 : index
        %c0_45 = arith.constant 0 : index
        %c64_46 = arith.constant 64 : index
        %c1_47 = arith.constant 1 : index
        %async_token_48, %results_49 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_50, %results_51 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_52, %results_53 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_54, %results_55 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %22 = air.wait_all async 
        %async_token_56, %results_57 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_58, %results_59 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_60, %results_61 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_62, %results_63 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_64, %results_65 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_66, %results_67 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %23 = scf.for %arg16 = %c0_16 to %c2_15 step %c1_14 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %44 = air.channel.get async [%arg17]  @VIn_0[%c0_16] (%results[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
          %45 = air.channel.put async [%44]  @V2L1_0[%c0_16, %c0_16] (%results[%c0_16, %c0_16, %c0_16, %c0_16] [%c8_13, %c8_13, %c8_13, %c8_13] [%c8_13, %c512_12, %c64_11, %c1_14]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %45 : !air.async.token
        }
        %24 = scf.for %arg16 = %c0_16 to %c2_15 step %c1_14 iter_args(%arg17 = %async_token_18) -> (!air.async.token) {
          %44 = air.channel.get async [%arg17]  @VIn_1[%c0_16] (%results_19[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %45 = air.channel.put async [%44]  @V2L1_1[%c0_16, %c0_16] (%results_19[%c0_16, %c0_16, %c0_16, %c0_16] [%c8_13, %c8_13, %c8_13, %c8_13] [%c8_13, %c512_12, %c64_11, %c1_14]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %45 : !air.async.token
        }
        %25 = scf.for %arg16 = %c0_16 to %c2_15 step %c1_14 iter_args(%arg17 = %async_token_20) -> (!air.async.token) {
          %44 = air.channel.get async [%arg17]  @VIn_2[%c0_16] (%results_21[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
          %45 = air.channel.put async [%44]  @V2L1_2[%c0_16, %c0_16] (%results_21[%c0_16, %c0_16, %c0_16, %c0_16] [%c8_13, %c8_13, %c8_13, %c8_13] [%c8_13, %c512_12, %c64_11, %c1_14]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %45 : !air.async.token
        }
        %26 = scf.for %arg16 = %c0_16 to %c2_15 step %c1_14 iter_args(%arg17 = %async_token_22) -> (!air.async.token) {
          %44 = air.channel.get async [%arg17]  @VIn_3[%c0_16] (%results_23[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
          %45 = air.channel.put async [%44]  @V2L1_3[%c0_16, %c0_16] (%results_23[%c0_16, %c0_16, %c0_16, %c0_16] [%c8_13, %c8_13, %c8_13, %c8_13] [%c8_13, %c512_12, %c64_11, %c1_14]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %45 : !air.async.token
        }
        %27 = air.channel.get async [%async_token_48]  @Gp2L2[%c0_16, %c0_16] (%results_49[%c0_45, %c0_16] [%c64_11, %c64_11] [%c64_46, %c1_47]) {id = 25 : i32} : (memref<64x64xbf16, 1 : i32>)
        %28 = air.wait_all async [%27] 
        %29 = air.channel.get async [%async_token_50]  @Gp2L2[%c1_14, %c0_16] (%results_51[%c0_39, %c0_16] [%c64_11, %c64_11] [%c64_40, %c1_41]) {id = 26 : i32} : (memref<64x64xbf16, 1 : i32>)
        %30 = air.wait_all async [%29] 
        %31 = air.channel.get async [%async_token_52]  @Gp2L2[%c2_15, %c0_16] (%results_53[%c0_33, %c0_16] [%c64_11, %c64_11] [%c64_34, %c1_35]) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
        %32 = air.wait_all async [%31] 
        %33 = air.channel.get async [%async_token_54]  @Gp2L2[%c3_10, %c0_16] (%results_55[%c0_27, %c0_16] [%c64_11, %c64_11] [%c64_28, %c1_29]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
        %34 = air.wait_all async [%33] 
        %35 = air.wait_all async [%28, %30, %32, %34] 
        %36 = air.wait_all async 
        %37 = air.channel.put async [%35]  @channel_0[%c0_16, %c0_16] (%results_49[%c0_42, %c0_16] [%c64_11, %c64_11] [%c64_43, %c1_44]) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
        %38 = air.channel.put async [%35]  @channel_0[%c1_14, %c0_16] (%results_51[%c0_36, %c0_16] [%c64_11, %c64_11] [%c64_37, %c1_38]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
        %39 = air.channel.put async [%35]  @channel_0[%c2_15, %c0_16] (%results_53[%c0_30, %c0_16] [%c64_11, %c64_11] [%c64_31, %c1_32]) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
        %40 = air.channel.put async [%35]  @channel_0[%c3_10, %c0_16] (%results_55[%c0_24, %c0_16] [%c64_11, %c64_11] [%c64_25, %c1_26]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
        %41 = air.wait_all async [%37, %38, %39, %40] 
        %42 = air.wait_all async 
        %43 = air.herd @herd_0 async [%async_token_56, %async_token_58, %async_token_60, %async_token_62, %async_token_64, %async_token_66, %async_token_68]  tile (%arg16, %arg17) in (%arg18=%c4_17, %arg19=%c4_17) args(%arg20=%results_57, %arg21=%results_59, %arg22=%results_61, %arg23=%results_63, %arg24=%results_65, %arg25=%results_67, %arg26=%results_69) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32> attributes {id = 3 : i32, link_with = "attn.o"} {
          %c512_85 = arith.constant 512 : index
          %c64_86 = arith.constant 64 : index
          %c8_87 = arith.constant 8 : index
          %c1_88 = arith.constant 1 : index
          %c0_89 = arith.constant 0 : index
          %c2_90 = arith.constant 2 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_91 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_92 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_93 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %44 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %56 = air.channel.get async  @QK2L1_0[%arg16, %c0_89] (%arg21[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %56 : !air.async.token
          } else {
            %56 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %57 = air.channel.get async  @QK2L1_1[%arg16, %c0_89] (%arg21[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %57 : !air.async.token
            } else {
              %57 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %58 = air.channel.get async  @QK2L1_2[%arg16, %c0_89] (%arg21[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              } else {
                %58 = air.channel.get async  @QK2L1_3[%arg16, %c0_89] (%arg21[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              }
              affine.yield %57 : !air.async.token
            }
            affine.yield %56 : !air.async.token
          }
          %45 = arith.index_cast %arg16 : index to i32
          %46 = arith.cmpi eq, %45, %c0_i32 : i32
          scf.if %46 {
            %async_token_94 = air.execute [%44] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %47 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %56 = air.channel.get async  @QK2L1_0[%arg16, %c0_89] (%arg21[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %56 : !air.async.token
          } else {
            %56 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %57 = air.channel.get async  @QK2L1_1[%arg16, %c0_89] (%arg21[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %57 : !air.async.token
            } else {
              %57 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %58 = air.channel.get async  @QK2L1_2[%arg16, %c0_89] (%arg21[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              } else {
                %58 = air.channel.get async  @QK2L1_3[%arg16, %c0_89] (%arg21[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              }
              affine.yield %57 : !air.async.token
            }
            affine.yield %56 : !air.async.token
          }
          %48 = arith.cmpi eq, %45, %c1_i32 : i32
          scf.if %48 {
            %async_token_94 = air.execute [%47] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %49 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %56 = air.channel.get async  @QK2L1_0[%arg16, %c0_89] (%arg21[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %56 : !air.async.token
          } else {
            %56 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %57 = air.channel.get async  @QK2L1_1[%arg16, %c0_89] (%arg21[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %57 : !air.async.token
            } else {
              %57 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %58 = air.channel.get async  @QK2L1_2[%arg16, %c0_89] (%arg21[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              } else {
                %58 = air.channel.get async  @QK2L1_3[%arg16, %c0_89] (%arg21[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              }
              affine.yield %57 : !air.async.token
            }
            affine.yield %56 : !air.async.token
          }
          %50 = arith.cmpi eq, %45, %c2_i32 : i32
          scf.if %50 {
            %async_token_94 = air.execute [%49] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %51 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %56 = air.channel.get async  @QK2L1_0[%arg16, %c0_89] (%arg21[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %56 : !air.async.token
          } else {
            %56 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %57 = air.channel.get async  @QK2L1_1[%arg16, %c0_89] (%arg21[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %57 : !air.async.token
            } else {
              %57 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %58 = air.channel.get async  @QK2L1_2[%arg16, %c0_89] (%arg21[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              } else {
                %58 = air.channel.get async  @QK2L1_3[%arg16, %c0_89] (%arg21[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %58 : !air.async.token
              }
              affine.yield %57 : !air.async.token
            }
            affine.yield %56 : !air.async.token
          }
          %52 = arith.cmpi eq, %45, %c3_i32 : i32
          scf.if %52 {
            %async_token_94 = air.execute [%51] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %53 = air.wait_all async [%async_token_91, %async_token_92, %async_token_93] 
          %54 = scf.for %arg27 = %c0_89 to %c2_90 step %c1_88 iter_args(%arg28 = %53) -> (!air.async.token) {
            %async_token_94 = air.execute [%arg28] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %56 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %62 = air.channel.get async [%arg28]  @QK2L1_0[%arg16, %c0_89] (%arg21[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %62 : !air.async.token
            } else {
              %62 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %63 = air.channel.get async [%arg28]  @QK2L1_1[%arg16, %c0_89] (%arg21[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %63 : !air.async.token
              } else {
                %63 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %64 = air.channel.get async [%arg28]  @QK2L1_2[%arg16, %c0_89] (%arg21[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %64 : !air.async.token
                } else {
                  %64 = air.channel.get async [%arg28]  @QK2L1_3[%arg16, %c0_89] (%arg21[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
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
            %async_token_95 = air.execute [%async_token_94, %56] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_96, %results_97 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_98, %results_99 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_100 = air.execute [%async_token_98, %async_token_96, %async_token_95] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_97, %results_99) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_101 = air.execute [%async_token_100] {
              func.call @mul_r_gp(%results_99, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_102 = air.execute [%60, %async_token_101] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_103 = air.execute [%async_token_101] {
              func.call @accum_sp_r_s(%arg26, %results_99, %results_97) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_104 = air.execute [%async_token_103] {
              func.call @vector_copy_32elems(%c0_i32, %results_97, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_105 = air.execute [%async_token_104] {
              memref.dealloc %results_97 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_106 = air.execute [%async_token_103] {
              memref.dealloc %results_99 : memref<64x1xbf16, 2 : i32>
            }
            %61 = air.wait_all async [%async_token_102, %async_token_104] 
            scf.yield %61 : !air.async.token
          }
          %55 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %56 = arith.subi %arg17, %c1_88 : index
            %57 = air.channel.put async [%54]  @cascade_gp[%arg16, %56] (%arg24[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %58 = air.channel.put async [%54]  @cascade_up[%arg16, %56] (%arg25[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
            %59 = air.channel.put async [%54]  @cascade_sp[%arg16, %56] (%arg26[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
            %60 = air.wait_all async [%57, %58, %59] 
            affine.yield %60 : !air.async.token
          } else {
            %56 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_94, %results_95 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_96, %results_97 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_98, %results_99 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %57 = air.channel.get async [%async_token_94]  @cascade_gp[%arg16, %arg17] (%results_95[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %58 = air.channel.get async [%async_token_96]  @cascade_up[%arg16, %arg17] (%results_97[] [] []) {id = 61 : i32} : (memref<64x1xbf16, 2 : i32>)
              %59 = air.channel.get async [%async_token_98]  @cascade_sp[%arg16, %arg17] (%results_99[] [] []) {id = 62 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_100, %results_101 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_102 = air.execute [%async_token_100, %54] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_101) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_103 = air.execute [%async_token_102, %58] {
                func.call @maximum_up_u_bf16(%results_97, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_104, %results_105 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_106 = air.execute [%async_token_104, %async_token_103] {
                func.call @exp_up_minus_u(%results_97, %arg25, %results_105) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_107, %results_108 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_109 = air.execute [%async_token_107, %async_token_106] {
                func.call @exp_up_minus_u(%results_101, %arg25, %results_108) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_110 = air.execute [%async_token_106, %57] {
                func.call @mul_r_gp(%results_105, %results_95) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_111 = air.execute [%async_token_109] {
                func.call @mul_r_gp(%results_108, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_112 = air.execute [%async_token_111, %async_token_110] {
                func.call @add_gp_g(%arg24, %results_95) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_113, %results_114 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_115 = air.execute [%async_token_113] {
                func.call @zero_fill_sp_bf16(%results_114) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_116 = air.execute [%async_token_115, %async_token_110, %59] {
                func.call @accum_sp_r_s(%results_99, %results_105, %results_114) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_117 = air.execute [%async_token_116, %async_token_111] {
                func.call @accum_sp_r_s(%arg26, %results_108, %results_114) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_118 = air.execute [%async_token_117] {
                func.call @vector_copy_32elems(%c0_i32, %results_114, %results_99) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %60 = arith.subi %arg17, %c1_88 : index
              %61 = air.channel.put async [%async_token_112]  @cascade_gp[%arg16, %60] (%results_95[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              %62 = air.channel.put async [%async_token_109]  @cascade_up[%arg16, %60] (%arg25[] [] []) {id = 64 : i32} : (memref<64x1xbf16, 2 : i32>)
              %63 = air.channel.put async [%async_token_118]  @cascade_sp[%arg16, %60] (%results_99[] [] []) {id = 65 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_119 = air.execute [%61] {
                memref.dealloc %results_95 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_120 = air.execute [%async_token_106] {
                memref.dealloc %results_97 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_121 = air.execute [%63] {
                memref.dealloc %results_99 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_122 = air.execute [%async_token_109] {
                memref.dealloc %results_101 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_123 = air.execute [%async_token_116] {
                memref.dealloc %results_105 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_124 = air.execute [%async_token_117] {
                memref.dealloc %results_108 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_125 = air.execute [%async_token_118] {
                memref.dealloc %results_114 : memref<64x1xbf16, 2 : i32>
              }
              %64 = air.wait_all async [%61, %62, %63] 
              affine.yield %64 : !air.async.token
            } else {
              %async_token_94, %results_95 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_96, %results_97 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_98, %results_99 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %57 = air.channel.get async [%async_token_94]  @cascade_gp[%arg16, %arg17] (%results_95[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              %58 = air.channel.get async [%async_token_96]  @cascade_up[%arg16, %arg17] (%results_97[] [] []) {id = 67 : i32} : (memref<64x1xbf16, 2 : i32>)
              %59 = air.channel.get async [%async_token_98]  @cascade_sp[%arg16, %arg17] (%results_99[] [] []) {id = 68 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_100, %results_101 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_102 = air.execute [%async_token_100, %54] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_101) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_103 = air.execute [%async_token_102, %58] {
                func.call @maximum_up_u_bf16(%results_97, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_104, %results_105 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_106 = air.execute [%async_token_104, %async_token_103] {
                func.call @exp_up_minus_u(%results_97, %arg25, %results_105) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_107, %results_108 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_109 = air.execute [%async_token_107, %async_token_106] {
                func.call @exp_up_minus_u(%results_101, %arg25, %results_108) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_110 = air.execute [%async_token_106, %57] {
                func.call @mul_r_gp(%results_105, %results_95) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_111 = air.execute [%async_token_109] {
                func.call @mul_r_gp(%results_108, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_112 = air.execute [%async_token_111, %async_token_110] {
                func.call @add_gp_g(%arg24, %results_95) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_113, %results_114 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_115 = air.execute [%async_token_113] {
                func.call @zero_fill_sp_bf16(%results_114) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_116 = air.execute [%async_token_115, %async_token_110, %59] {
                func.call @accum_sp_r_s(%results_99, %results_105, %results_114) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_117 = air.execute [%async_token_116, %async_token_111] {
                func.call @accum_sp_r_s(%arg26, %results_108, %results_114) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_118 = air.execute [%async_token_117] {
                func.call @vector_copy_32elems(%c0_i32, %results_114, %results_99) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_119 = air.execute [%async_token_118, %async_token_112] {
                func.call @div_gp_sp(%results_99, %results_95) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %60 = air.channel.put async [%async_token_119]  @Gp2L2[%arg16, %c0_89] (%results_95[%c0_89, %c0_89, %c0_89, %c0_89] [%c8_87, %c8_87, %c8_87, %c8_87] [%c64_86, %c8_87, %c512_85, %c1_88]) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_120 = air.execute [%60] {
                memref.dealloc %results_95 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_121 = air.execute [%async_token_106] {
                memref.dealloc %results_97 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_122 = air.execute [%async_token_119] {
                memref.dealloc %results_99 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_123 = air.execute [%async_token_109] {
                memref.dealloc %results_101 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_124 = air.execute [%async_token_116] {
                memref.dealloc %results_105 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_125 = air.execute [%async_token_117] {
                memref.dealloc %results_108 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_126 = air.execute [%async_token_118] {
                memref.dealloc %results_114 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %60 : !air.async.token
            }
            affine.yield %54 : !air.async.token
          }
        }
        %async_token_70 = air.execute [%43] {
          memref.dealloc %results_57 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_71 = air.execute [%43] {
          memref.dealloc %results_59 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_72 = air.execute [%43] {
          memref.dealloc %results_61 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_73 = air.execute [%43] {
          memref.dealloc %results_63 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_74 = air.execute [%43] {
          memref.dealloc %results_65 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_75 = air.execute [%43] {
          memref.dealloc %results_67 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_76 = air.execute [%43] {
          memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_77 = air.execute [%23] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_78 = air.execute [%24] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_79 = air.execute [%25] {
          memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_80 = air.execute [%26] {
          memref.dealloc %results_23 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_81 = air.execute [%40, %39, %38, %37] {
          memref.dealloc %results_55 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_82 = air.execute [%40, %39, %38, %37] {
          memref.dealloc %results_53 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_83 = air.execute [%40, %39, %38, %37] {
          memref.dealloc %results_51 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_84 = air.execute [%40, %39, %38, %37] {
          memref.dealloc %results_49 : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
