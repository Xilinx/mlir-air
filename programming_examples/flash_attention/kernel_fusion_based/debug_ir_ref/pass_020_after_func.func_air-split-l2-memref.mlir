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
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<2x256x64xbf16>, %arg1: memref<2x256x64xbf16>, %arg2: memref<2x256x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c1_2 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1_2, %arg7=%c1_2) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c16384 = arith.constant 16384 : index
      %c4096 = arith.constant 4096 : index
      %c1_3 = arith.constant 1 : index
      %c64_4 = arith.constant 64 : index
      %c256_5 = arith.constant 256 : index
      %c0_6 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QKIn_0[%c0_6] (%arg8[%c0_6, %1] [%c256_5, %c64_4] [%c64_4, %c1_3]) {id = 1 : i32} : (memref<2x256x64xbf16>)
      %3 = air.channel.put async  @QKIn_1[%c0_6] (%arg8[%c0_6, %1] [%c256_5, %c64_4] [%c64_4, %c1_3]) {id = 2 : i32} : (memref<2x256x64xbf16>)
      %4 = air.channel.put async  @QKIn_2[%c0_6] (%arg8[%c0_6, %1] [%c256_5, %c64_4] [%c64_4, %c1_3]) {id = 3 : i32} : (memref<2x256x64xbf16>)
      %5 = air.channel.put async  @QKIn_3[%c0_6] (%arg8[%c0_6, %1] [%c256_5, %c64_4] [%c64_4, %c1_3]) {id = 4 : i32} : (memref<2x256x64xbf16>)
      %6 = affine.apply #map1()[%arg5]
      %7 = air.channel.put async  @QKIn_0[%c0_6] (%arg9[%c0_6, %6] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 5 : i32} : (memref<2x256x64xbf16>)
      %8 = affine.apply #map2()[%arg5]
      %9 = air.channel.put async  @QKIn_1[%c0_6] (%arg9[%c0_6, %8] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 6 : i32} : (memref<2x256x64xbf16>)
      %10 = affine.apply #map3()[%arg5]
      %11 = air.channel.put async  @QKIn_2[%c0_6] (%arg9[%c0_6, %10] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 7 : i32} : (memref<2x256x64xbf16>)
      %12 = affine.apply #map4()[%arg5]
      %13 = air.channel.put async  @QKIn_3[%c0_6] (%arg9[%c0_6, %12] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 8 : i32} : (memref<2x256x64xbf16>)
      %14 = air.channel.put async  @VIn_0[%c0_6] (%arg10[%c0_6, %c0_6, %6] [%c1_3, %c64_4, %c64_4] [%c4096, %c64_4, %c1_3]) {id = 9 : i32} : (memref<2x256x64xbf16>)
      %15 = air.channel.put async  @VIn_1[%c0_6] (%arg10[%c0_6, %c0_6, %8] [%c1_3, %c64_4, %c64_4] [%c4096, %c64_4, %c1_3]) {id = 10 : i32} : (memref<2x256x64xbf16>)
      %16 = air.channel.put async  @VIn_2[%c0_6] (%arg10[%c0_6, %c0_6, %10] [%c1_3, %c64_4, %c64_4] [%c4096, %c64_4, %c1_3]) {id = 11 : i32} : (memref<2x256x64xbf16>)
      %17 = air.channel.put async  @VIn_3[%c0_6] (%arg10[%c0_6, %c0_6, %12] [%c1_3, %c64_4, %c64_4] [%c4096, %c64_4, %c1_3]) {id = 12 : i32} : (memref<2x256x64xbf16>)
      %18 = air.channel.get async  @channel_0[%c0_6, %c0_6] (%arg11[%c0_6, %1] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 13 : i32} : (memref<2x256x64xbf16>)
      %19 = air.channel.get async  @channel_0[%c1_3, %c0_6] (%arg11[%c64_4, %1] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 14 : i32} : (memref<2x256x64xbf16>)
      %20 = air.channel.get async  @channel_0[%c2, %c0_6] (%arg11[%c128, %1] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 15 : i32} : (memref<2x256x64xbf16>)
      %21 = air.channel.get async  @channel_0[%c3, %c0_6] (%arg11[%c192, %1] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 16 : i32} : (memref<2x256x64xbf16>)
      %22 = air.wait_all async [%18, %19, %20, %21] 
      %23 = air.wait_all async 
      %24 = air.wait_all async 
      %25 = affine.apply #map5()[%arg5, %arg4]
      %26 = air.channel.put async  @QKIn_0[%c1_3] (%arg8[%c0_6, %25] [%c256_5, %c64_4] [%c64_4, %c1_3]) {id = 17 : i32} : (memref<2x256x64xbf16>)
      %27 = air.channel.put async  @QKIn_1[%c1_3] (%arg8[%c0_6, %25] [%c256_5, %c64_4] [%c64_4, %c1_3]) {id = 18 : i32} : (memref<2x256x64xbf16>)
      %28 = air.channel.put async  @QKIn_2[%c1_3] (%arg8[%c0_6, %25] [%c256_5, %c64_4] [%c64_4, %c1_3]) {id = 19 : i32} : (memref<2x256x64xbf16>)
      %29 = air.channel.put async  @QKIn_3[%c1_3] (%arg8[%c0_6, %25] [%c256_5, %c64_4] [%c64_4, %c1_3]) {id = 20 : i32} : (memref<2x256x64xbf16>)
      %30 = affine.apply #map6()[%arg5]
      %31 = air.channel.put async  @QKIn_0[%c1_3] (%arg9[%c0_6, %30] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 21 : i32} : (memref<2x256x64xbf16>)
      %32 = affine.apply #map7()[%arg5]
      %33 = air.channel.put async  @QKIn_1[%c1_3] (%arg9[%c0_6, %32] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 22 : i32} : (memref<2x256x64xbf16>)
      %34 = affine.apply #map8()[%arg5]
      %35 = air.channel.put async  @QKIn_2[%c1_3] (%arg9[%c0_6, %34] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 23 : i32} : (memref<2x256x64xbf16>)
      %36 = affine.apply #map9()[%arg5]
      %37 = air.channel.put async  @QKIn_3[%c1_3] (%arg9[%c0_6, %36] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 24 : i32} : (memref<2x256x64xbf16>)
      %38 = air.channel.put async  @VIn_0[%c1_3] (%arg10[%c0_6, %c0_6, %30] [%c1_3, %c64_4, %c64_4] [%c4096, %c64_4, %c1_3]) {id = 25 : i32} : (memref<2x256x64xbf16>)
      %39 = air.channel.put async  @VIn_1[%c1_3] (%arg10[%c0_6, %c0_6, %32] [%c1_3, %c64_4, %c64_4] [%c4096, %c64_4, %c1_3]) {id = 26 : i32} : (memref<2x256x64xbf16>)
      %40 = air.channel.put async  @VIn_2[%c1_3] (%arg10[%c0_6, %c0_6, %34] [%c1_3, %c64_4, %c64_4] [%c4096, %c64_4, %c1_3]) {id = 27 : i32} : (memref<2x256x64xbf16>)
      %41 = air.channel.put async  @VIn_3[%c1_3] (%arg10[%c0_6, %c0_6, %36] [%c1_3, %c64_4, %c64_4] [%c4096, %c64_4, %c1_3]) {id = 28 : i32} : (memref<2x256x64xbf16>)
      %42 = air.channel.get async  @channel_0[%c0_6, %c1_3] (%arg11[%c0_6, %25] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 29 : i32} : (memref<2x256x64xbf16>)
      %43 = air.channel.get async  @channel_0[%c1_3, %c1_3] (%arg11[%c64_4, %25] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 30 : i32} : (memref<2x256x64xbf16>)
      %44 = air.channel.get async  @channel_0[%c2, %c1_3] (%arg11[%c128, %25] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 31 : i32} : (memref<2x256x64xbf16>)
      %45 = air.channel.get async  @channel_0[%c3, %c1_3] (%arg11[%c192, %25] [%c64_4, %c64_4] [%c64_4, %c1_3]) {id = 32 : i32} : (memref<2x256x64xbf16>)
      %46 = air.wait_all async [%42, %43, %44, %45] 
      %47 = air.wait_all async 
      %48 = air.wait_all async 
      %49 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_3) attributes {id = 2 : i32} {
        %c192_7 = arith.constant 192 : index
        %c128_8 = arith.constant 128 : index
        %c3_9 = arith.constant 3 : index
        %c2_10 = arith.constant 2 : index
        %c64_11 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c8 = arith.constant 8 : index
        %c1_12 = arith.constant 1 : index
        %c0_13 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
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
        %c0_28 = arith.constant 0 : index
        %c64_29 = arith.constant 64 : index
        %c1_30 = arith.constant 1 : index
        %c0_31 = arith.constant 0 : index
        %c64_32 = arith.constant 64 : index
        %c1_33 = arith.constant 1 : index
        %c0_34 = arith.constant 0 : index
        %c64_35 = arith.constant 64 : index
        %c1_36 = arith.constant 1 : index
        %c0_37 = arith.constant 0 : index
        %c64_38 = arith.constant 64 : index
        %c1_39 = arith.constant 1 : index
        %c0_40 = arith.constant 0 : index
        %c64_41 = arith.constant 64 : index
        %c1_42 = arith.constant 1 : index
        %c0_43 = arith.constant 0 : index
        %c64_44 = arith.constant 64 : index
        %c1_45 = arith.constant 1 : index
        %c0_46 = arith.constant 0 : index
        %c64_47 = arith.constant 64 : index
        %c1_48 = arith.constant 1 : index
        %c0_49 = arith.constant 0 : index
        %c64_50 = arith.constant 64 : index
        %c1_51 = arith.constant 1 : index
        %async_token_52, %results_53 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_54, %results_55 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_56, %results_57 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_58, %results_59 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %50 = air.wait_all async 
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
        %async_token_66, %results_67 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_68, %results_69 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_70, %results_71 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_72, %results_73 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %51 = scf.for %arg16 = %c0_13 to %c4 step %c1_12 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %89 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %90 = arith.cmpi eq, %arg12, %c0_13 : index
          %91 = scf.if %90 -> (!air.async.token) {
            %92 = air.channel.put async [%89]  @QK2L1_0_0[%c0_13, %c0_13, %c0_13] (%results[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %92 : !air.async.token
          } else {
            %92 = air.channel.put async [%89]  @QK2L1_0_1[%c0_13, %c0_13, %c0_13] (%results[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %92 : !air.async.token
          }
          scf.yield %91 : !air.async.token
        }
        %52 = air.channel.get async [%51]  @QKIn_0[%arg12] (%results[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
        %53 = arith.cmpi eq, %arg12, %c0_13 : index
        %54 = scf.if %53 -> (!air.async.token) {
          %89 = air.channel.put async [%52]  @QK2L1_0_0[%c0_13, %c0_13, %c0_13] (%results[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        } else {
          %89 = air.channel.put async [%52]  @QK2L1_0_1[%c0_13, %c0_13, %c0_13] (%results[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        }
        %55 = scf.for %arg16 = %c0_13 to %c4 step %c1_12 iter_args(%arg17 = %async_token_14) -> (!air.async.token) {
          %89 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_15[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %90 = scf.if %53 -> (!air.async.token) {
            %91 = air.channel.put async [%89]  @QK2L1_1_0[%c0_13, %c0_13, %c0_13] (%results_15[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %91 : !air.async.token
          } else {
            %91 = air.channel.put async [%89]  @QK2L1_1_1[%c0_13, %c0_13, %c0_13] (%results_15[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %91 : !air.async.token
          }
          scf.yield %90 : !air.async.token
        }
        %56 = air.channel.get async [%55]  @QKIn_1[%arg12] (%results_15[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
        %57 = scf.if %53 -> (!air.async.token) {
          %89 = air.channel.put async [%56]  @QK2L1_1_0[%c0_13, %c0_13, %c0_13] (%results_15[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        } else {
          %89 = air.channel.put async [%56]  @QK2L1_1_1[%c0_13, %c0_13, %c0_13] (%results_15[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        }
        %58 = scf.for %arg16 = %c0_13 to %c4 step %c1_12 iter_args(%arg17 = %async_token_16) -> (!air.async.token) {
          %89 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_17[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
          %90 = scf.if %53 -> (!air.async.token) {
            %91 = air.channel.put async [%89]  @QK2L1_2_0[%c0_13, %c0_13, %c0_13] (%results_17[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %91 : !air.async.token
          } else {
            %91 = air.channel.put async [%89]  @QK2L1_2_1[%c0_13, %c0_13, %c0_13] (%results_17[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %91 : !air.async.token
          }
          scf.yield %90 : !air.async.token
        }
        %59 = air.channel.get async [%58]  @QKIn_2[%arg12] (%results_17[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
        %60 = scf.if %53 -> (!air.async.token) {
          %89 = air.channel.put async [%59]  @QK2L1_2_0[%c0_13, %c0_13, %c0_13] (%results_17[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        } else {
          %89 = air.channel.put async [%59]  @QK2L1_2_1[%c0_13, %c0_13, %c0_13] (%results_17[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        }
        %61 = scf.for %arg16 = %c0_13 to %c4 step %c1_12 iter_args(%arg17 = %async_token_18) -> (!air.async.token) {
          %89 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_19[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
          %90 = scf.if %53 -> (!air.async.token) {
            %91 = air.channel.put async [%89]  @QK2L1_3_0[%c0_13, %c0_13, %c0_13] (%results_19[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %91 : !air.async.token
          } else {
            %91 = air.channel.put async [%89]  @QK2L1_3_1[%c0_13, %c0_13, %c0_13] (%results_19[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 53 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %91 : !air.async.token
          }
          scf.yield %90 : !air.async.token
        }
        %62 = air.channel.get async [%61]  @QKIn_3[%arg12] (%results_19[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 1 : i32>)
        %63 = scf.if %53 -> (!air.async.token) {
          %89 = air.channel.put async [%62]  @QK2L1_3_0[%c0_13, %c0_13, %c0_13] (%results_19[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 55 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        } else {
          %89 = air.channel.put async [%62]  @QK2L1_3_1[%c0_13, %c0_13, %c0_13] (%results_19[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 56 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        }
        %64 = air.channel.get async [%async_token_20]  @VIn_0[%arg12] (%results_21[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 1 : i32>)
        %65 = scf.if %53 -> (!air.async.token) {
          %89 = air.channel.put async [%64]  @V2L1_0_0[%c0_13, %c0_13, %c0_13] (%results_21[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 58 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        } else {
          %89 = air.channel.put async [%64]  @V2L1_0_1[%c0_13, %c0_13, %c0_13] (%results_21[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 59 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        }
        %66 = air.channel.get async [%async_token_22]  @VIn_1[%arg12] (%results_23[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 1 : i32>)
        %67 = scf.if %53 -> (!air.async.token) {
          %89 = air.channel.put async [%66]  @V2L1_1_0[%c0_13, %c0_13, %c0_13] (%results_23[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 61 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        } else {
          %89 = air.channel.put async [%66]  @V2L1_1_1[%c0_13, %c0_13, %c0_13] (%results_23[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 62 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        }
        %68 = air.channel.get async [%async_token_24]  @VIn_2[%arg12] (%results_25[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 1 : i32>)
        %69 = scf.if %53 -> (!air.async.token) {
          %89 = air.channel.put async [%68]  @V2L1_2_0[%c0_13, %c0_13, %c0_13] (%results_25[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 64 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        } else {
          %89 = air.channel.put async [%68]  @V2L1_2_1[%c0_13, %c0_13, %c0_13] (%results_25[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 65 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        }
        %70 = air.channel.get async [%async_token_26]  @VIn_3[%arg12] (%results_27[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 1 : i32>)
        %71 = scf.if %53 -> (!air.async.token) {
          %89 = air.channel.put async [%70]  @V2L1_3_0[%c0_13, %c0_13, %c0_13] (%results_27[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 67 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        } else {
          %89 = air.channel.put async [%70]  @V2L1_3_1[%c0_13, %c0_13, %c0_13] (%results_27[%c0_13, %c0_13, %c0_13, %c0_13] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_11, %c1_12]) {id = 68 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %89 : !air.async.token
        }
        %72 = air.channel.get async [%async_token_52]  @Gp2L2[%c0_13, %c0_13] (%results_53[%c0_49, %c0_13] [%c64_11, %c64_11] [%c64_50, %c1_51]) {id = 69 : i32} : (memref<64x64xbf16, 1 : i32>)
        %73 = air.wait_all async [%72] 
        %74 = air.channel.get async [%async_token_54]  @Gp2L2[%c1_12, %c0_13] (%results_55[%c0_43, %c0_13] [%c64_11, %c64_11] [%c64_44, %c1_45]) {id = 70 : i32} : (memref<64x64xbf16, 1 : i32>)
        %75 = air.wait_all async [%74] 
        %76 = air.channel.get async [%async_token_56]  @Gp2L2[%c2_10, %c0_13] (%results_57[%c0_37, %c0_13] [%c64_11, %c64_11] [%c64_38, %c1_39]) {id = 71 : i32} : (memref<64x64xbf16, 1 : i32>)
        %77 = air.wait_all async [%76] 
        %78 = air.channel.get async [%async_token_58]  @Gp2L2[%c3_9, %c0_13] (%results_59[%c0_31, %c0_13] [%c64_11, %c64_11] [%c64_32, %c1_33]) {id = 72 : i32} : (memref<64x64xbf16, 1 : i32>)
        %79 = air.wait_all async [%78] 
        %80 = air.wait_all async [%73, %75, %77, %79] 
        %81 = air.wait_all async 
        %82 = air.channel.put async [%80]  @channel_0[%c0_13, %arg12] (%results_53[%c0_46, %c0_13] [%c64_11, %c64_11] [%c64_47, %c1_48]) {id = 73 : i32} : (memref<64x64xbf16, 1 : i32>)
        %83 = air.channel.put async [%80]  @channel_0[%c1_12, %arg12] (%results_55[%c0_40, %c0_13] [%c64_11, %c64_11] [%c64_41, %c1_42]) {id = 74 : i32} : (memref<64x64xbf16, 1 : i32>)
        %84 = air.channel.put async [%80]  @channel_0[%c2_10, %arg12] (%results_57[%c0_34, %c0_13] [%c64_11, %c64_11] [%c64_35, %c1_36]) {id = 75 : i32} : (memref<64x64xbf16, 1 : i32>)
        %85 = air.channel.put async [%80]  @channel_0[%c3_9, %arg12] (%results_59[%c0_28, %c0_13] [%c64_11, %c64_11] [%c64_29, %c1_30]) {id = 76 : i32} : (memref<64x64xbf16, 1 : i32>)
        %86 = air.wait_all async [%82, %83, %84, %85] 
        %87 = air.wait_all async 
        %88 = air.herd @herd_0 async [%async_token_60, %async_token_62, %async_token_64, %async_token_66, %async_token_68, %async_token_70, %async_token_72]  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%results_61, %arg21=%results_63, %arg22=%results_65, %arg23=%results_67, %arg24=%results_69, %arg25=%results_71, %arg26=%results_73, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {id = 3 : i32, link_with = "attn.o"} {
          %c512_93 = arith.constant 512 : index
          %c64_94 = arith.constant 64 : index
          %c8_95 = arith.constant 8 : index
          %c0_96 = arith.constant 0 : index
          %c1_97 = arith.constant 1 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_98 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_99 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_100 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %89 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async  @QK2L1_0_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async  @QK2L1_0_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %90 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%89]  @QK2L1_1_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%89]  @QK2L1_1_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %91 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%90]  @QK2L1_2_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%90]  @QK2L1_2_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %92 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%91]  @QK2L1_3_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%91]  @QK2L1_3_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %93 = arith.index_cast %arg16 : index to i32
          %94 = arith.cmpi eq, %93, %c0_i32 : i32
          scf.if %94 {
            %async_token_114 = air.execute [%92] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %95 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async  @QK2L1_0_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async  @QK2L1_0_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %96 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%95]  @QK2L1_1_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%95]  @QK2L1_1_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %97 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%96]  @QK2L1_2_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%96]  @QK2L1_2_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %98 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%97]  @QK2L1_3_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%97]  @QK2L1_3_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %99 = arith.cmpi eq, %93, %c1_i32 : i32
          scf.if %99 {
            %async_token_114 = air.execute [%98] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %100 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async  @QK2L1_0_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async  @QK2L1_0_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %101 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%100]  @QK2L1_1_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%100]  @QK2L1_1_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %102 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%101]  @QK2L1_2_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%101]  @QK2L1_2_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %103 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%102]  @QK2L1_3_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%102]  @QK2L1_3_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %104 = arith.cmpi eq, %93, %c2_i32 : i32
          scf.if %104 {
            %async_token_114 = air.execute [%103] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %105 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async  @QK2L1_0_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async  @QK2L1_0_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 102 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %106 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%105]  @QK2L1_1_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 103 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%105]  @QK2L1_1_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %107 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%106]  @QK2L1_2_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 105 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%106]  @QK2L1_2_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 106 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %108 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%107]  @QK2L1_3_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%107]  @QK2L1_3_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 108 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %109 = arith.cmpi eq, %93, %c3_i32 : i32
          scf.if %109 {
            %async_token_114 = air.execute [%108] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %async_token_101 = air.execute {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          %110 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async  @QK2L1_0_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 109 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async  @QK2L1_0_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %111 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%110]  @QK2L1_1_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 111 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%110]  @QK2L1_1_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 112 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %112 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%111]  @QK2L1_2_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%111]  @QK2L1_2_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 114 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %113 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%112]  @QK2L1_3_0[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 115 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%112]  @QK2L1_3_1[%c0_96, %arg17, %arg16] (%arg21[] [] []) {id = 116 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %async_token_102 = air.execute [%113, %async_token_101] {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %114 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async  @V2L1_0_0[%c0_96, %arg17, %arg16] (%arg22[] [] []) {id = 117 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async  @V2L1_0_1[%c0_96, %arg17, %arg16] (%arg22[] [] []) {id = 118 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %115 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%114]  @V2L1_1_0[%c0_96, %arg17, %arg16] (%arg22[] [] []) {id = 119 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%114]  @V2L1_1_1[%c0_96, %arg17, %arg16] (%arg22[] [] []) {id = 120 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %116 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%115]  @V2L1_2_0[%c0_96, %arg17, %arg16] (%arg22[] [] []) {id = 121 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%115]  @V2L1_2_1[%c0_96, %arg17, %arg16] (%arg22[] [] []) {id = 122 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %117 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.cmpi eq, %arg27, %c0_96 : index
            %120 = scf.if %119 -> (!air.async.token) {
              %121 = air.channel.get async [%116]  @V2L1_3_0[%c0_96, %arg17, %arg16] (%arg22[] [] []) {id = 123 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            } else {
              %121 = air.channel.get async [%116]  @V2L1_3_1[%c0_96, %arg17, %arg16] (%arg22[] [] []) {id = 124 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %121 : !air.async.token
            }
            affine.yield %120 : !air.async.token
          } else {
            %119 = air.wait_all async 
            affine.yield %119 : !air.async.token
          }
          %async_token_103, %results_104 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_105, %results_106 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_107 = air.execute [%async_token_105, %async_token_103, %async_token_102, %async_token_100] {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %results_104, %results_106) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_108 = air.execute [%async_token_107, %async_token_98] {
            func.call @mul_r_gp(%results_106, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_109 = air.execute [%async_token_108, %117] {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_110 = air.execute [%async_token_108, %async_token_99] {
            func.call @accum_sp_r_s(%arg26, %results_106, %results_104) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_111 = air.execute [%async_token_110] {
            func.call @vector_copy_32elems(%c0_i32, %results_104, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_112 = air.execute [%async_token_111] {
            memref.dealloc %results_104 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_113 = air.execute [%async_token_110] {
            memref.dealloc %results_106 : memref<64x1xbf16, 2 : i32>
          }
          %118 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %119 = arith.subi %arg17, %c1_97 : index
            %120 = air.channel.put async [%async_token_109]  @cascade_gp[%arg16, %119] (%arg24[] [] []) {id = 125 : i32} : (memref<64x64xbf16, 2 : i32>)
            %121 = air.channel.put async [%async_token_100]  @cascade_up[%arg16, %119] (%arg25[] [] []) {id = 126 : i32} : (memref<64x1xbf16, 2 : i32>)
            %122 = air.channel.put async [%async_token_111]  @cascade_sp[%arg16, %119] (%arg26[] [] []) {id = 127 : i32} : (memref<64x1xbf16, 2 : i32>)
            %123 = air.wait_all async [%120, %121, %122] 
            affine.yield %123 : !air.async.token
          } else {
            %119 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_114, %results_115 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_116, %results_117 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_118, %results_119 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %121 = air.channel.get async [%async_token_114]  @cascade_gp[%arg16, %arg17] (%results_115[] [] []) {id = 128 : i32} : (memref<64x64xbf16, 2 : i32>)
              %122 = air.channel.get async [%async_token_116]  @cascade_up[%arg16, %arg17] (%results_117[] [] []) {id = 129 : i32} : (memref<64x1xbf16, 2 : i32>)
              %123 = air.channel.get async [%async_token_118]  @cascade_sp[%arg16, %arg17] (%results_119[] [] []) {id = 130 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_120, %results_121 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_122 = air.execute [%async_token_120, %async_token_100] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_121) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_123 = air.execute [%async_token_122, %122] {
                func.call @maximum_up_u_bf16(%results_117, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_124, %results_125 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_126 = air.execute [%async_token_124, %async_token_123] {
                func.call @exp_up_minus_u(%results_117, %arg25, %results_125) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127, %results_128 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_129 = air.execute [%async_token_127, %async_token_126] {
                func.call @exp_up_minus_u(%results_121, %arg25, %results_128) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_130 = air.execute [%async_token_126, %121] {
                func.call @mul_r_gp(%results_125, %results_115) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_131 = air.execute [%async_token_129, %async_token_109] {
                func.call @mul_r_gp(%results_128, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_132 = air.execute [%async_token_131, %async_token_130] {
                func.call @add_gp_g(%arg24, %results_115) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_133, %results_134 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_135 = air.execute [%async_token_133] {
                func.call @zero_fill_sp_bf16(%results_134) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_136 = air.execute [%async_token_135, %async_token_130, %123] {
                func.call @accum_sp_r_s(%results_119, %results_125, %results_134) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_137 = air.execute [%async_token_136, %async_token_131, %async_token_111] {
                func.call @accum_sp_r_s(%arg26, %results_128, %results_134) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_138 = air.execute [%async_token_137] {
                func.call @vector_copy_32elems(%c0_i32, %results_134, %results_119) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %124 = arith.subi %arg17, %c1_97 : index
              %125 = air.channel.put async [%async_token_132]  @cascade_gp[%arg16, %124] (%results_115[] [] []) {id = 131 : i32} : (memref<64x64xbf16, 2 : i32>)
              %126 = air.channel.put async [%async_token_129]  @cascade_up[%arg16, %124] (%arg25[] [] []) {id = 132 : i32} : (memref<64x1xbf16, 2 : i32>)
              %127 = air.channel.put async [%async_token_138]  @cascade_sp[%arg16, %124] (%results_119[] [] []) {id = 133 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_139 = air.execute [%125] {
                memref.dealloc %results_115 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_140 = air.execute [%async_token_126] {
                memref.dealloc %results_117 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_141 = air.execute [%127] {
                memref.dealloc %results_119 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_142 = air.execute [%async_token_129] {
                memref.dealloc %results_121 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_143 = air.execute [%async_token_136] {
                memref.dealloc %results_125 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_144 = air.execute [%async_token_137] {
                memref.dealloc %results_128 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_145 = air.execute [%async_token_138] {
                memref.dealloc %results_134 : memref<64x1xbf16, 2 : i32>
              }
              %128 = air.wait_all async [%125, %126, %127] 
              affine.yield %128 : !air.async.token
            } else {
              %async_token_114, %results_115 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_116, %results_117 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_118, %results_119 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %121 = air.channel.get async [%async_token_114]  @cascade_gp[%arg16, %arg17] (%results_115[] [] []) {id = 134 : i32} : (memref<64x64xbf16, 2 : i32>)
              %122 = air.channel.get async [%async_token_116]  @cascade_up[%arg16, %arg17] (%results_117[] [] []) {id = 135 : i32} : (memref<64x1xbf16, 2 : i32>)
              %123 = air.channel.get async [%async_token_118]  @cascade_sp[%arg16, %arg17] (%results_119[] [] []) {id = 136 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_120, %results_121 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_122 = air.execute [%async_token_120, %async_token_100] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_121) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_123 = air.execute [%async_token_122, %122] {
                func.call @maximum_up_u_bf16(%results_117, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_124, %results_125 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_126 = air.execute [%async_token_124, %async_token_123] {
                func.call @exp_up_minus_u(%results_117, %arg25, %results_125) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127, %results_128 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_129 = air.execute [%async_token_127, %async_token_126] {
                func.call @exp_up_minus_u(%results_121, %arg25, %results_128) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_130 = air.execute [%async_token_126, %121] {
                func.call @mul_r_gp(%results_125, %results_115) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_131 = air.execute [%async_token_129, %async_token_109] {
                func.call @mul_r_gp(%results_128, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_132 = air.execute [%async_token_131, %async_token_130] {
                func.call @add_gp_g(%arg24, %results_115) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_133, %results_134 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_135 = air.execute [%async_token_133] {
                func.call @zero_fill_sp_bf16(%results_134) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_136 = air.execute [%async_token_135, %async_token_130, %123] {
                func.call @accum_sp_r_s(%results_119, %results_125, %results_134) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_137 = air.execute [%async_token_136, %async_token_131, %async_token_111] {
                func.call @accum_sp_r_s(%arg26, %results_128, %results_134) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_138 = air.execute [%async_token_137] {
                func.call @vector_copy_32elems(%c0_i32, %results_134, %results_119) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_139 = air.execute [%async_token_138, %async_token_132] {
                func.call @div_gp_sp(%results_119, %results_115) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %124 = air.channel.put async [%async_token_139]  @Gp2L2[%arg16, %c0_96] (%results_115[%c0_96, %c0_96, %c0_96, %c0_96] [%c8_95, %c8_95, %c8_95, %c8_95] [%c64_94, %c8_95, %c512_93, %c1_97]) {id = 137 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_140 = air.execute [%124] {
                memref.dealloc %results_115 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_141 = air.execute [%async_token_126] {
                memref.dealloc %results_117 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_142 = air.execute [%async_token_139] {
                memref.dealloc %results_119 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_143 = air.execute [%async_token_129] {
                memref.dealloc %results_121 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_144 = air.execute [%async_token_136] {
                memref.dealloc %results_125 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_145 = air.execute [%async_token_137] {
                memref.dealloc %results_128 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_146 = air.execute [%async_token_138] {
                memref.dealloc %results_134 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %124 : !air.async.token
            }
            %120 = air.wait_all async [%110, %111, %112, %114, %115, %116, %async_token_109, %async_token_111] 
            affine.yield %120 : !air.async.token
          }
        }
        %async_token_74 = air.execute [%88] {
          memref.dealloc %results_61 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_75 = air.execute [%88] {
          memref.dealloc %results_63 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_76 = air.execute [%88] {
          memref.dealloc %results_65 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_77 = air.execute [%88] {
          memref.dealloc %results_67 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_78 = air.execute [%88] {
          memref.dealloc %results_69 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_79 = air.execute [%88] {
          memref.dealloc %results_71 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_80 = air.execute [%88] {
          memref.dealloc %results_73 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_81 = air.execute [%54] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_82 = air.execute [%65] {
          memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_83 = air.execute [%57] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_84 = air.execute [%67] {
          memref.dealloc %results_23 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_85 = air.execute [%60] {
          memref.dealloc %results_17 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_86 = air.execute [%69] {
          memref.dealloc %results_25 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_87 = air.execute [%63] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_88 = air.execute [%71] {
          memref.dealloc %results_27 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_89 = air.execute [%85, %84, %83, %82] {
          memref.dealloc %results_59 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_90 = air.execute [%85, %84, %83, %82] {
          memref.dealloc %results_57 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_91 = air.execute [%85, %84, %83, %82] {
          memref.dealloc %results_55 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_92 = air.execute [%85, %84, %83, %82] {
          memref.dealloc %results_53 : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
