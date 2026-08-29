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
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set6 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set7 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
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
  air.channel @QK2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
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
  func.func @attention_bf16(%arg0: memref<2x512x64xbf16>, %arg1: memref<2x512x64xbf16>, %arg2: memref<2x512x64xbf16>, %arg3: memref<2x512x64xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c2_0 = arith.constant 2 : index
      %c1_1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QK2L1_0_0[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 1 : i32} : (memref<2x512x64xbf16>)
      %3 = air.channel.put async  @QK2L1_0_1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 2 : i32} : (memref<2x512x64xbf16>)
      %4 = air.channel.put async  @QK2L1_0_2[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 3 : i32} : (memref<2x512x64xbf16>)
      %5 = air.channel.put async  @QK2L1_0_3[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 4 : i32} : (memref<2x512x64xbf16>)
      %6 = affine.apply #map1()[%arg5]
      %7 = air.channel.put async  @QK2L1_0_0[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %6] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 5 : i32} : (memref<2x512x64xbf16>)
      %8 = affine.apply #map2()[%arg5]
      %9 = air.channel.put async  @QK2L1_0_1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %8] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 6 : i32} : (memref<2x512x64xbf16>)
      %10 = affine.apply #map3()[%arg5]
      %11 = air.channel.put async  @QK2L1_0_2[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %10] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 7 : i32} : (memref<2x512x64xbf16>)
      %12 = affine.apply #map4()[%arg5]
      %13 = air.channel.put async  @QK2L1_0_3[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %12] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 8 : i32} : (memref<2x512x64xbf16>)
      %14 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %6] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 9 : i32} : (memref<2x512x64xbf16>)
      %15 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %8] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 10 : i32} : (memref<2x512x64xbf16>)
      %16 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %10] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 11 : i32} : (memref<2x512x64xbf16>)
      %17 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %12] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 12 : i32} : (memref<2x512x64xbf16>)
      %18 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %1] [%c64, %c64] [%c64, %c1_1]) {id = 13 : i32} : (memref<2x512x64xbf16>)
      %19 = air.channel.get async  @channel_0[%c1_1, %c0] (%arg11[%c64, %1] [%c64, %c64] [%c64, %c1_1]) {id = 14 : i32} : (memref<2x512x64xbf16>)
      %20 = air.channel.get async  @channel_0[%c2_0, %c0] (%arg11[%c128, %1] [%c64, %c64] [%c64, %c1_1]) {id = 15 : i32} : (memref<2x512x64xbf16>)
      %21 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %1] [%c64, %c64] [%c64, %c1_1]) {id = 16 : i32} : (memref<2x512x64xbf16>)
      %22 = affine.apply #map5()[%arg5, %arg4]
      %23 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 17 : i32} : (memref<2x512x64xbf16>)
      %24 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 18 : i32} : (memref<2x512x64xbf16>)
      %25 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 19 : i32} : (memref<2x512x64xbf16>)
      %26 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 20 : i32} : (memref<2x512x64xbf16>)
      %27 = affine.apply #map6()[%arg5]
      %28 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %27] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 21 : i32} : (memref<2x512x64xbf16>)
      %29 = affine.apply #map7()[%arg5]
      %30 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %29] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 22 : i32} : (memref<2x512x64xbf16>)
      %31 = affine.apply #map8()[%arg5]
      %32 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %31] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 23 : i32} : (memref<2x512x64xbf16>)
      %33 = affine.apply #map9()[%arg5]
      %34 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %33] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 24 : i32} : (memref<2x512x64xbf16>)
      %35 = air.channel.put async  @VIn_0[%c1_1] (%arg10[%c0, %c0, %27] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 25 : i32} : (memref<2x512x64xbf16>)
      %36 = air.channel.put async  @VIn_1[%c1_1] (%arg10[%c0, %c0, %29] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 26 : i32} : (memref<2x512x64xbf16>)
      %37 = air.channel.put async  @VIn_2[%c1_1] (%arg10[%c0, %c0, %31] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 27 : i32} : (memref<2x512x64xbf16>)
      %38 = air.channel.put async  @VIn_3[%c1_1] (%arg10[%c0, %c0, %33] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 28 : i32} : (memref<2x512x64xbf16>)
      %39 = air.channel.get async  @channel_0[%c0, %c1_1] (%arg11[%c0, %22] [%c64, %c64] [%c64, %c1_1]) {id = 29 : i32} : (memref<2x512x64xbf16>)
      %40 = air.channel.get async  @channel_0[%c1_1, %c1_1] (%arg11[%c64, %22] [%c64, %c64] [%c64, %c1_1]) {id = 30 : i32} : (memref<2x512x64xbf16>)
      %41 = air.channel.get async  @channel_0[%c2_0, %c1_1] (%arg11[%c128, %22] [%c64, %c64] [%c64, %c1_1]) {id = 31 : i32} : (memref<2x512x64xbf16>)
      %42 = air.channel.get async  @channel_0[%c3, %c1_1] (%arg11[%c192, %22] [%c64, %c64] [%c64, %c1_1]) {id = 32 : i32} : (memref<2x512x64xbf16>)
      %43 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2_0, %arg15=%c1_1) attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c3_2 = arith.constant 3 : index
        %c64_3 = arith.constant 64 : index
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
        %44 = air.channel.get async [%async_token_15]  @VIn_0[%arg12] (%results_16[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
        %45 = arith.cmpi eq, %arg12, %c0_7 : index
        %46 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%44]  @V2L1_0_0[%c0_7, %c0_7, %c0_7] (%results_16[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%44]  @V2L1_0_1[%c0_7, %c0_7, %c0_7] (%results_16[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_17 = air.execute [%46, %44] {
          memref.dealloc %results_16 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %47 = air.channel.get async [%async_token_18]  @VIn_0[%arg12] (%results_19[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
        %48 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%47]  @V2L1_0_0[%c0_7, %c0_7, %c0_7] (%results_19[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%47]  @V2L1_0_1[%c0_7, %c0_7, %c0_7] (%results_19[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_20 = air.execute [%48, %47] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %49 = air.channel.get async [%async_token_21]  @VIn_1[%arg12] (%results_22[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
        %50 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%49]  @V2L1_1_0[%c0_7, %c0_7, %c0_7] (%results_22[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%49]  @V2L1_1_1[%c0_7, %c0_7, %c0_7] (%results_22[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_23 = air.execute [%50, %49] {
          memref.dealloc %results_22 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_24, %results_25 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %51 = air.channel.get async [%async_token_24]  @VIn_1[%arg12] (%results_25[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
        %52 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%51]  @V2L1_1_0[%c0_7, %c0_7, %c0_7] (%results_25[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%51]  @V2L1_1_1[%c0_7, %c0_7, %c0_7] (%results_25[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_26 = air.execute [%52, %51] {
          memref.dealloc %results_25 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_27, %results_28 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %53 = air.channel.get async [%async_token_27]  @VIn_2[%arg12] (%results_28[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
        %54 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%53]  @V2L1_2_0[%c0_7, %c0_7, %c0_7] (%results_28[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%53]  @V2L1_2_1[%c0_7, %c0_7, %c0_7] (%results_28[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_29 = air.execute [%54, %53] {
          memref.dealloc %results_28 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_30, %results_31 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %55 = air.channel.get async [%async_token_30]  @VIn_2[%arg12] (%results_31[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
        %56 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%55]  @V2L1_2_0[%c0_7, %c0_7, %c0_7] (%results_31[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%55]  @V2L1_2_1[%c0_7, %c0_7, %c0_7] (%results_31[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_32 = air.execute [%56, %55] {
          memref.dealloc %results_31 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_33, %results_34 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %57 = air.channel.get async [%async_token_33]  @VIn_3[%arg12] (%results_34[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
        %58 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%57]  @V2L1_3_0[%c0_7, %c0_7, %c0_7] (%results_34[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%57]  @V2L1_3_1[%c0_7, %c0_7, %c0_7] (%results_34[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_35 = air.execute [%58, %57] {
          memref.dealloc %results_34 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_36, %results_37 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %59 = air.channel.get async [%async_token_36]  @VIn_3[%arg12] (%results_37[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
        %60 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%59]  @V2L1_3_0[%c0_7, %c0_7, %c0_7] (%results_37[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%59]  @V2L1_3_1[%c0_7, %c0_7, %c0_7] (%results_37[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_38 = air.execute [%60, %59] {
          memref.dealloc %results_37 : memref<64x64xbf16, 1 : i32>
        }
        %61 = air.channel.get async [%async_token]  @Gp2L2[%c0_7, %c0_7] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %62 = air.channel.get async [%async_token_9]  @Gp2L2[%c1_5, %c0_7] (%results_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %63 = air.channel.get async [%async_token_11]  @Gp2L2[%c2_6, %c0_7] (%results_12[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %64 = air.channel.get async [%async_token_13]  @Gp2L2[%c3_2, %c0_7] (%results_14[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %65 = air.channel.put async [%61]  @channel_0[%c0_7, %arg12] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %66 = air.channel.put async [%62]  @channel_0[%c1_5, %arg12] (%results_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %67 = air.channel.put async [%63]  @channel_0[%c2_6, %arg12] (%results_12[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %68 = air.channel.put async [%64]  @channel_0[%c3_2, %arg12] (%results_14[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %69 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c64_43 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c0_44 = arith.constant 0 : index
          %c1_45 = arith.constant 1 : index
          %c8_46 = arith.constant 8 : index
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
          %70 = arith.cmpi eq, %arg20, %c0_44 : index
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %71 = arith.index_cast %arg16 : index to i32
          %72 = arith.cmpi eq, %71, %c0_i32 : i32
          scf.if %72 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %73 = arith.cmpi eq, %71, %c1_i32 : i32
          scf.if %73 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %74 = arith.cmpi eq, %71, %c2_i32 : i32
          scf.if %74 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %75 = arith.cmpi eq, %71, %c3_i32 : i32
          scf.if %75 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56] {
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
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %76 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_61]  @V2L1_0_0[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_61]  @V2L1_0_1[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %77 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_61, %76]  @V2L1_1_0[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_61, %76]  @V2L1_1_1[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %78 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_61, %77]  @V2L1_2_0[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_61, %77]  @V2L1_2_1[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %79 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_61, %78]  @V2L1_3_0[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_61, %78]  @V2L1_3_1[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %async_token_66 = air.execute [%async_token_65, %async_token_56, %async_token_54] {
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
          %async_token_73 = air.execute [%79, %async_token_72, %async_token_61, %async_token_63] {
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
          %async_token_79 = air.execute [%76, %77, %78, %async_token_73] {
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
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %80 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_80]  @V2L1_0_0[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_80]  @V2L1_0_1[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %81 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%77, %async_token_80, %80]  @V2L1_1_0[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%77, %async_token_80, %80]  @V2L1_1_1[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %82 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%78, %async_token_80, %81]  @V2L1_2_0[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%78, %async_token_80, %81]  @V2L1_2_1[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %83 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_80, %82]  @V2L1_3_0[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_80, %82]  @V2L1_3_1[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %async_token_85 = air.execute [%async_token_84, %async_token_56, %async_token_54] {
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
          %async_token_92 = air.execute [%83, %async_token_91, %async_token_80, %async_token_82] {
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
          %84 = air.wait_all async [%async_token_92, %async_token_94] 
          %async_token_97 = air.execute [%async_token_90, %async_token_92] {
            memref.dealloc %results_83 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_98 = air.execute [%80, %81, %82, %async_token_92] {
            memref.dealloc %results_81 : memref<64x64xbf16, 2 : i32>
          }
          %85 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %86 = arith.subi %arg17, %c1_45 : index
            %87 = air.channel.put async [%async_token_52, %84]  @cascade_gp[%arg16, %86] (%results_53[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
            %88 = air.channel.put async [%async_token_50]  @cascade_up[%arg16, %86] (%results_51[] [] []) {id = 102 : i32} : (memref<64x1xbf16, 2 : i32>)
            %89 = air.channel.put async [%async_token_48, %84]  @cascade_sp[%arg16, %86] (%results_49[] [] []) {id = 103 : i32} : (memref<64x1xbf16, 2 : i32>)
            %90 = air.wait_all async [%87, %88, %89] 
            affine.yield %90 : !air.async.token
          } else {
            %86 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
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
              %87 = air.channel.get async [%async_token_104]  @cascade_gp[%arg16, %arg17] (%results_105[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              %88 = air.channel.get async [%async_token_106]  @cascade_up[%arg16, %arg17] (%results_107[] [] []) {id = 105 : i32} : (memref<64x1xbf16, 2 : i32>)
              %89 = air.channel.get async [%async_token_108]  @cascade_sp[%arg16, %arg17] (%results_109[] [] []) {id = 106 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_110, %results_111 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_110, %async_token_50] {
                func.call @vector_copy_32elems(%c0_i32, %results_51, %results_111) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_113 = air.execute [%88, %async_token_112] {
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
              %async_token_120 = air.execute [%async_token_116, %87] {
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
              %async_token_126 = air.execute [%async_token_125, %async_token_120, %89] {
                func.call @accum_sp_r_s(%results_109, %results_115, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127 = air.execute [%async_token_48, %async_token_126, %async_token_121] {
                func.call @accum_sp_r_s(%results_49, %results_118, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_128 = air.execute [%async_token_127] {
                func.call @vector_copy_32elems(%c0_i32, %results_124, %results_109) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %90 = arith.subi %arg17, %c1_45 : index
              %91 = air.channel.put async [%async_token_122]  @cascade_gp[%arg16, %90] (%results_105[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              %92 = air.channel.put async [%async_token_50, %async_token_119]  @cascade_up[%arg16, %90] (%results_51[] [] []) {id = 108 : i32} : (memref<64x1xbf16, 2 : i32>)
              %93 = air.channel.put async [%async_token_128]  @cascade_sp[%arg16, %90] (%results_109[] [] []) {id = 109 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_129 = air.execute [%91] {
                memref.dealloc %results_105 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_130 = air.execute [%async_token_116] {
                memref.dealloc %results_107 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_131 = air.execute [%93] {
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
              %94 = air.wait_all async [%91, %92, %93] 
              affine.yield %94 : !air.async.token
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
              %87 = air.channel.get async [%async_token_104]  @cascade_gp[%arg16, %arg17] (%results_105[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              %88 = air.channel.get async [%async_token_106]  @cascade_up[%arg16, %arg17] (%results_107[] [] []) {id = 111 : i32} : (memref<64x1xbf16, 2 : i32>)
              %89 = air.channel.get async [%async_token_108]  @cascade_sp[%arg16, %arg17] (%results_109[] [] []) {id = 112 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_110, %results_111 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_110, %async_token_50] {
                func.call @vector_copy_32elems(%c0_i32, %results_51, %results_111) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_113 = air.execute [%88, %async_token_112] {
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
              %async_token_120 = air.execute [%async_token_116, %87] {
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
              %async_token_126 = air.execute [%async_token_125, %async_token_120, %89] {
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
              %90 = air.channel.put async [%async_token_129]  @Gp2L2[%arg16, %c0_44] (%results_105[%c0_44, %c0_44, %c0_44] [%c64_43, %c8_46, %c8_46] [%c8_46, %c512_47, %c1_45]) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_130 = air.execute [%90] {
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
              affine.yield %90 : !air.async.token
            }
            affine.yield %84 : !air.async.token
          }
          %async_token_99 = air.execute {
            memref.dealloc %results_57 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_100 = air.execute {
            memref.dealloc %results_55 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_101 = air.execute [%85, %84, %async_token_58] {
            memref.dealloc %results_53 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_102 = air.execute [%async_token_60, %85] {
            memref.dealloc %results_51 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_103 = air.execute [%85, %84, %async_token_59] {
            memref.dealloc %results_49 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_39 = air.execute [%68] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_40 = air.execute [%67] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_41 = air.execute [%66] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_42 = air.execute [%65] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        air.wait_all [%async_token_17, %async_token_20, %async_token_23, %async_token_26, %async_token_29, %async_token_32, %async_token_35, %async_token_38, %69, %async_token_39, %async_token_40, %async_token_41, %async_token_42]  {air.segment_end}
      }
    }
    return
  }
}
