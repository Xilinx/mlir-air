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
      %43 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2_0, %arg15=%c1_1) attributes {id = 2 : i32} {
        %c3_2 = arith.constant 3 : index
        %c64_3 = arith.constant 64 : index
        %c512_4 = arith.constant 512 : index
        %c8_5 = arith.constant 8 : index
        %c1_6 = arith.constant 1 : index
        %c2_7 = arith.constant 2 : index
        %c0_8 = arith.constant 0 : index
        %c4_9 = arith.constant 4 : index
        %44 = air.wait_all async 
        %45 = air.wait_all async 
        %46 = air.wait_all async 
        %47 = air.wait_all async 
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
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
        %48 = air.wait_all async 
        %49 = air.wait_all async 
        %50 = air.wait_all async 
        %51 = air.wait_all async 
        %52 = air.wait_all async 
        %53 = air.wait_all async 
        %54 = air.wait_all async 
        %async_token_16, %results_17 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %55 = air.channel.get async [%44, %async_token_16]  @VIn_0[%arg12] (%results_17[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
        %56 = arith.cmpi eq, %arg12, %c0_8 : index
        %57 = scf.if %56 -> (!air.async.token) {
          %99 = air.channel.put async [%async_token_16, %55]  @V2L1_0_0[%c0_8, %c0_8, %c0_8] (%results_17[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        } else {
          %99 = air.channel.put async [%async_token_16, %55]  @V2L1_0_1[%c0_8, %c0_8, %c0_8] (%results_17[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        }
        %async_token_18 = air.execute [%57, %55] {
          memref.dealloc %results_17 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %58 = air.channel.get async [%57, %async_token_19]  @VIn_0[%arg12] (%results_20[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
        %59 = arith.cmpi eq, %arg12, %c0_8 : index
        %60 = scf.if %59 -> (!air.async.token) {
          %99 = air.channel.put async [%async_token_19, %58]  @V2L1_0_0[%c0_8, %c0_8, %c0_8] (%results_20[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        } else {
          %99 = air.channel.put async [%async_token_19, %58]  @V2L1_0_1[%c0_8, %c0_8, %c0_8] (%results_20[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        }
        %async_token_21 = air.execute [%60, %58] {
          memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_22, %results_23 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %61 = air.channel.get async [%45, %async_token_22]  @VIn_1[%arg12] (%results_23[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
        %62 = arith.cmpi eq, %arg12, %c0_8 : index
        %63 = scf.if %62 -> (!air.async.token) {
          %99 = air.channel.put async [%async_token_22, %61]  @V2L1_1_0[%c0_8, %c0_8, %c0_8] (%results_23[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        } else {
          %99 = air.channel.put async [%async_token_22, %61]  @V2L1_1_1[%c0_8, %c0_8, %c0_8] (%results_23[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        }
        %async_token_24 = air.execute [%63, %61] {
          memref.dealloc %results_23 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_25, %results_26 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %64 = air.channel.get async [%63, %async_token_25]  @VIn_1[%arg12] (%results_26[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
        %65 = arith.cmpi eq, %arg12, %c0_8 : index
        %66 = scf.if %65 -> (!air.async.token) {
          %99 = air.channel.put async [%async_token_25, %64]  @V2L1_1_0[%c0_8, %c0_8, %c0_8] (%results_26[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        } else {
          %99 = air.channel.put async [%async_token_25, %64]  @V2L1_1_1[%c0_8, %c0_8, %c0_8] (%results_26[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        }
        %async_token_27 = air.execute [%66, %64] {
          memref.dealloc %results_26 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_28, %results_29 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %67 = air.channel.get async [%46, %async_token_28]  @VIn_2[%arg12] (%results_29[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
        %68 = arith.cmpi eq, %arg12, %c0_8 : index
        %69 = scf.if %68 -> (!air.async.token) {
          %99 = air.channel.put async [%async_token_28, %67]  @V2L1_2_0[%c0_8, %c0_8, %c0_8] (%results_29[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        } else {
          %99 = air.channel.put async [%async_token_28, %67]  @V2L1_2_1[%c0_8, %c0_8, %c0_8] (%results_29[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        }
        %async_token_30 = air.execute [%69, %67] {
          memref.dealloc %results_29 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_31, %results_32 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %70 = air.channel.get async [%69, %async_token_31]  @VIn_2[%arg12] (%results_32[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
        %71 = arith.cmpi eq, %arg12, %c0_8 : index
        %72 = scf.if %71 -> (!air.async.token) {
          %99 = air.channel.put async [%async_token_31, %70]  @V2L1_2_0[%c0_8, %c0_8, %c0_8] (%results_32[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        } else {
          %99 = air.channel.put async [%async_token_31, %70]  @V2L1_2_1[%c0_8, %c0_8, %c0_8] (%results_32[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        }
        %async_token_33 = air.execute [%72, %70] {
          memref.dealloc %results_32 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_34, %results_35 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %73 = air.channel.get async [%47, %async_token_34]  @VIn_3[%arg12] (%results_35[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
        %74 = arith.cmpi eq, %arg12, %c0_8 : index
        %75 = scf.if %74 -> (!air.async.token) {
          %99 = air.channel.put async [%async_token_34, %73]  @V2L1_3_0[%c0_8, %c0_8, %c0_8] (%results_35[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        } else {
          %99 = air.channel.put async [%async_token_34, %73]  @V2L1_3_1[%c0_8, %c0_8, %c0_8] (%results_35[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        }
        %async_token_36 = air.execute [%75, %73] {
          memref.dealloc %results_35 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_37, %results_38 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %76 = air.channel.get async [%75, %async_token_37]  @VIn_3[%arg12] (%results_38[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
        %77 = arith.cmpi eq, %arg12, %c0_8 : index
        %78 = scf.if %77 -> (!air.async.token) {
          %99 = air.channel.put async [%async_token_37, %76]  @V2L1_3_0[%c0_8, %c0_8, %c0_8] (%results_38[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        } else {
          %99 = air.channel.put async [%async_token_37, %76]  @V2L1_3_1[%c0_8, %c0_8, %c0_8] (%results_38[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %99 : !air.async.token
        }
        %async_token_39 = air.execute [%78, %76] {
          memref.dealloc %results_38 : memref<64x64xbf16, 1 : i32>
        }
        %79 = air.channel.get async [%async_token]  @Gp2L2[%c0_8, %c0_8] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %80 = air.channel.get async [%async_token_10]  @Gp2L2[%c1_6, %c0_8] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %81 = air.channel.get async [%async_token_12]  @Gp2L2[%c2_7, %c0_8] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %82 = air.channel.get async [%async_token_14]  @Gp2L2[%c3_2, %c0_8] (%results_15[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %83 = air.channel.put async [%79]  @channel_0[%c0_8, %arg12] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %84 = air.channel.put async [%80]  @channel_0[%c1_6, %arg12] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %85 = air.channel.put async [%81]  @channel_0[%c2_7, %arg12] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %86 = air.channel.put async [%82]  @channel_0[%c3_2, %arg12] (%results_15[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %87 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_9, %arg19=%c4_9) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn.o"} {
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c0_44 = arith.constant 0 : index
          %c1_45 = arith.constant 1 : index
          %c8_46 = arith.constant 8 : index
          %c64_47 = arith.constant 64 : index
          %c512_48 = arith.constant 512 : index
          %async_token_49, %results_50 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_51, %results_52 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_53, %results_54 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %99 = air.wait_all async 
          %100 = air.wait_all async 
          %async_token_55, %results_56 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_57, %results_58 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_59 = air.execute [%async_token_53] {
            func.call @zero_fill_gp_bf16(%results_54) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_60 = air.execute [%async_token_49] {
            func.call @zero_fill_sp_bf16(%results_50) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_61 = air.execute [%async_token_51] {
            func.call @neg_inf_fill_up_bf16(%results_52) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %101 = arith.cmpi eq, %arg20, %c0_44 : index
          scf.if %101 {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          } else {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          }
          %102 = arith.index_cast %arg16 : index to i32
          %103 = arith.cmpi eq, %102, %c0_i32 : i32
          scf.if %103 {
            %async_token_105 = air.execute [%async_token_55, %async_token_57] {
              func.call @copy_tile(%results_56, %results_58) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %101 {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          } else {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          }
          %104 = arith.cmpi eq, %102, %c1_i32 : i32
          scf.if %104 {
            %async_token_105 = air.execute [%async_token_55, %async_token_57] {
              func.call @copy_tile(%results_56, %results_58) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %101 {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          } else {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          }
          %105 = arith.cmpi eq, %102, %c2_i32 : i32
          scf.if %105 {
            %async_token_105 = air.execute [%async_token_55, %async_token_57] {
              func.call @copy_tile(%results_56, %results_58) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %101 {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          } else {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          }
          %106 = arith.cmpi eq, %102, %c3_i32 : i32
          scf.if %106 {
            %async_token_105 = air.execute [%async_token_55, %async_token_57] {
              func.call @copy_tile(%results_56, %results_58) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %107 = air.wait_all async [%async_token_59, %async_token_60, %async_token_61] 
          %async_token_62, %results_63 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_64, %results_65 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_66 = air.execute [%async_token_64, %107] {
            %collapse_shape = memref.collapse_shape %results_65 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          scf.if %101 {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55, %107]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55, %107]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55, %107]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55, %107]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          } else {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55, %107]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55, %107]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55, %107]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55, %107]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          }
          %108 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %121 = scf.if %101 -> (!air.async.token) {
              %122 = air.channel.get async [%async_token_62]  @V2L1_0_0[%c0_44, %arg17, %arg16] (%results_63[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            } else {
              %122 = air.channel.get async [%async_token_62]  @V2L1_0_1[%c0_44, %arg17, %arg16] (%results_63[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            }
            affine.yield %121 : !air.async.token
          } else {
            %121 = air.wait_all async 
            affine.yield %121 : !air.async.token
          }
          %109 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %121 = scf.if %101 -> (!air.async.token) {
              %122 = air.channel.get async [%107, %async_token_62, %108]  @V2L1_1_0[%c0_44, %arg17, %arg16] (%results_63[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            } else {
              %122 = air.channel.get async [%107, %async_token_62, %108]  @V2L1_1_1[%c0_44, %arg17, %arg16] (%results_63[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            }
            affine.yield %121 : !air.async.token
          } else {
            %121 = air.wait_all async 
            affine.yield %121 : !air.async.token
          }
          %110 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %121 = scf.if %101 -> (!air.async.token) {
              %122 = air.channel.get async [%107, %async_token_62, %109]  @V2L1_2_0[%c0_44, %arg17, %arg16] (%results_63[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            } else {
              %122 = air.channel.get async [%107, %async_token_62, %109]  @V2L1_2_1[%c0_44, %arg17, %arg16] (%results_63[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            }
            affine.yield %121 : !air.async.token
          } else {
            %121 = air.wait_all async 
            affine.yield %121 : !air.async.token
          }
          %111 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %121 = scf.if %101 -> (!air.async.token) {
              %122 = air.channel.get async [%107, %async_token_62, %110]  @V2L1_3_0[%c0_44, %arg17, %arg16] (%results_63[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            } else {
              %122 = air.channel.get async [%107, %async_token_62, %110]  @V2L1_3_1[%c0_44, %arg17, %arg16] (%results_63[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            }
            affine.yield %121 : !air.async.token
          } else {
            %121 = air.wait_all async 
            affine.yield %121 : !air.async.token
          }
          %async_token_67 = air.execute [%async_token_64, %async_token_55, %async_token_57, %async_token_66] {
            %collapse_shape = memref.collapse_shape %results_65 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%results_58, %results_56, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_70, %results_71 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_72 = air.execute [%async_token_51, %async_token_64, %async_token_70, %async_token_68, %async_token_67] {
            %collapse_shape = memref.collapse_shape %results_65 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %results_52, %results_69, %results_71) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_73 = air.execute [%async_token_53, %async_token_72] {
            func.call @mul_r_gp(%results_71, %results_54) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_74 = air.execute [%async_token_53, %async_token_64, %async_token_62, %async_token_73, %111] {
            %collapse_shape = memref.collapse_shape %results_65 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %results_63, %results_54) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_75 = air.execute [%async_token_49, %async_token_73] {
            func.call @accum_sp_r_s(%results_50, %results_71, %results_69) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_76 = air.execute [%async_token_49, %async_token_75] {
            func.call @vector_copy_32elems(%c0_i32, %results_69, %results_50) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_77 = air.execute [%async_token_76] {
            memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_78 = air.execute [%async_token_75] {
            memref.dealloc %results_71 : memref<64x1xbf16, 2 : i32>
          }
          %112 = air.wait_all async [%108, %109, %110, %async_token_74, %async_token_76] 
          %async_token_79 = air.execute [%async_token_74, %async_token_72, %async_token_67, %async_token_66] {
            memref.dealloc %results_65 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_80 = air.execute [%async_token_74, %111, %110, %109, %108] {
            memref.dealloc %results_63 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_81, %results_82 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_83, %results_84 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_85 = air.execute [%async_token_83, %112] {
            %collapse_shape = memref.collapse_shape %results_84 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          scf.if %101 {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55, %112]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55, %112]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55, %112]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55, %112]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          } else {
            %121 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %122 = air.channel.get async [%async_token_55, %112]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %122 : !air.async.token
            } else {
              %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %123 = air.channel.get async [%async_token_55, %112]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %123 : !air.async.token
              } else {
                %123 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %124 = air.channel.get async [%async_token_55, %112]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                } else {
                  %124 = air.channel.get async [%async_token_55, %112]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_56[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %124 : !air.async.token
                }
                affine.yield %123 : !air.async.token
              }
              affine.yield %122 : !air.async.token
            }
          }
          %113 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %121 = scf.if %101 -> (!air.async.token) {
              %122 = air.channel.get async [%async_token_81]  @V2L1_0_0[%c0_44, %arg17, %arg16] (%results_82[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            } else {
              %122 = air.channel.get async [%async_token_81]  @V2L1_0_1[%c0_44, %arg17, %arg16] (%results_82[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            }
            affine.yield %121 : !air.async.token
          } else {
            %121 = air.wait_all async 
            affine.yield %121 : !air.async.token
          }
          %114 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %121 = scf.if %101 -> (!air.async.token) {
              %122 = air.channel.get async [%112, %async_token_81, %113]  @V2L1_1_0[%c0_44, %arg17, %arg16] (%results_82[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            } else {
              %122 = air.channel.get async [%112, %async_token_81, %113]  @V2L1_1_1[%c0_44, %arg17, %arg16] (%results_82[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            }
            affine.yield %121 : !air.async.token
          } else {
            %121 = air.wait_all async 
            affine.yield %121 : !air.async.token
          }
          %115 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %121 = scf.if %101 -> (!air.async.token) {
              %122 = air.channel.get async [%112, %async_token_81, %114]  @V2L1_2_0[%c0_44, %arg17, %arg16] (%results_82[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            } else {
              %122 = air.channel.get async [%112, %async_token_81, %114]  @V2L1_2_1[%c0_44, %arg17, %arg16] (%results_82[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            }
            affine.yield %121 : !air.async.token
          } else {
            %121 = air.wait_all async 
            affine.yield %121 : !air.async.token
          }
          %116 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %121 = scf.if %101 -> (!air.async.token) {
              %122 = air.channel.get async [%112, %async_token_81, %115]  @V2L1_3_0[%c0_44, %arg17, %arg16] (%results_82[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            } else {
              %122 = air.channel.get async [%112, %async_token_81, %115]  @V2L1_3_1[%c0_44, %arg17, %arg16] (%results_82[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %122 : !air.async.token
            }
            affine.yield %121 : !air.async.token
          } else {
            %121 = air.wait_all async 
            affine.yield %121 : !air.async.token
          }
          %async_token_86 = air.execute [%async_token_83, %async_token_55, %async_token_57, %async_token_85] {
            %collapse_shape = memref.collapse_shape %results_84 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%results_58, %results_56, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %async_token_87, %results_88 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_89, %results_90 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_91 = air.execute [%async_token_51, %async_token_83, %async_token_89, %async_token_87, %async_token_86] {
            %collapse_shape = memref.collapse_shape %results_84 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %results_52, %results_88, %results_90) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_92 = air.execute [%async_token_53, %async_token_91] {
            func.call @mul_r_gp(%results_90, %results_54) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_93 = air.execute [%async_token_53, %async_token_83, %async_token_81, %async_token_92, %116] {
            %collapse_shape = memref.collapse_shape %results_84 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %results_82, %results_54) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_94 = air.execute [%async_token_49, %async_token_92] {
            func.call @accum_sp_r_s(%results_50, %results_90, %results_88) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_95 = air.execute [%async_token_49, %async_token_94] {
            func.call @vector_copy_32elems(%c0_i32, %results_88, %results_50) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_96 = air.execute [%async_token_95] {
            memref.dealloc %results_88 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_97 = air.execute [%async_token_94] {
            memref.dealloc %results_90 : memref<64x1xbf16, 2 : i32>
          }
          %117 = air.wait_all async [%113, %114, %115, %async_token_93, %async_token_95] 
          %async_token_98 = air.execute [%async_token_93, %async_token_91, %async_token_86, %async_token_85] {
            memref.dealloc %results_84 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_99 = air.execute [%async_token_93, %116, %115, %114, %113] {
            memref.dealloc %results_82 : memref<64x64xbf16, 2 : i32>
          }
          %118 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %121 = arith.subi %arg17, %c1_45 : index
            %122 = air.channel.put async [%async_token_53, %117]  @cascade_gp[%arg16, %121] (%results_54[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
            %123 = air.channel.put async [%async_token_51, %117]  @cascade_up[%arg16, %121] (%results_52[] [] []) {id = 102 : i32} : (memref<64x1xbf16, 2 : i32>)
            %124 = air.channel.put async [%async_token_49, %117]  @cascade_sp[%arg16, %121] (%results_50[] [] []) {id = 103 : i32} : (memref<64x1xbf16, 2 : i32>)
            %125 = air.wait_all async [%122, %123, %124] 
            affine.yield %125 : !air.async.token
          } else {
            %121 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_105, %results_106 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_107, %results_108 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_109, %results_110 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %122 = air.channel.get async [%async_token_105]  @cascade_gp[%arg16, %arg17] (%results_106[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              %123 = air.channel.get async [%async_token_107]  @cascade_up[%arg16, %arg17] (%results_108[] [] []) {id = 105 : i32} : (memref<64x1xbf16, 2 : i32>)
              %124 = air.channel.get async [%async_token_109]  @cascade_sp[%arg16, %arg17] (%results_110[] [] []) {id = 106 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_111, %results_112 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_113 = air.execute [%async_token_51, %async_token_111, %117] {
                func.call @vector_copy_32elems(%c0_i32, %results_52, %results_112) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_114 = air.execute [%async_token_51, %async_token_113, %123] {
                func.call @maximum_up_u_bf16(%results_108, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_115, %results_116 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_117 = air.execute [%async_token_51, %async_token_115, %async_token_114] {
                func.call @exp_up_minus_u(%results_108, %results_52, %results_116) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_118, %results_119 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_120 = air.execute [%async_token_51, %async_token_118, %async_token_117] {
                func.call @exp_up_minus_u(%results_112, %results_52, %results_119) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_121 = air.execute [%async_token_117, %122] {
                func.call @mul_r_gp(%results_116, %results_106) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_122 = air.execute [%async_token_53, %async_token_120] {
                func.call @mul_r_gp(%results_119, %results_54) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_123 = air.execute [%async_token_53, %async_token_122, %async_token_121] {
                func.call @add_gp_g(%results_54, %results_106) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_124, %results_125 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_126 = air.execute [%async_token_124] {
                func.call @zero_fill_sp_bf16(%results_125) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127 = air.execute [%async_token_126, %async_token_121, %124] {
                func.call @accum_sp_r_s(%results_110, %results_116, %results_125) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_128 = air.execute [%async_token_49, %async_token_127, %async_token_122] {
                func.call @accum_sp_r_s(%results_50, %results_119, %results_125) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_129 = air.execute [%async_token_128] {
                func.call @vector_copy_32elems(%c0_i32, %results_125, %results_110) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %125 = arith.subi %arg17, %c1_45 : index
              %126 = air.channel.put async [%async_token_123]  @cascade_gp[%arg16, %125] (%results_106[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              %127 = air.channel.put async [%async_token_51, %async_token_120]  @cascade_up[%arg16, %125] (%results_52[] [] []) {id = 108 : i32} : (memref<64x1xbf16, 2 : i32>)
              %128 = air.channel.put async [%async_token_129]  @cascade_sp[%arg16, %125] (%results_110[] [] []) {id = 109 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_130 = air.execute [%126] {
                memref.dealloc %results_106 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_131 = air.execute [%async_token_117] {
                memref.dealloc %results_108 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_132 = air.execute [%128] {
                memref.dealloc %results_110 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_133 = air.execute [%async_token_120] {
                memref.dealloc %results_112 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_134 = air.execute [%async_token_127] {
                memref.dealloc %results_116 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_135 = air.execute [%async_token_128] {
                memref.dealloc %results_119 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_136 = air.execute [%async_token_129] {
                memref.dealloc %results_125 : memref<64x1xbf16, 2 : i32>
              }
              %129 = air.wait_all async [%126, %127, %128] 
              affine.yield %129 : !air.async.token
            } else {
              %async_token_105, %results_106 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_107, %results_108 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_109, %results_110 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %122 = air.channel.get async [%async_token_105]  @cascade_gp[%arg16, %arg17] (%results_106[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              %123 = air.channel.get async [%async_token_107]  @cascade_up[%arg16, %arg17] (%results_108[] [] []) {id = 111 : i32} : (memref<64x1xbf16, 2 : i32>)
              %124 = air.channel.get async [%async_token_109]  @cascade_sp[%arg16, %arg17] (%results_110[] [] []) {id = 112 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_111, %results_112 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_113 = air.execute [%async_token_51, %async_token_111, %117] {
                func.call @vector_copy_32elems(%c0_i32, %results_52, %results_112) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_114 = air.execute [%async_token_51, %async_token_113, %123] {
                func.call @maximum_up_u_bf16(%results_108, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_115, %results_116 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_117 = air.execute [%async_token_51, %async_token_115, %async_token_114] {
                func.call @exp_up_minus_u(%results_108, %results_52, %results_116) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_118, %results_119 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_120 = air.execute [%async_token_51, %async_token_118, %async_token_117] {
                func.call @exp_up_minus_u(%results_112, %results_52, %results_119) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_121 = air.execute [%async_token_117, %122] {
                func.call @mul_r_gp(%results_116, %results_106) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_122 = air.execute [%async_token_53, %async_token_120] {
                func.call @mul_r_gp(%results_119, %results_54) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_123 = air.execute [%async_token_53, %async_token_122, %async_token_121] {
                func.call @add_gp_g(%results_54, %results_106) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_124, %results_125 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_126 = air.execute [%async_token_124] {
                func.call @zero_fill_sp_bf16(%results_125) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127 = air.execute [%async_token_126, %async_token_121, %124] {
                func.call @accum_sp_r_s(%results_110, %results_116, %results_125) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_128 = air.execute [%async_token_49, %async_token_127, %async_token_122] {
                func.call @accum_sp_r_s(%results_50, %results_119, %results_125) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_129 = air.execute [%async_token_128] {
                func.call @vector_copy_32elems(%c0_i32, %results_125, %results_110) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_130 = air.execute [%async_token_129, %async_token_123] {
                func.call @div_gp_sp(%results_110, %results_106) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %125 = air.channel.put async [%async_token_130]  @Gp2L2[%arg16, %c0_44] (%results_106[%c0_44, %c0_44, %c0_44, %c0_44] [%c8_46, %c8_46, %c8_46, %c8_46] [%c64_47, %c8_46, %c512_48, %c1_45]) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_131 = air.execute [%125] {
                memref.dealloc %results_106 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_132 = air.execute [%async_token_117] {
                memref.dealloc %results_108 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_133 = air.execute [%async_token_130] {
                memref.dealloc %results_110 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_134 = air.execute [%async_token_120] {
                memref.dealloc %results_112 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_135 = air.execute [%async_token_127] {
                memref.dealloc %results_116 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_136 = air.execute [%async_token_128] {
                memref.dealloc %results_119 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_137 = air.execute [%async_token_129] {
                memref.dealloc %results_125 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %125 : !air.async.token
            }
            affine.yield %117 : !air.async.token
          }
          %async_token_100 = air.execute [%117] {
            memref.dealloc %results_58 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_101 = air.execute [%117] {
            memref.dealloc %results_56 : memref<64x64xbf16, 2 : i32>
          }
          %119 = air.wait_all async 
          %120 = air.wait_all async 
          %async_token_102 = air.execute [%118, %117, %async_token_59] {
            memref.dealloc %results_54 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_103 = air.execute [%118, %117, %async_token_61] {
            memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_104 = air.execute [%118, %117, %async_token_60] {
            memref.dealloc %results_50 : memref<64x1xbf16, 2 : i32>
          }
        }
        %88 = air.wait_all async 
        %89 = air.wait_all async 
        %90 = air.wait_all async 
        %91 = air.wait_all async 
        %92 = air.wait_all async 
        %93 = air.wait_all async 
        %94 = air.wait_all async 
        %95 = air.wait_all async 
        %96 = air.wait_all async 
        %97 = air.wait_all async 
        %98 = air.wait_all async 
        %async_token_40 = air.execute [%86] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_41 = air.execute [%85] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_42 = air.execute [%84] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_43 = air.execute [%83] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
