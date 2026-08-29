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
        %55 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %44) -> (!air.async.token) {
          %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %79 = air.channel.get async [%async_token_20, %arg17]  @VIn_0[%arg12] (%results_21[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %80 = arith.cmpi eq, %arg12, %c0_8 : index
          %81 = scf.if %80 -> (!air.async.token) {
            %82 = air.channel.put async [%async_token_20, %79]  @V2L1_0_0[%c0_8, %c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %82 : !air.async.token
          } else {
            %82 = air.channel.put async [%async_token_20, %79]  @V2L1_0_1[%c0_8, %c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %82 : !air.async.token
          }
          %async_token_22 = air.execute [%81, %79] {
            memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %81 : !air.async.token
        }
        %56 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %45) -> (!air.async.token) {
          %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %79 = air.channel.get async [%async_token_20, %arg17]  @VIn_1[%arg12] (%results_21[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
          %80 = arith.cmpi eq, %arg12, %c0_8 : index
          %81 = scf.if %80 -> (!air.async.token) {
            %82 = air.channel.put async [%async_token_20, %79]  @V2L1_1_0[%c0_8, %c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %82 : !air.async.token
          } else {
            %82 = air.channel.put async [%async_token_20, %79]  @V2L1_1_1[%c0_8, %c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %82 : !air.async.token
          }
          %async_token_22 = air.execute [%81, %79] {
            memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %81 : !air.async.token
        }
        %57 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %46) -> (!air.async.token) {
          %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %79 = air.channel.get async [%async_token_20, %arg17]  @VIn_2[%arg12] (%results_21[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %80 = arith.cmpi eq, %arg12, %c0_8 : index
          %81 = scf.if %80 -> (!air.async.token) {
            %82 = air.channel.put async [%async_token_20, %79]  @V2L1_2_0[%c0_8, %c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %82 : !air.async.token
          } else {
            %82 = air.channel.put async [%async_token_20, %79]  @V2L1_2_1[%c0_8, %c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %82 : !air.async.token
          }
          %async_token_22 = air.execute [%81, %79] {
            memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %81 : !air.async.token
        }
        %58 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %47) -> (!air.async.token) {
          %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %79 = air.channel.get async [%async_token_20, %arg17]  @VIn_3[%arg12] (%results_21[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
          %80 = arith.cmpi eq, %arg12, %c0_8 : index
          %81 = scf.if %80 -> (!air.async.token) {
            %82 = air.channel.put async [%async_token_20, %79]  @V2L1_3_0[%c0_8, %c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %82 : !air.async.token
          } else {
            %82 = air.channel.put async [%async_token_20, %79]  @V2L1_3_1[%c0_8, %c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %82 : !air.async.token
          }
          %async_token_22 = air.execute [%81, %79] {
            memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %81 : !air.async.token
        }
        %59 = air.channel.get async [%async_token]  @Gp2L2[%c0_8, %c0_8] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %60 = air.channel.get async [%async_token_10]  @Gp2L2[%c1_6, %c0_8] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %61 = air.channel.get async [%async_token_12]  @Gp2L2[%c2_7, %c0_8] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %62 = air.channel.get async [%async_token_14]  @Gp2L2[%c3_2, %c0_8] (%results_15[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %63 = air.channel.put async [%59]  @channel_0[%c0_8, %arg12] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %64 = air.channel.put async [%60]  @channel_0[%c1_6, %arg12] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %65 = air.channel.put async [%61]  @channel_0[%c2_7, %arg12] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %66 = air.channel.put async [%62]  @channel_0[%c3_2, %arg12] (%results_15[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %67 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_9, %arg19=%c4_9) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn.o"} {
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2_20 = arith.constant 2 : index
          %c0_21 = arith.constant 0 : index
          %c1_22 = arith.constant 1 : index
          %c8_23 = arith.constant 8 : index
          %c64_24 = arith.constant 64 : index
          %c512_25 = arith.constant 512 : index
          %async_token_26, %results_27 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_28, %results_29 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_30, %results_31 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %79 = air.wait_all async 
          %80 = air.wait_all async 
          %async_token_32, %results_33 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_34, %results_35 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_36 = air.execute [%async_token_30] {
            func.call @zero_fill_gp_bf16(%results_31) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_37 = air.execute [%async_token_26] {
            func.call @zero_fill_sp_bf16(%results_27) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_38 = air.execute [%async_token_28] {
            func.call @neg_inf_fill_up_bf16(%results_29) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %81 = arith.cmpi eq, %arg20, %c0_21 : index
          scf.if %81 {
            %92 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %93 = air.channel.get async [%async_token_32]  @QK2L1_0_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %93 : !air.async.token
            } else {
              %93 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %94 = air.channel.get async [%async_token_32]  @QK2L1_0_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %94 : !air.async.token
              } else {
                %94 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_0_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                } else {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_0_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                }
                affine.yield %94 : !air.async.token
              }
              affine.yield %93 : !air.async.token
            }
          } else {
            %92 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %93 = air.channel.get async [%async_token_32]  @QK2L1_1_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %93 : !air.async.token
            } else {
              %93 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %94 = air.channel.get async [%async_token_32]  @QK2L1_1_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %94 : !air.async.token
              } else {
                %94 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_1_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                } else {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_1_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                }
                affine.yield %94 : !air.async.token
              }
              affine.yield %93 : !air.async.token
            }
          }
          %82 = arith.index_cast %arg16 : index to i32
          %83 = arith.cmpi eq, %82, %c0_i32 : i32
          scf.if %83 {
            %async_token_44 = air.execute [%async_token_32, %async_token_34] {
              func.call @copy_tile(%results_33, %results_35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %81 {
            %92 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %93 = air.channel.get async [%async_token_32]  @QK2L1_0_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %93 : !air.async.token
            } else {
              %93 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %94 = air.channel.get async [%async_token_32]  @QK2L1_0_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %94 : !air.async.token
              } else {
                %94 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_0_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                } else {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_0_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                }
                affine.yield %94 : !air.async.token
              }
              affine.yield %93 : !air.async.token
            }
          } else {
            %92 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %93 = air.channel.get async [%async_token_32]  @QK2L1_1_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %93 : !air.async.token
            } else {
              %93 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %94 = air.channel.get async [%async_token_32]  @QK2L1_1_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %94 : !air.async.token
              } else {
                %94 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_1_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                } else {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_1_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                }
                affine.yield %94 : !air.async.token
              }
              affine.yield %93 : !air.async.token
            }
          }
          %84 = arith.cmpi eq, %82, %c1_i32 : i32
          scf.if %84 {
            %async_token_44 = air.execute [%async_token_32, %async_token_34] {
              func.call @copy_tile(%results_33, %results_35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %81 {
            %92 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %93 = air.channel.get async [%async_token_32]  @QK2L1_0_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %93 : !air.async.token
            } else {
              %93 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %94 = air.channel.get async [%async_token_32]  @QK2L1_0_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %94 : !air.async.token
              } else {
                %94 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_0_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                } else {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_0_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                }
                affine.yield %94 : !air.async.token
              }
              affine.yield %93 : !air.async.token
            }
          } else {
            %92 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %93 = air.channel.get async [%async_token_32]  @QK2L1_1_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %93 : !air.async.token
            } else {
              %93 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %94 = air.channel.get async [%async_token_32]  @QK2L1_1_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %94 : !air.async.token
              } else {
                %94 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_1_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                } else {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_1_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                }
                affine.yield %94 : !air.async.token
              }
              affine.yield %93 : !air.async.token
            }
          }
          %85 = arith.cmpi eq, %82, %c2_i32 : i32
          scf.if %85 {
            %async_token_44 = air.execute [%async_token_32, %async_token_34] {
              func.call @copy_tile(%results_33, %results_35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %81 {
            %92 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %93 = air.channel.get async [%async_token_32]  @QK2L1_0_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %93 : !air.async.token
            } else {
              %93 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %94 = air.channel.get async [%async_token_32]  @QK2L1_0_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %94 : !air.async.token
              } else {
                %94 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_0_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                } else {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_0_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                }
                affine.yield %94 : !air.async.token
              }
              affine.yield %93 : !air.async.token
            }
          } else {
            %92 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %93 = air.channel.get async [%async_token_32]  @QK2L1_1_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %93 : !air.async.token
            } else {
              %93 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %94 = air.channel.get async [%async_token_32]  @QK2L1_1_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %94 : !air.async.token
              } else {
                %94 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_1_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                } else {
                  %95 = air.channel.get async [%async_token_32]  @QK2L1_1_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %95 : !air.async.token
                }
                affine.yield %94 : !air.async.token
              }
              affine.yield %93 : !air.async.token
            }
          }
          %86 = arith.cmpi eq, %82, %c3_i32 : i32
          scf.if %86 {
            %async_token_44 = air.execute [%async_token_32, %async_token_34] {
              func.call @copy_tile(%results_33, %results_35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %87 = air.wait_all async [%async_token_36, %async_token_37, %async_token_38] 
          %88 = scf.for %arg21 = %c0_21 to %c2_20 step %c1_22 iter_args(%arg22 = %87) -> (!air.async.token) {
            %async_token_44, %results_45 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_46, %results_47 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_48 = air.execute [%async_token_46, %arg22] {
              %collapse_shape = memref.collapse_shape %results_47 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %81 {
              %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %98 = air.channel.get async [%async_token_32, %arg22]  @QK2L1_0_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %98 : !air.async.token
              } else {
                %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %99 = air.channel.get async [%async_token_32, %arg22]  @QK2L1_0_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %99 : !air.async.token
                } else {
                  %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %100 = air.channel.get async [%async_token_32, %arg22]  @QK2L1_0_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %100 : !air.async.token
                  } else {
                    %100 = air.channel.get async [%async_token_32, %arg22]  @QK2L1_0_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %100 : !air.async.token
                  }
                  affine.yield %99 : !air.async.token
                }
                affine.yield %98 : !air.async.token
              }
            } else {
              %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %98 = air.channel.get async [%async_token_32, %arg22]  @QK2L1_1_0[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %98 : !air.async.token
              } else {
                %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %99 = air.channel.get async [%async_token_32, %arg22]  @QK2L1_1_1[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %99 : !air.async.token
                } else {
                  %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %100 = air.channel.get async [%async_token_32, %arg22]  @QK2L1_1_2[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %100 : !air.async.token
                  } else {
                    %100 = air.channel.get async [%async_token_32, %arg22]  @QK2L1_1_3[%c0_21, %c0_21, %arg16] (%results_33[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %100 : !air.async.token
                  }
                  affine.yield %99 : !air.async.token
                }
                affine.yield %98 : !air.async.token
              }
            }
            %92 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %97 = scf.if %81 -> (!air.async.token) {
                %98 = air.channel.get async [%async_token_44]  @V2L1_0_0[%c0_21, %arg17, %arg16] (%results_45[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %98 : !air.async.token
              } else {
                %98 = air.channel.get async [%async_token_44]  @V2L1_0_1[%c0_21, %arg17, %arg16] (%results_45[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %98 : !air.async.token
              }
              affine.yield %97 : !air.async.token
            } else {
              %97 = air.wait_all async 
              affine.yield %97 : !air.async.token
            }
            %93 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %97 = scf.if %81 -> (!air.async.token) {
                %98 = air.channel.get async [%async_token_44, %arg22, %92]  @V2L1_1_0[%c0_21, %arg17, %arg16] (%results_45[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %98 : !air.async.token
              } else {
                %98 = air.channel.get async [%async_token_44, %arg22, %92]  @V2L1_1_1[%c0_21, %arg17, %arg16] (%results_45[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %98 : !air.async.token
              }
              affine.yield %97 : !air.async.token
            } else {
              %97 = air.wait_all async 
              affine.yield %97 : !air.async.token
            }
            %94 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %97 = scf.if %81 -> (!air.async.token) {
                %98 = air.channel.get async [%async_token_44, %arg22, %93]  @V2L1_2_0[%c0_21, %arg17, %arg16] (%results_45[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %98 : !air.async.token
              } else {
                %98 = air.channel.get async [%async_token_44, %arg22, %93]  @V2L1_2_1[%c0_21, %arg17, %arg16] (%results_45[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %98 : !air.async.token
              }
              affine.yield %97 : !air.async.token
            } else {
              %97 = air.wait_all async 
              affine.yield %97 : !air.async.token
            }
            %95 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %97 = scf.if %81 -> (!air.async.token) {
                %98 = air.channel.get async [%async_token_44, %arg22, %94]  @V2L1_3_0[%c0_21, %arg17, %arg16] (%results_45[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %98 : !air.async.token
              } else {
                %98 = air.channel.get async [%async_token_44, %arg22, %94]  @V2L1_3_1[%c0_21, %arg17, %arg16] (%results_45[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %98 : !air.async.token
              }
              affine.yield %97 : !air.async.token
            } else {
              %97 = air.wait_all async 
              affine.yield %97 : !air.async.token
            }
            %async_token_49 = air.execute [%async_token_46, %async_token_32, %async_token_34, %async_token_48] {
              %collapse_shape = memref.collapse_shape %results_47 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_35, %results_33, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_50, %results_51 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_52, %results_53 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_54 = air.execute [%async_token_28, %async_token_46, %async_token_52, %async_token_50, %async_token_49] {
              %collapse_shape = memref.collapse_shape %results_47 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_29, %results_51, %results_53) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_55 = air.execute [%async_token_30, %async_token_54] {
              func.call @mul_r_gp(%results_53, %results_31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_56 = air.execute [%async_token_30, %async_token_46, %async_token_44, %async_token_55, %95] {
              %collapse_shape = memref.collapse_shape %results_47 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_45, %results_31) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_57 = air.execute [%async_token_26, %async_token_55] {
              func.call @accum_sp_r_s(%results_27, %results_53, %results_51) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_58 = air.execute [%async_token_26, %async_token_57] {
              func.call @vector_copy_32elems(%c0_i32, %results_51, %results_27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_59 = air.execute [%async_token_58] {
              memref.dealloc %results_51 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_60 = air.execute [%async_token_57] {
              memref.dealloc %results_53 : memref<64x1xbf16, 2 : i32>
            }
            %96 = air.wait_all async [%92, %93, %94, %async_token_56, %async_token_58] 
            %async_token_61 = air.execute [%async_token_56, %async_token_54, %async_token_49, %async_token_48] {
              memref.dealloc %results_47 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_62 = air.execute [%async_token_56, %95, %94, %93, %92] {
              memref.dealloc %results_45 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %96 : !air.async.token
          }
          %89 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %92 = arith.subi %arg17, %c1_22 : index
            %93 = air.channel.put async [%async_token_30, %88]  @cascade_gp[%arg16, %92] (%results_31[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
            %94 = air.channel.put async [%async_token_28, %88]  @cascade_up[%arg16, %92] (%results_29[] [] []) {id = 102 : i32} : (memref<64x1xbf16, 2 : i32>)
            %95 = air.channel.put async [%async_token_26, %88]  @cascade_sp[%arg16, %92] (%results_27[] [] []) {id = 103 : i32} : (memref<64x1xbf16, 2 : i32>)
            %96 = air.wait_all async [%93, %94, %95] 
            affine.yield %96 : !air.async.token
          } else {
            %92 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_44, %results_45 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_46, %results_47 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_48, %results_49 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %93 = air.channel.get async [%async_token_44]  @cascade_gp[%arg16, %arg17] (%results_45[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              %94 = air.channel.get async [%async_token_46]  @cascade_up[%arg16, %arg17] (%results_47[] [] []) {id = 105 : i32} : (memref<64x1xbf16, 2 : i32>)
              %95 = air.channel.get async [%async_token_48]  @cascade_sp[%arg16, %arg17] (%results_49[] [] []) {id = 106 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_50, %results_51 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_52 = air.execute [%async_token_28, %async_token_50, %88] {
                func.call @vector_copy_32elems(%c0_i32, %results_29, %results_51) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_53 = air.execute [%async_token_28, %async_token_52, %94] {
                func.call @maximum_up_u_bf16(%results_47, %results_29) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56 = air.execute [%async_token_28, %async_token_54, %async_token_53] {
                func.call @exp_up_minus_u(%results_47, %results_29, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_57, %results_58 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_59 = air.execute [%async_token_28, %async_token_57, %async_token_56] {
                func.call @exp_up_minus_u(%results_51, %results_29, %results_58) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_60 = air.execute [%async_token_56, %93] {
                func.call @mul_r_gp(%results_55, %results_45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_30, %async_token_59] {
                func.call @mul_r_gp(%results_58, %results_31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_62 = air.execute [%async_token_30, %async_token_61, %async_token_60] {
                func.call @add_gp_g(%results_31, %results_45) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_65 = air.execute [%async_token_63] {
                func.call @zero_fill_sp_bf16(%results_64) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_65, %async_token_60, %95] {
                func.call @accum_sp_r_s(%results_49, %results_55, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67 = air.execute [%async_token_26, %async_token_66, %async_token_61] {
                func.call @accum_sp_r_s(%results_27, %results_58, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_68 = air.execute [%async_token_67] {
                func.call @vector_copy_32elems(%c0_i32, %results_64, %results_49) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %96 = arith.subi %arg17, %c1_22 : index
              %97 = air.channel.put async [%async_token_62]  @cascade_gp[%arg16, %96] (%results_45[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              %98 = air.channel.put async [%async_token_28, %async_token_59]  @cascade_up[%arg16, %96] (%results_29[] [] []) {id = 108 : i32} : (memref<64x1xbf16, 2 : i32>)
              %99 = air.channel.put async [%async_token_68]  @cascade_sp[%arg16, %96] (%results_49[] [] []) {id = 109 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_69 = air.execute [%97] {
                memref.dealloc %results_45 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%async_token_56] {
                memref.dealloc %results_47 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%99] {
                memref.dealloc %results_49 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_59] {
                memref.dealloc %results_51 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_66] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_74 = air.execute [%async_token_67] {
                memref.dealloc %results_58 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_68] {
                memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
              }
              %100 = air.wait_all async [%97, %98, %99] 
              affine.yield %100 : !air.async.token
            } else {
              %async_token_44, %results_45 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_46, %results_47 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_48, %results_49 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %93 = air.channel.get async [%async_token_44]  @cascade_gp[%arg16, %arg17] (%results_45[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              %94 = air.channel.get async [%async_token_46]  @cascade_up[%arg16, %arg17] (%results_47[] [] []) {id = 111 : i32} : (memref<64x1xbf16, 2 : i32>)
              %95 = air.channel.get async [%async_token_48]  @cascade_sp[%arg16, %arg17] (%results_49[] [] []) {id = 112 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_50, %results_51 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_52 = air.execute [%async_token_28, %async_token_50, %88] {
                func.call @vector_copy_32elems(%c0_i32, %results_29, %results_51) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_53 = air.execute [%async_token_28, %async_token_52, %94] {
                func.call @maximum_up_u_bf16(%results_47, %results_29) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56 = air.execute [%async_token_28, %async_token_54, %async_token_53] {
                func.call @exp_up_minus_u(%results_47, %results_29, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_57, %results_58 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_59 = air.execute [%async_token_28, %async_token_57, %async_token_56] {
                func.call @exp_up_minus_u(%results_51, %results_29, %results_58) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_60 = air.execute [%async_token_56, %93] {
                func.call @mul_r_gp(%results_55, %results_45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_30, %async_token_59] {
                func.call @mul_r_gp(%results_58, %results_31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_62 = air.execute [%async_token_30, %async_token_61, %async_token_60] {
                func.call @add_gp_g(%results_31, %results_45) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_65 = air.execute [%async_token_63] {
                func.call @zero_fill_sp_bf16(%results_64) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_65, %async_token_60, %95] {
                func.call @accum_sp_r_s(%results_49, %results_55, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67 = air.execute [%async_token_26, %async_token_66, %async_token_61] {
                func.call @accum_sp_r_s(%results_27, %results_58, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_68 = air.execute [%async_token_67] {
                func.call @vector_copy_32elems(%c0_i32, %results_64, %results_49) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_69 = air.execute [%async_token_68, %async_token_62] {
                func.call @div_gp_sp(%results_49, %results_45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %96 = air.channel.put async [%async_token_69]  @Gp2L2[%arg16, %c0_21] (%results_45[%c0_21, %c0_21, %c0_21, %c0_21] [%c8_23, %c8_23, %c8_23, %c8_23] [%c64_24, %c8_23, %c512_25, %c1_22]) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_70 = air.execute [%96] {
                memref.dealloc %results_45 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%async_token_56] {
                memref.dealloc %results_47 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_69] {
                memref.dealloc %results_49 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_59] {
                memref.dealloc %results_51 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_74 = air.execute [%async_token_66] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_67] {
                memref.dealloc %results_58 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_76 = air.execute [%async_token_68] {
                memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %96 : !air.async.token
            }
            affine.yield %88 : !air.async.token
          }
          %async_token_39 = air.execute [%88] {
            memref.dealloc %results_35 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_40 = air.execute [%88] {
            memref.dealloc %results_33 : memref<64x64xbf16, 2 : i32>
          }
          %90 = air.wait_all async 
          %91 = air.wait_all async 
          %async_token_41 = air.execute [%89, %88, %async_token_36] {
            memref.dealloc %results_31 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_42 = air.execute [%89, %88, %async_token_38] {
            memref.dealloc %results_29 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_43 = air.execute [%89, %88, %async_token_37] {
            memref.dealloc %results_27 : memref<64x1xbf16, 2 : i32>
          }
        }
        %68 = air.wait_all async 
        %69 = air.wait_all async 
        %70 = air.wait_all async 
        %71 = air.wait_all async 
        %72 = air.wait_all async 
        %73 = air.wait_all async 
        %74 = air.wait_all async 
        %75 = air.wait_all async 
        %76 = air.wait_all async 
        %77 = air.wait_all async 
        %78 = air.wait_all async 
        %async_token_16 = air.execute [%66] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17 = air.execute [%65] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18 = air.execute [%64] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_19 = air.execute [%63] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
