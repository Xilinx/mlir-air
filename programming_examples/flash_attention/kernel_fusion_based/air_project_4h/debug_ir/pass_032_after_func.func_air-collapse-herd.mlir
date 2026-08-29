#map = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 262144)>
#map2 = affine_map<()[s0] -> (s0 * 262144 + 32768)>
#map3 = affine_map<()[s0] -> (s0 * 262144 + 65536)>
#map4 = affine_map<()[s0] -> (s0 * 262144 + 98304)>
#map5 = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 16384 + 131072)>
#map6 = affine_map<()[s0] -> (s0 * 262144 + 131072)>
#map7 = affine_map<()[s0] -> (s0 * 262144 + 163840)>
#map8 = affine_map<()[s0] -> (s0 * 262144 + 196608)>
#map9 = affine_map<()[s0] -> (s0 * 262144 + 229376)>
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
  func.func @attention_bf16(%arg0: memref<12x2048x64xbf16>, %arg1: memref<12x2048x64xbf16>, %arg2: memref<12x2048x64xbf16>, %arg3: memref<12x2048x64xbf16>) {
    %c8 = arith.constant 8 : index
    %c6 = arith.constant 6 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c8, %arg7=%c6) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<12x2048x64xbf16>, memref<12x2048x64xbf16>, memref<12x2048x64xbf16>, memref<12x2048x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8_0 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QK2L1_0_0[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 1 : i32} : (memref<12x2048x64xbf16>)
      %3 = air.channel.put async  @QK2L1_0_1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 2 : i32} : (memref<12x2048x64xbf16>)
      %4 = air.channel.put async  @QK2L1_0_2[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 3 : i32} : (memref<12x2048x64xbf16>)
      %5 = air.channel.put async  @QK2L1_0_3[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 4 : i32} : (memref<12x2048x64xbf16>)
      %6 = affine.apply #map1()[%arg5]
      %7 = air.channel.put async  @QK2L1_0_0[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %6] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 5 : i32} : (memref<12x2048x64xbf16>)
      %8 = affine.apply #map2()[%arg5]
      %9 = air.channel.put async  @QK2L1_0_1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %8] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 6 : i32} : (memref<12x2048x64xbf16>)
      %10 = affine.apply #map3()[%arg5]
      %11 = air.channel.put async  @QK2L1_0_2[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %10] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 7 : i32} : (memref<12x2048x64xbf16>)
      %12 = affine.apply #map4()[%arg5]
      %13 = air.channel.put async  @QK2L1_0_3[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %12] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 8 : i32} : (memref<12x2048x64xbf16>)
      %14 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %6] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 9 : i32} : (memref<12x2048x64xbf16>)
      %15 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %8] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 10 : i32} : (memref<12x2048x64xbf16>)
      %16 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %10] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 11 : i32} : (memref<12x2048x64xbf16>)
      %17 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %12] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 12 : i32} : (memref<12x2048x64xbf16>)
      %18 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %1] [%c64, %c64] [%c64, %c1]) {id = 13 : i32} : (memref<12x2048x64xbf16>)
      %19 = air.channel.get async  @channel_0[%c1, %c0] (%arg11[%c64, %1] [%c64, %c64] [%c64, %c1]) {id = 14 : i32} : (memref<12x2048x64xbf16>)
      %20 = air.channel.get async  @channel_0[%c2, %c0] (%arg11[%c128, %1] [%c64, %c64] [%c64, %c1]) {id = 15 : i32} : (memref<12x2048x64xbf16>)
      %21 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %1] [%c64, %c64] [%c64, %c1]) {id = 16 : i32} : (memref<12x2048x64xbf16>)
      %22 = affine.apply #map5()[%arg5, %arg4]
      %23 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 17 : i32} : (memref<12x2048x64xbf16>)
      %24 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 18 : i32} : (memref<12x2048x64xbf16>)
      %25 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 19 : i32} : (memref<12x2048x64xbf16>)
      %26 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 20 : i32} : (memref<12x2048x64xbf16>)
      %27 = affine.apply #map6()[%arg5]
      %28 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %27] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 21 : i32} : (memref<12x2048x64xbf16>)
      %29 = affine.apply #map7()[%arg5]
      %30 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %29] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 22 : i32} : (memref<12x2048x64xbf16>)
      %31 = affine.apply #map8()[%arg5]
      %32 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %31] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 23 : i32} : (memref<12x2048x64xbf16>)
      %33 = affine.apply #map9()[%arg5]
      %34 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %33] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 24 : i32} : (memref<12x2048x64xbf16>)
      %35 = air.channel.put async  @VIn_0[%c1] (%arg10[%c0, %c0, %27] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 25 : i32} : (memref<12x2048x64xbf16>)
      %36 = air.channel.put async  @VIn_1[%c1] (%arg10[%c0, %c0, %29] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 26 : i32} : (memref<12x2048x64xbf16>)
      %37 = air.channel.put async  @VIn_2[%c1] (%arg10[%c0, %c0, %31] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 27 : i32} : (memref<12x2048x64xbf16>)
      %38 = air.channel.put async  @VIn_3[%c1] (%arg10[%c0, %c0, %33] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 28 : i32} : (memref<12x2048x64xbf16>)
      %39 = air.channel.get async  @channel_0[%c0, %c1] (%arg11[%c0, %22] [%c64, %c64] [%c64, %c1]) {id = 29 : i32} : (memref<12x2048x64xbf16>)
      %40 = air.channel.get async  @channel_0[%c1, %c1] (%arg11[%c64, %22] [%c64, %c64] [%c64, %c1]) {id = 30 : i32} : (memref<12x2048x64xbf16>)
      %41 = air.channel.get async  @channel_0[%c2, %c1] (%arg11[%c128, %22] [%c64, %c64] [%c64, %c1]) {id = 31 : i32} : (memref<12x2048x64xbf16>)
      %42 = air.channel.get async  @channel_0[%c3, %c1] (%arg11[%c192, %22] [%c64, %c64] [%c64, %c1]) {id = 32 : i32} : (memref<12x2048x64xbf16>)
      %43 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1) attributes {id = 2 : i32} {
        %c3_1 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c64_3 = arith.constant 64 : index
        %c1_4 = arith.constant 1 : index
        %c8_5 = arith.constant 8 : index
        %c0_6 = arith.constant 0 : index
        %c4_7 = arith.constant 4 : index
        %44 = air.wait_all async 
        %45 = air.wait_all async 
        %46 = air.wait_all async 
        %47 = air.wait_all async 
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
        %48 = scf.for %arg16 = %c0_6 to %c8_5 step %c1_4 iter_args(%arg17 = %44) -> (!air.async.token) {
          %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %61 = air.channel.get async [%async_token_18, %arg17]  @VIn_0[%arg12] (%results_19[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_6 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @V2L1_0_0[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @V2L1_0_1[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          %async_token_20 = air.execute [%63, %61] {
            memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %63 : !air.async.token
        }
        %49 = scf.for %arg16 = %c0_6 to %c8_5 step %c1_4 iter_args(%arg17 = %45) -> (!air.async.token) {
          %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %61 = air.channel.get async [%async_token_18, %arg17]  @VIn_1[%arg12] (%results_19[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_6 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @V2L1_1_0[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @V2L1_1_1[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          %async_token_20 = air.execute [%63, %61] {
            memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %63 : !air.async.token
        }
        %50 = scf.for %arg16 = %c0_6 to %c8_5 step %c1_4 iter_args(%arg17 = %46) -> (!air.async.token) {
          %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %61 = air.channel.get async [%async_token_18, %arg17]  @VIn_2[%arg12] (%results_19[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_6 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @V2L1_2_0[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @V2L1_2_1[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          %async_token_20 = air.execute [%63, %61] {
            memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %63 : !air.async.token
        }
        %51 = scf.for %arg16 = %c0_6 to %c8_5 step %c1_4 iter_args(%arg17 = %47) -> (!air.async.token) {
          %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %61 = air.channel.get async [%async_token_18, %arg17]  @VIn_3[%arg12] (%results_19[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_6 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @V2L1_3_0[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @V2L1_3_1[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          %async_token_20 = air.execute [%63, %61] {
            memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %63 : !air.async.token
        }
        %52 = air.channel.get async [%async_token]  @Gp2L2[%c0_6, %c0_6] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %53 = air.channel.get async [%async_token_8]  @Gp2L2[%c1_4, %c0_6] (%results_9[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %54 = air.channel.get async [%async_token_10]  @Gp2L2[%c2_2, %c0_6] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %55 = air.channel.get async [%async_token_12]  @Gp2L2[%c3_1, %c0_6] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %56 = air.channel.put async [%52]  @channel_0[%c0_6, %arg12] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %57 = air.channel.put async [%53]  @channel_0[%c1_4, %arg12] (%results_9[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %58 = air.channel.put async [%54]  @channel_0[%c2_2, %arg12] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %59 = air.channel.put async [%55]  @channel_0[%c3_1, %arg12] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %60 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_7, %arg19=%c4_7) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn.o"} {
          %c64_18 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c8_19 = arith.constant 8 : index
          %c0_20 = arith.constant 0 : index
          %c1_21 = arith.constant 1 : index
          %c512_22 = arith.constant 512 : index
          %async_token_23, %results_24 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_25, %results_26 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_27, %results_28 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_31, %results_32 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_33 = air.execute [%async_token_27] {
            func.call @zero_fill_gp_bf16(%results_28) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_34 = air.execute [%async_token_23] {
            func.call @zero_fill_sp_bf16(%results_24) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_35 = air.execute [%async_token_25] {
            func.call @neg_inf_fill_up_bf16(%results_26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %61 = arith.cmpi eq, %arg20, %c0_20 : index
          scf.if %61 {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          } else {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          }
          %62 = arith.index_cast %arg16 : index to i32
          %63 = arith.cmpi eq, %62, %c0_i32 : i32
          scf.if %63 {
            %async_token_41 = air.execute [%async_token_29, %async_token_31] {
              func.call @copy_tile(%results_30, %results_32) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %61 {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          } else {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          }
          %64 = arith.cmpi eq, %62, %c1_i32 : i32
          scf.if %64 {
            %async_token_41 = air.execute [%async_token_29, %async_token_31] {
              func.call @copy_tile(%results_30, %results_32) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %61 {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          } else {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          }
          %65 = arith.cmpi eq, %62, %c2_i32 : i32
          scf.if %65 {
            %async_token_41 = air.execute [%async_token_29, %async_token_31] {
              func.call @copy_tile(%results_30, %results_32) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %61 {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          } else {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          }
          %66 = arith.cmpi eq, %62, %c3_i32 : i32
          scf.if %66 {
            %async_token_41 = air.execute [%async_token_29, %async_token_31] {
              func.call @copy_tile(%results_30, %results_32) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %67 = air.wait_all async [%async_token_29, %async_token_31, %async_token_33, %async_token_34, %async_token_35] 
          %68 = scf.for %arg21 = %c0_20 to %c8_19 step %c1_21 iter_args(%arg22 = %67) -> (!air.async.token) {
            %async_token_41, %results_42 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_43, %results_44 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_45 = air.execute [%async_token_43, %arg22] {
              %collapse_shape = memref.collapse_shape %results_44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %61 {
              %75 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%arg22]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%arg22]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %78 = air.channel.get async [%arg22]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %78 : !air.async.token
                  } else {
                    %78 = air.channel.get async [%arg22]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %78 : !air.async.token
                  }
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
            } else {
              %75 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%arg22]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%arg22]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %78 = air.channel.get async [%arg22]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %78 : !air.async.token
                  } else {
                    %78 = air.channel.get async [%arg22]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %78 : !air.async.token
                  }
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
            }
            %70 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %75 = scf.if %61 -> (!air.async.token) {
                %76 = air.channel.get async [%async_token_41]  @V2L1_0_0[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              } else {
                %76 = air.channel.get async [%async_token_41]  @V2L1_0_1[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            } else {
              %75 = air.wait_all async 
              affine.yield %75 : !air.async.token
            }
            %71 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %75 = scf.if %61 -> (!air.async.token) {
                %76 = air.channel.get async [%async_token_41, %arg22, %70]  @V2L1_1_0[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              } else {
                %76 = air.channel.get async [%async_token_41, %arg22, %70]  @V2L1_1_1[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            } else {
              %75 = air.wait_all async 
              affine.yield %75 : !air.async.token
            }
            %72 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %75 = scf.if %61 -> (!air.async.token) {
                %76 = air.channel.get async [%async_token_41, %arg22, %71]  @V2L1_2_0[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              } else {
                %76 = air.channel.get async [%async_token_41, %arg22, %71]  @V2L1_2_1[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            } else {
              %75 = air.wait_all async 
              affine.yield %75 : !air.async.token
            }
            %73 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %75 = scf.if %61 -> (!air.async.token) {
                %76 = air.channel.get async [%async_token_41, %arg22, %72]  @V2L1_3_0[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              } else {
                %76 = air.channel.get async [%async_token_41, %arg22, %72]  @V2L1_3_1[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            } else {
              %75 = air.wait_all async 
              affine.yield %75 : !air.async.token
            }
            %async_token_46 = air.execute [%async_token_45] {
              %collapse_shape = memref.collapse_shape %results_44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_32, %results_30, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_47, %results_48 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_49, %results_50 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_51 = air.execute [%async_token_46, %async_token_47, %async_token_49] {
              %collapse_shape = memref.collapse_shape %results_44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_26, %results_48, %results_50) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_52 = air.execute [%async_token_51] {
              func.call @mul_r_gp(%results_50, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_53 = air.execute [%73, %async_token_52, %async_token_41, %async_token_43] {
              %collapse_shape = memref.collapse_shape %results_44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_42, %results_28) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_54 = air.execute [%async_token_52] {
              func.call @accum_sp_r_s(%results_24, %results_50, %results_48) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_55 = air.execute [%async_token_54] {
              func.call @vector_copy_32elems(%c0_i32, %results_48, %results_24) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_56 = air.execute [%async_token_55] {
              memref.dealloc %results_48 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_57 = air.execute [%async_token_54] {
              memref.dealloc %results_50 : memref<64x1xbf16, 2 : i32>
            }
            %74 = air.wait_all async [%70, %71, %72, %async_token_53, %async_token_55] 
            %async_token_58 = air.execute [%async_token_51, %async_token_53] {
              memref.dealloc %results_44 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_59 = air.execute [%70, %71, %72, %async_token_53] {
              memref.dealloc %results_42 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %74 : !air.async.token
          }
          %69 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %70 = arith.subi %arg17, %c1_21 : index
            %71 = air.channel.put async [%async_token_27, %68]  @cascade_gp[%arg16, %70] (%results_28[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
            %72 = air.channel.put async [%async_token_25, %68]  @cascade_up[%arg16, %70] (%results_26[] [] []) {id = 102 : i32} : (memref<64x1xbf16, 2 : i32>)
            %73 = air.channel.put async [%async_token_23, %68]  @cascade_sp[%arg16, %70] (%results_24[] [] []) {id = 103 : i32} : (memref<64x1xbf16, 2 : i32>)
            %74 = air.wait_all async [%71, %72, %73] 
            affine.yield %74 : !air.async.token
          } else {
            %70 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_41, %results_42 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_43, %results_44 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_45, %results_46 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %71 = air.channel.get async [%async_token_41]  @cascade_gp[%arg16, %arg17] (%results_42[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              %72 = air.channel.get async [%async_token_43]  @cascade_up[%arg16, %arg17] (%results_44[] [] []) {id = 105 : i32} : (memref<64x1xbf16, 2 : i32>)
              %73 = air.channel.get async [%async_token_45]  @cascade_sp[%arg16, %arg17] (%results_46[] [] []) {id = 106 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_47, %results_48 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_49 = air.execute [%async_token_25, %async_token_47, %68] {
                func.call @vector_copy_32elems(%c0_i32, %results_26, %results_48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_50 = air.execute [%72, %async_token_49] {
                func.call @maximum_up_u_bf16(%results_44, %results_26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_51, %results_52 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_53 = air.execute [%async_token_50, %async_token_51] {
                func.call @exp_up_minus_u(%results_44, %results_26, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56 = air.execute [%async_token_53, %async_token_54] {
                func.call @exp_up_minus_u(%results_48, %results_26, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_57 = air.execute [%async_token_53, %71] {
                func.call @mul_r_gp(%results_52, %results_42) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_58 = air.execute [%async_token_27, %async_token_56] {
                func.call @mul_r_gp(%results_55, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_59 = air.execute [%async_token_57, %async_token_58] {
                func.call @add_gp_g(%results_28, %results_42) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_62 = air.execute [%async_token_60] {
                func.call @zero_fill_sp_bf16(%results_61) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_63 = air.execute [%async_token_62, %async_token_57, %73] {
                func.call @accum_sp_r_s(%results_46, %results_52, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_64 = air.execute [%async_token_23, %async_token_63, %async_token_58] {
                func.call @accum_sp_r_s(%results_24, %results_55, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65 = air.execute [%async_token_64] {
                func.call @vector_copy_32elems(%c0_i32, %results_61, %results_46) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %74 = arith.subi %arg17, %c1_21 : index
              %75 = air.channel.put async [%async_token_59]  @cascade_gp[%arg16, %74] (%results_42[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              %76 = air.channel.put async [%async_token_25, %async_token_56]  @cascade_up[%arg16, %74] (%results_26[] [] []) {id = 108 : i32} : (memref<64x1xbf16, 2 : i32>)
              %77 = air.channel.put async [%async_token_65]  @cascade_sp[%arg16, %74] (%results_46[] [] []) {id = 109 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_66 = air.execute [%75] {
                memref.dealloc %results_42 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_67 = air.execute [%async_token_53] {
                memref.dealloc %results_44 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_68 = air.execute [%77] {
                memref.dealloc %results_46 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_56] {
                memref.dealloc %results_48 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%async_token_63] {
                memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%async_token_64] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_65] {
                memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
              }
              %78 = air.wait_all async [%75, %76, %77] 
              affine.yield %78 : !air.async.token
            } else {
              %async_token_41, %results_42 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_43, %results_44 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_45, %results_46 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %71 = air.channel.get async [%async_token_41]  @cascade_gp[%arg16, %arg17] (%results_42[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              %72 = air.channel.get async [%async_token_43]  @cascade_up[%arg16, %arg17] (%results_44[] [] []) {id = 111 : i32} : (memref<64x1xbf16, 2 : i32>)
              %73 = air.channel.get async [%async_token_45]  @cascade_sp[%arg16, %arg17] (%results_46[] [] []) {id = 112 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_47, %results_48 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_49 = air.execute [%async_token_25, %async_token_47, %68] {
                func.call @vector_copy_32elems(%c0_i32, %results_26, %results_48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_50 = air.execute [%72, %async_token_49] {
                func.call @maximum_up_u_bf16(%results_44, %results_26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_51, %results_52 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_53 = air.execute [%async_token_50, %async_token_51] {
                func.call @exp_up_minus_u(%results_44, %results_26, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56 = air.execute [%async_token_53, %async_token_54] {
                func.call @exp_up_minus_u(%results_48, %results_26, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_57 = air.execute [%async_token_53, %71] {
                func.call @mul_r_gp(%results_52, %results_42) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_58 = air.execute [%async_token_27, %async_token_56] {
                func.call @mul_r_gp(%results_55, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_59 = air.execute [%async_token_57, %async_token_58] {
                func.call @add_gp_g(%results_28, %results_42) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_62 = air.execute [%async_token_60] {
                func.call @zero_fill_sp_bf16(%results_61) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_63 = air.execute [%async_token_62, %async_token_57, %73] {
                func.call @accum_sp_r_s(%results_46, %results_52, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_64 = air.execute [%async_token_23, %async_token_63, %async_token_58] {
                func.call @accum_sp_r_s(%results_24, %results_55, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65 = air.execute [%async_token_64] {
                func.call @vector_copy_32elems(%c0_i32, %results_61, %results_46) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_65, %async_token_59] {
                func.call @div_gp_sp(%results_46, %results_42) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %74 = air.channel.put async [%async_token_66]  @Gp2L2[%arg16, %c0_20] (%results_42[%c0_20, %c0_20, %c0_20] [%c64_18, %c8_19, %c8_19] [%c8_19, %c512_22, %c1_21]) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_67 = air.execute [%74] {
                memref.dealloc %results_42 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_68 = air.execute [%async_token_53] {
                memref.dealloc %results_44 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_66] {
                memref.dealloc %results_46 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%async_token_56] {
                memref.dealloc %results_48 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%async_token_63] {
                memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_64] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_65] {
                memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %74 : !air.async.token
            }
            affine.yield %68 : !air.async.token
          }
          %async_token_36 = air.execute [%68] {
            memref.dealloc %results_32 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_37 = air.execute [%68] {
            memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_38 = air.execute [%69, %68, %async_token_33] {
            memref.dealloc %results_28 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_39 = air.execute [%69, %68, %async_token_35] {
            memref.dealloc %results_26 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_40 = air.execute [%69, %68, %async_token_34] {
            memref.dealloc %results_24 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_14 = air.execute [%59] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_15 = air.execute [%58] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_16 = air.execute [%57] {
          memref.dealloc %results_9 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17 = air.execute [%56] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        air.wait_all [%48, %49, %50, %51, %60, %async_token_14, %async_token_15, %async_token_16, %async_token_17]  {air.segment_end}
      }
    }
    return
  }
}
