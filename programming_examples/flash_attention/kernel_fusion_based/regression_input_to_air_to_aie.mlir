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
  func.func @attention_bf16(%arg0: memref<2x512x64xbf16>, %arg1: memref<2x512x64xbf16>, %arg2: memref<2x512x64xbf16>, %arg3: memref<2x512x64xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c4096 = arith.constant 4096 : index
      %c2_0 = arith.constant 2 : index
      %c1_1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_1]) {id = 1 : i32} : (memref<2x512x64xbf16>)
      %3 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_1]) {id = 2 : i32} : (memref<2x512x64xbf16>)
      %4 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_1]) {id = 3 : i32} : (memref<2x512x64xbf16>)
      %5 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_1]) {id = 4 : i32} : (memref<2x512x64xbf16>)
      %6 = affine.apply #map1()[%arg5]
      %7 = air.channel.put async  @QKIn_0[%c0] (%arg9[%c0, %c0, %c0, %6] [%c2_0, %c1_1, %c64, %c64] [%c4096, %c64, %c64, %c1_1]) {id = 5 : i32} : (memref<2x512x64xbf16>)
      %8 = affine.apply #map2()[%arg5]
      %9 = air.channel.put async  @QKIn_1[%c0] (%arg9[%c0, %c0, %c0, %8] [%c2_0, %c1_1, %c64, %c64] [%c4096, %c64, %c64, %c1_1]) {id = 6 : i32} : (memref<2x512x64xbf16>)
      %10 = affine.apply #map3()[%arg5]
      %11 = air.channel.put async  @QKIn_2[%c0] (%arg9[%c0, %c0, %c0, %10] [%c2_0, %c1_1, %c64, %c64] [%c4096, %c64, %c64, %c1_1]) {id = 7 : i32} : (memref<2x512x64xbf16>)
      %12 = affine.apply #map4()[%arg5]
      %13 = air.channel.put async  @QKIn_3[%c0] (%arg9[%c0, %c0, %c0, %12] [%c2_0, %c1_1, %c64, %c64] [%c4096, %c64, %c64, %c1_1]) {id = 8 : i32} : (memref<2x512x64xbf16>)
      %14 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %6] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 9 : i32} : (memref<2x512x64xbf16>)
      %15 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %8] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 10 : i32} : (memref<2x512x64xbf16>)
      %16 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %10] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 11 : i32} : (memref<2x512x64xbf16>)
      %17 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %12] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 12 : i32} : (memref<2x512x64xbf16>)
      %18 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %1] [%c64, %c64] [%c64, %c1_1]) {id = 13 : i32} : (memref<2x512x64xbf16>)
      %19 = air.channel.get async  @channel_0[%c1_1, %c0] (%arg11[%c64, %1] [%c64, %c64] [%c64, %c1_1]) {id = 14 : i32} : (memref<2x512x64xbf16>)
      %20 = air.channel.get async  @channel_0[%c2_0, %c0] (%arg11[%c128, %1] [%c64, %c64] [%c64, %c1_1]) {id = 15 : i32} : (memref<2x512x64xbf16>)
      %21 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %1] [%c64, %c64] [%c64, %c1_1]) {id = 16 : i32} : (memref<2x512x64xbf16>)
      %22 = affine.apply #map5()[%arg5, %arg4]
      %23 = air.channel.put async  @QKIn_0[%c1_1] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_1]) {id = 17 : i32} : (memref<2x512x64xbf16>)
      %24 = air.channel.put async  @QKIn_1[%c1_1] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_1]) {id = 18 : i32} : (memref<2x512x64xbf16>)
      %25 = air.channel.put async  @QKIn_2[%c1_1] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_1]) {id = 19 : i32} : (memref<2x512x64xbf16>)
      %26 = air.channel.put async  @QKIn_3[%c1_1] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_1]) {id = 20 : i32} : (memref<2x512x64xbf16>)
      %27 = affine.apply #map6()[%arg5]
      %28 = air.channel.put async  @QKIn_0[%c1_1] (%arg9[%c0, %c0, %c0, %27] [%c2_0, %c1_1, %c64, %c64] [%c4096, %c64, %c64, %c1_1]) {id = 21 : i32} : (memref<2x512x64xbf16>)
      %29 = affine.apply #map7()[%arg5]
      %30 = air.channel.put async  @QKIn_1[%c1_1] (%arg9[%c0, %c0, %c0, %29] [%c2_0, %c1_1, %c64, %c64] [%c4096, %c64, %c64, %c1_1]) {id = 22 : i32} : (memref<2x512x64xbf16>)
      %31 = affine.apply #map8()[%arg5]
      %32 = air.channel.put async  @QKIn_2[%c1_1] (%arg9[%c0, %c0, %c0, %31] [%c2_0, %c1_1, %c64, %c64] [%c4096, %c64, %c64, %c1_1]) {id = 23 : i32} : (memref<2x512x64xbf16>)
      %33 = affine.apply #map9()[%arg5]
      %34 = air.channel.put async  @QKIn_3[%c1_1] (%arg9[%c0, %c0, %c0, %33] [%c2_0, %c1_1, %c64, %c64] [%c4096, %c64, %c64, %c1_1]) {id = 24 : i32} : (memref<2x512x64xbf16>)
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
        %c8 = arith.constant 8 : index
        %c1_4 = arith.constant 1 : index
        %c2_5 = arith.constant 2 : index
        %c0_6 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_7, %results_8 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
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
        %44 = air.wait_all async 
        %45 = air.wait_all async 
        %46 = air.wait_all async 
        %47 = air.wait_all async 
        %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_15, %results_16 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17, %results_18 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %48 = scf.for %arg16 = %c0_6 to %c4 step %c1_4 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %69 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @QK2L1_0_0[%c0_6, %c0_6, %c0_6] (%results[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @QK2L1_0_1[%c0_6, %c0_6, %c0_6] (%results[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          scf.yield %71 : !air.async.token
        }
        %49 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %48) -> (!air.async.token) {
          %69 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @QK2L1_0_0[%c0_6, %c0_6, %c0_6] (%results[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @QK2L1_0_1[%c0_6, %c0_6, %c0_6] (%results[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          scf.yield %71 : !air.async.token
        }
        %50 = scf.for %arg16 = %c0_6 to %c4 step %c1_4 iter_args(%arg17 = %async_token_7) -> (!air.async.token) {
          %69 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_8[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @QK2L1_1_0[%c0_6, %c0_6, %c0_6] (%results_8[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @QK2L1_1_1[%c0_6, %c0_6, %c0_6] (%results_8[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          scf.yield %71 : !air.async.token
        }
        %51 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %50) -> (!air.async.token) {
          %69 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_8[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @QK2L1_1_0[%c0_6, %c0_6, %c0_6] (%results_8[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @QK2L1_1_1[%c0_6, %c0_6, %c0_6] (%results_8[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          scf.yield %71 : !air.async.token
        }
        %52 = scf.for %arg16 = %c0_6 to %c4 step %c1_4 iter_args(%arg17 = %async_token_9) -> (!air.async.token) {
          %69 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_10[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @QK2L1_2_0[%c0_6, %c0_6, %c0_6] (%results_10[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @QK2L1_2_1[%c0_6, %c0_6, %c0_6] (%results_10[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          scf.yield %71 : !air.async.token
        }
        %53 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %52) -> (!air.async.token) {
          %69 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_10[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @QK2L1_2_0[%c0_6, %c0_6, %c0_6] (%results_10[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @QK2L1_2_1[%c0_6, %c0_6, %c0_6] (%results_10[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          scf.yield %71 : !air.async.token
        }
        %54 = scf.for %arg16 = %c0_6 to %c4 step %c1_4 iter_args(%arg17 = %async_token_11) -> (!air.async.token) {
          %69 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_12[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @QK2L1_3_0[%c0_6, %c0_6, %c0_6] (%results_12[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @QK2L1_3_1[%c0_6, %c0_6, %c0_6] (%results_12[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 53 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          scf.yield %71 : !air.async.token
        }
        %55 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %54) -> (!air.async.token) {
          %69 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_12[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @QK2L1_3_0[%c0_6, %c0_6, %c0_6] (%results_12[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 55 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @QK2L1_3_1[%c0_6, %c0_6, %c0_6] (%results_12[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 56 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          scf.yield %71 : !air.async.token
        }
        %56 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %44) -> (!air.async.token) {
          %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %69 = air.channel.get async [%async_token_29, %arg17]  @VIn_0[%arg12] (%results_30[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @V2L1_0_0[%c0_6, %c0_6, %c0_6] (%results_30[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 58 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @V2L1_0_1[%c0_6, %c0_6, %c0_6] (%results_30[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 59 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          %async_token_31 = air.execute [%71, %69] {
            memref.dealloc %results_30 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %71 : !air.async.token
        }
        %57 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %45) -> (!air.async.token) {
          %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %69 = air.channel.get async [%async_token_29, %arg17]  @VIn_1[%arg12] (%results_30[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @V2L1_1_0[%c0_6, %c0_6, %c0_6] (%results_30[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 61 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @V2L1_1_1[%c0_6, %c0_6, %c0_6] (%results_30[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 62 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          %async_token_31 = air.execute [%71, %69] {
            memref.dealloc %results_30 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %71 : !air.async.token
        }
        %58 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %46) -> (!air.async.token) {
          %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %69 = air.channel.get async [%async_token_29, %arg17]  @VIn_2[%arg12] (%results_30[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @V2L1_2_0[%c0_6, %c0_6, %c0_6] (%results_30[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 64 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @V2L1_2_1[%c0_6, %c0_6, %c0_6] (%results_30[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 65 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          %async_token_31 = air.execute [%71, %69] {
            memref.dealloc %results_30 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %71 : !air.async.token
        }
        %59 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %47) -> (!air.async.token) {
          %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %69 = air.channel.get async [%async_token_29, %arg17]  @VIn_3[%arg12] (%results_30[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 1 : i32>)
          %70 = arith.cmpi eq, %arg12, %c0_6 : index
          %71 = scf.if %70 -> (!air.async.token) {
            %72 = air.channel.put async [%69]  @V2L1_3_0[%c0_6, %c0_6, %c0_6] (%results_30[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 67 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          } else {
            %72 = air.channel.put async [%69]  @V2L1_3_1[%c0_6, %c0_6, %c0_6] (%results_30[%c0_6, %c0_6, %c0_6] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 68 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %72 : !air.async.token
          }
          %async_token_31 = air.execute [%71, %69] {
            memref.dealloc %results_30 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %71 : !air.async.token
        }
        %60 = air.channel.get async [%async_token_13]  @Gp2L2[%c0_6, %c0_6] (%results_14[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 1 : i32>)
        %61 = air.channel.get async [%async_token_15]  @Gp2L2[%c1_4, %c0_6] (%results_16[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 1 : i32>)
        %62 = air.channel.get async [%async_token_17]  @Gp2L2[%c2_5, %c0_6] (%results_18[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 1 : i32>)
        %63 = air.channel.get async [%async_token_19]  @Gp2L2[%c3_2, %c0_6] (%results_20[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 1 : i32>)
        %64 = air.channel.put async [%60]  @channel_0[%c0_6, %arg12] (%results_14[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 1 : i32>)
        %65 = air.channel.put async [%61]  @channel_0[%c1_4, %arg12] (%results_16[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 1 : i32>)
        %66 = air.channel.put async [%62]  @channel_0[%c2_5, %arg12] (%results_18[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 1 : i32>)
        %67 = air.channel.put async [%63]  @channel_0[%c3_2, %arg12] (%results_20[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 1 : i32>)
        %68 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c64_29 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2_30 = arith.constant 2 : index
          %c0_31 = arith.constant 0 : index
          %c1_32 = arith.constant 1 : index
          %c8_33 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %async_token_34, %results_35 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_36, %results_37 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_38, %results_39 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_40, %results_41 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_42, %results_43 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_44 = air.execute [%async_token_38] {
            func.call @zero_fill_gp_bf16(%results_39) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_45 = air.execute [%async_token_34] {
            func.call @zero_fill_sp_bf16(%results_35) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_46 = air.execute [%async_token_36] {
            func.call @neg_inf_fill_up_bf16(%results_37) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %69 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40]  @QK2L1_0_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40]  @QK2L1_0_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %70 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %69]  @QK2L1_1_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %69]  @QK2L1_1_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %71 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %70]  @QK2L1_2_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %70]  @QK2L1_2_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %72 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %71]  @QK2L1_3_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %71]  @QK2L1_3_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %73 = arith.index_cast %arg16 : index to i32
          %74 = arith.cmpi eq, %73, %c0_i32 : i32
          scf.if %74 {
            %async_token_52 = air.execute [%async_token_40, %async_token_42, %72] {
              func.call @copy_tile(%results_41, %results_43) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %75 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40]  @QK2L1_0_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40]  @QK2L1_0_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %76 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %75]  @QK2L1_1_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %75]  @QK2L1_1_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %77 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %76]  @QK2L1_2_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %76]  @QK2L1_2_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %78 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %77]  @QK2L1_3_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %77]  @QK2L1_3_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %79 = arith.cmpi eq, %73, %c1_i32 : i32
          scf.if %79 {
            %async_token_52 = air.execute [%async_token_40, %async_token_42, %78] {
              func.call @copy_tile(%results_41, %results_43) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %80 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40]  @QK2L1_0_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40]  @QK2L1_0_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %81 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %80]  @QK2L1_1_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %80]  @QK2L1_1_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %82 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %81]  @QK2L1_2_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %81]  @QK2L1_2_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %83 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %82]  @QK2L1_3_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %82]  @QK2L1_3_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %84 = arith.cmpi eq, %73, %c2_i32 : i32
          scf.if %84 {
            %async_token_52 = air.execute [%async_token_40, %async_token_42, %83] {
              func.call @copy_tile(%results_41, %results_43) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %85 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40]  @QK2L1_0_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40]  @QK2L1_0_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 102 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %86 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %85]  @QK2L1_1_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 103 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %85]  @QK2L1_1_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %87 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %86]  @QK2L1_2_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 105 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %86]  @QK2L1_2_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 106 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %88 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.cmpi eq, %arg20, %c0_31 : index
            %94 = scf.if %93 -> (!air.async.token) {
              %95 = air.channel.get async [%async_token_40, %87]  @QK2L1_3_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            } else {
              %95 = air.channel.get async [%async_token_40, %87]  @QK2L1_3_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 108 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %95 : !air.async.token
            }
            affine.yield %94 : !air.async.token
          } else {
            %93 = air.wait_all async 
            affine.yield %93 : !air.async.token
          }
          %89 = arith.cmpi eq, %73, %c3_i32 : i32
          scf.if %89 {
            %async_token_52 = air.execute [%async_token_40, %async_token_42, %88] {
              func.call @copy_tile(%results_41, %results_43) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %90 = air.wait_all async [%async_token_40, %async_token_42, %async_token_44, %async_token_45, %async_token_46] 
          %91 = scf.for %arg21 = %c0_31 to %c2_30 step %c1_32 iter_args(%arg22 = %90) -> (!air.async.token) {
            %async_token_52, %results_53 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_54, %results_55 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_56 = air.execute [%async_token_54, %arg22] {
              %collapse_shape = memref.collapse_shape %results_55 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %93 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %102 = arith.cmpi eq, %arg20, %c0_31 : index
              %103 = scf.if %102 -> (!air.async.token) {
                %104 = air.channel.get async [%arg22]  @QK2L1_0_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 109 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              } else {
                %104 = air.channel.get async [%arg22]  @QK2L1_0_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              }
              affine.yield %103 : !air.async.token
            } else {
              %102 = air.wait_all async 
              affine.yield %102 : !air.async.token
            }
            %94 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %102 = arith.cmpi eq, %arg20, %c0_31 : index
              %103 = scf.if %102 -> (!air.async.token) {
                %104 = air.channel.get async [%arg22, %93]  @QK2L1_1_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 111 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              } else {
                %104 = air.channel.get async [%arg22, %93]  @QK2L1_1_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 112 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              }
              affine.yield %103 : !air.async.token
            } else {
              %102 = air.wait_all async 
              affine.yield %102 : !air.async.token
            }
            %95 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %102 = arith.cmpi eq, %arg20, %c0_31 : index
              %103 = scf.if %102 -> (!air.async.token) {
                %104 = air.channel.get async [%arg22, %94]  @QK2L1_2_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              } else {
                %104 = air.channel.get async [%arg22, %94]  @QK2L1_2_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 114 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              }
              affine.yield %103 : !air.async.token
            } else {
              %102 = air.wait_all async 
              affine.yield %102 : !air.async.token
            }
            %96 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %102 = arith.cmpi eq, %arg20, %c0_31 : index
              %103 = scf.if %102 -> (!air.async.token) {
                %104 = air.channel.get async [%arg22, %95]  @QK2L1_3_0[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 115 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              } else {
                %104 = air.channel.get async [%arg22, %95]  @QK2L1_3_1[%c0_31, %arg17, %arg16] (%results_41[] [] []) {id = 116 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              }
              affine.yield %103 : !air.async.token
            } else {
              %102 = air.wait_all async 
              affine.yield %102 : !air.async.token
            }
            %async_token_57 = air.execute [%async_token_56, %96] {
              %collapse_shape = memref.collapse_shape %results_55 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_43, %results_41, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %102 = arith.cmpi eq, %arg20, %c0_31 : index
              %103 = scf.if %102 -> (!air.async.token) {
                %104 = air.channel.get async [%async_token_52]  @V2L1_0_0[%c0_31, %arg17, %arg16] (%results_53[] [] []) {id = 117 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              } else {
                %104 = air.channel.get async [%async_token_52]  @V2L1_0_1[%c0_31, %arg17, %arg16] (%results_53[] [] []) {id = 118 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              }
              affine.yield %103 : !air.async.token
            } else {
              %102 = air.wait_all async 
              affine.yield %102 : !air.async.token
            }
            %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %102 = arith.cmpi eq, %arg20, %c0_31 : index
              %103 = scf.if %102 -> (!air.async.token) {
                %104 = air.channel.get async [%async_token_52, %arg22, %97]  @V2L1_1_0[%c0_31, %arg17, %arg16] (%results_53[] [] []) {id = 119 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              } else {
                %104 = air.channel.get async [%async_token_52, %arg22, %97]  @V2L1_1_1[%c0_31, %arg17, %arg16] (%results_53[] [] []) {id = 120 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              }
              affine.yield %103 : !air.async.token
            } else {
              %102 = air.wait_all async 
              affine.yield %102 : !air.async.token
            }
            %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %102 = arith.cmpi eq, %arg20, %c0_31 : index
              %103 = scf.if %102 -> (!air.async.token) {
                %104 = air.channel.get async [%async_token_52, %arg22, %98]  @V2L1_2_0[%c0_31, %arg17, %arg16] (%results_53[] [] []) {id = 121 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              } else {
                %104 = air.channel.get async [%async_token_52, %arg22, %98]  @V2L1_2_1[%c0_31, %arg17, %arg16] (%results_53[] [] []) {id = 122 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              }
              affine.yield %103 : !air.async.token
            } else {
              %102 = air.wait_all async 
              affine.yield %102 : !air.async.token
            }
            %100 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %102 = arith.cmpi eq, %arg20, %c0_31 : index
              %103 = scf.if %102 -> (!air.async.token) {
                %104 = air.channel.get async [%async_token_52, %arg22, %99]  @V2L1_3_0[%c0_31, %arg17, %arg16] (%results_53[] [] []) {id = 123 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              } else {
                %104 = air.channel.get async [%async_token_52, %arg22, %99]  @V2L1_3_1[%c0_31, %arg17, %arg16] (%results_53[] [] []) {id = 124 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %104 : !air.async.token
              }
              affine.yield %103 : !air.async.token
            } else {
              %102 = air.wait_all async 
              affine.yield %102 : !air.async.token
            }
            %async_token_58, %results_59 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_62 = air.execute [%async_token_57, %async_token_58, %async_token_60] {
              %collapse_shape = memref.collapse_shape %results_55 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_37, %results_59, %results_61) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_63 = air.execute [%async_token_62] {
              func.call @mul_r_gp(%results_61, %results_39) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_64 = air.execute [%100, %async_token_63, %async_token_52, %async_token_54] {
              %collapse_shape = memref.collapse_shape %results_55 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_53, %results_39) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_65 = air.execute [%async_token_63] {
              func.call @accum_sp_r_s(%results_35, %results_61, %results_59) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_66 = air.execute [%async_token_65] {
              func.call @vector_copy_32elems(%c0_i32, %results_59, %results_35) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_67 = air.execute [%async_token_66] {
              memref.dealloc %results_59 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_68 = air.execute [%async_token_65] {
              memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
            }
            %101 = air.wait_all async [%93, %94, %95, %97, %98, %99, %async_token_64, %async_token_66] 
            %async_token_69 = air.execute [%async_token_62, %async_token_64] {
              memref.dealloc %results_55 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_70 = air.execute [%97, %98, %99, %async_token_64] {
              memref.dealloc %results_53 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %101 : !air.async.token
          }
          %92 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %93 = arith.subi %arg17, %c1_32 : index
            %94 = air.channel.put async [%async_token_38, %91]  @cascade_gp[%arg16, %93] (%results_39[] [] []) {id = 125 : i32} : (memref<64x64xbf16, 2 : i32>)
            %95 = air.channel.put async [%async_token_36, %91]  @cascade_up[%arg16, %93] (%results_37[] [] []) {id = 126 : i32} : (memref<64x1xbf16, 2 : i32>)
            %96 = air.channel.put async [%async_token_34, %91]  @cascade_sp[%arg16, %93] (%results_35[] [] []) {id = 127 : i32} : (memref<64x1xbf16, 2 : i32>)
            %97 = air.wait_all async [%94, %95, %96] 
            affine.yield %97 : !air.async.token
          } else {
            %93 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_52, %results_53 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %94 = air.channel.get async [%async_token_52]  @cascade_gp[%arg16, %arg17] (%results_53[] [] []) {id = 128 : i32} : (memref<64x64xbf16, 2 : i32>)
              %95 = air.channel.get async [%async_token_54]  @cascade_up[%arg16, %arg17] (%results_55[] [] []) {id = 129 : i32} : (memref<64x1xbf16, 2 : i32>)
              %96 = air.channel.get async [%async_token_56]  @cascade_sp[%arg16, %arg17] (%results_57[] [] []) {id = 130 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_58, %results_59 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_60 = air.execute [%async_token_36, %async_token_58, %91] {
                func.call @vector_copy_32elems(%c0_i32, %results_37, %results_59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%95, %async_token_60] {
                func.call @maximum_up_u_bf16(%results_55, %results_37) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64 = air.execute [%async_token_61, %async_token_62] {
                func.call @exp_up_minus_u(%results_55, %results_37, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_67 = air.execute [%async_token_64, %async_token_65] {
                func.call @exp_up_minus_u(%results_59, %results_37, %results_66) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_68 = air.execute [%async_token_64, %94] {
                func.call @mul_r_gp(%results_63, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_69 = air.execute [%async_token_38, %async_token_67] {
                func.call @mul_r_gp(%results_66, %results_39) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_68, %async_token_69] {
                func.call @add_gp_g(%results_39, %results_53) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_71] {
                func.call @zero_fill_sp_bf16(%results_72) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_74 = air.execute [%async_token_73, %async_token_68, %96] {
                func.call @accum_sp_r_s(%results_57, %results_63, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_75 = air.execute [%async_token_34, %async_token_74, %async_token_69] {
                func.call @accum_sp_r_s(%results_35, %results_66, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_75] {
                func.call @vector_copy_32elems(%c0_i32, %results_72, %results_57) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %97 = arith.subi %arg17, %c1_32 : index
              %98 = air.channel.put async [%async_token_70]  @cascade_gp[%arg16, %97] (%results_53[] [] []) {id = 131 : i32} : (memref<64x64xbf16, 2 : i32>)
              %99 = air.channel.put async [%async_token_36, %async_token_67]  @cascade_up[%arg16, %97] (%results_37[] [] []) {id = 132 : i32} : (memref<64x1xbf16, 2 : i32>)
              %100 = air.channel.put async [%async_token_76]  @cascade_sp[%arg16, %97] (%results_57[] [] []) {id = 133 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_77 = air.execute [%98] {
                memref.dealloc %results_53 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_78 = air.execute [%async_token_64] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_79 = air.execute [%100] {
                memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_80 = air.execute [%async_token_67] {
                memref.dealloc %results_59 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_81 = air.execute [%async_token_74] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_82 = air.execute [%async_token_75] {
                memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_76] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              }
              %101 = air.wait_all async [%98, %99, %100] 
              affine.yield %101 : !air.async.token
            } else {
              %async_token_52, %results_53 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %94 = air.channel.get async [%async_token_52]  @cascade_gp[%arg16, %arg17] (%results_53[] [] []) {id = 134 : i32} : (memref<64x64xbf16, 2 : i32>)
              %95 = air.channel.get async [%async_token_54]  @cascade_up[%arg16, %arg17] (%results_55[] [] []) {id = 135 : i32} : (memref<64x1xbf16, 2 : i32>)
              %96 = air.channel.get async [%async_token_56]  @cascade_sp[%arg16, %arg17] (%results_57[] [] []) {id = 136 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_58, %results_59 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_60 = air.execute [%async_token_36, %async_token_58, %91] {
                func.call @vector_copy_32elems(%c0_i32, %results_37, %results_59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%95, %async_token_60] {
                func.call @maximum_up_u_bf16(%results_55, %results_37) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64 = air.execute [%async_token_61, %async_token_62] {
                func.call @exp_up_minus_u(%results_55, %results_37, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_67 = air.execute [%async_token_64, %async_token_65] {
                func.call @exp_up_minus_u(%results_59, %results_37, %results_66) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_68 = air.execute [%async_token_64, %94] {
                func.call @mul_r_gp(%results_63, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_69 = air.execute [%async_token_38, %async_token_67] {
                func.call @mul_r_gp(%results_66, %results_39) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_68, %async_token_69] {
                func.call @add_gp_g(%results_39, %results_53) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_71] {
                func.call @zero_fill_sp_bf16(%results_72) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_74 = air.execute [%async_token_73, %async_token_68, %96] {
                func.call @accum_sp_r_s(%results_57, %results_63, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_75 = air.execute [%async_token_34, %async_token_74, %async_token_69] {
                func.call @accum_sp_r_s(%results_35, %results_66, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_75] {
                func.call @vector_copy_32elems(%c0_i32, %results_72, %results_57) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_76, %async_token_70] {
                func.call @div_gp_sp(%results_57, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %97 = air.channel.put async [%async_token_77]  @Gp2L2[%arg16, %c0_31] (%results_53[%c0_31, %c0_31, %c0_31] [%c64_29, %c8_33, %c8_33] [%c8_33, %c512, %c1_32]) {id = 137 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_78 = air.execute [%97] {
                memref.dealloc %results_53 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_79 = air.execute [%async_token_64] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_80 = air.execute [%async_token_77] {
                memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_81 = air.execute [%async_token_67] {
                memref.dealloc %results_59 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_82 = air.execute [%async_token_74] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_75] {
                memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_84 = air.execute [%async_token_76] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %97 : !air.async.token
            }
            affine.yield %91 : !air.async.token
          }
          %async_token_47 = air.execute [%91] {
            memref.dealloc %results_43 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_48 = air.execute [%91, %88, %87, %86, %85, %83, %82, %81, %80, %78, %77, %76, %75, %72, %71, %70, %69] {
            memref.dealloc %results_41 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_49 = air.execute [%92, %91, %async_token_44] {
            memref.dealloc %results_39 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_50 = air.execute [%92, %91, %async_token_46] {
            memref.dealloc %results_37 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_51 = air.execute [%92, %91, %async_token_45] {
            memref.dealloc %results_35 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_21 = air.execute [%49] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_22 = air.execute [%51] {
          memref.dealloc %results_8 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_23 = air.execute [%53] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_24 = air.execute [%55] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_25 = air.execute [%67] {
          memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_26 = air.execute [%66] {
          memref.dealloc %results_18 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_27 = air.execute [%65] {
          memref.dealloc %results_16 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_28 = air.execute [%64] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        air.wait_all [%56, %57, %58, %59, %68, %async_token_21, %async_token_22, %async_token_23, %async_token_24, %async_token_25, %async_token_26, %async_token_27, %async_token_28]  {air.segment_end}
      }
    }
    return
  }
}

