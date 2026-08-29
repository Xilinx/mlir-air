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
        %c512_4 = arith.constant 512 : index
        %c1_5 = arith.constant 1 : index
        %c8_6 = arith.constant 8 : index
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
        %async_token_17, %results_18 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_23, %results_24 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_25, %results_26 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
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
        %async_token_33, %results_34 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_35, %results_36 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %44 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %57 = air.channel.get async [%arg17]  @VIn_0[%arg12] (%results[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %58 = arith.cmpi eq, %arg12, %c0_7 : index
          %59 = scf.if %58 -> (!air.async.token) {
            %60 = air.channel.put async [%57]  @V2L1_0_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %60 : !air.async.token
          } else {
            %60 = air.channel.put async [%57]  @V2L1_0_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %60 : !air.async.token
          }
          scf.yield %59 : !air.async.token
        }
        %45 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %async_token_9) -> (!air.async.token) {
          %57 = air.channel.get async [%arg17]  @VIn_1[%arg12] (%results_10[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
          %58 = arith.cmpi eq, %arg12, %c0_7 : index
          %59 = scf.if %58 -> (!air.async.token) {
            %60 = air.channel.put async [%57]  @V2L1_1_0[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %60 : !air.async.token
          } else {
            %60 = air.channel.put async [%57]  @V2L1_1_1[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %60 : !air.async.token
          }
          scf.yield %59 : !air.async.token
        }
        %46 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %async_token_11) -> (!air.async.token) {
          %57 = air.channel.get async [%arg17]  @VIn_2[%arg12] (%results_12[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %58 = arith.cmpi eq, %arg12, %c0_7 : index
          %59 = scf.if %58 -> (!air.async.token) {
            %60 = air.channel.put async [%57]  @V2L1_2_0[%c0_7, %c0_7, %c0_7] (%results_12[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %60 : !air.async.token
          } else {
            %60 = air.channel.put async [%57]  @V2L1_2_1[%c0_7, %c0_7, %c0_7] (%results_12[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %60 : !air.async.token
          }
          scf.yield %59 : !air.async.token
        }
        %47 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %async_token_13) -> (!air.async.token) {
          %57 = air.channel.get async [%arg17]  @VIn_3[%arg12] (%results_14[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
          %58 = arith.cmpi eq, %arg12, %c0_7 : index
          %59 = scf.if %58 -> (!air.async.token) {
            %60 = air.channel.put async [%57]  @V2L1_3_0[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %60 : !air.async.token
          } else {
            %60 = air.channel.put async [%57]  @V2L1_3_1[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %60 : !air.async.token
          }
          scf.yield %59 : !air.async.token
        }
        %48 = air.channel.get async [%async_token_15]  @Gp2L2[%c0_7, %c0_7] (%results_16[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %49 = air.channel.get async [%async_token_17]  @Gp2L2[%c1_5, %c0_7] (%results_18[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %50 = air.channel.get async [%async_token_19]  @Gp2L2[%c2_2, %c0_7] (%results_20[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %51 = air.channel.get async [%async_token_21]  @Gp2L2[%c3_1, %c0_7] (%results_22[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %52 = air.channel.put async [%48]  @channel_0[%c0_7, %arg12] (%results_16[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %53 = air.channel.put async [%49]  @channel_0[%c1_5, %arg12] (%results_18[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %54 = air.channel.put async [%50]  @channel_0[%c2_2, %arg12] (%results_20[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %55 = air.channel.put async [%51]  @channel_0[%c3_1, %arg12] (%results_22[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %56 = air.herd @herd_0 async [%async_token_23, %async_token_25, %async_token_27, %async_token_29, %async_token_31, %async_token_33, %async_token_35]  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%results_24, %arg21=%results_26, %arg22=%results_28, %arg23=%results_30, %arg24=%results_32, %arg25=%results_34, %arg26=%results_36, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {id = 3 : i32, link_with = "attn.o"} {
          %c512_52 = arith.constant 512 : index
          %c64_53 = arith.constant 64 : index
          %c1_54 = arith.constant 1 : index
          %c0_55 = arith.constant 0 : index
          %c8_56 = arith.constant 8 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_57 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_58 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_59 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %57 = arith.cmpi eq, %arg27, %c0_55 : index
          scf.if %57 {
            %66 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %67 = air.channel.get async  @QK2L1_0_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %67 : !air.async.token
            } else {
              %67 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %68 = air.channel.get async  @QK2L1_0_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %68 : !air.async.token
              } else {
                %68 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %69 = air.channel.get async  @QK2L1_0_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                } else {
                  %69 = air.channel.get async  @QK2L1_0_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                }
                affine.yield %68 : !air.async.token
              }
              affine.yield %67 : !air.async.token
            }
          } else {
            %66 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %67 = air.channel.get async  @QK2L1_1_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %67 : !air.async.token
            } else {
              %67 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %68 = air.channel.get async  @QK2L1_1_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %68 : !air.async.token
              } else {
                %68 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %69 = air.channel.get async  @QK2L1_1_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                } else {
                  %69 = air.channel.get async  @QK2L1_1_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                }
                affine.yield %68 : !air.async.token
              }
              affine.yield %67 : !air.async.token
            }
          }
          %58 = arith.index_cast %arg16 : index to i32
          %59 = arith.cmpi eq, %58, %c0_i32 : i32
          scf.if %59 {
            %async_token_60 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %57 {
            %66 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %67 = air.channel.get async  @QK2L1_0_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %67 : !air.async.token
            } else {
              %67 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %68 = air.channel.get async  @QK2L1_0_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %68 : !air.async.token
              } else {
                %68 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %69 = air.channel.get async  @QK2L1_0_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                } else {
                  %69 = air.channel.get async  @QK2L1_0_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                }
                affine.yield %68 : !air.async.token
              }
              affine.yield %67 : !air.async.token
            }
          } else {
            %66 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %67 = air.channel.get async  @QK2L1_1_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %67 : !air.async.token
            } else {
              %67 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %68 = air.channel.get async  @QK2L1_1_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %68 : !air.async.token
              } else {
                %68 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %69 = air.channel.get async  @QK2L1_1_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                } else {
                  %69 = air.channel.get async  @QK2L1_1_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                }
                affine.yield %68 : !air.async.token
              }
              affine.yield %67 : !air.async.token
            }
          }
          %60 = arith.cmpi eq, %58, %c1_i32 : i32
          scf.if %60 {
            %async_token_60 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %57 {
            %66 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %67 = air.channel.get async  @QK2L1_0_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %67 : !air.async.token
            } else {
              %67 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %68 = air.channel.get async  @QK2L1_0_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %68 : !air.async.token
              } else {
                %68 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %69 = air.channel.get async  @QK2L1_0_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                } else {
                  %69 = air.channel.get async  @QK2L1_0_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                }
                affine.yield %68 : !air.async.token
              }
              affine.yield %67 : !air.async.token
            }
          } else {
            %66 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %67 = air.channel.get async  @QK2L1_1_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %67 : !air.async.token
            } else {
              %67 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %68 = air.channel.get async  @QK2L1_1_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %68 : !air.async.token
              } else {
                %68 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %69 = air.channel.get async  @QK2L1_1_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                } else {
                  %69 = air.channel.get async  @QK2L1_1_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                }
                affine.yield %68 : !air.async.token
              }
              affine.yield %67 : !air.async.token
            }
          }
          %61 = arith.cmpi eq, %58, %c2_i32 : i32
          scf.if %61 {
            %async_token_60 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %57 {
            %66 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %67 = air.channel.get async  @QK2L1_0_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %67 : !air.async.token
            } else {
              %67 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %68 = air.channel.get async  @QK2L1_0_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %68 : !air.async.token
              } else {
                %68 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %69 = air.channel.get async  @QK2L1_0_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                } else {
                  %69 = air.channel.get async  @QK2L1_0_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                }
                affine.yield %68 : !air.async.token
              }
              affine.yield %67 : !air.async.token
            }
          } else {
            %66 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %67 = air.channel.get async  @QK2L1_1_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %67 : !air.async.token
            } else {
              %67 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %68 = air.channel.get async  @QK2L1_1_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %68 : !air.async.token
              } else {
                %68 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %69 = air.channel.get async  @QK2L1_1_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                } else {
                  %69 = air.channel.get async  @QK2L1_1_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %69 : !air.async.token
                }
                affine.yield %68 : !air.async.token
              }
              affine.yield %67 : !air.async.token
            }
          }
          %62 = arith.cmpi eq, %58, %c3_i32 : i32
          scf.if %62 {
            %async_token_60 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %63 = air.wait_all async [%async_token_57, %async_token_58, %async_token_59] 
          %64 = scf.for %arg28 = %c0_55 to %c8_56 step %c1_54 iter_args(%arg29 = %63) -> (!air.async.token) {
            %async_token_60 = air.execute [%arg29] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %57 {
              %71 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%arg29]  @QK2L1_0_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%arg29]  @QK2L1_0_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %74 = air.channel.get async [%arg29]  @QK2L1_0_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %74 : !air.async.token
                  } else {
                    %74 = air.channel.get async [%arg29]  @QK2L1_0_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %74 : !air.async.token
                  }
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
            } else {
              %71 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%arg29]  @QK2L1_1_0[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%arg29]  @QK2L1_1_1[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %74 = air.channel.get async [%arg29]  @QK2L1_1_2[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %74 : !air.async.token
                  } else {
                    %74 = air.channel.get async [%arg29]  @QK2L1_1_3[%c0_55, %c0_55, %arg16] (%arg21[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %74 : !air.async.token
                  }
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
            }
            %66 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %71 = scf.if %57 -> (!air.async.token) {
                %72 = air.channel.get async  @V2L1_0_0[%c0_55, %arg17, %arg16] (%arg22[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async  @V2L1_0_1[%c0_55, %arg17, %arg16] (%arg22[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            } else {
              %71 = air.wait_all async 
              affine.yield %71 : !air.async.token
            }
            %67 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %71 = scf.if %57 -> (!air.async.token) {
                %72 = air.channel.get async [%arg29, %66]  @V2L1_1_0[%c0_55, %arg17, %arg16] (%arg22[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%arg29, %66]  @V2L1_1_1[%c0_55, %arg17, %arg16] (%arg22[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            } else {
              %71 = air.wait_all async 
              affine.yield %71 : !air.async.token
            }
            %68 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %71 = scf.if %57 -> (!air.async.token) {
                %72 = air.channel.get async [%arg29, %67]  @V2L1_2_0[%c0_55, %arg17, %arg16] (%arg22[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%arg29, %67]  @V2L1_2_1[%c0_55, %arg17, %arg16] (%arg22[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            } else {
              %71 = air.wait_all async 
              affine.yield %71 : !air.async.token
            }
            %69 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %71 = scf.if %57 -> (!air.async.token) {
                %72 = air.channel.get async [%arg29, %68]  @V2L1_3_0[%c0_55, %arg17, %arg16] (%arg22[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%arg29, %68]  @V2L1_3_1[%c0_55, %arg17, %arg16] (%arg22[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            } else {
              %71 = air.wait_all async 
              affine.yield %71 : !air.async.token
            }
            %async_token_61 = air.execute [%async_token_60] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_66 = air.execute [%async_token_64, %async_token_62, %async_token_61] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_63, %results_65) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_67 = air.execute [%async_token_66] {
              func.call @mul_r_gp(%results_65, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_68 = air.execute [%async_token_67, %69] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_69 = air.execute [%async_token_67] {
              func.call @accum_sp_r_s(%arg26, %results_65, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_70 = air.execute [%async_token_69] {
              func.call @vector_copy_32elems(%c0_i32, %results_63, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_71 = air.execute [%async_token_70] {
              memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_72 = air.execute [%async_token_69] {
              memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
            }
            %70 = air.wait_all async [%66, %67, %68, %async_token_68, %async_token_70] 
            scf.yield %70 : !air.async.token
          }
          %65 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %66 = arith.subi %arg17, %c1_54 : index
            %67 = air.channel.put async [%64]  @cascade_gp[%arg16, %66] (%arg24[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
            %68 = air.channel.put async [%64]  @cascade_up[%arg16, %66] (%arg25[] [] []) {id = 102 : i32} : (memref<64x1xbf16, 2 : i32>)
            %69 = air.channel.put async [%64]  @cascade_sp[%arg16, %66] (%arg26[] [] []) {id = 103 : i32} : (memref<64x1xbf16, 2 : i32>)
            %70 = air.wait_all async [%67, %68, %69] 
            affine.yield %70 : !air.async.token
          } else {
            %66 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_60, %results_61 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %67 = air.channel.get async [%async_token_60]  @cascade_gp[%arg16, %arg17] (%results_61[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              %68 = air.channel.get async [%async_token_62]  @cascade_up[%arg16, %arg17] (%results_63[] [] []) {id = 105 : i32} : (memref<64x1xbf16, 2 : i32>)
              %69 = air.channel.get async [%async_token_64]  @cascade_sp[%arg16, %arg17] (%results_65[] [] []) {id = 106 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_66, %results_67 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_68 = air.execute [%async_token_66, %64] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_67) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_69 = air.execute [%async_token_68, %68] {
                func.call @maximum_up_u_bf16(%results_63, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_70, %results_71 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_70, %async_token_69] {
                func.call @exp_up_minus_u(%results_63, %arg25, %results_71) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_73, %async_token_72] {
                func.call @exp_up_minus_u(%results_67, %arg25, %results_74) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_72, %67] {
                func.call @mul_r_gp(%results_71, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_75] {
                func.call @mul_r_gp(%results_74, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_78 = air.execute [%async_token_77, %async_token_76] {
                func.call @add_gp_g(%arg24, %results_61) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_79, %results_80 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_81 = air.execute [%async_token_79] {
                func.call @zero_fill_sp_bf16(%results_80) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_82 = air.execute [%async_token_81, %async_token_76, %69] {
                func.call @accum_sp_r_s(%results_65, %results_71, %results_80) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_83 = air.execute [%async_token_82, %async_token_77] {
                func.call @accum_sp_r_s(%arg26, %results_74, %results_80) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_84 = air.execute [%async_token_83] {
                func.call @vector_copy_32elems(%c0_i32, %results_80, %results_65) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %70 = arith.subi %arg17, %c1_54 : index
              %71 = air.channel.put async [%async_token_78]  @cascade_gp[%arg16, %70] (%results_61[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              %72 = air.channel.put async [%async_token_75]  @cascade_up[%arg16, %70] (%arg25[] [] []) {id = 108 : i32} : (memref<64x1xbf16, 2 : i32>)
              %73 = air.channel.put async [%async_token_84]  @cascade_sp[%arg16, %70] (%results_65[] [] []) {id = 109 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_85 = air.execute [%71] {
                memref.dealloc %results_61 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_86 = air.execute [%async_token_72] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_87 = air.execute [%73] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_88 = air.execute [%async_token_75] {
                memref.dealloc %results_67 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_89 = air.execute [%async_token_82] {
                memref.dealloc %results_71 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_90 = air.execute [%async_token_83] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_91 = air.execute [%async_token_84] {
                memref.dealloc %results_80 : memref<64x1xbf16, 2 : i32>
              }
              %74 = air.wait_all async [%71, %72, %73] 
              affine.yield %74 : !air.async.token
            } else {
              %async_token_60, %results_61 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %67 = air.channel.get async [%async_token_60]  @cascade_gp[%arg16, %arg17] (%results_61[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              %68 = air.channel.get async [%async_token_62]  @cascade_up[%arg16, %arg17] (%results_63[] [] []) {id = 111 : i32} : (memref<64x1xbf16, 2 : i32>)
              %69 = air.channel.get async [%async_token_64]  @cascade_sp[%arg16, %arg17] (%results_65[] [] []) {id = 112 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_66, %results_67 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_68 = air.execute [%async_token_66, %64] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_67) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_69 = air.execute [%async_token_68, %68] {
                func.call @maximum_up_u_bf16(%results_63, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_70, %results_71 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_70, %async_token_69] {
                func.call @exp_up_minus_u(%results_63, %arg25, %results_71) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_73, %async_token_72] {
                func.call @exp_up_minus_u(%results_67, %arg25, %results_74) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_72, %67] {
                func.call @mul_r_gp(%results_71, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_75] {
                func.call @mul_r_gp(%results_74, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_78 = air.execute [%async_token_77, %async_token_76] {
                func.call @add_gp_g(%arg24, %results_61) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_79, %results_80 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_81 = air.execute [%async_token_79] {
                func.call @zero_fill_sp_bf16(%results_80) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_82 = air.execute [%async_token_81, %async_token_76, %69] {
                func.call @accum_sp_r_s(%results_65, %results_71, %results_80) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_83 = air.execute [%async_token_82, %async_token_77] {
                func.call @accum_sp_r_s(%arg26, %results_74, %results_80) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_84 = air.execute [%async_token_83] {
                func.call @vector_copy_32elems(%c0_i32, %results_80, %results_65) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_85 = air.execute [%async_token_84, %async_token_78] {
                func.call @div_gp_sp(%results_65, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %70 = air.channel.put async [%async_token_85]  @Gp2L2[%arg16, %c0_55] (%results_61[%c0_55, %c0_55, %c0_55, %c0_55] [%c8_56, %c8_56, %c8_56, %c8_56] [%c64_53, %c8_56, %c512_52, %c1_54]) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_86 = air.execute [%70] {
                memref.dealloc %results_61 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_87 = air.execute [%async_token_72] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_88 = air.execute [%async_token_85] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_89 = air.execute [%async_token_75] {
                memref.dealloc %results_67 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_90 = air.execute [%async_token_82] {
                memref.dealloc %results_71 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_91 = air.execute [%async_token_83] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_92 = air.execute [%async_token_84] {
                memref.dealloc %results_80 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %70 : !air.async.token
            }
            affine.yield %64 : !air.async.token
          }
        }
        %async_token_37 = air.execute [%56] {
          memref.dealloc %results_24 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_38 = air.execute [%56] {
          memref.dealloc %results_26 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_39 = air.execute [%56] {
          memref.dealloc %results_28 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_40 = air.execute [%56] {
          memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_41 = air.execute [%56] {
          memref.dealloc %results_32 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_42 = air.execute [%56] {
          memref.dealloc %results_34 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_43 = air.execute [%56] {
          memref.dealloc %results_36 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_44 = air.execute [%44] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_45 = air.execute [%45] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_46 = air.execute [%46] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_47 = air.execute [%47] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_48 = air.execute [%55] {
          memref.dealloc %results_22 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_49 = air.execute [%54] {
          memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_50 = air.execute [%53] {
          memref.dealloc %results_18 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_51 = air.execute [%52] {
          memref.dealloc %results_16 : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
