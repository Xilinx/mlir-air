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
#map10 = affine_map<()[s0] -> (s0 * 64)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set6 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set7 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
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
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<12x2048x64xbf16>, %arg1: memref<12x2048x64xbf16>, %arg2: memref<12x2048x64xbf16>, %arg3: memref<12x2048x64xbf16>) {
    %c8 = arith.constant 8 : index
    %c6 = arith.constant 6 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c8, %arg7=%c6) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<12x2048x64xbf16>, memref<12x2048x64xbf16>, memref<12x2048x64xbf16>, memref<12x2048x64xbf16> attributes {id = 3 : i32} {
      %c16384 = arith.constant 16384 : index
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
      %18 = air.channel.get async  @GpOut[%c0] (%arg11[%1] [%c16384] [%c1]) {id = 13 : i32} : (memref<12x2048x64xbf16>)
      %19 = affine.apply #map5()[%arg5, %arg4]
      %20 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %19] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 14 : i32} : (memref<12x2048x64xbf16>)
      %21 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %19] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 15 : i32} : (memref<12x2048x64xbf16>)
      %22 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %19] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 16 : i32} : (memref<12x2048x64xbf16>)
      %23 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %19] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 17 : i32} : (memref<12x2048x64xbf16>)
      %24 = affine.apply #map6()[%arg5]
      %25 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %24] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 18 : i32} : (memref<12x2048x64xbf16>)
      %26 = affine.apply #map7()[%arg5]
      %27 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %26] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 19 : i32} : (memref<12x2048x64xbf16>)
      %28 = affine.apply #map8()[%arg5]
      %29 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %28] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 20 : i32} : (memref<12x2048x64xbf16>)
      %30 = affine.apply #map9()[%arg5]
      %31 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %30] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 21 : i32} : (memref<12x2048x64xbf16>)
      %32 = air.channel.put async  @VIn_0[%c1] (%arg10[%c0, %c0, %24] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 22 : i32} : (memref<12x2048x64xbf16>)
      %33 = air.channel.put async  @VIn_1[%c1] (%arg10[%c0, %c0, %26] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 23 : i32} : (memref<12x2048x64xbf16>)
      %34 = air.channel.put async  @VIn_2[%c1] (%arg10[%c0, %c0, %28] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 24 : i32} : (memref<12x2048x64xbf16>)
      %35 = air.channel.put async  @VIn_3[%c1] (%arg10[%c0, %c0, %30] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 25 : i32} : (memref<12x2048x64xbf16>)
      %36 = air.channel.get async  @GpOut[%c1] (%arg11[%19] [%c16384] [%c1]) {id = 26 : i32} : (memref<12x2048x64xbf16>)
      %37 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1) attributes {id = 2 : i32} {
        %c64_1 = arith.constant 64 : index
        %c512_2 = arith.constant 512 : index
        %c1_3 = arith.constant 1 : index
        %c8_4 = arith.constant 8 : index
        %c0_5 = arith.constant 0 : index
        %c4_6 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 1 : i32}
        %async_token_7, %results_8 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 2 : i32}
        %async_token_9, %results_10 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 3 : i32}
        %async_token_11, %results_12 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 4 : i32}
        %async_token_13, %results_14 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        } {id = 5 : i32}
        %async_token_15, %results_16 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 6 : i32}
        %async_token_17, %results_18 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 7 : i32}
        %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 8 : i32}
        %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 9 : i32}
        %async_token_23, %results_24 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 10 : i32}
        %async_token_25, %results_26 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 11 : i32}
        %async_token_27, %results_28 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 12 : i32}
        %38 = scf.for %arg16 = %c0_5 to %c8_4 step %c1_3 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %45 = air.channel.get async [%arg17]  @VIn_0[%arg12] (%results[] [] []) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
          %46 = arith.cmpi eq, %arg12, %c0_5 : index
          %47 = scf.if %46 -> (!air.async.token) {
            %48 = air.channel.put async [%45]  @V2L1_0_0[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5, %c0_5] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_2, %c64_1, %c1_3]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %48 : !air.async.token
          } else {
            %48 = air.channel.put async [%45]  @V2L1_0_1[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5, %c0_5] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_2, %c64_1, %c1_3]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %48 : !air.async.token
          }
          scf.yield %47 : !air.async.token
        }
        %39 = scf.for %arg16 = %c0_5 to %c8_4 step %c1_3 iter_args(%arg17 = %async_token_7) -> (!air.async.token) {
          %45 = air.channel.get async [%arg17]  @VIn_1[%arg12] (%results_8[] [] []) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
          %46 = arith.cmpi eq, %arg12, %c0_5 : index
          %47 = scf.if %46 -> (!air.async.token) {
            %48 = air.channel.put async [%45]  @V2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_8[%c0_5, %c0_5, %c0_5, %c0_5] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_2, %c64_1, %c1_3]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %48 : !air.async.token
          } else {
            %48 = air.channel.put async [%45]  @V2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_8[%c0_5, %c0_5, %c0_5, %c0_5] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_2, %c64_1, %c1_3]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %48 : !air.async.token
          }
          scf.yield %47 : !air.async.token
        }
        %40 = scf.for %arg16 = %c0_5 to %c8_4 step %c1_3 iter_args(%arg17 = %async_token_9) -> (!air.async.token) {
          %45 = air.channel.get async [%arg17]  @VIn_2[%arg12] (%results_10[] [] []) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
          %46 = arith.cmpi eq, %arg12, %c0_5 : index
          %47 = scf.if %46 -> (!air.async.token) {
            %48 = air.channel.put async [%45]  @V2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_10[%c0_5, %c0_5, %c0_5, %c0_5] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_2, %c64_1, %c1_3]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %48 : !air.async.token
          } else {
            %48 = air.channel.put async [%45]  @V2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_10[%c0_5, %c0_5, %c0_5, %c0_5] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_2, %c64_1, %c1_3]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %48 : !air.async.token
          }
          scf.yield %47 : !air.async.token
        }
        %41 = scf.for %arg16 = %c0_5 to %c8_4 step %c1_3 iter_args(%arg17 = %async_token_11) -> (!air.async.token) {
          %45 = air.channel.get async [%arg17]  @VIn_3[%arg12] (%results_12[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %46 = arith.cmpi eq, %arg12, %c0_5 : index
          %47 = scf.if %46 -> (!air.async.token) {
            %48 = air.channel.put async [%45]  @V2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_12[%c0_5, %c0_5, %c0_5, %c0_5] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_2, %c64_1, %c1_3]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %48 : !air.async.token
          } else {
            %48 = air.channel.put async [%45]  @V2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_12[%c0_5, %c0_5, %c0_5, %c0_5] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_2, %c64_1, %c1_3]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %48 : !air.async.token
          }
          scf.yield %47 : !air.async.token
        }
        %42 = scf.parallel (%arg16) = (%c0_5) to (%c4_6) step (%c1_3) init (%async_token_13) -> !air.async.token {
          %45 = affine.apply #map10()[%arg16]
          %46 = air.channel.get async [%async_token_13]  @Gp2L2[%arg16, %c0_5] (%results_14[%45, %c0_5] [%c64_1, %c64_1] [%c64_1, %c1_3]) {id = 35 : i32} : (memref<256x64xbf16, 1 : i32>)
          scf.reduce(%46 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %47 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %47 : !air.async.token
          }
        }
        %43 = air.channel.put async [%42]  @GpOut[%arg12] (%results_14[] [] []) {id = 36 : i32} : (memref<256x64xbf16, 1 : i32>)
        %44 = air.herd @herd_0 async [%async_token_15, %async_token_17, %async_token_19, %async_token_21, %async_token_23, %async_token_25, %async_token_27]  tile (%arg16, %arg17) in (%arg18=%c4_6, %arg19=%c4_6) args(%arg20=%results_16, %arg21=%results_18, %arg22=%results_20, %arg23=%results_22, %arg24=%results_24, %arg25=%results_26, %arg26=%results_28, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {id = 1 : i32, link_with = "attn.o"} {
          %c512_41 = arith.constant 512 : index
          %c64_42 = arith.constant 64 : index
          %c1_43 = arith.constant 1 : index
          %c0_44 = arith.constant 0 : index
          %c8_45 = arith.constant 8 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_46 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          } {id = 13 : i32}
          %async_token_47 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 14 : i32}
          %async_token_48 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 15 : i32}
          %45 = arith.cmpi eq, %arg27, %c0_44 : index
          %46 = scf.if %45 -> (!air.async.token) {
            %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %59 = air.channel.get async  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
              affine.yield %59 : !air.async.token
            } else {
              %59 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %60 = air.channel.get async  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %60 : !air.async.token
              } else {
                %60 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %61 = air.channel.get async  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                } else {
                  %61 = air.channel.get async  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                }
                affine.yield %60 : !air.async.token
              }
              affine.yield %59 : !air.async.token
            }
            scf.yield %58 : !air.async.token
          } else {
            %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %59 = air.channel.get async  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
              affine.yield %59 : !air.async.token
            } else {
              %59 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %60 = air.channel.get async  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %60 : !air.async.token
              } else {
                %60 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %61 = air.channel.get async  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                } else {
                  %61 = air.channel.get async  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                }
                affine.yield %60 : !air.async.token
              }
              affine.yield %59 : !air.async.token
            }
            scf.yield %58 : !air.async.token
          }
          %47 = arith.index_cast %arg16 : index to i32
          %48 = arith.cmpi eq, %47, %c0_i32 : i32
          scf.if %48 {
            %async_token_49 = air.execute [%46] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 16 : i32}
          }
          %49 = scf.if %45 -> (!air.async.token) {
            %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %59 = air.channel.get async  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
              affine.yield %59 : !air.async.token
            } else {
              %59 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %60 = air.channel.get async  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %60 : !air.async.token
              } else {
                %60 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %61 = air.channel.get async  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                } else {
                  %61 = air.channel.get async  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                }
                affine.yield %60 : !air.async.token
              }
              affine.yield %59 : !air.async.token
            }
            scf.yield %58 : !air.async.token
          } else {
            %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %59 = air.channel.get async  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
              affine.yield %59 : !air.async.token
            } else {
              %59 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %60 = air.channel.get async  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %60 : !air.async.token
              } else {
                %60 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %61 = air.channel.get async  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                } else {
                  %61 = air.channel.get async  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                }
                affine.yield %60 : !air.async.token
              }
              affine.yield %59 : !air.async.token
            }
            scf.yield %58 : !air.async.token
          }
          %50 = arith.cmpi eq, %47, %c1_i32 : i32
          scf.if %50 {
            %async_token_49 = air.execute [%49] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 17 : i32}
          }
          %51 = scf.if %45 -> (!air.async.token) {
            %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %59 = air.channel.get async  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
              affine.yield %59 : !air.async.token
            } else {
              %59 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %60 = air.channel.get async  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %60 : !air.async.token
              } else {
                %60 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %61 = air.channel.get async  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                } else {
                  %61 = air.channel.get async  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                }
                affine.yield %60 : !air.async.token
              }
              affine.yield %59 : !air.async.token
            }
            scf.yield %58 : !air.async.token
          } else {
            %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %59 = air.channel.get async  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
              affine.yield %59 : !air.async.token
            } else {
              %59 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %60 = air.channel.get async  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %60 : !air.async.token
              } else {
                %60 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %61 = air.channel.get async  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                } else {
                  %61 = air.channel.get async  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                }
                affine.yield %60 : !air.async.token
              }
              affine.yield %59 : !air.async.token
            }
            scf.yield %58 : !air.async.token
          }
          %52 = arith.cmpi eq, %47, %c2_i32 : i32
          scf.if %52 {
            %async_token_49 = air.execute [%51] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 18 : i32}
          }
          %53 = scf.if %45 -> (!air.async.token) {
            %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %59 = air.channel.get async  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
              affine.yield %59 : !air.async.token
            } else {
              %59 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %60 = air.channel.get async  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %60 : !air.async.token
              } else {
                %60 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %61 = air.channel.get async  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                } else {
                  %61 = air.channel.get async  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                }
                affine.yield %60 : !air.async.token
              }
              affine.yield %59 : !air.async.token
            }
            scf.yield %58 : !air.async.token
          } else {
            %58 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %59 = air.channel.get async  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
              affine.yield %59 : !air.async.token
            } else {
              %59 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %60 = air.channel.get async  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %60 : !air.async.token
              } else {
                %60 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %61 = air.channel.get async  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                } else {
                  %61 = air.channel.get async  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %61 : !air.async.token
                }
                affine.yield %60 : !air.async.token
              }
              affine.yield %59 : !air.async.token
            }
            scf.yield %58 : !air.async.token
          }
          %54 = arith.cmpi eq, %47, %c3_i32 : i32
          scf.if %54 {
            %async_token_49 = air.execute [%53] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 19 : i32}
          }
          %55 = air.wait_all async [%async_token_46, %async_token_47, %async_token_48]  {id = 35 : i32}
          %56 = scf.for %arg28 = %c0_44 to %c8_45 step %c1_43 iter_args(%arg29 = %55) -> (!air.async.token) {
            %async_token_49 = air.execute [%arg29] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            } {id = 20 : i32}
            %58 = scf.if %45 -> (!air.async.token) {
              %64 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %65 = air.channel.get async [%arg29]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %65 : !air.async.token
              } else {
                %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %66 = air.channel.get async [%arg29]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %66 : !air.async.token
                } else {
                  %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %67 = air.channel.get async [%arg29]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %67 : !air.async.token
                  } else {
                    %67 = air.channel.get async [%arg29]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %67 : !air.async.token
                  }
                  affine.yield %66 : !air.async.token
                }
                affine.yield %65 : !air.async.token
              }
              scf.yield %64 : !air.async.token
            } else {
              %64 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %65 = air.channel.get async [%arg29]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                affine.yield %65 : !air.async.token
              } else {
                %65 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %66 = air.channel.get async [%arg29]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %66 : !air.async.token
                } else {
                  %66 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %67 = air.channel.get async [%arg29]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %67 : !air.async.token
                  } else {
                    %67 = air.channel.get async [%arg29]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %67 : !air.async.token
                  }
                  affine.yield %66 : !air.async.token
                }
                affine.yield %65 : !air.async.token
              }
              scf.yield %64 : !air.async.token
            }
            %59 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %64 = scf.if %45 -> (!air.async.token) {
                %65 = air.channel.get async  @V2L1_0_0[%c0_44, %arg17, %arg16] (%arg22[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %65 : !air.async.token
              } else {
                %65 = air.channel.get async  @V2L1_0_1[%c0_44, %arg17, %arg16] (%arg22[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %65 : !air.async.token
              }
              affine.yield %64 : !air.async.token
            } else {
              %64 = air.wait_all async  {id = 25 : i32}
              affine.yield %64 : !air.async.token
            }
            %60 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %64 = scf.if %45 -> (!air.async.token) {
                %65 = air.channel.get async [%arg29, %59]  @V2L1_1_0[%c0_44, %arg17, %arg16] (%arg22[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %65 : !air.async.token
              } else {
                %65 = air.channel.get async [%arg29, %59]  @V2L1_1_1[%c0_44, %arg17, %arg16] (%arg22[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %65 : !air.async.token
              }
              affine.yield %64 : !air.async.token
            } else {
              %64 = air.wait_all async  {id = 28 : i32}
              affine.yield %64 : !air.async.token
            }
            %61 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %64 = scf.if %45 -> (!air.async.token) {
                %65 = air.channel.get async [%arg29, %60]  @V2L1_2_0[%c0_44, %arg17, %arg16] (%arg22[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %65 : !air.async.token
              } else {
                %65 = air.channel.get async [%arg29, %60]  @V2L1_2_1[%c0_44, %arg17, %arg16] (%arg22[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %65 : !air.async.token
              }
              affine.yield %64 : !air.async.token
            } else {
              %64 = air.wait_all async  {id = 31 : i32}
              affine.yield %64 : !air.async.token
            }
            %62 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %64 = scf.if %45 -> (!air.async.token) {
                %65 = air.channel.get async [%arg29, %61]  @V2L1_3_0[%c0_44, %arg17, %arg16] (%arg22[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %65 : !air.async.token
              } else {
                %65 = air.channel.get async [%arg29, %61]  @V2L1_3_1[%c0_44, %arg17, %arg16] (%arg22[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %65 : !air.async.token
              }
              affine.yield %64 : !air.async.token
            } else {
              %64 = air.wait_all async  {id = 34 : i32}
              affine.yield %64 : !air.async.token
            }
            %async_token_50 = air.execute [%async_token_49, %58] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            } {id = 21 : i32}
            %async_token_51, %results_52 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 22 : i32}
            %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 23 : i32}
            %async_token_55 = air.execute [%async_token_53, %async_token_51, %async_token_50, %arg29] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_52, %results_54) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 24 : i32}
            %async_token_56 = air.execute [%async_token_55] {
              func.call @mul_r_gp(%results_54, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 25 : i32}
            %async_token_57 = air.execute [%arg29, %async_token_56, %62] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 26 : i32}
            %async_token_58 = air.execute [%async_token_56, %arg29] {
              func.call @accum_sp_r_s(%arg26, %results_54, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 27 : i32}
            %async_token_59 = air.execute [%async_token_58] {
              func.call @vector_copy_32elems(%c0_i32, %results_52, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 28 : i32}
            %async_token_60 = air.execute [%async_token_59] {
              memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
            } {id = 29 : i32}
            %async_token_61 = air.execute [%async_token_58] {
              memref.dealloc %results_54 : memref<64x1xbf16, 2 : i32>
            } {id = 30 : i32}
            %63 = air.wait_all async [%59, %60, %61, %async_token_57, %async_token_59]  {id = 36 : i32}
            scf.yield %63 : !air.async.token
          }
          %57 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %58 = arith.subi %arg17, %c1_43 : index
            %59 = air.channel.put async [%56]  @cascade_gp[%arg16, %58] (%arg24[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
            %60 = air.channel.put async [%56]  @cascade_up[%arg16, %58] (%arg25[] [] []) {id = 47 : i32} : (memref<64x1xbf16, 2 : i32>)
            %61 = air.channel.put async [%56]  @cascade_sp[%arg16, %58] (%arg26[] [] []) {id = 48 : i32} : (memref<64x1xbf16, 2 : i32>)
            %62 = air.wait_all async [%59, %60, %61]  {id = 41 : i32}
            affine.yield %62 : !air.async.token
          } else {
            %58 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_49, %results_50 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 31 : i32}
              %async_token_51, %results_52 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 32 : i32}
              %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 33 : i32}
              %59 = air.channel.get async [%async_token_49]  @cascade_gp[%arg16, %arg17] (%results_50[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
              %60 = air.channel.get async [%async_token_51]  @cascade_up[%arg16, %arg17] (%results_52[] [] []) {id = 50 : i32} : (memref<64x1xbf16, 2 : i32>)
              %61 = air.channel.get async [%async_token_53]  @cascade_sp[%arg16, %arg17] (%results_54[] [] []) {id = 51 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_55, %results_56 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 34 : i32}
              %async_token_57 = air.execute [%async_token_55, %56] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_56) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 35 : i32}
              %async_token_58 = air.execute [%async_token_57, %60] {
                func.call @maximum_up_u_bf16(%results_52, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 36 : i32}
              %async_token_59, %results_60 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 37 : i32}
              %async_token_61 = air.execute [%async_token_59, %async_token_58] {
                func.call @exp_up_minus_u(%results_52, %arg25, %results_60) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 38 : i32}
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 39 : i32}
              %async_token_64 = air.execute [%async_token_62, %async_token_61] {
                func.call @exp_up_minus_u(%results_56, %arg25, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 40 : i32}
              %async_token_65 = air.execute [%async_token_61, %59] {
                func.call @mul_r_gp(%results_60, %results_50) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 41 : i32}
              %async_token_66 = air.execute [%async_token_64, %56] {
                func.call @mul_r_gp(%results_63, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 42 : i32}
              %async_token_67 = air.execute [%async_token_66, %async_token_65] {
                func.call @add_gp_g(%arg24, %results_50) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 43 : i32}
              %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 44 : i32}
              %async_token_70 = air.execute [%async_token_68] {
                func.call @zero_fill_sp_bf16(%results_69) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 45 : i32}
              %async_token_71 = air.execute [%async_token_70, %async_token_65, %61] {
                func.call @accum_sp_r_s(%results_54, %results_60, %results_69) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 46 : i32}
              %async_token_72 = air.execute [%async_token_66, %async_token_71] {
                func.call @accum_sp_r_s(%arg26, %results_63, %results_69) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 47 : i32}
              %async_token_73 = air.execute [%async_token_72] {
                func.call @vector_copy_32elems(%c0_i32, %results_69, %results_54) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 48 : i32}
              %62 = arith.subi %arg17, %c1_43 : index
              %63 = air.channel.put async [%async_token_67]  @cascade_gp[%arg16, %62] (%results_50[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
              %64 = air.channel.put async [%async_token_64]  @cascade_up[%arg16, %62] (%arg25[] [] []) {id = 53 : i32} : (memref<64x1xbf16, 2 : i32>)
              %65 = air.channel.put async [%async_token_73]  @cascade_sp[%arg16, %62] (%results_54[] [] []) {id = 54 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_74 = air.execute [%63] {
                memref.dealloc %results_50 : memref<64x64xbf16, 2 : i32>
              } {id = 49 : i32}
              %async_token_75 = air.execute [%async_token_61] {
                memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
              } {id = 50 : i32}
              %async_token_76 = air.execute [%65] {
                memref.dealloc %results_54 : memref<64x1xbf16, 2 : i32>
              } {id = 51 : i32}
              %async_token_77 = air.execute [%async_token_64] {
                memref.dealloc %results_56 : memref<64x1xbf16, 2 : i32>
              } {id = 52 : i32}
              %async_token_78 = air.execute [%async_token_71] {
                memref.dealloc %results_60 : memref<64x1xbf16, 2 : i32>
              } {id = 53 : i32}
              %async_token_79 = air.execute [%async_token_72] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              } {id = 54 : i32}
              %async_token_80 = air.execute [%async_token_73] {
                memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
              } {id = 55 : i32}
              %66 = air.wait_all async [%63, %64, %65]  {id = 38 : i32}
              affine.yield %66 : !air.async.token
            } else {
              %async_token_49, %results_50 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 56 : i32}
              %async_token_51, %results_52 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 57 : i32}
              %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 58 : i32}
              %59 = air.channel.get async [%async_token_49]  @cascade_gp[%arg16, %arg17] (%results_50[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              %60 = air.channel.get async [%async_token_51]  @cascade_up[%arg16, %arg17] (%results_52[] [] []) {id = 56 : i32} : (memref<64x1xbf16, 2 : i32>)
              %61 = air.channel.get async [%async_token_53]  @cascade_sp[%arg16, %arg17] (%results_54[] [] []) {id = 57 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_55, %results_56 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 59 : i32}
              %async_token_57 = air.execute [%async_token_55, %56] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_56) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 60 : i32}
              %async_token_58 = air.execute [%async_token_57, %60] {
                func.call @maximum_up_u_bf16(%results_52, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 61 : i32}
              %async_token_59, %results_60 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 62 : i32}
              %async_token_61 = air.execute [%async_token_59, %async_token_58] {
                func.call @exp_up_minus_u(%results_52, %arg25, %results_60) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 63 : i32}
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 64 : i32}
              %async_token_64 = air.execute [%async_token_62, %async_token_61] {
                func.call @exp_up_minus_u(%results_56, %arg25, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 65 : i32}
              %async_token_65 = air.execute [%async_token_61, %59] {
                func.call @mul_r_gp(%results_60, %results_50) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 66 : i32}
              %async_token_66 = air.execute [%async_token_64, %56] {
                func.call @mul_r_gp(%results_63, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 67 : i32}
              %async_token_67 = air.execute [%async_token_66, %async_token_65] {
                func.call @add_gp_g(%arg24, %results_50) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 68 : i32}
              %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 69 : i32}
              %async_token_70 = air.execute [%async_token_68] {
                func.call @zero_fill_sp_bf16(%results_69) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 70 : i32}
              %async_token_71 = air.execute [%async_token_70, %async_token_65, %61] {
                func.call @accum_sp_r_s(%results_54, %results_60, %results_69) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 71 : i32}
              %async_token_72 = air.execute [%async_token_66, %async_token_71] {
                func.call @accum_sp_r_s(%arg26, %results_63, %results_69) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 72 : i32}
              %async_token_73 = air.execute [%async_token_72] {
                func.call @vector_copy_32elems(%c0_i32, %results_69, %results_54) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 73 : i32}
              %async_token_74 = air.execute [%async_token_73, %async_token_67] {
                func.call @div_gp_sp(%results_54, %results_50) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 74 : i32}
              %62 = air.channel.put async [%async_token_74]  @Gp2L2[%arg16, %c0_44] (%results_50[%c0_44, %c0_44, %c0_44, %c0_44] [%c8_45, %c8_45, %c8_45, %c8_45] [%c64_42, %c8_45, %c512_41, %c1_43]) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_75 = air.execute [%62] {
                memref.dealloc %results_50 : memref<64x64xbf16, 2 : i32>
              } {id = 75 : i32}
              %async_token_76 = air.execute [%async_token_61] {
                memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
              } {id = 76 : i32}
              %async_token_77 = air.execute [%async_token_74] {
                memref.dealloc %results_54 : memref<64x1xbf16, 2 : i32>
              } {id = 77 : i32}
              %async_token_78 = air.execute [%async_token_64] {
                memref.dealloc %results_56 : memref<64x1xbf16, 2 : i32>
              } {id = 78 : i32}
              %async_token_79 = air.execute [%async_token_71] {
                memref.dealloc %results_60 : memref<64x1xbf16, 2 : i32>
              } {id = 79 : i32}
              %async_token_80 = air.execute [%async_token_72] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              } {id = 80 : i32}
              %async_token_81 = air.execute [%async_token_73] {
                memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
              } {id = 81 : i32}
              affine.yield %62 : !air.async.token
            }
            affine.yield %56 : !air.async.token
          }
        }
        %async_token_29 = air.execute [%44] {
          memref.dealloc %results_16 : memref<64x64xbf16, 2 : i32>
        } {id = 82 : i32}
        %async_token_30 = air.execute [%44] {
          memref.dealloc %results_18 : memref<64x64xbf16, 2 : i32>
        } {id = 83 : i32}
        %async_token_31 = air.execute [%44] {
          memref.dealloc %results_20 : memref<64x64xbf16, 2 : i32>
        } {id = 84 : i32}
        %async_token_32 = air.execute [%44] {
          memref.dealloc %results_22 : memref<64x64xbf16, 2 : i32>
        } {id = 85 : i32}
        %async_token_33 = air.execute [%44] {
          memref.dealloc %results_24 : memref<64x64xbf16, 2 : i32>
        } {id = 86 : i32}
        %async_token_34 = air.execute [%44] {
          memref.dealloc %results_26 : memref<64x1xbf16, 2 : i32>
        } {id = 87 : i32}
        %async_token_35 = air.execute [%44] {
          memref.dealloc %results_28 : memref<64x1xbf16, 2 : i32>
        } {id = 88 : i32}
        %async_token_36 = air.execute [%38] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        } {id = 89 : i32}
        %async_token_37 = air.execute [%39] {
          memref.dealloc %results_8 : memref<64x64xbf16, 1 : i32>
        } {id = 90 : i32}
        %async_token_38 = air.execute [%40] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        } {id = 91 : i32}
        %async_token_39 = air.execute [%41] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        } {id = 92 : i32}
        %async_token_40 = air.execute [%43] {
          memref.dealloc %results_14 : memref<256x64xbf16, 1 : i32>
        } {id = 93 : i32}
      }
    }
    return
  }
}
