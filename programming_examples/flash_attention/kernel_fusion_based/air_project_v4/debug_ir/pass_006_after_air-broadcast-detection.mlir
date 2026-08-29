#map = affine_map<()[s0] -> (s0 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 2)>
#map2 = affine_map<()[s0] -> (s0 * 32768)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1)>
#map4 = affine_map<()[s0] -> (s0 + 1)>
#map5 = affine_map<()[s0] -> (s0 * 64)>
#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set4 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
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
  air.channel @QK2L1 [2, 1, 4] {broadcast_shape = [2 : index, 4 : index, 4 : index]}
  air.channel @V2L1_0 [2, 1, 1] {broadcast_shape = [2 : index, 4 : index, 1 : index]}
  air.channel @VIn_0 [2]
  air.channel @V2L1_1 [2, 1, 1] {broadcast_shape = [2 : index, 4 : index, 1 : index]}
  air.channel @VIn_1 [2]
  air.channel @V2L1_2 [2, 1, 1] {broadcast_shape = [2 : index, 4 : index, 1 : index]}
  air.channel @VIn_2 [2]
  air.channel @V2L1_3 [2, 1, 1] {broadcast_shape = [2 : index, 4 : index, 1 : index]}
  air.channel @VIn_3 [2]
  air.channel @cascade_gp [4, 3] {channel_type = "cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<2x256x64xbf16>, %arg1: memref<2x512x64xbf16>, %arg2: memref<2x512x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x256x64xbf16> attributes {id = 3 : i32} {
      %c24576 = arith.constant 24576 : index
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = affine.apply #map1()[%arg5]
      %3 = affine.apply #map()[%2]
      %4 = affine.apply #map2()[%2]
      %5 = affine.apply #map2()[%2]
      %6 = affine.apply #map3()[%3, %1]
      %7 = air.channel.put async  @QK2L1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %6] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 1 : i32} : (memref<2x256x64xbf16>)
      %8 = air.channel.put async  @QK2L1[%c0, %c0, %c1_0] (%arg8[%c0, %c0, %c0, %c0, %6] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 2 : i32} : (memref<2x256x64xbf16>)
      %9 = air.channel.put async  @QK2L1[%c0, %c0, %c2] (%arg8[%c0, %c0, %c0, %c0, %6] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 3 : i32} : (memref<2x256x64xbf16>)
      %10 = air.channel.put async  @QK2L1[%c0, %c0, %c3] (%arg8[%c0, %c0, %c0, %c0, %6] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 4 : i32} : (memref<2x256x64xbf16>)
      %11 = affine.apply #map3()[%4, %c0]
      %12 = air.channel.put async  @QK2L1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %11] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 5 : i32} : (memref<2x512x64xbf16>)
      %13 = affine.apply #map3()[%4, %c8192]
      %14 = air.channel.put async  @QK2L1[%c0, %c0, %c1_0] (%arg9[%c0, %c0, %c0, %c0, %13] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 6 : i32} : (memref<2x512x64xbf16>)
      %15 = affine.apply #map3()[%4, %c16384]
      %16 = air.channel.put async  @QK2L1[%c0, %c0, %c2] (%arg9[%c0, %c0, %c0, %c0, %15] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 7 : i32} : (memref<2x512x64xbf16>)
      %17 = affine.apply #map3()[%4, %c24576]
      %18 = air.channel.put async  @QK2L1[%c0, %c0, %c3] (%arg9[%c0, %c0, %c0, %c0, %17] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 8 : i32} : (memref<2x512x64xbf16>)
      %19 = affine.apply #map3()[%5, %c0]
      %20 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %19] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 9 : i32} : (memref<2x512x64xbf16>)
      %21 = affine.apply #map3()[%5, %c8192]
      %22 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %21] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 10 : i32} : (memref<2x512x64xbf16>)
      %23 = affine.apply #map3()[%5, %c16384]
      %24 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %23] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 11 : i32} : (memref<2x512x64xbf16>)
      %25 = affine.apply #map3()[%5, %c24576]
      %26 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %25] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 12 : i32} : (memref<2x512x64xbf16>)
      %27 = air.channel.get async  @GpOut[%c0] (%arg11[%6] [%c16384] [%c1_0]) {id = 13 : i32} : (memref<2x256x64xbf16>)
      %28 = affine.apply #map4()[%2]
      %29 = affine.apply #map()[%28]
      %30 = affine.apply #map2()[%28]
      %31 = affine.apply #map2()[%28]
      %32 = affine.apply #map3()[%29, %1]
      %33 = air.channel.put async  @QK2L1[%c1_0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %32] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 14 : i32} : (memref<2x256x64xbf16>)
      %34 = air.channel.put async  @QK2L1[%c1_0, %c0, %c1_0] (%arg8[%c0, %c0, %c0, %c0, %32] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 15 : i32} : (memref<2x256x64xbf16>)
      %35 = air.channel.put async  @QK2L1[%c1_0, %c0, %c2] (%arg8[%c0, %c0, %c0, %c0, %32] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 16 : i32} : (memref<2x256x64xbf16>)
      %36 = air.channel.put async  @QK2L1[%c1_0, %c0, %c3] (%arg8[%c0, %c0, %c0, %c0, %32] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 17 : i32} : (memref<2x256x64xbf16>)
      %37 = affine.apply #map3()[%30, %c0]
      %38 = air.channel.put async  @QK2L1[%c1_0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %37] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 18 : i32} : (memref<2x512x64xbf16>)
      %39 = affine.apply #map3()[%30, %c8192]
      %40 = air.channel.put async  @QK2L1[%c1_0, %c0, %c1_0] (%arg9[%c0, %c0, %c0, %c0, %39] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 19 : i32} : (memref<2x512x64xbf16>)
      %41 = affine.apply #map3()[%30, %c16384]
      %42 = air.channel.put async  @QK2L1[%c1_0, %c0, %c2] (%arg9[%c0, %c0, %c0, %c0, %41] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 20 : i32} : (memref<2x512x64xbf16>)
      %43 = affine.apply #map3()[%30, %c24576]
      %44 = air.channel.put async  @QK2L1[%c1_0, %c0, %c3] (%arg9[%c0, %c0, %c0, %c0, %43] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 21 : i32} : (memref<2x512x64xbf16>)
      %45 = affine.apply #map3()[%31, %c0]
      %46 = air.channel.put async  @VIn_0[%c1_0] (%arg10[%c0, %c0, %45] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 22 : i32} : (memref<2x512x64xbf16>)
      %47 = affine.apply #map3()[%31, %c8192]
      %48 = air.channel.put async  @VIn_1[%c1_0] (%arg10[%c0, %c0, %47] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 23 : i32} : (memref<2x512x64xbf16>)
      %49 = affine.apply #map3()[%31, %c16384]
      %50 = air.channel.put async  @VIn_2[%c1_0] (%arg10[%c0, %c0, %49] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 24 : i32} : (memref<2x512x64xbf16>)
      %51 = affine.apply #map3()[%31, %c24576]
      %52 = air.channel.put async  @VIn_3[%c1_0] (%arg10[%c0, %c0, %51] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 25 : i32} : (memref<2x512x64xbf16>)
      %53 = air.channel.get async  @GpOut[%c1_0] (%arg11[%32] [%c16384] [%c1_0]) {id = 26 : i32} : (memref<2x256x64xbf16>)
      %54 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) attributes {id = 2 : i32} {
        %c64_1 = arith.constant 64 : index
        %c512_2 = arith.constant 512 : index
        %c8_3 = arith.constant 8 : index
        %c1_4 = arith.constant 1 : index
        %c2_5 = arith.constant 2 : index
        %c0_6 = arith.constant 0 : index
        %c4_7 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 1 : i32}
        %async_token_8, %results_9 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 2 : i32}
        %async_token_10, %results_11 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 3 : i32}
        %async_token_12, %results_13 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 4 : i32}
        %async_token_14, %results_15 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        } {id = 5 : i32}
        %async_token_16, %results_17 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 6 : i32}
        %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 7 : i32}
        %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 8 : i32}
        %async_token_22, %results_23 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 9 : i32}
        %async_token_24, %results_25 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 10 : i32}
        %async_token_26, %results_27 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 11 : i32}
        %async_token_28, %results_29 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 12 : i32}
        %55 = air.wait_all async [%async_token]  {id = 1 : i32}
        %56 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %55) -> (!air.async.token) {
          %67 = air.channel.get async [%arg17]  @VIn_0[%arg12] (%results[] [] []) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
          %68 = air.channel.put async [%arg17, %67]  @V2L1_0[%arg12, %c0_6, %c0_6] (%results[%c0_6, %c0_6, %c0_6, %c0_6] [%c8_3, %c8_3, %c8_3, %c8_3] [%c8_3, %c512_2, %c64_1, %c1_4]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
          %69 = air.wait_all async [%68]  {id = 2 : i32}
          scf.yield %69 : !air.async.token
        }
        %57 = air.wait_all async [%async_token_8]  {id = 3 : i32}
        %58 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %57) -> (!air.async.token) {
          %67 = air.channel.get async [%arg17]  @VIn_1[%arg12] (%results_9[] [] []) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
          %68 = air.channel.put async [%arg17, %67]  @V2L1_1[%arg12, %c0_6, %c0_6] (%results_9[%c0_6, %c0_6, %c0_6, %c0_6] [%c8_3, %c8_3, %c8_3, %c8_3] [%c8_3, %c512_2, %c64_1, %c1_4]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
          %69 = air.wait_all async [%68]  {id = 4 : i32}
          scf.yield %69 : !air.async.token
        }
        %59 = air.wait_all async [%async_token_10]  {id = 5 : i32}
        %60 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %59) -> (!air.async.token) {
          %67 = air.channel.get async [%arg17]  @VIn_2[%arg12] (%results_11[] [] []) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
          %68 = air.channel.put async [%arg17, %67]  @V2L1_2[%arg12, %c0_6, %c0_6] (%results_11[%c0_6, %c0_6, %c0_6, %c0_6] [%c8_3, %c8_3, %c8_3, %c8_3] [%c8_3, %c512_2, %c64_1, %c1_4]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
          %69 = air.wait_all async [%68]  {id = 6 : i32}
          scf.yield %69 : !air.async.token
        }
        %61 = air.wait_all async [%async_token_12]  {id = 7 : i32}
        %62 = scf.for %arg16 = %c0_6 to %c2_5 step %c1_4 iter_args(%arg17 = %61) -> (!air.async.token) {
          %67 = air.channel.get async [%arg17]  @VIn_3[%arg12] (%results_13[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %68 = air.channel.put async [%arg17, %67]  @V2L1_3[%arg12, %c0_6, %c0_6] (%results_13[%c0_6, %c0_6, %c0_6, %c0_6] [%c8_3, %c8_3, %c8_3, %c8_3] [%c8_3, %c512_2, %c64_1, %c1_4]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          %69 = air.wait_all async [%68]  {id = 8 : i32}
          scf.yield %69 : !air.async.token
        }
        %63 = air.wait_all async [%async_token_14]  {id = 9 : i32}
        %64 = scf.parallel (%arg16) = (%c0_6) to (%c4_7) step (%c1_4) init (%63) -> !air.async.token {
          %67 = affine.apply #map5()[%arg16]
          %68 = air.channel.get async [%63]  @Gp2L2[%arg16, %c0_6] (%results_15[%67, %c0_6] [%c64_1, %c64_1] [%c64_1, %c1_4]) {id = 35 : i32} : (memref<256x64xbf16, 1 : i32>)
          %69 = air.wait_all async [%68]  {id = 10 : i32}
          scf.reduce(%69 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %70 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %70 : !air.async.token
          }
        }
        %65 = air.channel.put async [%64]  @GpOut[%arg12] (%results_15[] [] []) {id = 36 : i32} : (memref<256x64xbf16, 1 : i32>)
        %66 = air.herd @herd_0 async [%async_token_16, %async_token_18, %async_token_20, %async_token_22, %async_token_24, %async_token_26, %async_token_28]  tile (%arg16, %arg17) in (%arg18=%c4_7, %arg19=%c4_7) args(%arg20=%results_17, %arg21=%results_19, %arg22=%results_21, %arg23=%results_23, %arg24=%results_25, %arg25=%results_27, %arg26=%results_29, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {id = 1 : i32, link_with = "attn.o"} {
          %c512_42 = arith.constant 512 : index
          %c64_43 = arith.constant 64 : index
          %c8_44 = arith.constant 8 : index
          %c1_45 = arith.constant 1 : index
          %c0_46 = arith.constant 0 : index
          %c2_47 = arith.constant 2 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_48 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          } {id = 13 : i32}
          %async_token_49 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 14 : i32}
          %async_token_50 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 15 : i32}
          %67 = air.channel.get async  @QK2L1[%arg27, %arg16, %arg17] (%arg21[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
          %68 = arith.index_cast %arg16 : index to i32
          %69 = arith.cmpi eq, %68, %c0_i32 : i32
          %70 = air.wait_all async [%67]  {id = 11 : i32}
          %71 = scf.if %69 -> (!air.async.token) {
            %async_token_51 = air.execute [%67] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 16 : i32}
            %91 = air.wait_all async [%async_token_51]  {id = 12 : i32}
            scf.yield %91 : !air.async.token
          } else {
            %91 = air.wait_all async  {id = 13 : i32}
            scf.yield %91 : !air.async.token
          }
          %72 = air.channel.get async  @QK2L1[%arg27, %arg16, %arg17] (%arg21[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
          %73 = arith.index_cast %arg16 : index to i32
          %74 = arith.cmpi eq, %73, %c1_i32 : i32
          %75 = air.wait_all async [%72]  {id = 14 : i32}
          %76 = scf.if %74 -> (!air.async.token) {
            %async_token_51 = air.execute [%72] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 17 : i32}
            %91 = air.wait_all async [%async_token_51]  {id = 15 : i32}
            scf.yield %91 : !air.async.token
          } else {
            %91 = air.wait_all async  {id = 16 : i32}
            scf.yield %91 : !air.async.token
          }
          %77 = air.channel.get async  @QK2L1[%arg27, %arg16, %arg17] (%arg21[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
          %78 = arith.index_cast %arg16 : index to i32
          %79 = arith.cmpi eq, %78, %c2_i32 : i32
          %80 = air.wait_all async [%77]  {id = 17 : i32}
          %81 = scf.if %79 -> (!air.async.token) {
            %async_token_51 = air.execute [%77] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 18 : i32}
            %91 = air.wait_all async [%async_token_51]  {id = 18 : i32}
            scf.yield %91 : !air.async.token
          } else {
            %91 = air.wait_all async  {id = 19 : i32}
            scf.yield %91 : !air.async.token
          }
          %82 = air.channel.get async  @QK2L1[%arg27, %arg16, %arg17] (%arg21[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
          %83 = arith.index_cast %arg16 : index to i32
          %84 = arith.cmpi eq, %83, %c3_i32 : i32
          %85 = air.wait_all async [%82]  {id = 20 : i32}
          %86 = scf.if %84 -> (!air.async.token) {
            %async_token_51 = air.execute [%82] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 19 : i32}
            %91 = air.wait_all async [%async_token_51]  {id = 21 : i32}
            scf.yield %91 : !air.async.token
          } else {
            %91 = air.wait_all async  {id = 22 : i32}
            scf.yield %91 : !air.async.token
          }
          %87 = air.wait_all async [%async_token_48, %async_token_49, %async_token_50]  {id = 35 : i32}
          %88 = scf.for %arg28 = %c0_46 to %c2_47 step %c1_45 iter_args(%arg29 = %87) -> (!air.async.token) {
            %async_token_51 = air.execute [%arg29] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            } {id = 20 : i32}
            %91 = air.channel.get async [%arg29]  @QK2L1[%arg27, %arg16, %arg17] (%arg21[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            %92 = air.wait_all async [%arg29]  {id = 23 : i32}
            %93 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %101 = air.channel.get async  @V2L1_0[%arg27, %arg16, %arg17] (%arg22[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              %102 = air.wait_all async [%101]  {id = 24 : i32}
              affine.yield %102 : !air.async.token
            } else {
              %101 = air.wait_all async  {id = 25 : i32}
              affine.yield %101 : !air.async.token
            }
            %94 = air.wait_all async [%arg29, %93, %93]  {id = 26 : i32}
            %95 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %101 = air.channel.get async [%94]  @V2L1_1[%arg27, %arg16, %arg17] (%arg22[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
              %102 = air.wait_all async [%101]  {id = 27 : i32}
              affine.yield %102 : !air.async.token
            } else {
              %101 = air.wait_all async  {id = 28 : i32}
              affine.yield %101 : !air.async.token
            }
            %96 = air.wait_all async [%arg29, %95, %95]  {id = 29 : i32}
            %97 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %101 = air.channel.get async [%96]  @V2L1_2[%arg27, %arg16, %arg17] (%arg22[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
              %102 = air.wait_all async [%101]  {id = 30 : i32}
              affine.yield %102 : !air.async.token
            } else {
              %101 = air.wait_all async  {id = 31 : i32}
              affine.yield %101 : !air.async.token
            }
            %98 = air.wait_all async [%arg29, %97, %97]  {id = 32 : i32}
            %99 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %101 = air.channel.get async [%98]  @V2L1_3[%arg27, %arg16, %arg17] (%arg22[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
              %102 = air.wait_all async [%101]  {id = 33 : i32}
              affine.yield %102 : !air.async.token
            } else {
              %101 = air.wait_all async  {id = 34 : i32}
              affine.yield %101 : !air.async.token
            }
            %async_token_52 = air.execute [%arg29, %91, %async_token_51] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            } {id = 21 : i32}
            %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 22 : i32}
            %async_token_55, %results_56 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 23 : i32}
            %async_token_57 = air.execute [%async_token_55, %async_token_53, %async_token_52, %arg29] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_54, %results_56) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 24 : i32}
            %async_token_58 = air.execute [%async_token_57, %arg29] {
              func.call @mul_r_gp(%results_56, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 25 : i32}
            %async_token_59 = air.execute [%arg29, %async_token_58, %99] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 26 : i32}
            %async_token_60 = air.execute [%async_token_58, %arg29] {
              func.call @accum_sp_r_s(%arg26, %results_56, %results_54) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 27 : i32}
            %async_token_61 = air.execute [%arg29, %async_token_60] {
              func.call @vector_copy_32elems(%c0_i32, %results_54, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 28 : i32}
            %async_token_62 = air.execute [%async_token_61] {
              memref.dealloc %results_54 : memref<64x1xbf16, 2 : i32>
            } {id = 29 : i32}
            %async_token_63 = air.execute [%async_token_60] {
              memref.dealloc %results_56 : memref<64x1xbf16, 2 : i32>
            } {id = 30 : i32}
            %100 = air.wait_all async [%92, %94, %96, %98, %async_token_59, %async_token_61]  {id = 36 : i32}
            scf.yield %100 : !air.async.token
          }
          %89 = air.wait_all async [%88, %88]  {id = 40 : i32}
          %90 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %91 = arith.subi %arg17, %c1_45 : index
            %92 = air.channel.put async [%89]  @cascade_gp[%arg16, %91] (%arg24[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
            %93 = air.channel.put async [%89]  @cascade_up[%arg16, %91] (%arg25[] [] []) {id = 47 : i32} : (memref<64x1xbf16, 2 : i32>)
            %94 = air.channel.put async [%89]  @cascade_sp[%arg16, %91] (%arg26[] [] []) {id = 48 : i32} : (memref<64x1xbf16, 2 : i32>)
            %95 = air.wait_all async [%92, %93, %94]  {id = 41 : i32}
            affine.yield %95 : !air.async.token
          } else {
            %91 = air.wait_all async [%89, %89]  {id = 37 : i32}
            %92 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_51, %results_52 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 31 : i32}
              %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 32 : i32}
              %async_token_55, %results_56 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 33 : i32}
              %94 = air.channel.get async [%async_token_51]  @cascade_gp[%arg16, %arg17] (%results_52[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
              %95 = air.channel.get async [%async_token_53]  @cascade_up[%arg16, %arg17] (%results_54[] [] []) {id = 50 : i32} : (memref<64x1xbf16, 2 : i32>)
              %96 = air.channel.get async [%async_token_55]  @cascade_sp[%arg16, %arg17] (%results_56[] [] []) {id = 51 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_57, %results_58 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 34 : i32}
              %async_token_59 = air.execute [%async_token_57, %91] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_58) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 35 : i32}
              %async_token_60 = air.execute [%async_token_59, %95] {
                func.call @maximum_up_u_bf16(%results_54, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 36 : i32}
              %async_token_61, %results_62 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 37 : i32}
              %async_token_63 = air.execute [%async_token_61, %async_token_60] {
                func.call @exp_up_minus_u(%results_54, %arg25, %results_62) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 38 : i32}
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 39 : i32}
              %async_token_66 = air.execute [%async_token_64, %async_token_63] {
                func.call @exp_up_minus_u(%results_58, %arg25, %results_65) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 40 : i32}
              %async_token_67 = air.execute [%async_token_63, %94] {
                func.call @mul_r_gp(%results_62, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 41 : i32}
              %async_token_68 = air.execute [%async_token_66, %91] {
                func.call @mul_r_gp(%results_65, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 42 : i32}
              %async_token_69 = air.execute [%async_token_68, %async_token_67] {
                func.call @add_gp_g(%arg24, %results_52) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 43 : i32}
              %async_token_70, %results_71 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 44 : i32}
              %async_token_72 = air.execute [%async_token_70] {
                func.call @zero_fill_sp_bf16(%results_71) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 45 : i32}
              %async_token_73 = air.execute [%async_token_72, %async_token_67, %96] {
                func.call @accum_sp_r_s(%results_56, %results_62, %results_71) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 46 : i32}
              %async_token_74 = air.execute [%async_token_73, %async_token_68, %91] {
                func.call @accum_sp_r_s(%arg26, %results_65, %results_71) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 47 : i32}
              %async_token_75 = air.execute [%async_token_74] {
                func.call @vector_copy_32elems(%c0_i32, %results_71, %results_56) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 48 : i32}
              %97 = arith.subi %arg17, %c1_45 : index
              %98 = air.channel.put async [%async_token_69]  @cascade_gp[%arg16, %97] (%results_52[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
              %99 = air.channel.put async [%async_token_66]  @cascade_up[%arg16, %97] (%arg25[] [] []) {id = 53 : i32} : (memref<64x1xbf16, 2 : i32>)
              %100 = air.channel.put async [%async_token_75]  @cascade_sp[%arg16, %97] (%results_56[] [] []) {id = 54 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_76 = air.execute [%98] {
                memref.dealloc %results_52 : memref<64x64xbf16, 2 : i32>
              } {id = 49 : i32}
              %async_token_77 = air.execute [%async_token_63] {
                memref.dealloc %results_54 : memref<64x1xbf16, 2 : i32>
              } {id = 50 : i32}
              %async_token_78 = air.execute [%100] {
                memref.dealloc %results_56 : memref<64x1xbf16, 2 : i32>
              } {id = 51 : i32}
              %async_token_79 = air.execute [%async_token_66] {
                memref.dealloc %results_58 : memref<64x1xbf16, 2 : i32>
              } {id = 52 : i32}
              %async_token_80 = air.execute [%async_token_73] {
                memref.dealloc %results_62 : memref<64x1xbf16, 2 : i32>
              } {id = 53 : i32}
              %async_token_81 = air.execute [%async_token_74] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              } {id = 54 : i32}
              %async_token_82 = air.execute [%async_token_75] {
                memref.dealloc %results_71 : memref<64x1xbf16, 2 : i32>
              } {id = 55 : i32}
              %101 = air.wait_all async [%98, %99, %100]  {id = 38 : i32}
              affine.yield %101 : !air.async.token
            } else {
              %async_token_51, %results_52 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 56 : i32}
              %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 57 : i32}
              %async_token_55, %results_56 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 58 : i32}
              %94 = air.channel.get async [%async_token_51]  @cascade_gp[%arg16, %arg17] (%results_52[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              %95 = air.channel.get async [%async_token_53]  @cascade_up[%arg16, %arg17] (%results_54[] [] []) {id = 56 : i32} : (memref<64x1xbf16, 2 : i32>)
              %96 = air.channel.get async [%async_token_55]  @cascade_sp[%arg16, %arg17] (%results_56[] [] []) {id = 57 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_57, %results_58 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 59 : i32}
              %async_token_59 = air.execute [%async_token_57, %91] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_58) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 60 : i32}
              %async_token_60 = air.execute [%async_token_59, %95] {
                func.call @maximum_up_u_bf16(%results_54, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 61 : i32}
              %async_token_61, %results_62 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 62 : i32}
              %async_token_63 = air.execute [%async_token_61, %async_token_60] {
                func.call @exp_up_minus_u(%results_54, %arg25, %results_62) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 63 : i32}
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 64 : i32}
              %async_token_66 = air.execute [%async_token_64, %async_token_63] {
                func.call @exp_up_minus_u(%results_58, %arg25, %results_65) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 65 : i32}
              %async_token_67 = air.execute [%async_token_63, %94] {
                func.call @mul_r_gp(%results_62, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 66 : i32}
              %async_token_68 = air.execute [%async_token_66, %91] {
                func.call @mul_r_gp(%results_65, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 67 : i32}
              %async_token_69 = air.execute [%async_token_68, %async_token_67] {
                func.call @add_gp_g(%arg24, %results_52) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 68 : i32}
              %async_token_70, %results_71 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 69 : i32}
              %async_token_72 = air.execute [%async_token_70] {
                func.call @zero_fill_sp_bf16(%results_71) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 70 : i32}
              %async_token_73 = air.execute [%async_token_72, %async_token_67, %96] {
                func.call @accum_sp_r_s(%results_56, %results_62, %results_71) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 71 : i32}
              %async_token_74 = air.execute [%async_token_73, %async_token_68, %91] {
                func.call @accum_sp_r_s(%arg26, %results_65, %results_71) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 72 : i32}
              %async_token_75 = air.execute [%async_token_74] {
                func.call @vector_copy_32elems(%c0_i32, %results_71, %results_56) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 73 : i32}
              %async_token_76 = air.execute [%async_token_75, %async_token_69] {
                func.call @div_gp_sp(%results_56, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 74 : i32}
              %97 = air.channel.put async [%async_token_76]  @Gp2L2[%arg16, %c0_46] (%results_52[%c0_46, %c0_46, %c0_46, %c0_46] [%c8_44, %c8_44, %c8_44, %c8_44] [%c64_43, %c8_44, %c512_42, %c1_45]) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_77 = air.execute [%97] {
                memref.dealloc %results_52 : memref<64x64xbf16, 2 : i32>
              } {id = 75 : i32}
              %async_token_78 = air.execute [%async_token_63] {
                memref.dealloc %results_54 : memref<64x1xbf16, 2 : i32>
              } {id = 76 : i32}
              %async_token_79 = air.execute [%async_token_76] {
                memref.dealloc %results_56 : memref<64x1xbf16, 2 : i32>
              } {id = 77 : i32}
              %async_token_80 = air.execute [%async_token_66] {
                memref.dealloc %results_58 : memref<64x1xbf16, 2 : i32>
              } {id = 78 : i32}
              %async_token_81 = air.execute [%async_token_73] {
                memref.dealloc %results_62 : memref<64x1xbf16, 2 : i32>
              } {id = 79 : i32}
              %async_token_82 = air.execute [%async_token_74] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              } {id = 80 : i32}
              %async_token_83 = air.execute [%async_token_75] {
                memref.dealloc %results_71 : memref<64x1xbf16, 2 : i32>
              } {id = 81 : i32}
              %98 = air.wait_all async [%97]  {id = 39 : i32}
              affine.yield %98 : !air.async.token
            }
            %93 = air.wait_all async [%91]  {id = 42 : i32}
            affine.yield %93 : !air.async.token
          }
        }
        %async_token_30 = air.execute [%66] {
          memref.dealloc %results_17 : memref<64x64xbf16, 2 : i32>
        } {id = 82 : i32}
        %async_token_31 = air.execute [%66] {
          memref.dealloc %results_19 : memref<64x64xbf16, 2 : i32>
        } {id = 83 : i32}
        %async_token_32 = air.execute [%66] {
          memref.dealloc %results_21 : memref<64x64xbf16, 2 : i32>
        } {id = 84 : i32}
        %async_token_33 = air.execute [%66] {
          memref.dealloc %results_23 : memref<64x64xbf16, 2 : i32>
        } {id = 85 : i32}
        %async_token_34 = air.execute [%66] {
          memref.dealloc %results_25 : memref<64x64xbf16, 2 : i32>
        } {id = 86 : i32}
        %async_token_35 = air.execute [%66] {
          memref.dealloc %results_27 : memref<64x1xbf16, 2 : i32>
        } {id = 87 : i32}
        %async_token_36 = air.execute [%66] {
          memref.dealloc %results_29 : memref<64x1xbf16, 2 : i32>
        } {id = 88 : i32}
        %async_token_37 = air.execute [%56] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        } {id = 89 : i32}
        %async_token_38 = air.execute [%58] {
          memref.dealloc %results_9 : memref<64x64xbf16, 1 : i32>
        } {id = 90 : i32}
        %async_token_39 = air.execute [%60] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        } {id = 91 : i32}
        %async_token_40 = air.execute [%62] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        } {id = 92 : i32}
        %async_token_41 = air.execute [%65] {
          memref.dealloc %results_15 : memref<256x64xbf16, 1 : i32>
        } {id = 93 : i32}
      }
    }
    return
  }
}
