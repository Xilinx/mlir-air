#map = affine_map<()[s0] -> (s0 * 32768)>
#map1 = affine_map<()[s0] -> (s0 * 2)>
#map2 = affine_map<()[s0] -> (s0 * 65536)>
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
  air.channel @QK2L1_0 [2, 1, 1] {broadcast_shape = [2 : index, 1 : index, 4 : index]}
  air.channel @QKIn_0 [2]
  air.channel @QK2L1_1 [2, 1, 1] {broadcast_shape = [2 : index, 1 : index, 4 : index]}
  air.channel @QKIn_1 [2]
  air.channel @QK2L1_2 [2, 1, 1] {broadcast_shape = [2 : index, 1 : index, 4 : index]}
  air.channel @QKIn_2 [2]
  air.channel @QK2L1_3 [2, 1, 1] {broadcast_shape = [2 : index, 1 : index, 4 : index]}
  air.channel @QKIn_3 [2]
  air.channel @V2L1_0 [2, 1, 1] {broadcast_shape = [2 : index, 1 : index, 4 : index]}
  air.channel @VIn_0 [2]
  air.channel @V2L1_1 [2, 1, 1] {broadcast_shape = [2 : index, 1 : index, 4 : index]}
  air.channel @VIn_1 [2]
  air.channel @V2L1_2 [2, 1, 1] {broadcast_shape = [2 : index, 1 : index, 4 : index]}
  air.channel @VIn_2 [2]
  air.channel @V2L1_3 [2, 1, 1] {broadcast_shape = [2 : index, 1 : index, 4 : index]}
  air.channel @VIn_3 [2]
  air.channel @cascade_gp [4, 3] {channel_type = "cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<2x256x128xbf16>, %arg1: memref<2x512x128xbf16>, %arg2: memref<2x512x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x128xbf16>, memref<2x512x128xbf16>, memref<2x512x64xbf16>, memref<2x256x64xbf16> attributes {id = 3 : i32} {
      %c24576 = arith.constant 24576 : index
      %c8192 = arith.constant 8192 : index
      %c4096 = arith.constant 4096 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c49152 = arith.constant 49152 : index
      %c32768 = arith.constant 32768 : index
      %c16384 = arith.constant 16384 : index
      %c1_0 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = affine.apply #map1()[%arg5]
      %3 = affine.apply #map()[%2]
      %4 = affine.apply #map2()[%2]
      %5 = affine.apply #map()[%2]
      %6 = affine.apply #map3()[%3, %1]
      %7 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %6] [%c256, %c128] [%c128, %c1_0]) {id = 1 : i32} : (memref<2x256x128xbf16>)
      %8 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %6] [%c256, %c128] [%c128, %c1_0]) {id = 2 : i32} : (memref<2x256x128xbf16>)
      %9 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %6] [%c256, %c128] [%c128, %c1_0]) {id = 3 : i32} : (memref<2x256x128xbf16>)
      %10 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %6] [%c256, %c128] [%c128, %c1_0]) {id = 4 : i32} : (memref<2x256x128xbf16>)
      %11 = affine.apply #map3()[%4, %c0]
      %12 = air.channel.put async  @QKIn_0[%c0] (%arg9[%c0, %11] [%c128, %c128] [%c128, %c1_0]) {id = 5 : i32} : (memref<2x512x128xbf16>)
      %13 = affine.apply #map3()[%4, %c16384]
      %14 = air.channel.put async  @QKIn_1[%c0] (%arg9[%c0, %13] [%c128, %c128] [%c128, %c1_0]) {id = 6 : i32} : (memref<2x512x128xbf16>)
      %15 = affine.apply #map3()[%4, %c32768]
      %16 = air.channel.put async  @QKIn_2[%c0] (%arg9[%c0, %15] [%c128, %c128] [%c128, %c1_0]) {id = 7 : i32} : (memref<2x512x128xbf16>)
      %17 = affine.apply #map3()[%4, %c49152]
      %18 = air.channel.put async  @QKIn_3[%c0] (%arg9[%c0, %17] [%c128, %c128] [%c128, %c1_0]) {id = 8 : i32} : (memref<2x512x128xbf16>)
      %19 = affine.apply #map3()[%5, %c0]
      %20 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %19] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 9 : i32} : (memref<2x512x64xbf16>)
      %21 = affine.apply #map3()[%5, %c8192]
      %22 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %21] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 10 : i32} : (memref<2x512x64xbf16>)
      %23 = affine.apply #map3()[%5, %c16384]
      %24 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %23] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 11 : i32} : (memref<2x512x64xbf16>)
      %25 = affine.apply #map3()[%5, %c24576]
      %26 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %25] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 12 : i32} : (memref<2x512x64xbf16>)
      %27 = air.channel.get async  @GpOut[%c0] (%arg11[%6] [%c32768] [%c1_0]) {id = 13 : i32} : (memref<2x256x64xbf16>)
      %28 = affine.apply #map4()[%2]
      %29 = affine.apply #map()[%28]
      %30 = affine.apply #map2()[%28]
      %31 = affine.apply #map()[%28]
      %32 = affine.apply #map3()[%29, %1]
      %33 = air.channel.put async  @QKIn_0[%c1_0] (%arg8[%c0, %32] [%c256, %c128] [%c128, %c1_0]) {id = 14 : i32} : (memref<2x256x128xbf16>)
      %34 = air.channel.put async  @QKIn_1[%c1_0] (%arg8[%c0, %32] [%c256, %c128] [%c128, %c1_0]) {id = 15 : i32} : (memref<2x256x128xbf16>)
      %35 = air.channel.put async  @QKIn_2[%c1_0] (%arg8[%c0, %32] [%c256, %c128] [%c128, %c1_0]) {id = 16 : i32} : (memref<2x256x128xbf16>)
      %36 = air.channel.put async  @QKIn_3[%c1_0] (%arg8[%c0, %32] [%c256, %c128] [%c128, %c1_0]) {id = 17 : i32} : (memref<2x256x128xbf16>)
      %37 = affine.apply #map3()[%30, %c0]
      %38 = air.channel.put async  @QKIn_0[%c1_0] (%arg9[%c0, %37] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<2x512x128xbf16>)
      %39 = affine.apply #map3()[%30, %c16384]
      %40 = air.channel.put async  @QKIn_1[%c1_0] (%arg9[%c0, %39] [%c128, %c128] [%c128, %c1_0]) {id = 19 : i32} : (memref<2x512x128xbf16>)
      %41 = affine.apply #map3()[%30, %c32768]
      %42 = air.channel.put async  @QKIn_2[%c1_0] (%arg9[%c0, %41] [%c128, %c128] [%c128, %c1_0]) {id = 20 : i32} : (memref<2x512x128xbf16>)
      %43 = affine.apply #map3()[%30, %c49152]
      %44 = air.channel.put async  @QKIn_3[%c1_0] (%arg9[%c0, %43] [%c128, %c128] [%c128, %c1_0]) {id = 21 : i32} : (memref<2x512x128xbf16>)
      %45 = affine.apply #map3()[%31, %c0]
      %46 = air.channel.put async  @VIn_0[%c1_0] (%arg10[%c0, %c0, %45] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 22 : i32} : (memref<2x512x64xbf16>)
      %47 = affine.apply #map3()[%31, %c8192]
      %48 = air.channel.put async  @VIn_1[%c1_0] (%arg10[%c0, %c0, %47] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 23 : i32} : (memref<2x512x64xbf16>)
      %49 = affine.apply #map3()[%31, %c16384]
      %50 = air.channel.put async  @VIn_2[%c1_0] (%arg10[%c0, %c0, %49] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 24 : i32} : (memref<2x512x64xbf16>)
      %51 = affine.apply #map3()[%31, %c24576]
      %52 = air.channel.put async  @VIn_3[%c1_0] (%arg10[%c0, %c0, %51] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 25 : i32} : (memref<2x512x64xbf16>)
      %53 = air.channel.get async  @GpOut[%c1_0] (%arg11[%32] [%c32768] [%c1_0]) {id = 26 : i32} : (memref<2x256x64xbf16>)
      %54 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) attributes {id = 2 : i32} {
        %c64_1 = arith.constant 64 : index
        %c128_2 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c8 = arith.constant 8 : index
        %c1_3 = arith.constant 1 : index
        %c2_4 = arith.constant 2 : index
        %c0_5 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x128xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x128xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x128xbf16, 1 : i32>
        } {id = 1 : i32}
        %async_token_6, %results_7 = air.execute -> (memref<64x128xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x128xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x128xbf16, 1 : i32>
        } {id = 2 : i32}
        %async_token_8, %results_9 = air.execute -> (memref<64x128xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x128xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x128xbf16, 1 : i32>
        } {id = 3 : i32}
        %async_token_10, %results_11 = air.execute -> (memref<64x128xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x128xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x128xbf16, 1 : i32>
        } {id = 4 : i32}
        %async_token_12, %results_13 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 5 : i32}
        %async_token_14, %results_15 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 6 : i32}
        %async_token_16, %results_17 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 7 : i32}
        %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 8 : i32}
        %async_token_20, %results_21 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        } {id = 9 : i32}
        %async_token_22, %results_23 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 10 : i32}
        %async_token_24, %results_25 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 11 : i32}
        %async_token_26, %results_27 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 12 : i32}
        %async_token_28, %results_29 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 13 : i32}
        %async_token_30, %results_31 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 14 : i32}
        %async_token_32, %results_33 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 15 : i32}
        %async_token_34, %results_35 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 16 : i32}
        %async_token_36, %results_37 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 17 : i32}
        %55 = air.wait_all async [%async_token]  {id = 1 : i32}
        %56 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %55) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c128_61 = arith.constant 128 : index
          %c1_62 = arith.constant 1 : index
          %c64_63 = arith.constant 64 : index
          %83 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 27 : i32} : (memref<64x128xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @QK2L1_0[%arg12, %c0_58, %c0_58] (%results[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 28 : i32} : (memref<64x128xbf16, 1 : i32>)
          %85 = air.channel.put async [%arg17]  @QK2L1_0[%arg12, %c0_58, %c0_58] (%results[%c0_58, %c0_58, %c64_63, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 29 : i32} : (memref<64x128xbf16, 1 : i32>)
          %86 = air.wait_all async [%84, %85]  {id = 2 : i32}
          scf.yield %86 : !air.async.token
        }
        %57 = air.wait_all async [%async_token, %56]  {id = 3 : i32}
        %58 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %57) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c128_61 = arith.constant 128 : index
          %c1_62 = arith.constant 1 : index
          %c64_63 = arith.constant 64 : index
          %83 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 30 : i32} : (memref<64x128xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @QK2L1_0[%arg12, %c0_58, %c0_58] (%results[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 31 : i32} : (memref<64x128xbf16, 1 : i32>)
          %85 = air.channel.put async [%arg17]  @QK2L1_0[%arg12, %c0_58, %c0_58] (%results[%c0_58, %c0_58, %c64_63, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 32 : i32} : (memref<64x128xbf16, 1 : i32>)
          %86 = air.wait_all async [%84, %85]  {id = 4 : i32}
          scf.yield %86 : !air.async.token
        }
        %59 = air.wait_all async [%async_token_6]  {id = 5 : i32}
        %60 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %59) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c128_61 = arith.constant 128 : index
          %c1_62 = arith.constant 1 : index
          %c64_63 = arith.constant 64 : index
          %83 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 33 : i32} : (memref<64x128xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @QK2L1_1[%arg12, %c0_58, %c0_58] (%results_7[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 34 : i32} : (memref<64x128xbf16, 1 : i32>)
          %85 = air.channel.put async [%arg17]  @QK2L1_1[%arg12, %c0_58, %c0_58] (%results_7[%c0_58, %c0_58, %c64_63, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 35 : i32} : (memref<64x128xbf16, 1 : i32>)
          %86 = air.wait_all async [%84, %85]  {id = 6 : i32}
          scf.yield %86 : !air.async.token
        }
        %61 = air.wait_all async [%async_token_6, %60]  {id = 7 : i32}
        %62 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %61) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c128_61 = arith.constant 128 : index
          %c1_62 = arith.constant 1 : index
          %c64_63 = arith.constant 64 : index
          %83 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 36 : i32} : (memref<64x128xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @QK2L1_1[%arg12, %c0_58, %c0_58] (%results_7[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 37 : i32} : (memref<64x128xbf16, 1 : i32>)
          %85 = air.channel.put async [%arg17]  @QK2L1_1[%arg12, %c0_58, %c0_58] (%results_7[%c0_58, %c0_58, %c64_63, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 38 : i32} : (memref<64x128xbf16, 1 : i32>)
          %86 = air.wait_all async [%84, %85]  {id = 8 : i32}
          scf.yield %86 : !air.async.token
        }
        %63 = air.wait_all async [%async_token_8]  {id = 9 : i32}
        %64 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %63) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c128_61 = arith.constant 128 : index
          %c1_62 = arith.constant 1 : index
          %c64_63 = arith.constant 64 : index
          %83 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 39 : i32} : (memref<64x128xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @QK2L1_2[%arg12, %c0_58, %c0_58] (%results_9[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 40 : i32} : (memref<64x128xbf16, 1 : i32>)
          %85 = air.channel.put async [%arg17]  @QK2L1_2[%arg12, %c0_58, %c0_58] (%results_9[%c0_58, %c0_58, %c64_63, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 41 : i32} : (memref<64x128xbf16, 1 : i32>)
          %86 = air.wait_all async [%84, %85]  {id = 10 : i32}
          scf.yield %86 : !air.async.token
        }
        %65 = air.wait_all async [%async_token_8, %64]  {id = 11 : i32}
        %66 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %65) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c128_61 = arith.constant 128 : index
          %c1_62 = arith.constant 1 : index
          %c64_63 = arith.constant 64 : index
          %83 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 42 : i32} : (memref<64x128xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @QK2L1_2[%arg12, %c0_58, %c0_58] (%results_9[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 43 : i32} : (memref<64x128xbf16, 1 : i32>)
          %85 = air.channel.put async [%arg17]  @QK2L1_2[%arg12, %c0_58, %c0_58] (%results_9[%c0_58, %c0_58, %c64_63, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 44 : i32} : (memref<64x128xbf16, 1 : i32>)
          %86 = air.wait_all async [%84, %85]  {id = 12 : i32}
          scf.yield %86 : !air.async.token
        }
        %67 = air.wait_all async [%async_token_10]  {id = 13 : i32}
        %68 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %67) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c128_61 = arith.constant 128 : index
          %c1_62 = arith.constant 1 : index
          %c64_63 = arith.constant 64 : index
          %83 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 45 : i32} : (memref<64x128xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @QK2L1_3[%arg12, %c0_58, %c0_58] (%results_11[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 46 : i32} : (memref<64x128xbf16, 1 : i32>)
          %85 = air.channel.put async [%arg17]  @QK2L1_3[%arg12, %c0_58, %c0_58] (%results_11[%c0_58, %c0_58, %c64_63, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 47 : i32} : (memref<64x128xbf16, 1 : i32>)
          %86 = air.wait_all async [%84, %85]  {id = 14 : i32}
          scf.yield %86 : !air.async.token
        }
        %69 = air.wait_all async [%async_token_10, %68]  {id = 15 : i32}
        %70 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %69) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c128_61 = arith.constant 128 : index
          %c1_62 = arith.constant 1 : index
          %c64_63 = arith.constant 64 : index
          %83 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 48 : i32} : (memref<64x128xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @QK2L1_3[%arg12, %c0_58, %c0_58] (%results_11[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 49 : i32} : (memref<64x128xbf16, 1 : i32>)
          %85 = air.channel.put async [%arg17]  @QK2L1_3[%arg12, %c0_58, %c0_58] (%results_11[%c0_58, %c0_58, %c64_63, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c128_61, %c1_62]) {id = 50 : i32} : (memref<64x128xbf16, 1 : i32>)
          %86 = air.wait_all async [%84, %85]  {id = 16 : i32}
          scf.yield %86 : !air.async.token
        }
        %71 = air.wait_all async [%async_token_12]  {id = 17 : i32}
        %72 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %71) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c64_61 = arith.constant 64 : index
          %c1_62 = arith.constant 1 : index
          %83 = air.channel.get async [%arg17]  @VIn_0[%arg12] (%results_13[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @V2L1_0[%arg12, %c0_58, %c0_58] (%results_13[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c64_61, %c1_62]) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = air.wait_all async [%84]  {id = 18 : i32}
          scf.yield %85 : !air.async.token
        }
        %73 = air.wait_all async [%async_token_14]  {id = 19 : i32}
        %74 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %73) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c64_61 = arith.constant 64 : index
          %c1_62 = arith.constant 1 : index
          %83 = air.channel.get async [%arg17]  @VIn_1[%arg12] (%results_15[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @V2L1_1[%arg12, %c0_58, %c0_58] (%results_15[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c64_61, %c1_62]) {id = 54 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = air.wait_all async [%84]  {id = 20 : i32}
          scf.yield %85 : !air.async.token
        }
        %75 = air.wait_all async [%async_token_16]  {id = 21 : i32}
        %76 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %75) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c64_61 = arith.constant 64 : index
          %c1_62 = arith.constant 1 : index
          %83 = air.channel.get async [%arg17]  @VIn_2[%arg12] (%results_17[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @V2L1_2[%arg12, %c0_58, %c0_58] (%results_17[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c64_61, %c1_62]) {id = 56 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = air.wait_all async [%84]  {id = 22 : i32}
          scf.yield %85 : !air.async.token
        }
        %77 = air.wait_all async [%async_token_18]  {id = 23 : i32}
        %78 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %77) -> (!air.async.token) {
          %c0_58 = arith.constant 0 : index
          %c8_59 = arith.constant 8 : index
          %c512_60 = arith.constant 512 : index
          %c64_61 = arith.constant 64 : index
          %c1_62 = arith.constant 1 : index
          %83 = air.channel.get async [%arg17]  @VIn_3[%arg12] (%results_19[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 1 : i32>)
          %84 = air.channel.put async [%arg17, %83]  @V2L1_3[%arg12, %c0_58, %c0_58] (%results_19[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_59, %c8_59, %c8_59, %c8_59] [%c8_59, %c512_60, %c64_61, %c1_62]) {id = 58 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = air.wait_all async [%84]  {id = 24 : i32}
          scf.yield %85 : !air.async.token
        }
        %c0_38 = arith.constant 0 : index
        %c4_39 = arith.constant 4 : index
        %c1_40 = arith.constant 1 : index
        %79 = air.wait_all async [%async_token_20]  {id = 25 : i32}
        %80 = scf.parallel (%arg16) = (%c0_38) to (%c4_39) step (%c1_40) init (%79) -> !air.async.token {
          %c0_58 = arith.constant 0 : index
          %c64_59 = arith.constant 64 : index
          %c1_60 = arith.constant 1 : index
          %83 = affine.apply #map5()[%arg16]
          %84 = air.channel.get async [%79]  @Gp2L2[%arg16, %c0_58] (%results_21[%83, %c0_58] [%c64_59, %c64_59] [%c64_59, %c1_60]) {id = 59 : i32} : (memref<256x64xbf16, 1 : i32>)
          %85 = air.wait_all async [%84]  {id = 26 : i32}
          scf.reduce(%85 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %86 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %86 : !air.async.token
          }
        }
        %81 = air.channel.put async [%80]  @GpOut[%arg12] (%results_21[] [] []) {id = 60 : i32} : (memref<256x64xbf16, 1 : i32>)
        %82 = air.herd @herd_0 async [%async_token_22, %async_token_24, %async_token_26, %async_token_28, %async_token_30, %async_token_32, %async_token_34, %async_token_36]  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%results_23, %arg21=%results_25, %arg22=%results_27, %arg23=%results_29, %arg24=%results_31, %arg25=%results_33, %arg26=%results_35, %arg27=%results_37, %arg28=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {id = 1 : i32, link_with = "attn.o"} {
          %c512_58 = arith.constant 512 : index
          %c64_59 = arith.constant 64 : index
          %c8_60 = arith.constant 8 : index
          %c1_61 = arith.constant 1 : index
          %c0_62 = arith.constant 0 : index
          %c2_63 = arith.constant 2 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_64 = air.execute {
            func.call @zero_fill_gp_bf16(%arg25) : (memref<64x64xbf16, 2 : i32>) -> ()
          } {id = 18 : i32}
          %async_token_65 = air.execute {
            func.call @zero_fill_sp_bf16(%arg27) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 19 : i32}
          %async_token_66 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 20 : i32}
          %83 = air.wait_all async  {id = 27 : i32}
          %84 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 28 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 29 : i32}
            affine.yield %183 : !air.async.token
          }
          %85 = air.wait_all async [%84, %84]  {id = 30 : i32}
          %86 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%85]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 31 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 32 : i32}
            affine.yield %183 : !air.async.token
          }
          %87 = air.wait_all async [%86, %86]  {id = 33 : i32}
          %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%87]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 34 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 35 : i32}
            affine.yield %183 : !air.async.token
          }
          %89 = air.wait_all async [%88, %88]  {id = 36 : i32}
          %90 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%89]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 37 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 38 : i32}
            affine.yield %183 : !air.async.token
          }
          %91 = arith.index_cast %arg16 : index to i32
          %92 = arith.cmpi eq, %91, %c0_i32 : i32
          %93 = air.wait_all async [%90]  {id = 39 : i32}
          %94 = scf.if %92 -> (!air.async.token) {
            %async_token_67 = air.execute [%90] {
              func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 21 : i32}
            %183 = air.wait_all async [%async_token_67]  {id = 40 : i32}
            scf.yield %183 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 41 : i32}
            scf.yield %183 : !air.async.token
          }
          %95 = air.wait_all async  {id = 42 : i32}
          %96 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 43 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 44 : i32}
            affine.yield %183 : !air.async.token
          }
          %97 = air.wait_all async [%96, %96]  {id = 45 : i32}
          %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%97]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 46 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 47 : i32}
            affine.yield %183 : !air.async.token
          }
          %99 = air.wait_all async [%98, %98]  {id = 48 : i32}
          %100 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%99]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 49 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 50 : i32}
            affine.yield %183 : !air.async.token
          }
          %101 = air.wait_all async [%100, %100]  {id = 51 : i32}
          %102 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%101]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 52 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 53 : i32}
            affine.yield %183 : !air.async.token
          }
          %103 = arith.index_cast %arg16 : index to i32
          %104 = arith.cmpi eq, %103, %c1_i32 : i32
          %105 = air.wait_all async [%102]  {id = 54 : i32}
          %106 = scf.if %104 -> (!air.async.token) {
            %async_token_67 = air.execute [%102] {
              func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 22 : i32}
            %183 = air.wait_all async [%async_token_67]  {id = 55 : i32}
            scf.yield %183 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 56 : i32}
            scf.yield %183 : !air.async.token
          }
          %107 = air.wait_all async  {id = 57 : i32}
          %108 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 58 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 59 : i32}
            affine.yield %183 : !air.async.token
          }
          %109 = air.wait_all async [%108, %108]  {id = 60 : i32}
          %110 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%109]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 61 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 62 : i32}
            affine.yield %183 : !air.async.token
          }
          %111 = air.wait_all async [%110, %110]  {id = 63 : i32}
          %112 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%111]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 64 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 65 : i32}
            affine.yield %183 : !air.async.token
          }
          %113 = air.wait_all async [%112, %112]  {id = 66 : i32}
          %114 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%113]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 67 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 68 : i32}
            affine.yield %183 : !air.async.token
          }
          %115 = arith.index_cast %arg16 : index to i32
          %116 = arith.cmpi eq, %115, %c2_i32 : i32
          %117 = air.wait_all async [%114]  {id = 69 : i32}
          %118 = scf.if %116 -> (!air.async.token) {
            %async_token_67 = air.execute [%114] {
              func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 23 : i32}
            %183 = air.wait_all async [%async_token_67]  {id = 70 : i32}
            scf.yield %183 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 71 : i32}
            scf.yield %183 : !air.async.token
          }
          %119 = air.wait_all async  {id = 72 : i32}
          %120 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 73 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 74 : i32}
            affine.yield %183 : !air.async.token
          }
          %121 = air.wait_all async [%120, %120]  {id = 75 : i32}
          %122 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%121]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 76 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 77 : i32}
            affine.yield %183 : !air.async.token
          }
          %123 = air.wait_all async [%122, %122]  {id = 78 : i32}
          %124 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%123]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 79 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 80 : i32}
            affine.yield %183 : !air.async.token
          }
          %125 = air.wait_all async [%124, %124]  {id = 81 : i32}
          %126 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%125]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 82 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 83 : i32}
            affine.yield %183 : !air.async.token
          }
          %127 = arith.index_cast %arg16 : index to i32
          %128 = arith.cmpi eq, %127, %c3_i32 : i32
          %129 = air.wait_all async [%126]  {id = 84 : i32}
          %130 = scf.if %128 -> (!air.async.token) {
            %async_token_67 = air.execute [%126] {
              func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 24 : i32}
            %183 = air.wait_all async [%async_token_67]  {id = 85 : i32}
            scf.yield %183 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 86 : i32}
            scf.yield %183 : !air.async.token
          }
          %131 = air.wait_all async  {id = 87 : i32}
          %132 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 88 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 89 : i32}
            affine.yield %183 : !air.async.token
          }
          %133 = air.wait_all async [%132, %132]  {id = 90 : i32}
          %134 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%133]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 91 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 92 : i32}
            affine.yield %183 : !air.async.token
          }
          %135 = air.wait_all async [%134, %134]  {id = 93 : i32}
          %136 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%135]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 94 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 95 : i32}
            affine.yield %183 : !air.async.token
          }
          %137 = air.wait_all async [%136, %136]  {id = 96 : i32}
          %138 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%137]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 97 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 98 : i32}
            affine.yield %183 : !air.async.token
          }
          %139 = arith.index_cast %arg16 : index to i32
          %140 = arith.cmpi eq, %139, %c0_i32 : i32
          %141 = air.wait_all async [%138]  {id = 99 : i32}
          %142 = scf.if %140 -> (!air.async.token) {
            %async_token_67 = air.execute [%138] {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 25 : i32}
            %183 = air.wait_all async [%async_token_67]  {id = 100 : i32}
            scf.yield %183 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 101 : i32}
            scf.yield %183 : !air.async.token
          }
          %143 = air.wait_all async  {id = 102 : i32}
          %144 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 103 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 104 : i32}
            affine.yield %183 : !air.async.token
          }
          %145 = air.wait_all async [%144, %144]  {id = 105 : i32}
          %146 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%145]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 106 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 107 : i32}
            affine.yield %183 : !air.async.token
          }
          %147 = air.wait_all async [%146, %146]  {id = 108 : i32}
          %148 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%147]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 109 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 110 : i32}
            affine.yield %183 : !air.async.token
          }
          %149 = air.wait_all async [%148, %148]  {id = 111 : i32}
          %150 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%149]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 112 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 113 : i32}
            affine.yield %183 : !air.async.token
          }
          %151 = arith.index_cast %arg16 : index to i32
          %152 = arith.cmpi eq, %151, %c1_i32 : i32
          %153 = air.wait_all async [%150]  {id = 114 : i32}
          %154 = scf.if %152 -> (!air.async.token) {
            %async_token_67 = air.execute [%150] {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 26 : i32}
            %183 = air.wait_all async [%async_token_67]  {id = 115 : i32}
            scf.yield %183 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 116 : i32}
            scf.yield %183 : !air.async.token
          }
          %155 = air.wait_all async  {id = 117 : i32}
          %156 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 118 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 119 : i32}
            affine.yield %183 : !air.async.token
          }
          %157 = air.wait_all async [%156, %156]  {id = 120 : i32}
          %158 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%157]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 121 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 122 : i32}
            affine.yield %183 : !air.async.token
          }
          %159 = air.wait_all async [%158, %158]  {id = 123 : i32}
          %160 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%159]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 124 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 125 : i32}
            affine.yield %183 : !air.async.token
          }
          %161 = air.wait_all async [%160, %160]  {id = 126 : i32}
          %162 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%161]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 127 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 128 : i32}
            affine.yield %183 : !air.async.token
          }
          %163 = arith.index_cast %arg16 : index to i32
          %164 = arith.cmpi eq, %163, %c2_i32 : i32
          %165 = air.wait_all async [%162]  {id = 129 : i32}
          %166 = scf.if %164 -> (!air.async.token) {
            %async_token_67 = air.execute [%162] {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 27 : i32}
            %183 = air.wait_all async [%async_token_67]  {id = 130 : i32}
            scf.yield %183 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 131 : i32}
            scf.yield %183 : !air.async.token
          }
          %167 = air.wait_all async  {id = 132 : i32}
          %168 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 133 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 134 : i32}
            affine.yield %183 : !air.async.token
          }
          %169 = air.wait_all async [%168, %168]  {id = 135 : i32}
          %170 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%169]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 136 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 137 : i32}
            affine.yield %183 : !air.async.token
          }
          %171 = air.wait_all async [%170, %170]  {id = 138 : i32}
          %172 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%171]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 139 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 140 : i32}
            affine.yield %183 : !air.async.token
          }
          %173 = air.wait_all async [%172, %172]  {id = 141 : i32}
          %174 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %183 = air.channel.get async [%173]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
            %184 = air.wait_all async [%183]  {id = 142 : i32}
            affine.yield %184 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 143 : i32}
            affine.yield %183 : !air.async.token
          }
          %175 = arith.index_cast %arg16 : index to i32
          %176 = arith.cmpi eq, %175, %c3_i32 : i32
          %177 = air.wait_all async [%174]  {id = 144 : i32}
          %178 = scf.if %176 -> (!air.async.token) {
            %async_token_67 = air.execute [%174] {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 28 : i32}
            %183 = air.wait_all async [%async_token_67]  {id = 145 : i32}
            scf.yield %183 : !air.async.token
          } else {
            %183 = air.wait_all async  {id = 146 : i32}
            scf.yield %183 : !air.async.token
          }
          %179 = air.wait_all async [%async_token_64, %async_token_65, %async_token_66]  {id = 183 : i32}
          %180 = scf.for %arg29 = %c0_62 to %c2_63 step %c1_61 iter_args(%arg30 = %179) -> (!air.async.token) {
            %c0_i32_67 = arith.constant 0 : i32
            %async_token_68 = air.execute [%arg30] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            } {id = 29 : i32}
            %183 = air.wait_all async [%arg30]  {id = 147 : i32}
            %184 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 148 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 149 : i32}
              affine.yield %208 : !air.async.token
            }
            %185 = air.wait_all async [%arg30, %184, %184]  {id = 150 : i32}
            %186 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%185]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 151 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 152 : i32}
              affine.yield %208 : !air.async.token
            }
            %187 = air.wait_all async [%arg30, %186, %186]  {id = 153 : i32}
            %188 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%187]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 154 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 155 : i32}
              affine.yield %208 : !air.async.token
            }
            %189 = air.wait_all async [%arg30, %188, %188]  {id = 156 : i32}
            %190 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%189]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 157 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 158 : i32}
              affine.yield %208 : !air.async.token
            }
            %async_token_69 = air.execute [%arg30, %190, %async_token_68] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg22, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            } {id = 30 : i32}
            %191 = air.wait_all async [%arg30, %async_token_69, %async_token_69]  {id = 159 : i32}
            %192 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%191]  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 160 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 161 : i32}
              affine.yield %208 : !air.async.token
            }
            %193 = air.wait_all async [%arg30, %192, %192]  {id = 162 : i32}
            %194 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%193]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 163 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 164 : i32}
              affine.yield %208 : !air.async.token
            }
            %195 = air.wait_all async [%arg30, %194, %194]  {id = 165 : i32}
            %196 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%195]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 166 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 167 : i32}
              affine.yield %208 : !air.async.token
            }
            %197 = air.wait_all async [%arg30, %196, %196]  {id = 168 : i32}
            %198 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%197]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 169 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 170 : i32}
              affine.yield %208 : !air.async.token
            }
            %async_token_70 = air.execute [%arg30, %198] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg21, %arg22, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            } {id = 31 : i32}
            %199 = air.wait_all async [%arg30]  {id = 171 : i32}
            %200 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async  @V2L1_0[%arg28, %arg17, %arg16] (%arg23[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 172 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 173 : i32}
              affine.yield %208 : !air.async.token
            }
            %201 = air.wait_all async [%arg30, %200, %200]  {id = 174 : i32}
            %202 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%201]  @V2L1_1[%arg28, %arg17, %arg16] (%arg23[] [] []) {id = 102 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 175 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 176 : i32}
              affine.yield %208 : !air.async.token
            }
            %203 = air.wait_all async [%arg30, %202, %202]  {id = 177 : i32}
            %204 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%203]  @V2L1_2[%arg28, %arg17, %arg16] (%arg23[] [] []) {id = 103 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 178 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 179 : i32}
              affine.yield %208 : !air.async.token
            }
            %205 = air.wait_all async [%arg30, %204, %204]  {id = 180 : i32}
            %206 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %208 = air.channel.get async [%205]  @V2L1_3[%arg28, %arg17, %arg16] (%arg23[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              %209 = air.wait_all async [%208]  {id = 181 : i32}
              affine.yield %209 : !air.async.token
            } else {
              %208 = air.wait_all async  {id = 182 : i32}
              affine.yield %208 : !air.async.token
            }
            %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 32 : i32}
            %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 33 : i32}
            %async_token_75 = air.execute [%async_token_73, %async_token_71, %async_token_70, %arg30] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg26, %results_72, %results_74) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 34 : i32}
            %async_token_76 = air.execute [%async_token_75, %arg30] {
              func.call @mul_r_gp(%results_74, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 35 : i32}
            %async_token_77 = air.execute [%arg30, %async_token_76, %206] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg23, %arg25) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 36 : i32}
            %async_token_78 = air.execute [%async_token_76, %arg30] {
              func.call @accum_sp_r_s(%arg27, %results_74, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 37 : i32}
            %async_token_79 = air.execute [%arg30, %async_token_78] {
              func.call @vector_copy_32elems(%c0_i32_67, %results_72, %arg27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 38 : i32}
            %async_token_80 = air.execute [%async_token_79] {
              memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
            } {id = 39 : i32}
            %async_token_81 = air.execute [%async_token_78] {
              memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
            } {id = 40 : i32}
            %207 = air.wait_all async [%183, %185, %187, %189, %191, %193, %195, %197, %199, %201, %203, %205, %async_token_77, %async_token_79]  {id = 184 : i32}
            scf.yield %207 : !air.async.token
          }
          %181 = air.wait_all async [%180, %180]  {id = 188 : i32}
          %182 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %183 = arith.subi %arg17, %c1_61 : index
            %184 = air.channel.put async [%181]  @cascade_gp[%arg16, %183] (%arg25[] [] []) {id = 105 : i32} : (memref<64x64xbf16, 2 : i32>)
            %185 = air.channel.put async [%181]  @cascade_up[%arg16, %183] (%arg26[] [] []) {id = 106 : i32} : (memref<64x1xbf16, 2 : i32>)
            %186 = air.channel.put async [%181]  @cascade_sp[%arg16, %183] (%arg27[] [] []) {id = 107 : i32} : (memref<64x1xbf16, 2 : i32>)
            %187 = air.wait_all async [%184, %185, %186]  {id = 189 : i32}
            affine.yield %187 : !air.async.token
          } else {
            %183 = air.wait_all async [%181, %181]  {id = 185 : i32}
            %184 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_67, %results_68 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 41 : i32}
              %async_token_69, %results_70 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 42 : i32}
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 43 : i32}
              %186 = air.channel.get async [%async_token_67]  @cascade_gp[%arg16, %arg17] (%results_68[] [] []) {id = 108 : i32} : (memref<64x64xbf16, 2 : i32>)
              %187 = air.channel.get async [%async_token_69]  @cascade_up[%arg16, %arg17] (%results_70[] [] []) {id = 109 : i32} : (memref<64x1xbf16, 2 : i32>)
              %188 = air.channel.get async [%async_token_71]  @cascade_sp[%arg16, %arg17] (%results_72[] [] []) {id = 110 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 44 : i32}
              %async_token_75 = air.execute [%async_token_73, %183] {
                func.call @vector_copy_32elems(%c0_i32, %arg26, %results_74) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 45 : i32}
              %async_token_76 = air.execute [%async_token_75, %187] {
                func.call @maximum_up_u_bf16(%results_70, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 46 : i32}
              %async_token_77, %results_78 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 47 : i32}
              %async_token_79 = air.execute [%async_token_77, %async_token_76] {
                func.call @exp_up_minus_u(%results_70, %arg26, %results_78) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 48 : i32}
              %async_token_80, %results_81 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 49 : i32}
              %async_token_82 = air.execute [%async_token_80, %async_token_79] {
                func.call @exp_up_minus_u(%results_74, %arg26, %results_81) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 50 : i32}
              %async_token_83 = air.execute [%async_token_79, %186] {
                func.call @mul_r_gp(%results_78, %results_68) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 51 : i32}
              %async_token_84 = air.execute [%async_token_82, %183] {
                func.call @mul_r_gp(%results_81, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 52 : i32}
              %async_token_85 = air.execute [%async_token_84, %async_token_83] {
                func.call @add_gp_g(%arg25, %results_68) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 53 : i32}
              %async_token_86, %results_87 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 54 : i32}
              %async_token_88 = air.execute [%async_token_86] {
                func.call @zero_fill_sp_bf16(%results_87) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 55 : i32}
              %async_token_89 = air.execute [%async_token_88, %async_token_83, %188] {
                func.call @accum_sp_r_s(%results_72, %results_78, %results_87) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 56 : i32}
              %async_token_90 = air.execute [%async_token_89, %async_token_84, %183] {
                func.call @accum_sp_r_s(%arg27, %results_81, %results_87) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 57 : i32}
              %async_token_91 = air.execute [%async_token_90] {
                func.call @vector_copy_32elems(%c0_i32, %results_87, %results_72) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 58 : i32}
              %189 = arith.subi %arg17, %c1_61 : index
              %190 = air.channel.put async [%async_token_85]  @cascade_gp[%arg16, %189] (%results_68[] [] []) {id = 111 : i32} : (memref<64x64xbf16, 2 : i32>)
              %191 = air.channel.put async [%async_token_82]  @cascade_up[%arg16, %189] (%arg26[] [] []) {id = 112 : i32} : (memref<64x1xbf16, 2 : i32>)
              %192 = air.channel.put async [%async_token_91]  @cascade_sp[%arg16, %189] (%results_72[] [] []) {id = 113 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_92 = air.execute [%190] {
                memref.dealloc %results_68 : memref<64x64xbf16, 2 : i32>
              } {id = 59 : i32}
              %async_token_93 = air.execute [%async_token_79] {
                memref.dealloc %results_70 : memref<64x1xbf16, 2 : i32>
              } {id = 60 : i32}
              %async_token_94 = air.execute [%192] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              } {id = 61 : i32}
              %async_token_95 = air.execute [%async_token_82] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              } {id = 62 : i32}
              %async_token_96 = air.execute [%async_token_89] {
                memref.dealloc %results_78 : memref<64x1xbf16, 2 : i32>
              } {id = 63 : i32}
              %async_token_97 = air.execute [%async_token_90] {
                memref.dealloc %results_81 : memref<64x1xbf16, 2 : i32>
              } {id = 64 : i32}
              %async_token_98 = air.execute [%async_token_91] {
                memref.dealloc %results_87 : memref<64x1xbf16, 2 : i32>
              } {id = 65 : i32}
              %193 = air.wait_all async [%190, %191, %192]  {id = 186 : i32}
              affine.yield %193 : !air.async.token
            } else {
              %async_token_67, %results_68 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 66 : i32}
              %async_token_69, %results_70 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 67 : i32}
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 68 : i32}
              %186 = air.channel.get async [%async_token_67]  @cascade_gp[%arg16, %arg17] (%results_68[] [] []) {id = 114 : i32} : (memref<64x64xbf16, 2 : i32>)
              %187 = air.channel.get async [%async_token_69]  @cascade_up[%arg16, %arg17] (%results_70[] [] []) {id = 115 : i32} : (memref<64x1xbf16, 2 : i32>)
              %188 = air.channel.get async [%async_token_71]  @cascade_sp[%arg16, %arg17] (%results_72[] [] []) {id = 116 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 69 : i32}
              %async_token_75 = air.execute [%async_token_73, %183] {
                func.call @vector_copy_32elems(%c0_i32, %arg26, %results_74) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 70 : i32}
              %async_token_76 = air.execute [%async_token_75, %187] {
                func.call @maximum_up_u_bf16(%results_70, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 71 : i32}
              %async_token_77, %results_78 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 72 : i32}
              %async_token_79 = air.execute [%async_token_77, %async_token_76] {
                func.call @exp_up_minus_u(%results_70, %arg26, %results_78) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 73 : i32}
              %async_token_80, %results_81 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 74 : i32}
              %async_token_82 = air.execute [%async_token_80, %async_token_79] {
                func.call @exp_up_minus_u(%results_74, %arg26, %results_81) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 75 : i32}
              %async_token_83 = air.execute [%async_token_79, %186] {
                func.call @mul_r_gp(%results_78, %results_68) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 76 : i32}
              %async_token_84 = air.execute [%async_token_82, %183] {
                func.call @mul_r_gp(%results_81, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 77 : i32}
              %async_token_85 = air.execute [%async_token_84, %async_token_83] {
                func.call @add_gp_g(%arg25, %results_68) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 78 : i32}
              %async_token_86, %results_87 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 79 : i32}
              %async_token_88 = air.execute [%async_token_86] {
                func.call @zero_fill_sp_bf16(%results_87) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 80 : i32}
              %async_token_89 = air.execute [%async_token_88, %async_token_83, %188] {
                func.call @accum_sp_r_s(%results_72, %results_78, %results_87) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 81 : i32}
              %async_token_90 = air.execute [%async_token_89, %async_token_84, %183] {
                func.call @accum_sp_r_s(%arg27, %results_81, %results_87) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 82 : i32}
              %async_token_91 = air.execute [%async_token_90] {
                func.call @vector_copy_32elems(%c0_i32, %results_87, %results_72) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 83 : i32}
              %async_token_92 = air.execute [%async_token_91, %async_token_85] {
                func.call @div_gp_sp(%results_72, %results_68) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 84 : i32}
              %189 = air.channel.put async [%async_token_92]  @Gp2L2[%arg16, %c0_62] (%results_68[%c0_62, %c0_62, %c0_62, %c0_62] [%c8_60, %c8_60, %c8_60, %c8_60] [%c64_59, %c8_60, %c512_58, %c1_61]) {id = 117 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_93 = air.execute [%189] {
                memref.dealloc %results_68 : memref<64x64xbf16, 2 : i32>
              } {id = 85 : i32}
              %async_token_94 = air.execute [%async_token_79] {
                memref.dealloc %results_70 : memref<64x1xbf16, 2 : i32>
              } {id = 86 : i32}
              %async_token_95 = air.execute [%async_token_92] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              } {id = 87 : i32}
              %async_token_96 = air.execute [%async_token_82] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              } {id = 88 : i32}
              %async_token_97 = air.execute [%async_token_89] {
                memref.dealloc %results_78 : memref<64x1xbf16, 2 : i32>
              } {id = 89 : i32}
              %async_token_98 = air.execute [%async_token_90] {
                memref.dealloc %results_81 : memref<64x1xbf16, 2 : i32>
              } {id = 90 : i32}
              %async_token_99 = air.execute [%async_token_91] {
                memref.dealloc %results_87 : memref<64x1xbf16, 2 : i32>
              } {id = 91 : i32}
              %190 = air.wait_all async [%189]  {id = 187 : i32}
              affine.yield %190 : !air.async.token
            }
            %185 = air.wait_all async [%183]  {id = 190 : i32}
            affine.yield %185 : !air.async.token
          }
        }
        %async_token_41 = air.execute [%82] {
          memref.dealloc %results_23 : memref<64x64xbf16, 2 : i32>
        } {id = 92 : i32}
        %async_token_42 = air.execute [%82] {
          memref.dealloc %results_25 : memref<64x64xbf16, 2 : i32>
        } {id = 93 : i32}
        %async_token_43 = air.execute [%82] {
          memref.dealloc %results_27 : memref<64x64xbf16, 2 : i32>
        } {id = 94 : i32}
        %async_token_44 = air.execute [%82] {
          memref.dealloc %results_29 : memref<64x64xbf16, 2 : i32>
        } {id = 95 : i32}
        %async_token_45 = air.execute [%82] {
          memref.dealloc %results_31 : memref<64x64xbf16, 2 : i32>
        } {id = 96 : i32}
        %async_token_46 = air.execute [%82] {
          memref.dealloc %results_33 : memref<64x64xbf16, 2 : i32>
        } {id = 97 : i32}
        %async_token_47 = air.execute [%82] {
          memref.dealloc %results_35 : memref<64x1xbf16, 2 : i32>
        } {id = 98 : i32}
        %async_token_48 = air.execute [%82] {
          memref.dealloc %results_37 : memref<64x1xbf16, 2 : i32>
        } {id = 99 : i32}
        %async_token_49 = air.execute [%58] {
          memref.dealloc %results : memref<64x128xbf16, 1 : i32>
        } {id = 100 : i32}
        %async_token_50 = air.execute [%72] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        } {id = 101 : i32}
        %async_token_51 = air.execute [%62] {
          memref.dealloc %results_7 : memref<64x128xbf16, 1 : i32>
        } {id = 102 : i32}
        %async_token_52 = air.execute [%74] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        } {id = 103 : i32}
        %async_token_53 = air.execute [%66] {
          memref.dealloc %results_9 : memref<64x128xbf16, 1 : i32>
        } {id = 104 : i32}
        %async_token_54 = air.execute [%76] {
          memref.dealloc %results_17 : memref<64x64xbf16, 1 : i32>
        } {id = 105 : i32}
        %async_token_55 = air.execute [%70] {
          memref.dealloc %results_11 : memref<64x128xbf16, 1 : i32>
        } {id = 106 : i32}
        %async_token_56 = air.execute [%78] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        } {id = 107 : i32}
        %async_token_57 = air.execute [%81] {
          memref.dealloc %results_21 : memref<256x64xbf16, 1 : i32>
        } {id = 108 : i32}
      }
    }
    return
  }
}
