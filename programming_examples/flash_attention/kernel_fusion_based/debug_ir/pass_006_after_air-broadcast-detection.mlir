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
      %c4096 = arith.constant 4096 : index
      %c49152 = arith.constant 49152 : index
      %c32768 = arith.constant 32768 : index
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = affine.apply #map1()[%arg5]
      %3 = affine.apply #map()[%2]
      %4 = affine.apply #map2()[%2]
      %5 = affine.apply #map()[%2]
      %6 = affine.apply #map3()[%3, %1]
      %7 = affine.apply #map3()[%6, %c0]
      %8 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %7] [%c256, %c64] [%c128, %c1_0]) {id = 1 : i32} : (memref<2x256x128xbf16>)
      %9 = affine.apply #map3()[%6, %c64]
      %10 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %9] [%c256, %c64] [%c128, %c1_0]) {id = 2 : i32} : (memref<2x256x128xbf16>)
      %11 = affine.apply #map3()[%6, %c0]
      %12 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %11] [%c256, %c64] [%c128, %c1_0]) {id = 3 : i32} : (memref<2x256x128xbf16>)
      %13 = affine.apply #map3()[%6, %c64]
      %14 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %13] [%c256, %c64] [%c128, %c1_0]) {id = 4 : i32} : (memref<2x256x128xbf16>)
      %15 = affine.apply #map3()[%6, %c0]
      %16 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %15] [%c256, %c64] [%c128, %c1_0]) {id = 5 : i32} : (memref<2x256x128xbf16>)
      %17 = affine.apply #map3()[%6, %c64]
      %18 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %17] [%c256, %c64] [%c128, %c1_0]) {id = 6 : i32} : (memref<2x256x128xbf16>)
      %19 = affine.apply #map3()[%6, %c0]
      %20 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %19] [%c256, %c64] [%c128, %c1_0]) {id = 7 : i32} : (memref<2x256x128xbf16>)
      %21 = affine.apply #map3()[%6, %c64]
      %22 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %21] [%c256, %c64] [%c128, %c1_0]) {id = 8 : i32} : (memref<2x256x128xbf16>)
      %23 = affine.apply #map3()[%4, %c0]
      %24 = air.channel.put async  @QKIn_0[%c0] (%arg9[%c0, %c0, %c0, %23] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 9 : i32} : (memref<2x512x128xbf16>)
      %25 = affine.apply #map3()[%4, %c16384]
      %26 = air.channel.put async  @QKIn_1[%c0] (%arg9[%c0, %c0, %c0, %25] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 10 : i32} : (memref<2x512x128xbf16>)
      %27 = affine.apply #map3()[%4, %c32768]
      %28 = air.channel.put async  @QKIn_2[%c0] (%arg9[%c0, %c0, %c0, %27] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 11 : i32} : (memref<2x512x128xbf16>)
      %29 = affine.apply #map3()[%4, %c49152]
      %30 = air.channel.put async  @QKIn_3[%c0] (%arg9[%c0, %c0, %c0, %29] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 12 : i32} : (memref<2x512x128xbf16>)
      %31 = affine.apply #map3()[%5, %c0]
      %32 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %31] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 13 : i32} : (memref<2x512x64xbf16>)
      %33 = affine.apply #map3()[%5, %c8192]
      %34 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %33] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 14 : i32} : (memref<2x512x64xbf16>)
      %35 = affine.apply #map3()[%5, %c16384]
      %36 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %35] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 15 : i32} : (memref<2x512x64xbf16>)
      %37 = affine.apply #map3()[%5, %c24576]
      %38 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %37] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 16 : i32} : (memref<2x512x64xbf16>)
      %39 = air.channel.get async  @GpOut[%c0] (%arg11[%6] [%c32768] [%c1_0]) {id = 17 : i32} : (memref<2x256x64xbf16>)
      %40 = affine.apply #map4()[%2]
      %41 = affine.apply #map()[%40]
      %42 = affine.apply #map2()[%40]
      %43 = affine.apply #map()[%40]
      %44 = affine.apply #map3()[%41, %1]
      %45 = affine.apply #map3()[%44, %c0]
      %46 = air.channel.put async  @QKIn_0[%c1_0] (%arg8[%c0, %45] [%c256, %c64] [%c128, %c1_0]) {id = 18 : i32} : (memref<2x256x128xbf16>)
      %47 = affine.apply #map3()[%44, %c64]
      %48 = air.channel.put async  @QKIn_0[%c1_0] (%arg8[%c0, %47] [%c256, %c64] [%c128, %c1_0]) {id = 19 : i32} : (memref<2x256x128xbf16>)
      %49 = affine.apply #map3()[%44, %c0]
      %50 = air.channel.put async  @QKIn_1[%c1_0] (%arg8[%c0, %49] [%c256, %c64] [%c128, %c1_0]) {id = 20 : i32} : (memref<2x256x128xbf16>)
      %51 = affine.apply #map3()[%44, %c64]
      %52 = air.channel.put async  @QKIn_1[%c1_0] (%arg8[%c0, %51] [%c256, %c64] [%c128, %c1_0]) {id = 21 : i32} : (memref<2x256x128xbf16>)
      %53 = affine.apply #map3()[%44, %c0]
      %54 = air.channel.put async  @QKIn_2[%c1_0] (%arg8[%c0, %53] [%c256, %c64] [%c128, %c1_0]) {id = 22 : i32} : (memref<2x256x128xbf16>)
      %55 = affine.apply #map3()[%44, %c64]
      %56 = air.channel.put async  @QKIn_2[%c1_0] (%arg8[%c0, %55] [%c256, %c64] [%c128, %c1_0]) {id = 23 : i32} : (memref<2x256x128xbf16>)
      %57 = affine.apply #map3()[%44, %c0]
      %58 = air.channel.put async  @QKIn_3[%c1_0] (%arg8[%c0, %57] [%c256, %c64] [%c128, %c1_0]) {id = 24 : i32} : (memref<2x256x128xbf16>)
      %59 = affine.apply #map3()[%44, %c64]
      %60 = air.channel.put async  @QKIn_3[%c1_0] (%arg8[%c0, %59] [%c256, %c64] [%c128, %c1_0]) {id = 25 : i32} : (memref<2x256x128xbf16>)
      %61 = affine.apply #map3()[%42, %c0]
      %62 = air.channel.put async  @QKIn_0[%c1_0] (%arg9[%c0, %c0, %c0, %61] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 26 : i32} : (memref<2x512x128xbf16>)
      %63 = affine.apply #map3()[%42, %c16384]
      %64 = air.channel.put async  @QKIn_1[%c1_0] (%arg9[%c0, %c0, %c0, %63] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 27 : i32} : (memref<2x512x128xbf16>)
      %65 = affine.apply #map3()[%42, %c32768]
      %66 = air.channel.put async  @QKIn_2[%c1_0] (%arg9[%c0, %c0, %c0, %65] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 28 : i32} : (memref<2x512x128xbf16>)
      %67 = affine.apply #map3()[%42, %c49152]
      %68 = air.channel.put async  @QKIn_3[%c1_0] (%arg9[%c0, %c0, %c0, %67] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 29 : i32} : (memref<2x512x128xbf16>)
      %69 = affine.apply #map3()[%43, %c0]
      %70 = air.channel.put async  @VIn_0[%c1_0] (%arg10[%c0, %c0, %69] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 30 : i32} : (memref<2x512x64xbf16>)
      %71 = affine.apply #map3()[%43, %c8192]
      %72 = air.channel.put async  @VIn_1[%c1_0] (%arg10[%c0, %c0, %71] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 31 : i32} : (memref<2x512x64xbf16>)
      %73 = affine.apply #map3()[%43, %c16384]
      %74 = air.channel.put async  @VIn_2[%c1_0] (%arg10[%c0, %c0, %73] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 32 : i32} : (memref<2x512x64xbf16>)
      %75 = affine.apply #map3()[%43, %c24576]
      %76 = air.channel.put async  @VIn_3[%c1_0] (%arg10[%c0, %c0, %75] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 33 : i32} : (memref<2x512x64xbf16>)
      %77 = air.channel.get async  @GpOut[%c1_0] (%arg11[%44] [%c32768] [%c1_0]) {id = 34 : i32} : (memref<2x256x64xbf16>)
      %78 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) attributes {id = 2 : i32} {
        %c64_1 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c8 = arith.constant 8 : index
        %c1_2 = arith.constant 1 : index
        %c2_3 = arith.constant 2 : index
        %c0_4 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 1 : i32}
        %async_token_5, %results_6 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 2 : i32}
        %async_token_7, %results_8 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 3 : i32}
        %async_token_9, %results_10 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 4 : i32}
        %async_token_11, %results_12 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 5 : i32}
        %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 6 : i32}
        %async_token_15, %results_16 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 7 : i32}
        %async_token_17, %results_18 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 8 : i32}
        %async_token_19, %results_20 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        } {id = 9 : i32}
        %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 10 : i32}
        %async_token_23, %results_24 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 11 : i32}
        %async_token_25, %results_26 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 12 : i32}
        %async_token_27, %results_28 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 13 : i32}
        %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 14 : i32}
        %async_token_31, %results_32 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 15 : i32}
        %async_token_33, %results_34 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 16 : i32}
        %async_token_35, %results_36 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 17 : i32}
        %79 = air.wait_all async [%async_token]  {id = 1 : i32}
        %80 = scf.for %arg16 = %c0_4 to %c4 step %c1_2 iter_args(%arg17 = %79) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_0[%arg12, %c0_4, %c0_4] (%results[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 2 : i32}
          scf.yield %117 : !air.async.token
        }
        %81 = air.wait_all async [%80]  {id = 3 : i32}
        %82 = scf.for %arg16 = %c0_4 to %c4 step %c1_2 iter_args(%arg17 = %81) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_0[%arg12, %c0_4, %c0_4] (%results[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 4 : i32}
          scf.yield %117 : !air.async.token
        }
        %83 = air.wait_all async [%82]  {id = 5 : i32}
        %84 = scf.for %arg16 = %c0_4 to %c2_3 step %c1_2 iter_args(%arg17 = %83) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_0[%arg12, %c0_4, %c0_4] (%results[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.channel.get async [%arg17, %116]  @QKIn_0[%arg12] (%results[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
          %118 = air.channel.put async [%arg17, %117]  @QK2L1_0[%arg12, %c0_4, %c0_4] (%results[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
          %119 = air.wait_all async [%118]  {id = 6 : i32}
          scf.yield %119 : !air.async.token
        }
        %85 = air.wait_all async [%async_token_5]  {id = 7 : i32}
        %86 = scf.for %arg16 = %c0_4 to %c4 step %c1_2 iter_args(%arg17 = %85) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_6[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_1[%arg12, %c0_4, %c0_4] (%results_6[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 8 : i32}
          scf.yield %117 : !air.async.token
        }
        %87 = air.wait_all async [%86]  {id = 9 : i32}
        %88 = scf.for %arg16 = %c0_4 to %c4 step %c1_2 iter_args(%arg17 = %87) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_6[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_1[%arg12, %c0_4, %c0_4] (%results_6[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 10 : i32}
          scf.yield %117 : !air.async.token
        }
        %89 = air.wait_all async [%88]  {id = 11 : i32}
        %90 = scf.for %arg16 = %c0_4 to %c2_3 step %c1_2 iter_args(%arg17 = %89) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_6[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_1[%arg12, %c0_4, %c0_4] (%results_6[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.channel.get async [%arg17, %116]  @QKIn_1[%arg12] (%results_6[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
          %118 = air.channel.put async [%arg17, %117]  @QK2L1_1[%arg12, %c0_4, %c0_4] (%results_6[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
          %119 = air.wait_all async [%118]  {id = 12 : i32}
          scf.yield %119 : !air.async.token
        }
        %91 = air.wait_all async [%async_token_7]  {id = 13 : i32}
        %92 = scf.for %arg16 = %c0_4 to %c4 step %c1_2 iter_args(%arg17 = %91) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_8[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_2[%arg12, %c0_4, %c0_4] (%results_8[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 14 : i32}
          scf.yield %117 : !air.async.token
        }
        %93 = air.wait_all async [%92]  {id = 15 : i32}
        %94 = scf.for %arg16 = %c0_4 to %c4 step %c1_2 iter_args(%arg17 = %93) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_8[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_2[%arg12, %c0_4, %c0_4] (%results_8[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 54 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 16 : i32}
          scf.yield %117 : !air.async.token
        }
        %95 = air.wait_all async [%94]  {id = 17 : i32}
        %96 = scf.for %arg16 = %c0_4 to %c2_3 step %c1_2 iter_args(%arg17 = %95) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_8[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_2[%arg12, %c0_4, %c0_4] (%results_8[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 56 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.channel.get async [%arg17, %116]  @QKIn_2[%arg12] (%results_8[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 1 : i32>)
          %118 = air.channel.put async [%arg17, %117]  @QK2L1_2[%arg12, %c0_4, %c0_4] (%results_8[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 58 : i32} : (memref<64x64xbf16, 1 : i32>)
          %119 = air.wait_all async [%118]  {id = 18 : i32}
          scf.yield %119 : !air.async.token
        }
        %97 = air.wait_all async [%async_token_9]  {id = 19 : i32}
        %98 = scf.for %arg16 = %c0_4 to %c4 step %c1_2 iter_args(%arg17 = %97) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_10[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_3[%arg12, %c0_4, %c0_4] (%results_10[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 60 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 20 : i32}
          scf.yield %117 : !air.async.token
        }
        %99 = air.wait_all async [%98]  {id = 21 : i32}
        %100 = scf.for %arg16 = %c0_4 to %c4 step %c1_2 iter_args(%arg17 = %99) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_10[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_3[%arg12, %c0_4, %c0_4] (%results_10[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 62 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 22 : i32}
          scf.yield %117 : !air.async.token
        }
        %101 = air.wait_all async [%100]  {id = 23 : i32}
        %102 = scf.for %arg16 = %c0_4 to %c2_3 step %c1_2 iter_args(%arg17 = %101) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_10[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @QK2L1_3[%arg12, %c0_4, %c0_4] (%results_10[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 64 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.channel.get async [%arg17, %116]  @QKIn_3[%arg12] (%results_10[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 1 : i32>)
          %118 = air.channel.put async [%arg17, %117]  @QK2L1_3[%arg12, %c0_4, %c0_4] (%results_10[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 66 : i32} : (memref<64x64xbf16, 1 : i32>)
          %119 = air.wait_all async [%118]  {id = 24 : i32}
          scf.yield %119 : !air.async.token
        }
        %103 = air.wait_all async [%async_token_11]  {id = 25 : i32}
        %104 = scf.for %arg16 = %c0_4 to %c2_3 step %c1_2 iter_args(%arg17 = %103) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @VIn_0[%arg12] (%results_12[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @V2L1_0[%arg12, %c0_4, %c0_4] (%results_12[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 68 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 26 : i32}
          scf.yield %117 : !air.async.token
        }
        %105 = air.wait_all async [%async_token_13]  {id = 27 : i32}
        %106 = scf.for %arg16 = %c0_4 to %c2_3 step %c1_2 iter_args(%arg17 = %105) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @VIn_1[%arg12] (%results_14[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @V2L1_1[%arg12, %c0_4, %c0_4] (%results_14[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 70 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 28 : i32}
          scf.yield %117 : !air.async.token
        }
        %107 = air.wait_all async [%async_token_15]  {id = 29 : i32}
        %108 = scf.for %arg16 = %c0_4 to %c2_3 step %c1_2 iter_args(%arg17 = %107) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @VIn_2[%arg12] (%results_16[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @V2L1_2[%arg12, %c0_4, %c0_4] (%results_16[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 72 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 30 : i32}
          scf.yield %117 : !air.async.token
        }
        %109 = air.wait_all async [%async_token_17]  {id = 31 : i32}
        %110 = scf.for %arg16 = %c0_4 to %c2_3 step %c1_2 iter_args(%arg17 = %109) -> (!air.async.token) {
          %115 = air.channel.get async [%arg17]  @VIn_3[%arg12] (%results_18[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 1 : i32>)
          %116 = air.channel.put async [%arg17, %115]  @V2L1_3[%arg12, %c0_4, %c0_4] (%results_18[%c0_4, %c0_4, %c0_4, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 74 : i32} : (memref<64x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 32 : i32}
          scf.yield %117 : !air.async.token
        }
        %111 = air.wait_all async [%async_token_19]  {id = 33 : i32}
        %112 = scf.parallel (%arg16) = (%c0_4) to (%c4) step (%c1_2) init (%111) -> !air.async.token {
          %115 = affine.apply #map5()[%arg16]
          %116 = air.channel.get async [%111]  @Gp2L2[%arg16, %c0_4] (%results_20[%115, %c0_4] [%c64_1, %c64_1] [%c64_1, %c1_2]) {id = 75 : i32} : (memref<256x64xbf16, 1 : i32>)
          %117 = air.wait_all async [%116]  {id = 34 : i32}
          scf.reduce(%117 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %118 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %118 : !air.async.token
          }
        }
        %113 = air.channel.put async [%112]  @GpOut[%arg12] (%results_20[] [] []) {id = 76 : i32} : (memref<256x64xbf16, 1 : i32>)
        %114 = air.herd @herd_0 async [%async_token_21, %async_token_23, %async_token_25, %async_token_27, %async_token_29, %async_token_31, %async_token_33, %async_token_35]  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%results_22, %arg21=%results_24, %arg22=%results_26, %arg23=%results_28, %arg24=%results_30, %arg25=%results_32, %arg26=%results_34, %arg27=%results_36, %arg28=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {id = 1 : i32, link_with = "attn.o"} {
          %c512_54 = arith.constant 512 : index
          %c64_55 = arith.constant 64 : index
          %c8_56 = arith.constant 8 : index
          %c1_57 = arith.constant 1 : index
          %c0_58 = arith.constant 0 : index
          %c2_59 = arith.constant 2 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_60 = air.execute {
            func.call @zero_fill_gp_bf16(%arg25) : (memref<64x64xbf16, 2 : i32>) -> ()
          } {id = 18 : i32}
          %async_token_61 = air.execute {
            func.call @zero_fill_sp_bf16(%arg27) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 19 : i32}
          %async_token_62 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 20 : i32}
          %115 = air.wait_all async  {id = 35 : i32}
          %116 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 36 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 37 : i32}
            affine.yield %215 : !air.async.token
          }
          %117 = air.wait_all async [%116, %116]  {id = 38 : i32}
          %118 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%117]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 39 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 40 : i32}
            affine.yield %215 : !air.async.token
          }
          %119 = air.wait_all async [%118, %118]  {id = 41 : i32}
          %120 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%119]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 42 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 43 : i32}
            affine.yield %215 : !air.async.token
          }
          %121 = air.wait_all async [%120, %120]  {id = 44 : i32}
          %122 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%121]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 45 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 46 : i32}
            affine.yield %215 : !air.async.token
          }
          %123 = arith.index_cast %arg16 : index to i32
          %124 = arith.cmpi eq, %123, %c0_i32 : i32
          %125 = air.wait_all async [%122]  {id = 47 : i32}
          %126 = scf.if %124 -> (!air.async.token) {
            %async_token_63 = air.execute [%122] {
              func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 21 : i32}
            %215 = air.wait_all async [%async_token_63]  {id = 48 : i32}
            scf.yield %215 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 49 : i32}
            scf.yield %215 : !air.async.token
          }
          %127 = air.wait_all async  {id = 50 : i32}
          %128 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 51 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 52 : i32}
            affine.yield %215 : !air.async.token
          }
          %129 = air.wait_all async [%128, %128]  {id = 53 : i32}
          %130 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%129]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 54 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 55 : i32}
            affine.yield %215 : !air.async.token
          }
          %131 = air.wait_all async [%130, %130]  {id = 56 : i32}
          %132 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%131]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 57 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 58 : i32}
            affine.yield %215 : !air.async.token
          }
          %133 = air.wait_all async [%132, %132]  {id = 59 : i32}
          %134 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%133]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 60 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 61 : i32}
            affine.yield %215 : !air.async.token
          }
          %135 = arith.index_cast %arg16 : index to i32
          %136 = arith.cmpi eq, %135, %c1_i32 : i32
          %137 = air.wait_all async [%134]  {id = 62 : i32}
          %138 = scf.if %136 -> (!air.async.token) {
            %async_token_63 = air.execute [%134] {
              func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 22 : i32}
            %215 = air.wait_all async [%async_token_63]  {id = 63 : i32}
            scf.yield %215 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 64 : i32}
            scf.yield %215 : !air.async.token
          }
          %139 = air.wait_all async  {id = 65 : i32}
          %140 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 66 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 67 : i32}
            affine.yield %215 : !air.async.token
          }
          %141 = air.wait_all async [%140, %140]  {id = 68 : i32}
          %142 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%141]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 69 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 70 : i32}
            affine.yield %215 : !air.async.token
          }
          %143 = air.wait_all async [%142, %142]  {id = 71 : i32}
          %144 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%143]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 72 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 73 : i32}
            affine.yield %215 : !air.async.token
          }
          %145 = air.wait_all async [%144, %144]  {id = 74 : i32}
          %146 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%145]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 75 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 76 : i32}
            affine.yield %215 : !air.async.token
          }
          %147 = arith.index_cast %arg16 : index to i32
          %148 = arith.cmpi eq, %147, %c2_i32 : i32
          %149 = air.wait_all async [%146]  {id = 77 : i32}
          %150 = scf.if %148 -> (!air.async.token) {
            %async_token_63 = air.execute [%146] {
              func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 23 : i32}
            %215 = air.wait_all async [%async_token_63]  {id = 78 : i32}
            scf.yield %215 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 79 : i32}
            scf.yield %215 : !air.async.token
          }
          %151 = air.wait_all async  {id = 80 : i32}
          %152 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 81 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 82 : i32}
            affine.yield %215 : !air.async.token
          }
          %153 = air.wait_all async [%152, %152]  {id = 83 : i32}
          %154 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%153]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 84 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 85 : i32}
            affine.yield %215 : !air.async.token
          }
          %155 = air.wait_all async [%154, %154]  {id = 86 : i32}
          %156 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%155]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 87 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 88 : i32}
            affine.yield %215 : !air.async.token
          }
          %157 = air.wait_all async [%156, %156]  {id = 89 : i32}
          %158 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%157]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 90 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 91 : i32}
            affine.yield %215 : !air.async.token
          }
          %159 = arith.index_cast %arg16 : index to i32
          %160 = arith.cmpi eq, %159, %c3_i32 : i32
          %161 = air.wait_all async [%158]  {id = 92 : i32}
          %162 = scf.if %160 -> (!air.async.token) {
            %async_token_63 = air.execute [%158] {
              func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 24 : i32}
            %215 = air.wait_all async [%async_token_63]  {id = 93 : i32}
            scf.yield %215 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 94 : i32}
            scf.yield %215 : !air.async.token
          }
          %163 = air.wait_all async  {id = 95 : i32}
          %164 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 96 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 97 : i32}
            affine.yield %215 : !air.async.token
          }
          %165 = air.wait_all async [%164, %164]  {id = 98 : i32}
          %166 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%165]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 99 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 100 : i32}
            affine.yield %215 : !air.async.token
          }
          %167 = air.wait_all async [%166, %166]  {id = 101 : i32}
          %168 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%167]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 102 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 103 : i32}
            affine.yield %215 : !air.async.token
          }
          %169 = air.wait_all async [%168, %168]  {id = 104 : i32}
          %170 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%169]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 105 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 106 : i32}
            affine.yield %215 : !air.async.token
          }
          %171 = arith.index_cast %arg16 : index to i32
          %172 = arith.cmpi eq, %171, %c0_i32 : i32
          %173 = air.wait_all async [%170]  {id = 107 : i32}
          %174 = scf.if %172 -> (!air.async.token) {
            %async_token_63 = air.execute [%170] {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 25 : i32}
            %215 = air.wait_all async [%async_token_63]  {id = 108 : i32}
            scf.yield %215 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 109 : i32}
            scf.yield %215 : !air.async.token
          }
          %175 = air.wait_all async  {id = 110 : i32}
          %176 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 111 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 112 : i32}
            affine.yield %215 : !air.async.token
          }
          %177 = air.wait_all async [%176, %176]  {id = 113 : i32}
          %178 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%177]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 114 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 115 : i32}
            affine.yield %215 : !air.async.token
          }
          %179 = air.wait_all async [%178, %178]  {id = 116 : i32}
          %180 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%179]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 117 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 118 : i32}
            affine.yield %215 : !air.async.token
          }
          %181 = air.wait_all async [%180, %180]  {id = 119 : i32}
          %182 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%181]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 120 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 121 : i32}
            affine.yield %215 : !air.async.token
          }
          %183 = arith.index_cast %arg16 : index to i32
          %184 = arith.cmpi eq, %183, %c1_i32 : i32
          %185 = air.wait_all async [%182]  {id = 122 : i32}
          %186 = scf.if %184 -> (!air.async.token) {
            %async_token_63 = air.execute [%182] {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 26 : i32}
            %215 = air.wait_all async [%async_token_63]  {id = 123 : i32}
            scf.yield %215 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 124 : i32}
            scf.yield %215 : !air.async.token
          }
          %187 = air.wait_all async  {id = 125 : i32}
          %188 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 126 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 127 : i32}
            affine.yield %215 : !air.async.token
          }
          %189 = air.wait_all async [%188, %188]  {id = 128 : i32}
          %190 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%189]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 102 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 129 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 130 : i32}
            affine.yield %215 : !air.async.token
          }
          %191 = air.wait_all async [%190, %190]  {id = 131 : i32}
          %192 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%191]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 103 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 132 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 133 : i32}
            affine.yield %215 : !air.async.token
          }
          %193 = air.wait_all async [%192, %192]  {id = 134 : i32}
          %194 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%193]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 135 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 136 : i32}
            affine.yield %215 : !air.async.token
          }
          %195 = arith.index_cast %arg16 : index to i32
          %196 = arith.cmpi eq, %195, %c2_i32 : i32
          %197 = air.wait_all async [%194]  {id = 137 : i32}
          %198 = scf.if %196 -> (!air.async.token) {
            %async_token_63 = air.execute [%194] {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 27 : i32}
            %215 = air.wait_all async [%async_token_63]  {id = 138 : i32}
            scf.yield %215 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 139 : i32}
            scf.yield %215 : !air.async.token
          }
          %199 = air.wait_all async  {id = 140 : i32}
          %200 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 105 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 141 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 142 : i32}
            affine.yield %215 : !air.async.token
          }
          %201 = air.wait_all async [%200, %200]  {id = 143 : i32}
          %202 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%201]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 106 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 144 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 145 : i32}
            affine.yield %215 : !air.async.token
          }
          %203 = air.wait_all async [%202, %202]  {id = 146 : i32}
          %204 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%203]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 147 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 148 : i32}
            affine.yield %215 : !air.async.token
          }
          %205 = air.wait_all async [%204, %204]  {id = 149 : i32}
          %206 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %215 = air.channel.get async [%205]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 108 : i32} : (memref<64x64xbf16, 2 : i32>)
            %216 = air.wait_all async [%215]  {id = 150 : i32}
            affine.yield %216 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 151 : i32}
            affine.yield %215 : !air.async.token
          }
          %207 = arith.index_cast %arg16 : index to i32
          %208 = arith.cmpi eq, %207, %c3_i32 : i32
          %209 = air.wait_all async [%206]  {id = 152 : i32}
          %210 = scf.if %208 -> (!air.async.token) {
            %async_token_63 = air.execute [%206] {
              func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 28 : i32}
            %215 = air.wait_all async [%async_token_63]  {id = 153 : i32}
            scf.yield %215 : !air.async.token
          } else {
            %215 = air.wait_all async  {id = 154 : i32}
            scf.yield %215 : !air.async.token
          }
          %211 = air.wait_all async [%async_token_60, %async_token_61, %async_token_62]  {id = 191 : i32}
          %212 = scf.for %arg29 = %c0_58 to %c2_59 step %c1_57 iter_args(%arg30 = %211) -> (!air.async.token) {
            %async_token_63 = air.execute [%arg30] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            } {id = 29 : i32}
            %215 = air.wait_all async [%arg30]  {id = 155 : i32}
            %216 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 109 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 156 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 157 : i32}
              affine.yield %240 : !air.async.token
            }
            %217 = air.wait_all async [%arg30, %216, %216]  {id = 158 : i32}
            %218 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%217]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 159 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 160 : i32}
              affine.yield %240 : !air.async.token
            }
            %219 = air.wait_all async [%arg30, %218, %218]  {id = 161 : i32}
            %220 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%219]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 111 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 162 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 163 : i32}
              affine.yield %240 : !air.async.token
            }
            %221 = air.wait_all async [%arg30, %220, %220]  {id = 164 : i32}
            %222 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%221]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 112 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 165 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 166 : i32}
              affine.yield %240 : !air.async.token
            }
            %async_token_64 = air.execute [%arg30, %222, %async_token_63] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg22, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            } {id = 30 : i32}
            %223 = air.wait_all async [%arg30, %async_token_64, %async_token_64]  {id = 167 : i32}
            %224 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%223]  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 168 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 169 : i32}
              affine.yield %240 : !air.async.token
            }
            %225 = air.wait_all async [%arg30, %224, %224]  {id = 170 : i32}
            %226 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%225]  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 114 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 171 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 172 : i32}
              affine.yield %240 : !air.async.token
            }
            %227 = air.wait_all async [%arg30, %226, %226]  {id = 173 : i32}
            %228 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%227]  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 115 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 174 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 175 : i32}
              affine.yield %240 : !air.async.token
            }
            %229 = air.wait_all async [%arg30, %228, %228]  {id = 176 : i32}
            %230 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%229]  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) {id = 116 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 177 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 178 : i32}
              affine.yield %240 : !air.async.token
            }
            %async_token_65 = air.execute [%arg30, %230] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg21, %arg22, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            } {id = 31 : i32}
            %231 = air.wait_all async [%arg30]  {id = 179 : i32}
            %232 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async  @V2L1_0[%arg28, %arg17, %arg16] (%arg23[] [] []) {id = 117 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 180 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 181 : i32}
              affine.yield %240 : !air.async.token
            }
            %233 = air.wait_all async [%arg30, %232, %232]  {id = 182 : i32}
            %234 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%233]  @V2L1_1[%arg28, %arg17, %arg16] (%arg23[] [] []) {id = 118 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 183 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 184 : i32}
              affine.yield %240 : !air.async.token
            }
            %235 = air.wait_all async [%arg30, %234, %234]  {id = 185 : i32}
            %236 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%235]  @V2L1_2[%arg28, %arg17, %arg16] (%arg23[] [] []) {id = 119 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 186 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 187 : i32}
              affine.yield %240 : !air.async.token
            }
            %237 = air.wait_all async [%arg30, %236, %236]  {id = 188 : i32}
            %238 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %240 = air.channel.get async [%237]  @V2L1_3[%arg28, %arg17, %arg16] (%arg23[] [] []) {id = 120 : i32} : (memref<64x64xbf16, 2 : i32>)
              %241 = air.wait_all async [%240]  {id = 189 : i32}
              affine.yield %241 : !air.async.token
            } else {
              %240 = air.wait_all async  {id = 190 : i32}
              affine.yield %240 : !air.async.token
            }
            %async_token_66, %results_67 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 32 : i32}
            %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 33 : i32}
            %async_token_70 = air.execute [%async_token_68, %async_token_66, %async_token_65, %arg30] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg26, %results_67, %results_69) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 34 : i32}
            %async_token_71 = air.execute [%async_token_70, %arg30] {
              func.call @mul_r_gp(%results_69, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 35 : i32}
            %async_token_72 = air.execute [%arg30, %async_token_71, %238] {
              %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg23, %arg25) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 36 : i32}
            %async_token_73 = air.execute [%async_token_71, %arg30] {
              func.call @accum_sp_r_s(%arg27, %results_69, %results_67) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 37 : i32}
            %async_token_74 = air.execute [%arg30, %async_token_73] {
              func.call @vector_copy_32elems(%c0_i32, %results_67, %arg27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 38 : i32}
            %async_token_75 = air.execute [%async_token_74] {
              memref.dealloc %results_67 : memref<64x1xbf16, 2 : i32>
            } {id = 39 : i32}
            %async_token_76 = air.execute [%async_token_73] {
              memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
            } {id = 40 : i32}
            %239 = air.wait_all async [%215, %217, %219, %221, %223, %225, %227, %229, %231, %233, %235, %237, %async_token_72, %async_token_74]  {id = 192 : i32}
            scf.yield %239 : !air.async.token
          }
          %213 = air.wait_all async [%212, %212]  {id = 196 : i32}
          %214 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %215 = arith.subi %arg17, %c1_57 : index
            %216 = air.channel.put async [%213]  @cascade_gp[%arg16, %215] (%arg25[] [] []) {id = 121 : i32} : (memref<64x64xbf16, 2 : i32>)
            %217 = air.channel.put async [%213]  @cascade_up[%arg16, %215] (%arg26[] [] []) {id = 122 : i32} : (memref<64x1xbf16, 2 : i32>)
            %218 = air.channel.put async [%213]  @cascade_sp[%arg16, %215] (%arg27[] [] []) {id = 123 : i32} : (memref<64x1xbf16, 2 : i32>)
            %219 = air.wait_all async [%216, %217, %218]  {id = 197 : i32}
            affine.yield %219 : !air.async.token
          } else {
            %215 = air.wait_all async [%213, %213]  {id = 193 : i32}
            %216 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_63, %results_64 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 41 : i32}
              %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 42 : i32}
              %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 43 : i32}
              %218 = air.channel.get async [%async_token_63]  @cascade_gp[%arg16, %arg17] (%results_64[] [] []) {id = 124 : i32} : (memref<64x64xbf16, 2 : i32>)
              %219 = air.channel.get async [%async_token_65]  @cascade_up[%arg16, %arg17] (%results_66[] [] []) {id = 125 : i32} : (memref<64x1xbf16, 2 : i32>)
              %220 = air.channel.get async [%async_token_67]  @cascade_sp[%arg16, %arg17] (%results_68[] [] []) {id = 126 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_69, %results_70 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 44 : i32}
              %async_token_71 = air.execute [%async_token_69, %215] {
                func.call @vector_copy_32elems(%c0_i32, %arg26, %results_70) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 45 : i32}
              %async_token_72 = air.execute [%async_token_71, %219] {
                func.call @maximum_up_u_bf16(%results_66, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 46 : i32}
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 47 : i32}
              %async_token_75 = air.execute [%async_token_73, %async_token_72] {
                func.call @exp_up_minus_u(%results_66, %arg26, %results_74) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 48 : i32}
              %async_token_76, %results_77 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 49 : i32}
              %async_token_78 = air.execute [%async_token_76, %async_token_75] {
                func.call @exp_up_minus_u(%results_70, %arg26, %results_77) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 50 : i32}
              %async_token_79 = air.execute [%async_token_75, %218] {
                func.call @mul_r_gp(%results_74, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 51 : i32}
              %async_token_80 = air.execute [%async_token_78, %215] {
                func.call @mul_r_gp(%results_77, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 52 : i32}
              %async_token_81 = air.execute [%async_token_80, %async_token_79] {
                func.call @add_gp_g(%arg25, %results_64) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 53 : i32}
              %async_token_82, %results_83 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 54 : i32}
              %async_token_84 = air.execute [%async_token_82] {
                func.call @zero_fill_sp_bf16(%results_83) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 55 : i32}
              %async_token_85 = air.execute [%async_token_84, %async_token_79, %220] {
                func.call @accum_sp_r_s(%results_68, %results_74, %results_83) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 56 : i32}
              %async_token_86 = air.execute [%async_token_85, %async_token_80, %215] {
                func.call @accum_sp_r_s(%arg27, %results_77, %results_83) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 57 : i32}
              %async_token_87 = air.execute [%async_token_86] {
                func.call @vector_copy_32elems(%c0_i32, %results_83, %results_68) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 58 : i32}
              %221 = arith.subi %arg17, %c1_57 : index
              %222 = air.channel.put async [%async_token_81]  @cascade_gp[%arg16, %221] (%results_64[] [] []) {id = 127 : i32} : (memref<64x64xbf16, 2 : i32>)
              %223 = air.channel.put async [%async_token_78]  @cascade_up[%arg16, %221] (%arg26[] [] []) {id = 128 : i32} : (memref<64x1xbf16, 2 : i32>)
              %224 = air.channel.put async [%async_token_87]  @cascade_sp[%arg16, %221] (%results_68[] [] []) {id = 129 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_88 = air.execute [%222] {
                memref.dealloc %results_64 : memref<64x64xbf16, 2 : i32>
              } {id = 59 : i32}
              %async_token_89 = air.execute [%async_token_75] {
                memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
              } {id = 60 : i32}
              %async_token_90 = air.execute [%224] {
                memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
              } {id = 61 : i32}
              %async_token_91 = air.execute [%async_token_78] {
                memref.dealloc %results_70 : memref<64x1xbf16, 2 : i32>
              } {id = 62 : i32}
              %async_token_92 = air.execute [%async_token_85] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              } {id = 63 : i32}
              %async_token_93 = air.execute [%async_token_86] {
                memref.dealloc %results_77 : memref<64x1xbf16, 2 : i32>
              } {id = 64 : i32}
              %async_token_94 = air.execute [%async_token_87] {
                memref.dealloc %results_83 : memref<64x1xbf16, 2 : i32>
              } {id = 65 : i32}
              %225 = air.wait_all async [%222, %223, %224]  {id = 194 : i32}
              affine.yield %225 : !air.async.token
            } else {
              %async_token_63, %results_64 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 66 : i32}
              %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 67 : i32}
              %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 68 : i32}
              %218 = air.channel.get async [%async_token_63]  @cascade_gp[%arg16, %arg17] (%results_64[] [] []) {id = 130 : i32} : (memref<64x64xbf16, 2 : i32>)
              %219 = air.channel.get async [%async_token_65]  @cascade_up[%arg16, %arg17] (%results_66[] [] []) {id = 131 : i32} : (memref<64x1xbf16, 2 : i32>)
              %220 = air.channel.get async [%async_token_67]  @cascade_sp[%arg16, %arg17] (%results_68[] [] []) {id = 132 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_69, %results_70 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 69 : i32}
              %async_token_71 = air.execute [%async_token_69, %215] {
                func.call @vector_copy_32elems(%c0_i32, %arg26, %results_70) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 70 : i32}
              %async_token_72 = air.execute [%async_token_71, %219] {
                func.call @maximum_up_u_bf16(%results_66, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 71 : i32}
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 72 : i32}
              %async_token_75 = air.execute [%async_token_73, %async_token_72] {
                func.call @exp_up_minus_u(%results_66, %arg26, %results_74) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 73 : i32}
              %async_token_76, %results_77 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 74 : i32}
              %async_token_78 = air.execute [%async_token_76, %async_token_75] {
                func.call @exp_up_minus_u(%results_70, %arg26, %results_77) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 75 : i32}
              %async_token_79 = air.execute [%async_token_75, %218] {
                func.call @mul_r_gp(%results_74, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 76 : i32}
              %async_token_80 = air.execute [%async_token_78, %215] {
                func.call @mul_r_gp(%results_77, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 77 : i32}
              %async_token_81 = air.execute [%async_token_80, %async_token_79] {
                func.call @add_gp_g(%arg25, %results_64) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 78 : i32}
              %async_token_82, %results_83 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 79 : i32}
              %async_token_84 = air.execute [%async_token_82] {
                func.call @zero_fill_sp_bf16(%results_83) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 80 : i32}
              %async_token_85 = air.execute [%async_token_84, %async_token_79, %220] {
                func.call @accum_sp_r_s(%results_68, %results_74, %results_83) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 81 : i32}
              %async_token_86 = air.execute [%async_token_85, %async_token_80, %215] {
                func.call @accum_sp_r_s(%arg27, %results_77, %results_83) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 82 : i32}
              %async_token_87 = air.execute [%async_token_86] {
                func.call @vector_copy_32elems(%c0_i32, %results_83, %results_68) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 83 : i32}
              %async_token_88 = air.execute [%async_token_87, %async_token_81] {
                func.call @div_gp_sp(%results_68, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 84 : i32}
              %221 = air.channel.put async [%async_token_88]  @Gp2L2[%arg16, %c0_58] (%results_64[%c0_58, %c0_58, %c0_58, %c0_58] [%c8_56, %c8_56, %c8_56, %c8_56] [%c64_55, %c8_56, %c512_54, %c1_57]) {id = 133 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_89 = air.execute [%221] {
                memref.dealloc %results_64 : memref<64x64xbf16, 2 : i32>
              } {id = 85 : i32}
              %async_token_90 = air.execute [%async_token_75] {
                memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
              } {id = 86 : i32}
              %async_token_91 = air.execute [%async_token_88] {
                memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
              } {id = 87 : i32}
              %async_token_92 = air.execute [%async_token_78] {
                memref.dealloc %results_70 : memref<64x1xbf16, 2 : i32>
              } {id = 88 : i32}
              %async_token_93 = air.execute [%async_token_85] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              } {id = 89 : i32}
              %async_token_94 = air.execute [%async_token_86] {
                memref.dealloc %results_77 : memref<64x1xbf16, 2 : i32>
              } {id = 90 : i32}
              %async_token_95 = air.execute [%async_token_87] {
                memref.dealloc %results_83 : memref<64x1xbf16, 2 : i32>
              } {id = 91 : i32}
              %222 = air.wait_all async [%221]  {id = 195 : i32}
              affine.yield %222 : !air.async.token
            }
            %217 = air.wait_all async [%215]  {id = 198 : i32}
            affine.yield %217 : !air.async.token
          }
        }
        %async_token_37 = air.execute [%114] {
          memref.dealloc %results_22 : memref<64x64xbf16, 2 : i32>
        } {id = 92 : i32}
        %async_token_38 = air.execute [%114] {
          memref.dealloc %results_24 : memref<64x64xbf16, 2 : i32>
        } {id = 93 : i32}
        %async_token_39 = air.execute [%114] {
          memref.dealloc %results_26 : memref<64x64xbf16, 2 : i32>
        } {id = 94 : i32}
        %async_token_40 = air.execute [%114] {
          memref.dealloc %results_28 : memref<64x64xbf16, 2 : i32>
        } {id = 95 : i32}
        %async_token_41 = air.execute [%114] {
          memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
        } {id = 96 : i32}
        %async_token_42 = air.execute [%114] {
          memref.dealloc %results_32 : memref<64x64xbf16, 2 : i32>
        } {id = 97 : i32}
        %async_token_43 = air.execute [%114] {
          memref.dealloc %results_34 : memref<64x1xbf16, 2 : i32>
        } {id = 98 : i32}
        %async_token_44 = air.execute [%114] {
          memref.dealloc %results_36 : memref<64x1xbf16, 2 : i32>
        } {id = 99 : i32}
        %async_token_45 = air.execute [%84] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        } {id = 100 : i32}
        %async_token_46 = air.execute [%104] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        } {id = 101 : i32}
        %async_token_47 = air.execute [%90] {
          memref.dealloc %results_6 : memref<64x64xbf16, 1 : i32>
        } {id = 102 : i32}
        %async_token_48 = air.execute [%106] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        } {id = 103 : i32}
        %async_token_49 = air.execute [%96] {
          memref.dealloc %results_8 : memref<64x64xbf16, 1 : i32>
        } {id = 104 : i32}
        %async_token_50 = air.execute [%108] {
          memref.dealloc %results_16 : memref<64x64xbf16, 1 : i32>
        } {id = 105 : i32}
        %async_token_51 = air.execute [%102] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        } {id = 106 : i32}
        %async_token_52 = air.execute [%110] {
          memref.dealloc %results_18 : memref<64x64xbf16, 1 : i32>
        } {id = 107 : i32}
        %async_token_53 = air.execute [%113] {
          memref.dealloc %results_20 : memref<256x64xbf16, 1 : i32>
        } {id = 108 : i32}
      }
    }
    return
  }
}
