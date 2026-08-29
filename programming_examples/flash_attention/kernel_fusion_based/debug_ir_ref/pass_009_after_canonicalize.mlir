#map = affine_map<()[s0, s1] -> (s0 * 32768 + s1 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 32768)>
#map2 = affine_map<()[s0] -> (s0 * 32768 + 4096)>
#map3 = affine_map<()[s0] -> (s0 * 32768 + 8192)>
#map4 = affine_map<()[s0] -> (s0 * 32768 + 12288)>
#map5 = affine_map<()[s0, s1] -> (s0 * 32768 + s1 * 16384 + 16384)>
#map6 = affine_map<()[s0] -> (s0 * 32768 + 16384)>
#map7 = affine_map<()[s0] -> (s0 * 32768 + 20480)>
#map8 = affine_map<()[s0] -> (s0 * 32768 + 24576)>
#map9 = affine_map<()[s0] -> (s0 * 32768 + 28672)>
#map10 = affine_map<()[s0] -> (s0 * 64)>
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
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<2x256x64xbf16>, %arg1: memref<2x256x64xbf16>, %arg2: memref<2x256x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16> attributes {id = 3 : i32} {
      %c2 = arith.constant 2 : index
      %c16384 = arith.constant 16384 : index
      %c4096 = arith.constant 4096 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = affine.apply #map()[%arg5, %arg4]
      %3 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %2] [%c256, %c64] [%c64, %c1_0]) {id = 1 : i32} : (memref<2x256x64xbf16>)
      %4 = affine.apply #map()[%arg5, %arg4]
      %5 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %4] [%c256, %c64] [%c64, %c1_0]) {id = 2 : i32} : (memref<2x256x64xbf16>)
      %6 = affine.apply #map()[%arg5, %arg4]
      %7 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %6] [%c256, %c64] [%c64, %c1_0]) {id = 3 : i32} : (memref<2x256x64xbf16>)
      %8 = affine.apply #map()[%arg5, %arg4]
      %9 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %8] [%c256, %c64] [%c64, %c1_0]) {id = 4 : i32} : (memref<2x256x64xbf16>)
      %10 = affine.apply #map1()[%arg5]
      %11 = air.channel.put async  @QKIn_0[%c0] (%arg9[%c0, %10] [%c64, %c64] [%c64, %c1_0]) {id = 5 : i32} : (memref<2x256x64xbf16>)
      %12 = affine.apply #map2()[%arg5]
      %13 = air.channel.put async  @QKIn_1[%c0] (%arg9[%c0, %12] [%c64, %c64] [%c64, %c1_0]) {id = 6 : i32} : (memref<2x256x64xbf16>)
      %14 = affine.apply #map3()[%arg5]
      %15 = air.channel.put async  @QKIn_2[%c0] (%arg9[%c0, %14] [%c64, %c64] [%c64, %c1_0]) {id = 7 : i32} : (memref<2x256x64xbf16>)
      %16 = affine.apply #map4()[%arg5]
      %17 = air.channel.put async  @QKIn_3[%c0] (%arg9[%c0, %16] [%c64, %c64] [%c64, %c1_0]) {id = 8 : i32} : (memref<2x256x64xbf16>)
      %18 = affine.apply #map1()[%arg5]
      %19 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %18] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 9 : i32} : (memref<2x256x64xbf16>)
      %20 = affine.apply #map2()[%arg5]
      %21 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %20] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 10 : i32} : (memref<2x256x64xbf16>)
      %22 = affine.apply #map3()[%arg5]
      %23 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %22] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 11 : i32} : (memref<2x256x64xbf16>)
      %24 = affine.apply #map4()[%arg5]
      %25 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %24] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 12 : i32} : (memref<2x256x64xbf16>)
      %26 = air.channel.get async  @GpOut[%c0] (%arg11[%1] [%c16384] [%c1_0]) {id = 13 : i32} : (memref<2x256x64xbf16>)
      %27 = affine.apply #map5()[%arg5, %arg4]
      %28 = affine.apply #map5()[%arg5, %arg4]
      %29 = air.channel.put async  @QKIn_0[%c1_0] (%arg8[%c0, %28] [%c256, %c64] [%c64, %c1_0]) {id = 14 : i32} : (memref<2x256x64xbf16>)
      %30 = affine.apply #map5()[%arg5, %arg4]
      %31 = air.channel.put async  @QKIn_1[%c1_0] (%arg8[%c0, %30] [%c256, %c64] [%c64, %c1_0]) {id = 15 : i32} : (memref<2x256x64xbf16>)
      %32 = affine.apply #map5()[%arg5, %arg4]
      %33 = air.channel.put async  @QKIn_2[%c1_0] (%arg8[%c0, %32] [%c256, %c64] [%c64, %c1_0]) {id = 16 : i32} : (memref<2x256x64xbf16>)
      %34 = affine.apply #map5()[%arg5, %arg4]
      %35 = air.channel.put async  @QKIn_3[%c1_0] (%arg8[%c0, %34] [%c256, %c64] [%c64, %c1_0]) {id = 17 : i32} : (memref<2x256x64xbf16>)
      %36 = affine.apply #map6()[%arg5]
      %37 = air.channel.put async  @QKIn_0[%c1_0] (%arg9[%c0, %36] [%c64, %c64] [%c64, %c1_0]) {id = 18 : i32} : (memref<2x256x64xbf16>)
      %38 = affine.apply #map7()[%arg5]
      %39 = air.channel.put async  @QKIn_1[%c1_0] (%arg9[%c0, %38] [%c64, %c64] [%c64, %c1_0]) {id = 19 : i32} : (memref<2x256x64xbf16>)
      %40 = affine.apply #map8()[%arg5]
      %41 = air.channel.put async  @QKIn_2[%c1_0] (%arg9[%c0, %40] [%c64, %c64] [%c64, %c1_0]) {id = 20 : i32} : (memref<2x256x64xbf16>)
      %42 = affine.apply #map9()[%arg5]
      %43 = air.channel.put async  @QKIn_3[%c1_0] (%arg9[%c0, %42] [%c64, %c64] [%c64, %c1_0]) {id = 21 : i32} : (memref<2x256x64xbf16>)
      %44 = affine.apply #map6()[%arg5]
      %45 = air.channel.put async  @VIn_0[%c1_0] (%arg10[%c0, %c0, %44] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 22 : i32} : (memref<2x256x64xbf16>)
      %46 = affine.apply #map7()[%arg5]
      %47 = air.channel.put async  @VIn_1[%c1_0] (%arg10[%c0, %c0, %46] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 23 : i32} : (memref<2x256x64xbf16>)
      %48 = affine.apply #map8()[%arg5]
      %49 = air.channel.put async  @VIn_2[%c1_0] (%arg10[%c0, %c0, %48] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 24 : i32} : (memref<2x256x64xbf16>)
      %50 = affine.apply #map9()[%arg5]
      %51 = air.channel.put async  @VIn_3[%c1_0] (%arg10[%c0, %c0, %50] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 25 : i32} : (memref<2x256x64xbf16>)
      %52 = air.channel.get async  @GpOut[%c1_0] (%arg11[%27] [%c16384] [%c1_0]) {id = 26 : i32} : (memref<2x256x64xbf16>)
      %53 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) attributes {id = 2 : i32} {
        %c64_1 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c8 = arith.constant 8 : index
        %c1_2 = arith.constant 1 : index
        %c0_3 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 1 : i32}
        %async_token_4, %results_5 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 2 : i32}
        %async_token_6, %results_7 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 3 : i32}
        %async_token_8, %results_9 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 4 : i32}
        %async_token_10, %results_11 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 5 : i32}
        %async_token_12, %results_13 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 6 : i32}
        %async_token_14, %results_15 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 7 : i32}
        %async_token_16, %results_17 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 8 : i32}
        %async_token_18, %results_19 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        } {id = 9 : i32}
        %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 10 : i32}
        %async_token_22, %results_23 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 11 : i32}
        %async_token_24, %results_25 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 12 : i32}
        %async_token_26, %results_27 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 13 : i32}
        %async_token_28, %results_29 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        } {id = 14 : i32}
        %async_token_30, %results_31 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 15 : i32}
        %async_token_32, %results_33 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        } {id = 16 : i32}
        %54 = scf.for %arg16 = %c0_3 to %c4 step %c1_2 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %85 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
          %86 = arith.cmpi eq, %arg12, %c0_3 : index
          %87 = scf.if %86 -> (!air.async.token) {
            %88 = air.channel.put async [%85]  @QK2L1_0_0[%c0_3, %c0_3, %c0_3] (%results[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88 : !air.async.token
          } else {
            %88 = air.channel.put async [%85]  @QK2L1_0_1[%c0_3, %c0_3, %c0_3] (%results[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88 : !air.async.token
          }
          scf.yield %87 : !air.async.token
        }
        %55 = air.channel.get async [%54]  @QKIn_0[%arg12] (%results[] [] []) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
        %56 = arith.cmpi eq, %arg12, %c0_3 : index
        %57 = scf.if %56 -> (!air.async.token) {
          %85 = air.channel.put async [%55]  @QK2L1_0_0[%c0_3, %c0_3, %c0_3] (%results[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        } else {
          %85 = air.channel.put async [%55]  @QK2L1_0_1[%c0_3, %c0_3, %c0_3] (%results[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        }
        %58 = scf.for %arg16 = %c0_3 to %c4 step %c1_2 iter_args(%arg17 = %async_token_4) -> (!air.async.token) {
          %85 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_5[] [] []) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
          %86 = arith.cmpi eq, %arg12, %c0_3 : index
          %87 = scf.if %86 -> (!air.async.token) {
            %88 = air.channel.put async [%85]  @QK2L1_1_0[%c0_3, %c0_3, %c0_3] (%results_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88 : !air.async.token
          } else {
            %88 = air.channel.put async [%85]  @QK2L1_1_1[%c0_3, %c0_3, %c0_3] (%results_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88 : !air.async.token
          }
          scf.yield %87 : !air.async.token
        }
        %59 = air.channel.get async [%58]  @QKIn_1[%arg12] (%results_5[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
        %60 = arith.cmpi eq, %arg12, %c0_3 : index
        %61 = scf.if %60 -> (!air.async.token) {
          %85 = air.channel.put async [%59]  @QK2L1_1_0[%c0_3, %c0_3, %c0_3] (%results_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        } else {
          %85 = air.channel.put async [%59]  @QK2L1_1_1[%c0_3, %c0_3, %c0_3] (%results_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        }
        %62 = scf.for %arg16 = %c0_3 to %c4 step %c1_2 iter_args(%arg17 = %async_token_6) -> (!air.async.token) {
          %85 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_7[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
          %86 = arith.cmpi eq, %arg12, %c0_3 : index
          %87 = scf.if %86 -> (!air.async.token) {
            %88 = air.channel.put async [%85]  @QK2L1_2_0[%c0_3, %c0_3, %c0_3] (%results_7[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88 : !air.async.token
          } else {
            %88 = air.channel.put async [%85]  @QK2L1_2_1[%c0_3, %c0_3, %c0_3] (%results_7[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88 : !air.async.token
          }
          scf.yield %87 : !air.async.token
        }
        %63 = air.channel.get async [%62]  @QKIn_2[%arg12] (%results_7[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
        %64 = arith.cmpi eq, %arg12, %c0_3 : index
        %65 = scf.if %64 -> (!air.async.token) {
          %85 = air.channel.put async [%63]  @QK2L1_2_0[%c0_3, %c0_3, %c0_3] (%results_7[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        } else {
          %85 = air.channel.put async [%63]  @QK2L1_2_1[%c0_3, %c0_3, %c0_3] (%results_7[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        }
        %66 = scf.for %arg16 = %c0_3 to %c4 step %c1_2 iter_args(%arg17 = %async_token_8) -> (!air.async.token) {
          %85 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_9[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %86 = arith.cmpi eq, %arg12, %c0_3 : index
          %87 = scf.if %86 -> (!air.async.token) {
            %88 = air.channel.put async [%85]  @QK2L1_3_0[%c0_3, %c0_3, %c0_3] (%results_9[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88 : !air.async.token
          } else {
            %88 = air.channel.put async [%85]  @QK2L1_3_1[%c0_3, %c0_3, %c0_3] (%results_9[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88 : !air.async.token
          }
          scf.yield %87 : !air.async.token
        }
        %67 = air.channel.get async [%66]  @QKIn_3[%arg12] (%results_9[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
        %68 = arith.cmpi eq, %arg12, %c0_3 : index
        %69 = scf.if %68 -> (!air.async.token) {
          %85 = air.channel.put async [%67]  @QK2L1_3_0[%c0_3, %c0_3, %c0_3] (%results_9[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        } else {
          %85 = air.channel.put async [%67]  @QK2L1_3_1[%c0_3, %c0_3, %c0_3] (%results_9[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        }
        %70 = air.channel.get async [%async_token_10]  @VIn_0[%arg12] (%results_11[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
        %71 = arith.cmpi eq, %arg12, %c0_3 : index
        %72 = scf.if %71 -> (!air.async.token) {
          %85 = air.channel.put async [%70]  @V2L1_0_0[%c0_3, %c0_3, %c0_3] (%results_11[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        } else {
          %85 = air.channel.put async [%70]  @V2L1_0_1[%c0_3, %c0_3, %c0_3] (%results_11[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        }
        %73 = air.channel.get async [%async_token_12]  @VIn_1[%arg12] (%results_13[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
        %74 = arith.cmpi eq, %arg12, %c0_3 : index
        %75 = scf.if %74 -> (!air.async.token) {
          %85 = air.channel.put async [%73]  @V2L1_1_0[%c0_3, %c0_3, %c0_3] (%results_13[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        } else {
          %85 = air.channel.put async [%73]  @V2L1_1_1[%c0_3, %c0_3, %c0_3] (%results_13[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        }
        %76 = air.channel.get async [%async_token_14]  @VIn_2[%arg12] (%results_15[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
        %77 = arith.cmpi eq, %arg12, %c0_3 : index
        %78 = scf.if %77 -> (!air.async.token) {
          %85 = air.channel.put async [%76]  @V2L1_2_0[%c0_3, %c0_3, %c0_3] (%results_15[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        } else {
          %85 = air.channel.put async [%76]  @V2L1_2_1[%c0_3, %c0_3, %c0_3] (%results_15[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        }
        %79 = air.channel.get async [%async_token_16]  @VIn_3[%arg12] (%results_17[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
        %80 = arith.cmpi eq, %arg12, %c0_3 : index
        %81 = scf.if %80 -> (!air.async.token) {
          %85 = air.channel.put async [%79]  @V2L1_3_0[%c0_3, %c0_3, %c0_3] (%results_17[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        } else {
          %85 = air.channel.put async [%79]  @V2L1_3_1[%c0_3, %c0_3, %c0_3] (%results_17[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %85 : !air.async.token
        }
        %82 = scf.parallel (%arg16) = (%c0_3) to (%c4) step (%c1_2) init (%async_token_18) -> !air.async.token {
          %85 = affine.apply #map10()[%arg16]
          %86 = air.channel.get async [%async_token_18]  @Gp2L2[%arg16, %c0_3] (%results_19[%85, %c0_3] [%c64_1, %c64_1] [%c64_1, %c1_2]) {id = 51 : i32} : (memref<256x64xbf16, 1 : i32>)
          scf.reduce(%86 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %87 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %87 : !air.async.token
          }
        }
        %83 = air.channel.put async [%82]  @GpOut[%arg12] (%results_19[] [] []) {id = 52 : i32} : (memref<256x64xbf16, 1 : i32>)
        %84 = air.herd @herd_0 async [%async_token_20, %async_token_22, %async_token_24, %async_token_26, %async_token_28, %async_token_30, %async_token_32]  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%results_21, %arg21=%results_23, %arg22=%results_25, %arg23=%results_27, %arg24=%results_29, %arg25=%results_31, %arg26=%results_33, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {id = 1 : i32, link_with = "attn.o"} {
          %c512_50 = arith.constant 512 : index
          %c64_51 = arith.constant 64 : index
          %c8_52 = arith.constant 8 : index
          %c0_53 = arith.constant 0 : index
          %c1_54 = arith.constant 1 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_55 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          } {id = 17 : i32}
          %async_token_56 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 18 : i32}
          %async_token_57 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 19 : i32}
          %85 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 29 : i32}
            affine.yield %118 : !air.async.token
          }
          %86 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%85]  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%85]  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 32 : i32}
            affine.yield %118 : !air.async.token
          }
          %87 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%86]  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%86]  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 35 : i32}
            affine.yield %118 : !air.async.token
          }
          %88 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%87]  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%87]  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 38 : i32}
            affine.yield %118 : !air.async.token
          }
          %89 = arith.index_cast %arg16 : index to i32
          %90 = arith.cmpi eq, %89, %c0_i32 : i32
          scf.if %90 {
            %async_token_71 = air.execute [%88] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 20 : i32}
          }
          %91 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 44 : i32}
            affine.yield %118 : !air.async.token
          }
          %92 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%91]  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%91]  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 47 : i32}
            affine.yield %118 : !air.async.token
          }
          %93 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%92]  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%92]  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 50 : i32}
            affine.yield %118 : !air.async.token
          }
          %94 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%93]  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%93]  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 53 : i32}
            affine.yield %118 : !air.async.token
          }
          %95 = arith.index_cast %arg16 : index to i32
          %96 = arith.cmpi eq, %95, %c1_i32 : i32
          scf.if %96 {
            %async_token_71 = air.execute [%94] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 21 : i32}
          }
          %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 59 : i32}
            affine.yield %118 : !air.async.token
          }
          %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%97]  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%97]  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 62 : i32}
            affine.yield %118 : !air.async.token
          }
          %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%98]  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%98]  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 65 : i32}
            affine.yield %118 : !air.async.token
          }
          %100 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%99]  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%99]  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 68 : i32}
            affine.yield %118 : !air.async.token
          }
          %101 = arith.index_cast %arg16 : index to i32
          %102 = arith.cmpi eq, %101, %c2_i32 : i32
          scf.if %102 {
            %async_token_71 = air.execute [%100] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 22 : i32}
          }
          %103 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 74 : i32}
            affine.yield %118 : !air.async.token
          }
          %104 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%103]  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%103]  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 77 : i32}
            affine.yield %118 : !air.async.token
          }
          %105 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%104]  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%104]  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 80 : i32}
            affine.yield %118 : !air.async.token
          }
          %106 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%105]  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%105]  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 83 : i32}
            affine.yield %118 : !air.async.token
          }
          %107 = arith.index_cast %arg16 : index to i32
          %108 = arith.cmpi eq, %107, %c3_i32 : i32
          scf.if %108 {
            %async_token_71 = air.execute [%106] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 23 : i32}
          }
          %async_token_58 = air.execute {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          } {id = 24 : i32}
          %109 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async  @QK2L1_0_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async  @QK2L1_0_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 89 : i32}
            affine.yield %118 : !air.async.token
          }
          %110 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%109]  @QK2L1_1_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%109]  @QK2L1_1_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 92 : i32}
            affine.yield %118 : !air.async.token
          }
          %111 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%110]  @QK2L1_2_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%110]  @QK2L1_2_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 95 : i32}
            affine.yield %118 : !air.async.token
          }
          %112 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%111]  @QK2L1_3_0[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%111]  @QK2L1_3_1[%c0_53, %arg17, %arg16] (%arg21[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 98 : i32}
            affine.yield %118 : !air.async.token
          }
          %async_token_59 = air.execute [%async_token_58, %112] {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          } {id = 25 : i32}
          %113 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async  @V2L1_0_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async  @V2L1_0_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 101 : i32}
            affine.yield %118 : !air.async.token
          }
          %114 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%113]  @V2L1_1_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%113]  @V2L1_1_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 104 : i32}
            affine.yield %118 : !air.async.token
          }
          %115 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%114]  @V2L1_2_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%114]  @V2L1_2_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 107 : i32}
            affine.yield %118 : !air.async.token
          }
          %116 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.cmpi eq, %arg27, %c0_53 : index
            %119 = scf.if %118 -> (!air.async.token) {
              %120 = air.channel.get async [%115]  @V2L1_3_0[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            } else {
              %120 = air.channel.get async [%115]  @V2L1_3_1[%c0_53, %arg17, %arg16] (%arg22[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %120 : !air.async.token
            }
            affine.yield %119 : !air.async.token
          } else {
            %118 = air.wait_all async  {id = 110 : i32}
            affine.yield %118 : !air.async.token
          }
          %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          } {id = 26 : i32}
          %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          } {id = 27 : i32}
          %async_token_64 = air.execute [%async_token_59, %async_token_60, %async_token_62, %async_token_57] {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %results_61, %results_63) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 28 : i32}
          %async_token_65 = air.execute [%async_token_64, %async_token_55] {
            func.call @mul_r_gp(%results_63, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          } {id = 29 : i32}
          %async_token_66 = air.execute [%116, %async_token_65] {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          } {id = 30 : i32}
          %async_token_67 = air.execute [%async_token_65, %async_token_56] {
            func.call @accum_sp_r_s(%arg26, %results_63, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 31 : i32}
          %async_token_68 = air.execute [%async_token_67] {
            func.call @vector_copy_32elems(%c0_i32, %results_61, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          } {id = 32 : i32}
          %async_token_69 = air.execute [%async_token_68] {
            memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
          } {id = 33 : i32}
          %async_token_70 = air.execute [%async_token_67] {
            memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
          } {id = 34 : i32}
          %117 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %118 = arith.subi %arg17, %c1_54 : index
            %119 = air.channel.put async [%async_token_55, %async_token_66]  @cascade_gp[%arg16, %118] (%arg24[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
            %120 = air.channel.put async [%async_token_57]  @cascade_up[%arg16, %118] (%arg25[] [] []) {id = 78 : i32} : (memref<64x1xbf16, 2 : i32>)
            %121 = air.channel.put async [%async_token_56, %async_token_68]  @cascade_sp[%arg16, %118] (%arg26[] [] []) {id = 79 : i32} : (memref<64x1xbf16, 2 : i32>)
            %122 = air.wait_all async [%119, %120, %121]  {id = 117 : i32}
            affine.yield %122 : !air.async.token
          } else {
            %118 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_71, %results_72 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 35 : i32}
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 36 : i32}
              %async_token_75, %results_76 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 37 : i32}
              %120 = air.channel.get async [%async_token_71]  @cascade_gp[%arg16, %arg17] (%results_72[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
              %121 = air.channel.get async [%async_token_73]  @cascade_up[%arg16, %arg17] (%results_74[] [] []) {id = 81 : i32} : (memref<64x1xbf16, 2 : i32>)
              %122 = air.channel.get async [%async_token_75]  @cascade_sp[%arg16, %arg17] (%results_76[] [] []) {id = 82 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_77, %results_78 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 38 : i32}
              %async_token_79 = air.execute [%async_token_77, %async_token_57] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_78) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 39 : i32}
              %async_token_80 = air.execute [%async_token_79, %121] {
                func.call @maximum_up_u_bf16(%results_74, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 40 : i32}
              %async_token_81, %results_82 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 41 : i32}
              %async_token_83 = air.execute [%async_token_81, %async_token_80] {
                func.call @exp_up_minus_u(%results_74, %arg25, %results_82) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 42 : i32}
              %async_token_84, %results_85 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 43 : i32}
              %async_token_86 = air.execute [%async_token_84, %async_token_83] {
                func.call @exp_up_minus_u(%results_78, %arg25, %results_85) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 44 : i32}
              %async_token_87 = air.execute [%async_token_83, %120] {
                func.call @mul_r_gp(%results_82, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 45 : i32}
              %async_token_88 = air.execute [%async_token_86, %async_token_55, %async_token_66] {
                func.call @mul_r_gp(%results_85, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 46 : i32}
              %async_token_89 = air.execute [%async_token_88, %async_token_87] {
                func.call @add_gp_g(%arg24, %results_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 47 : i32}
              %async_token_90, %results_91 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 48 : i32}
              %async_token_92 = air.execute [%async_token_90] {
                func.call @zero_fill_sp_bf16(%results_91) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 49 : i32}
              %async_token_93 = air.execute [%async_token_92, %async_token_87, %122] {
                func.call @accum_sp_r_s(%results_76, %results_82, %results_91) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 50 : i32}
              %async_token_94 = air.execute [%async_token_88, %async_token_93, %async_token_56, %async_token_68] {
                func.call @accum_sp_r_s(%arg26, %results_85, %results_91) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 51 : i32}
              %async_token_95 = air.execute [%async_token_94] {
                func.call @vector_copy_32elems(%c0_i32, %results_91, %results_76) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 52 : i32}
              %123 = arith.subi %arg17, %c1_54 : index
              %124 = air.channel.put async [%async_token_89]  @cascade_gp[%arg16, %123] (%results_72[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
              %125 = air.channel.put async [%async_token_86]  @cascade_up[%arg16, %123] (%arg25[] [] []) {id = 84 : i32} : (memref<64x1xbf16, 2 : i32>)
              %126 = air.channel.put async [%async_token_95]  @cascade_sp[%arg16, %123] (%results_76[] [] []) {id = 85 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_96 = air.execute [%124] {
                memref.dealloc %results_72 : memref<64x64xbf16, 2 : i32>
              } {id = 53 : i32}
              %async_token_97 = air.execute [%async_token_83] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              } {id = 54 : i32}
              %async_token_98 = air.execute [%126] {
                memref.dealloc %results_76 : memref<64x1xbf16, 2 : i32>
              } {id = 55 : i32}
              %async_token_99 = air.execute [%async_token_86] {
                memref.dealloc %results_78 : memref<64x1xbf16, 2 : i32>
              } {id = 56 : i32}
              %async_token_100 = air.execute [%async_token_93] {
                memref.dealloc %results_82 : memref<64x1xbf16, 2 : i32>
              } {id = 57 : i32}
              %async_token_101 = air.execute [%async_token_94] {
                memref.dealloc %results_85 : memref<64x1xbf16, 2 : i32>
              } {id = 58 : i32}
              %async_token_102 = air.execute [%async_token_95] {
                memref.dealloc %results_91 : memref<64x1xbf16, 2 : i32>
              } {id = 59 : i32}
              %127 = air.wait_all async [%124, %125, %126]  {id = 114 : i32}
              affine.yield %127 : !air.async.token
            } else {
              %async_token_71, %results_72 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 60 : i32}
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 61 : i32}
              %async_token_75, %results_76 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 62 : i32}
              %120 = air.channel.get async [%async_token_71]  @cascade_gp[%arg16, %arg17] (%results_72[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
              %121 = air.channel.get async [%async_token_73]  @cascade_up[%arg16, %arg17] (%results_74[] [] []) {id = 87 : i32} : (memref<64x1xbf16, 2 : i32>)
              %122 = air.channel.get async [%async_token_75]  @cascade_sp[%arg16, %arg17] (%results_76[] [] []) {id = 88 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_77, %results_78 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 63 : i32}
              %async_token_79 = air.execute [%async_token_77, %async_token_57] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_78) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 64 : i32}
              %async_token_80 = air.execute [%async_token_79, %121] {
                func.call @maximum_up_u_bf16(%results_74, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 65 : i32}
              %async_token_81, %results_82 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 66 : i32}
              %async_token_83 = air.execute [%async_token_81, %async_token_80] {
                func.call @exp_up_minus_u(%results_74, %arg25, %results_82) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 67 : i32}
              %async_token_84, %results_85 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 68 : i32}
              %async_token_86 = air.execute [%async_token_84, %async_token_83] {
                func.call @exp_up_minus_u(%results_78, %arg25, %results_85) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 69 : i32}
              %async_token_87 = air.execute [%async_token_83, %120] {
                func.call @mul_r_gp(%results_82, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 70 : i32}
              %async_token_88 = air.execute [%async_token_86, %async_token_55, %async_token_66] {
                func.call @mul_r_gp(%results_85, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 71 : i32}
              %async_token_89 = air.execute [%async_token_88, %async_token_87] {
                func.call @add_gp_g(%arg24, %results_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 72 : i32}
              %async_token_90, %results_91 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 73 : i32}
              %async_token_92 = air.execute [%async_token_90] {
                func.call @zero_fill_sp_bf16(%results_91) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 74 : i32}
              %async_token_93 = air.execute [%async_token_92, %async_token_87, %122] {
                func.call @accum_sp_r_s(%results_76, %results_82, %results_91) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 75 : i32}
              %async_token_94 = air.execute [%async_token_88, %async_token_93, %async_token_56, %async_token_68] {
                func.call @accum_sp_r_s(%arg26, %results_85, %results_91) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 76 : i32}
              %async_token_95 = air.execute [%async_token_94] {
                func.call @vector_copy_32elems(%c0_i32, %results_91, %results_76) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 77 : i32}
              %async_token_96 = air.execute [%async_token_95, %async_token_89] {
                func.call @div_gp_sp(%results_76, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 78 : i32}
              %123 = air.channel.put async [%async_token_96]  @Gp2L2[%arg16, %c0_53] (%results_72[%c0_53, %c0_53, %c0_53, %c0_53] [%c8_52, %c8_52, %c8_52, %c8_52] [%c64_51, %c8_52, %c512_50, %c1_54]) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_97 = air.execute [%123] {
                memref.dealloc %results_72 : memref<64x64xbf16, 2 : i32>
              } {id = 79 : i32}
              %async_token_98 = air.execute [%async_token_83] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              } {id = 80 : i32}
              %async_token_99 = air.execute [%async_token_96] {
                memref.dealloc %results_76 : memref<64x1xbf16, 2 : i32>
              } {id = 81 : i32}
              %async_token_100 = air.execute [%async_token_86] {
                memref.dealloc %results_78 : memref<64x1xbf16, 2 : i32>
              } {id = 82 : i32}
              %async_token_101 = air.execute [%async_token_93] {
                memref.dealloc %results_82 : memref<64x1xbf16, 2 : i32>
              } {id = 83 : i32}
              %async_token_102 = air.execute [%async_token_94] {
                memref.dealloc %results_85 : memref<64x1xbf16, 2 : i32>
              } {id = 84 : i32}
              %async_token_103 = air.execute [%async_token_95] {
                memref.dealloc %results_91 : memref<64x1xbf16, 2 : i32>
              } {id = 85 : i32}
              affine.yield %123 : !air.async.token
            }
            %119 = air.wait_all async [%async_token_55, %async_token_56, %async_token_57, %109, %110, %111, %113, %114, %115, %async_token_66, %async_token_68]  {id = 118 : i32}
            affine.yield %119 : !air.async.token
          }
        }
        %async_token_34 = air.execute [%84] {
          memref.dealloc %results_21 : memref<64x64xbf16, 2 : i32>
        } {id = 86 : i32}
        %async_token_35 = air.execute [%84] {
          memref.dealloc %results_23 : memref<64x64xbf16, 2 : i32>
        } {id = 87 : i32}
        %async_token_36 = air.execute [%84] {
          memref.dealloc %results_25 : memref<64x64xbf16, 2 : i32>
        } {id = 88 : i32}
        %async_token_37 = air.execute [%84] {
          memref.dealloc %results_27 : memref<64x64xbf16, 2 : i32>
        } {id = 89 : i32}
        %async_token_38 = air.execute [%84] {
          memref.dealloc %results_29 : memref<64x64xbf16, 2 : i32>
        } {id = 90 : i32}
        %async_token_39 = air.execute [%84] {
          memref.dealloc %results_31 : memref<64x1xbf16, 2 : i32>
        } {id = 91 : i32}
        %async_token_40 = air.execute [%84] {
          memref.dealloc %results_33 : memref<64x1xbf16, 2 : i32>
        } {id = 92 : i32}
        %async_token_41 = air.execute [%57] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        } {id = 93 : i32}
        %async_token_42 = air.execute [%72] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        } {id = 94 : i32}
        %async_token_43 = air.execute [%61] {
          memref.dealloc %results_5 : memref<64x64xbf16, 1 : i32>
        } {id = 95 : i32}
        %async_token_44 = air.execute [%75] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        } {id = 96 : i32}
        %async_token_45 = air.execute [%65] {
          memref.dealloc %results_7 : memref<64x64xbf16, 1 : i32>
        } {id = 97 : i32}
        %async_token_46 = air.execute [%78] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        } {id = 98 : i32}
        %async_token_47 = air.execute [%69] {
          memref.dealloc %results_9 : memref<64x64xbf16, 1 : i32>
        } {id = 99 : i32}
        %async_token_48 = air.execute [%81] {
          memref.dealloc %results_17 : memref<64x64xbf16, 1 : i32>
        } {id = 100 : i32}
        %async_token_49 = air.execute [%83] {
          memref.dealloc %results_19 : memref<256x64xbf16, 1 : i32>
        } {id = 101 : i32}
      }
    }
    return
  }
}
