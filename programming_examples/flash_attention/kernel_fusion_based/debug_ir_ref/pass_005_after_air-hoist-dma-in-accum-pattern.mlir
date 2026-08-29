#map = affine_map<()[s0] -> (s0 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 2)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1)>
#map3 = affine_map<()[s0] -> (s0 + 1)>
#map4 = affine_map<()[s0] -> (s0 * 64)>
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
  func.func @attention_bf16(%arg0: memref<2x256x64xbf16>, %arg1: memref<2x256x64xbf16>, %arg2: memref<2x256x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16> attributes {id = 3 : i32} {
      %c2 = arith.constant 2 : index
      %c16384 = arith.constant 16384 : index
      %c12288 = arith.constant 12288 : index
      %c8192 = arith.constant 8192 : index
      %c4096 = arith.constant 4096 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = affine.apply #map1()[%arg5]
      %3 = affine.apply #map()[%2]
      %4 = affine.apply #map()[%2]
      %5 = affine.apply #map()[%2]
      %6 = affine.apply #map2()[%3, %1]
      %7 = affine.apply #map2()[%6, %c0]
      %8 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %7] [%c256, %c64] [%c64, %c1_0]) {id = 1 : i32} : (memref<2x256x64xbf16>)
      %9 = affine.apply #map2()[%6, %c0]
      %10 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %9] [%c256, %c64] [%c64, %c1_0]) {id = 2 : i32} : (memref<2x256x64xbf16>)
      %11 = affine.apply #map2()[%6, %c0]
      %12 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %11] [%c256, %c64] [%c64, %c1_0]) {id = 3 : i32} : (memref<2x256x64xbf16>)
      %13 = affine.apply #map2()[%6, %c0]
      %14 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %13] [%c256, %c64] [%c64, %c1_0]) {id = 4 : i32} : (memref<2x256x64xbf16>)
      %15 = affine.apply #map2()[%4, %c0]
      %16 = air.channel.put async  @QKIn_0[%c0] (%arg9[%c0, %15] [%c64, %c64] [%c64, %c1_0]) {id = 5 : i32} : (memref<2x256x64xbf16>)
      %17 = affine.apply #map2()[%4, %c4096]
      %18 = air.channel.put async  @QKIn_1[%c0] (%arg9[%c0, %17] [%c64, %c64] [%c64, %c1_0]) {id = 6 : i32} : (memref<2x256x64xbf16>)
      %19 = affine.apply #map2()[%4, %c8192]
      %20 = air.channel.put async  @QKIn_2[%c0] (%arg9[%c0, %19] [%c64, %c64] [%c64, %c1_0]) {id = 7 : i32} : (memref<2x256x64xbf16>)
      %21 = affine.apply #map2()[%4, %c12288]
      %22 = air.channel.put async  @QKIn_3[%c0] (%arg9[%c0, %21] [%c64, %c64] [%c64, %c1_0]) {id = 8 : i32} : (memref<2x256x64xbf16>)
      %23 = affine.apply #map2()[%5, %c0]
      %24 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %23] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 9 : i32} : (memref<2x256x64xbf16>)
      %25 = affine.apply #map2()[%5, %c4096]
      %26 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %25] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 10 : i32} : (memref<2x256x64xbf16>)
      %27 = affine.apply #map2()[%5, %c8192]
      %28 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %27] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 11 : i32} : (memref<2x256x64xbf16>)
      %29 = affine.apply #map2()[%5, %c12288]
      %30 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %29] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 12 : i32} : (memref<2x256x64xbf16>)
      %31 = air.channel.get async  @GpOut[%c0] (%arg11[%6] [%c16384] [%c1_0]) {id = 13 : i32} : (memref<2x256x64xbf16>)
      %32 = affine.apply #map3()[%2]
      %33 = affine.apply #map()[%32]
      %34 = affine.apply #map()[%32]
      %35 = affine.apply #map()[%32]
      %36 = affine.apply #map2()[%33, %1]
      %37 = affine.apply #map2()[%36, %c0]
      %38 = air.channel.put async  @QKIn_0[%c1_0] (%arg8[%c0, %37] [%c256, %c64] [%c64, %c1_0]) {id = 14 : i32} : (memref<2x256x64xbf16>)
      %39 = affine.apply #map2()[%36, %c0]
      %40 = air.channel.put async  @QKIn_1[%c1_0] (%arg8[%c0, %39] [%c256, %c64] [%c64, %c1_0]) {id = 15 : i32} : (memref<2x256x64xbf16>)
      %41 = affine.apply #map2()[%36, %c0]
      %42 = air.channel.put async  @QKIn_2[%c1_0] (%arg8[%c0, %41] [%c256, %c64] [%c64, %c1_0]) {id = 16 : i32} : (memref<2x256x64xbf16>)
      %43 = affine.apply #map2()[%36, %c0]
      %44 = air.channel.put async  @QKIn_3[%c1_0] (%arg8[%c0, %43] [%c256, %c64] [%c64, %c1_0]) {id = 17 : i32} : (memref<2x256x64xbf16>)
      %45 = affine.apply #map2()[%34, %c0]
      %46 = air.channel.put async  @QKIn_0[%c1_0] (%arg9[%c0, %45] [%c64, %c64] [%c64, %c1_0]) {id = 18 : i32} : (memref<2x256x64xbf16>)
      %47 = affine.apply #map2()[%34, %c4096]
      %48 = air.channel.put async  @QKIn_1[%c1_0] (%arg9[%c0, %47] [%c64, %c64] [%c64, %c1_0]) {id = 19 : i32} : (memref<2x256x64xbf16>)
      %49 = affine.apply #map2()[%34, %c8192]
      %50 = air.channel.put async  @QKIn_2[%c1_0] (%arg9[%c0, %49] [%c64, %c64] [%c64, %c1_0]) {id = 20 : i32} : (memref<2x256x64xbf16>)
      %51 = affine.apply #map2()[%34, %c12288]
      %52 = air.channel.put async  @QKIn_3[%c1_0] (%arg9[%c0, %51] [%c64, %c64] [%c64, %c1_0]) {id = 21 : i32} : (memref<2x256x64xbf16>)
      %53 = affine.apply #map2()[%35, %c0]
      %54 = air.channel.put async  @VIn_0[%c1_0] (%arg10[%c0, %c0, %53] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 22 : i32} : (memref<2x256x64xbf16>)
      %55 = affine.apply #map2()[%35, %c4096]
      %56 = air.channel.put async  @VIn_1[%c1_0] (%arg10[%c0, %c0, %55] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 23 : i32} : (memref<2x256x64xbf16>)
      %57 = affine.apply #map2()[%35, %c8192]
      %58 = air.channel.put async  @VIn_2[%c1_0] (%arg10[%c0, %c0, %57] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 24 : i32} : (memref<2x256x64xbf16>)
      %59 = affine.apply #map2()[%35, %c12288]
      %60 = air.channel.put async  @VIn_3[%c1_0] (%arg10[%c0, %c0, %59] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 25 : i32} : (memref<2x256x64xbf16>)
      %61 = air.channel.get async  @GpOut[%c1_0] (%arg11[%36] [%c16384] [%c1_0]) {id = 26 : i32} : (memref<2x256x64xbf16>)
      %62 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) attributes {id = 2 : i32} {
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
        %63 = air.wait_all async [%async_token]  {id = 1 : i32}
        %64 = scf.for %arg16 = %c0_3 to %c4 step %c1_2 iter_args(%arg17 = %63) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @QK2L1_0[%arg12, %c0_3, %c0_3] (%results[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 2 : i32}
          scf.yield %93 : !air.async.token
        }
        %65 = air.wait_all async [%64]  {id = 3 : i32}
        %66 = scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 iter_args(%arg17 = %65) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @QK2L1_0[%arg12, %c0_3, %c0_3] (%results[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 4 : i32}
          scf.yield %93 : !air.async.token
        }
        %67 = air.wait_all async [%async_token_4]  {id = 5 : i32}
        %68 = scf.for %arg16 = %c0_3 to %c4 step %c1_2 iter_args(%arg17 = %67) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_5[] [] []) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @QK2L1_1[%arg12, %c0_3, %c0_3] (%results_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 6 : i32}
          scf.yield %93 : !air.async.token
        }
        %69 = air.wait_all async [%68]  {id = 7 : i32}
        %70 = scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 iter_args(%arg17 = %69) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_5[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @QK2L1_1[%arg12, %c0_3, %c0_3] (%results_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 8 : i32}
          scf.yield %93 : !air.async.token
        }
        %71 = air.wait_all async [%async_token_6]  {id = 9 : i32}
        %72 = scf.for %arg16 = %c0_3 to %c4 step %c1_2 iter_args(%arg17 = %71) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_7[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @QK2L1_2[%arg12, %c0_3, %c0_3] (%results_7[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 10 : i32}
          scf.yield %93 : !air.async.token
        }
        %73 = air.wait_all async [%72]  {id = 11 : i32}
        %74 = scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 iter_args(%arg17 = %73) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_7[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @QK2L1_2[%arg12, %c0_3, %c0_3] (%results_7[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 12 : i32}
          scf.yield %93 : !air.async.token
        }
        %75 = air.wait_all async [%async_token_8]  {id = 13 : i32}
        %76 = scf.for %arg16 = %c0_3 to %c4 step %c1_2 iter_args(%arg17 = %75) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_9[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @QK2L1_3[%arg12, %c0_3, %c0_3] (%results_9[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 14 : i32}
          scf.yield %93 : !air.async.token
        }
        %77 = air.wait_all async [%76]  {id = 15 : i32}
        %78 = scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 iter_args(%arg17 = %77) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_9[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @QK2L1_3[%arg12, %c0_3, %c0_3] (%results_9[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 16 : i32}
          scf.yield %93 : !air.async.token
        }
        %79 = air.wait_all async [%async_token_10]  {id = 17 : i32}
        %80 = scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 iter_args(%arg17 = %79) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @VIn_0[%arg12] (%results_11[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @V2L1_0[%arg12, %c0_3, %c0_3] (%results_11[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 18 : i32}
          scf.yield %93 : !air.async.token
        }
        %81 = air.wait_all async [%async_token_12]  {id = 19 : i32}
        %82 = scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 iter_args(%arg17 = %81) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @VIn_1[%arg12] (%results_13[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @V2L1_1[%arg12, %c0_3, %c0_3] (%results_13[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 20 : i32}
          scf.yield %93 : !air.async.token
        }
        %83 = air.wait_all async [%async_token_14]  {id = 21 : i32}
        %84 = scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 iter_args(%arg17 = %83) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @VIn_2[%arg12] (%results_15[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @V2L1_2[%arg12, %c0_3, %c0_3] (%results_15[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 22 : i32}
          scf.yield %93 : !air.async.token
        }
        %85 = air.wait_all async [%async_token_16]  {id = 23 : i32}
        %86 = scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 iter_args(%arg17 = %85) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @VIn_3[%arg12] (%results_17[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = air.channel.put async [%arg17, %91]  @V2L1_3[%arg12, %c0_3, %c0_3] (%results_17[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 24 : i32}
          scf.yield %93 : !air.async.token
        }
        %87 = air.wait_all async [%async_token_18]  {id = 25 : i32}
        %88 = scf.parallel (%arg16) = (%c0_3) to (%c4) step (%c1_2) init (%87) -> !air.async.token {
          %91 = affine.apply #map4()[%arg16]
          %92 = air.channel.get async [%87]  @Gp2L2[%arg16, %c0_3] (%results_19[%91, %c0_3] [%c64_1, %c64_1] [%c64_1, %c1_2]) {id = 51 : i32} : (memref<256x64xbf16, 1 : i32>)
          %93 = air.wait_all async [%92]  {id = 26 : i32}
          scf.reduce(%93 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %94 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %94 : !air.async.token
          }
        }
        %89 = air.channel.put async [%88]  @GpOut[%arg12] (%results_19[] [] []) {id = 52 : i32} : (memref<256x64xbf16, 1 : i32>)
        %90 = air.herd @herd_0 async [%async_token_20, %async_token_22, %async_token_24, %async_token_26, %async_token_28, %async_token_30, %async_token_32]  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%results_21, %arg21=%results_23, %arg22=%results_25, %arg23=%results_27, %arg24=%results_29, %arg25=%results_31, %arg26=%results_33, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {id = 1 : i32, link_with = "attn.o"} {
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
          %91 = air.wait_all async  {id = 27 : i32}
          %92 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 28 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 29 : i32}
            affine.yield %143 : !air.async.token
          }
          %93 = air.wait_all async [%92, %92]  {id = 30 : i32}
          %94 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%93]  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 31 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 32 : i32}
            affine.yield %143 : !air.async.token
          }
          %95 = air.wait_all async [%94, %94]  {id = 33 : i32}
          %96 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%95]  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 34 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 35 : i32}
            affine.yield %143 : !air.async.token
          }
          %97 = air.wait_all async [%96, %96]  {id = 36 : i32}
          %98 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%97]  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 37 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 38 : i32}
            affine.yield %143 : !air.async.token
          }
          %99 = arith.index_cast %arg16 : index to i32
          %100 = arith.cmpi eq, %99, %c0_i32 : i32
          %101 = air.wait_all async [%98]  {id = 39 : i32}
          %102 = scf.if %100 -> (!air.async.token) {
            %async_token_58 = air.execute [%98] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 20 : i32}
            %143 = air.wait_all async [%async_token_58]  {id = 40 : i32}
            scf.yield %143 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 41 : i32}
            scf.yield %143 : !air.async.token
          }
          %103 = air.wait_all async  {id = 42 : i32}
          %104 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 43 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 44 : i32}
            affine.yield %143 : !air.async.token
          }
          %105 = air.wait_all async [%104, %104]  {id = 45 : i32}
          %106 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%105]  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 46 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 47 : i32}
            affine.yield %143 : !air.async.token
          }
          %107 = air.wait_all async [%106, %106]  {id = 48 : i32}
          %108 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%107]  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 49 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 50 : i32}
            affine.yield %143 : !air.async.token
          }
          %109 = air.wait_all async [%108, %108]  {id = 51 : i32}
          %110 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%109]  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 52 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 53 : i32}
            affine.yield %143 : !air.async.token
          }
          %111 = arith.index_cast %arg16 : index to i32
          %112 = arith.cmpi eq, %111, %c1_i32 : i32
          %113 = air.wait_all async [%110]  {id = 54 : i32}
          %114 = scf.if %112 -> (!air.async.token) {
            %async_token_58 = air.execute [%110] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 21 : i32}
            %143 = air.wait_all async [%async_token_58]  {id = 55 : i32}
            scf.yield %143 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 56 : i32}
            scf.yield %143 : !air.async.token
          }
          %115 = air.wait_all async  {id = 57 : i32}
          %116 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 58 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 59 : i32}
            affine.yield %143 : !air.async.token
          }
          %117 = air.wait_all async [%116, %116]  {id = 60 : i32}
          %118 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%117]  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 61 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 62 : i32}
            affine.yield %143 : !air.async.token
          }
          %119 = air.wait_all async [%118, %118]  {id = 63 : i32}
          %120 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%119]  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 64 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 65 : i32}
            affine.yield %143 : !air.async.token
          }
          %121 = air.wait_all async [%120, %120]  {id = 66 : i32}
          %122 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%121]  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 67 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 68 : i32}
            affine.yield %143 : !air.async.token
          }
          %123 = arith.index_cast %arg16 : index to i32
          %124 = arith.cmpi eq, %123, %c2_i32 : i32
          %125 = air.wait_all async [%122]  {id = 69 : i32}
          %126 = scf.if %124 -> (!air.async.token) {
            %async_token_58 = air.execute [%122] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 22 : i32}
            %143 = air.wait_all async [%async_token_58]  {id = 70 : i32}
            scf.yield %143 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 71 : i32}
            scf.yield %143 : !air.async.token
          }
          %127 = air.wait_all async  {id = 72 : i32}
          %128 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 73 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 74 : i32}
            affine.yield %143 : !air.async.token
          }
          %129 = air.wait_all async [%128, %128]  {id = 75 : i32}
          %130 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%129]  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 76 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 77 : i32}
            affine.yield %143 : !air.async.token
          }
          %131 = air.wait_all async [%130, %130]  {id = 78 : i32}
          %132 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%131]  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 79 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 80 : i32}
            affine.yield %143 : !air.async.token
          }
          %133 = air.wait_all async [%132, %132]  {id = 81 : i32}
          %134 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %143 = air.channel.get async [%133]  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
            %144 = air.wait_all async [%143]  {id = 82 : i32}
            affine.yield %144 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 83 : i32}
            affine.yield %143 : !air.async.token
          }
          %135 = arith.index_cast %arg16 : index to i32
          %136 = arith.cmpi eq, %135, %c3_i32 : i32
          %137 = air.wait_all async [%134]  {id = 84 : i32}
          %138 = scf.if %136 -> (!air.async.token) {
            %async_token_58 = air.execute [%134] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 23 : i32}
            %143 = air.wait_all async [%async_token_58]  {id = 85 : i32}
            scf.yield %143 : !air.async.token
          } else {
            %143 = air.wait_all async  {id = 86 : i32}
            scf.yield %143 : !air.async.token
          }
          %139 = air.wait_all async [%async_token_55, %async_token_56, %async_token_57]  {id = 111 : i32}
          %140 = scf.for %arg28 = %c0_53 to %c1_54 step %c1_54 iter_args(%arg29 = %139) -> (!air.async.token) {
            %async_token_58 = air.execute [%arg29] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            } {id = 24 : i32}
            %143 = air.wait_all async [%arg29]  {id = 87 : i32}
            %144 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %160 = air.channel.get async  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %161 = air.wait_all async [%160]  {id = 88 : i32}
              affine.yield %161 : !air.async.token
            } else {
              %160 = air.wait_all async  {id = 89 : i32}
              affine.yield %160 : !air.async.token
            }
            %145 = air.wait_all async [%arg29, %144, %144]  {id = 90 : i32}
            %146 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %160 = air.channel.get async [%145]  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
              %161 = air.wait_all async [%160]  {id = 91 : i32}
              affine.yield %161 : !air.async.token
            } else {
              %160 = air.wait_all async  {id = 92 : i32}
              affine.yield %160 : !air.async.token
            }
            %147 = air.wait_all async [%arg29, %146, %146]  {id = 93 : i32}
            %148 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %160 = air.channel.get async [%147]  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
              %161 = air.wait_all async [%160]  {id = 94 : i32}
              affine.yield %161 : !air.async.token
            } else {
              %160 = air.wait_all async  {id = 95 : i32}
              affine.yield %160 : !air.async.token
            }
            %149 = air.wait_all async [%arg29, %148, %148]  {id = 96 : i32}
            %150 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %160 = air.channel.get async [%149]  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
              %161 = air.wait_all async [%160]  {id = 97 : i32}
              affine.yield %161 : !air.async.token
            } else {
              %160 = air.wait_all async  {id = 98 : i32}
              affine.yield %160 : !air.async.token
            }
            %async_token_59 = air.execute [%arg29, %150, %async_token_58] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            } {id = 25 : i32}
            %151 = air.wait_all async [%arg29]  {id = 99 : i32}
            %152 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %160 = air.channel.get async  @V2L1_0[%arg27, %arg17, %arg16] (%arg22[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              %161 = air.wait_all async [%160]  {id = 100 : i32}
              affine.yield %161 : !air.async.token
            } else {
              %160 = air.wait_all async  {id = 101 : i32}
              affine.yield %160 : !air.async.token
            }
            %153 = air.wait_all async [%arg29, %152, %152]  {id = 102 : i32}
            %154 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %160 = air.channel.get async [%153]  @V2L1_1[%arg27, %arg17, %arg16] (%arg22[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
              %161 = air.wait_all async [%160]  {id = 103 : i32}
              affine.yield %161 : !air.async.token
            } else {
              %160 = air.wait_all async  {id = 104 : i32}
              affine.yield %160 : !air.async.token
            }
            %155 = air.wait_all async [%arg29, %154, %154]  {id = 105 : i32}
            %156 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %160 = air.channel.get async [%155]  @V2L1_2[%arg27, %arg17, %arg16] (%arg22[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
              %161 = air.wait_all async [%160]  {id = 106 : i32}
              affine.yield %161 : !air.async.token
            } else {
              %160 = air.wait_all async  {id = 107 : i32}
              affine.yield %160 : !air.async.token
            }
            %157 = air.wait_all async [%arg29, %156, %156]  {id = 108 : i32}
            %158 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %160 = air.channel.get async [%157]  @V2L1_3[%arg27, %arg17, %arg16] (%arg22[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
              %161 = air.wait_all async [%160]  {id = 109 : i32}
              affine.yield %161 : !air.async.token
            } else {
              %160 = air.wait_all async  {id = 110 : i32}
              affine.yield %160 : !air.async.token
            }
            %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 26 : i32}
            %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            } {id = 27 : i32}
            %async_token_64 = air.execute [%async_token_62, %async_token_60, %async_token_59, %arg29] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_61, %results_63) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 28 : i32}
            %async_token_65 = air.execute [%async_token_64, %arg29] {
              func.call @mul_r_gp(%results_63, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 29 : i32}
            %async_token_66 = air.execute [%arg29, %async_token_65, %158] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            } {id = 30 : i32}
            %async_token_67 = air.execute [%async_token_65, %arg29] {
              func.call @accum_sp_r_s(%arg26, %results_63, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 31 : i32}
            %async_token_68 = air.execute [%arg29, %async_token_67] {
              func.call @vector_copy_32elems(%c0_i32, %results_61, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            } {id = 32 : i32}
            %async_token_69 = air.execute [%async_token_68] {
              memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
            } {id = 33 : i32}
            %async_token_70 = air.execute [%async_token_67] {
              memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
            } {id = 34 : i32}
            %159 = air.wait_all async [%143, %145, %147, %149, %151, %153, %155, %157, %async_token_66, %async_token_68]  {id = 112 : i32}
            scf.yield %159 : !air.async.token
          }
          %141 = air.wait_all async [%140, %140]  {id = 116 : i32}
          %142 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %143 = arith.subi %arg17, %c1_54 : index
            %144 = air.channel.put async [%141]  @cascade_gp[%arg16, %143] (%arg24[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
            %145 = air.channel.put async [%141]  @cascade_up[%arg16, %143] (%arg25[] [] []) {id = 78 : i32} : (memref<64x1xbf16, 2 : i32>)
            %146 = air.channel.put async [%141]  @cascade_sp[%arg16, %143] (%arg26[] [] []) {id = 79 : i32} : (memref<64x1xbf16, 2 : i32>)
            %147 = air.wait_all async [%144, %145, %146]  {id = 117 : i32}
            affine.yield %147 : !air.async.token
          } else {
            %143 = air.wait_all async [%141, %141]  {id = 113 : i32}
            %144 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_58, %results_59 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 35 : i32}
              %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 36 : i32}
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 37 : i32}
              %146 = air.channel.get async [%async_token_58]  @cascade_gp[%arg16, %arg17] (%results_59[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
              %147 = air.channel.get async [%async_token_60]  @cascade_up[%arg16, %arg17] (%results_61[] [] []) {id = 81 : i32} : (memref<64x1xbf16, 2 : i32>)
              %148 = air.channel.get async [%async_token_62]  @cascade_sp[%arg16, %arg17] (%results_63[] [] []) {id = 82 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 38 : i32}
              %async_token_66 = air.execute [%async_token_64, %143] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_65) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 39 : i32}
              %async_token_67 = air.execute [%async_token_66, %147] {
                func.call @maximum_up_u_bf16(%results_61, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 40 : i32}
              %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 41 : i32}
              %async_token_70 = air.execute [%async_token_68, %async_token_67] {
                func.call @exp_up_minus_u(%results_61, %arg25, %results_69) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 42 : i32}
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 43 : i32}
              %async_token_73 = air.execute [%async_token_71, %async_token_70] {
                func.call @exp_up_minus_u(%results_65, %arg25, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 44 : i32}
              %async_token_74 = air.execute [%async_token_70, %146] {
                func.call @mul_r_gp(%results_69, %results_59) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 45 : i32}
              %async_token_75 = air.execute [%async_token_73, %143] {
                func.call @mul_r_gp(%results_72, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 46 : i32}
              %async_token_76 = air.execute [%async_token_75, %async_token_74] {
                func.call @add_gp_g(%arg24, %results_59) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 47 : i32}
              %async_token_77, %results_78 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 48 : i32}
              %async_token_79 = air.execute [%async_token_77] {
                func.call @zero_fill_sp_bf16(%results_78) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 49 : i32}
              %async_token_80 = air.execute [%async_token_79, %async_token_74, %148] {
                func.call @accum_sp_r_s(%results_63, %results_69, %results_78) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 50 : i32}
              %async_token_81 = air.execute [%async_token_80, %async_token_75, %143] {
                func.call @accum_sp_r_s(%arg26, %results_72, %results_78) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 51 : i32}
              %async_token_82 = air.execute [%async_token_81] {
                func.call @vector_copy_32elems(%c0_i32, %results_78, %results_63) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 52 : i32}
              %149 = arith.subi %arg17, %c1_54 : index
              %150 = air.channel.put async [%async_token_76]  @cascade_gp[%arg16, %149] (%results_59[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
              %151 = air.channel.put async [%async_token_73]  @cascade_up[%arg16, %149] (%arg25[] [] []) {id = 84 : i32} : (memref<64x1xbf16, 2 : i32>)
              %152 = air.channel.put async [%async_token_82]  @cascade_sp[%arg16, %149] (%results_63[] [] []) {id = 85 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_83 = air.execute [%150] {
                memref.dealloc %results_59 : memref<64x64xbf16, 2 : i32>
              } {id = 53 : i32}
              %async_token_84 = air.execute [%async_token_70] {
                memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
              } {id = 54 : i32}
              %async_token_85 = air.execute [%152] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              } {id = 55 : i32}
              %async_token_86 = air.execute [%async_token_73] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              } {id = 56 : i32}
              %async_token_87 = air.execute [%async_token_80] {
                memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
              } {id = 57 : i32}
              %async_token_88 = air.execute [%async_token_81] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              } {id = 58 : i32}
              %async_token_89 = air.execute [%async_token_82] {
                memref.dealloc %results_78 : memref<64x1xbf16, 2 : i32>
              } {id = 59 : i32}
              %153 = air.wait_all async [%150, %151, %152]  {id = 114 : i32}
              affine.yield %153 : !air.async.token
            } else {
              %async_token_58, %results_59 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              } {id = 60 : i32}
              %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 61 : i32}
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 62 : i32}
              %146 = air.channel.get async [%async_token_58]  @cascade_gp[%arg16, %arg17] (%results_59[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
              %147 = air.channel.get async [%async_token_60]  @cascade_up[%arg16, %arg17] (%results_61[] [] []) {id = 87 : i32} : (memref<64x1xbf16, 2 : i32>)
              %148 = air.channel.get async [%async_token_62]  @cascade_sp[%arg16, %arg17] (%results_63[] [] []) {id = 88 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 63 : i32}
              %async_token_66 = air.execute [%async_token_64, %143] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_65) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 64 : i32}
              %async_token_67 = air.execute [%async_token_66, %147] {
                func.call @maximum_up_u_bf16(%results_61, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 65 : i32}
              %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 66 : i32}
              %async_token_70 = air.execute [%async_token_68, %async_token_67] {
                func.call @exp_up_minus_u(%results_61, %arg25, %results_69) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 67 : i32}
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 68 : i32}
              %async_token_73 = air.execute [%async_token_71, %async_token_70] {
                func.call @exp_up_minus_u(%results_65, %arg25, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 69 : i32}
              %async_token_74 = air.execute [%async_token_70, %146] {
                func.call @mul_r_gp(%results_69, %results_59) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 70 : i32}
              %async_token_75 = air.execute [%async_token_73, %143] {
                func.call @mul_r_gp(%results_72, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 71 : i32}
              %async_token_76 = air.execute [%async_token_75, %async_token_74] {
                func.call @add_gp_g(%arg24, %results_59) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 72 : i32}
              %async_token_77, %results_78 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              } {id = 73 : i32}
              %async_token_79 = air.execute [%async_token_77] {
                func.call @zero_fill_sp_bf16(%results_78) : (memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 74 : i32}
              %async_token_80 = air.execute [%async_token_79, %async_token_74, %148] {
                func.call @accum_sp_r_s(%results_63, %results_69, %results_78) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 75 : i32}
              %async_token_81 = air.execute [%async_token_80, %async_token_75, %143] {
                func.call @accum_sp_r_s(%arg26, %results_72, %results_78) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 76 : i32}
              %async_token_82 = air.execute [%async_token_81] {
                func.call @vector_copy_32elems(%c0_i32, %results_78, %results_63) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              } {id = 77 : i32}
              %async_token_83 = air.execute [%async_token_82, %async_token_76] {
                func.call @div_gp_sp(%results_63, %results_59) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              } {id = 78 : i32}
              %149 = air.channel.put async [%async_token_83]  @Gp2L2[%arg16, %c0_53] (%results_59[%c0_53, %c0_53, %c0_53, %c0_53] [%c8_52, %c8_52, %c8_52, %c8_52] [%c64_51, %c8_52, %c512_50, %c1_54]) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_84 = air.execute [%149] {
                memref.dealloc %results_59 : memref<64x64xbf16, 2 : i32>
              } {id = 79 : i32}
              %async_token_85 = air.execute [%async_token_70] {
                memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
              } {id = 80 : i32}
              %async_token_86 = air.execute [%async_token_83] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              } {id = 81 : i32}
              %async_token_87 = air.execute [%async_token_73] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              } {id = 82 : i32}
              %async_token_88 = air.execute [%async_token_80] {
                memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
              } {id = 83 : i32}
              %async_token_89 = air.execute [%async_token_81] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              } {id = 84 : i32}
              %async_token_90 = air.execute [%async_token_82] {
                memref.dealloc %results_78 : memref<64x1xbf16, 2 : i32>
              } {id = 85 : i32}
              %150 = air.wait_all async [%149]  {id = 115 : i32}
              affine.yield %150 : !air.async.token
            }
            %145 = air.wait_all async [%143]  {id = 118 : i32}
            affine.yield %145 : !air.async.token
          }
        }
        %async_token_34 = air.execute [%90] {
          memref.dealloc %results_21 : memref<64x64xbf16, 2 : i32>
        } {id = 86 : i32}
        %async_token_35 = air.execute [%90] {
          memref.dealloc %results_23 : memref<64x64xbf16, 2 : i32>
        } {id = 87 : i32}
        %async_token_36 = air.execute [%90] {
          memref.dealloc %results_25 : memref<64x64xbf16, 2 : i32>
        } {id = 88 : i32}
        %async_token_37 = air.execute [%90] {
          memref.dealloc %results_27 : memref<64x64xbf16, 2 : i32>
        } {id = 89 : i32}
        %async_token_38 = air.execute [%90] {
          memref.dealloc %results_29 : memref<64x64xbf16, 2 : i32>
        } {id = 90 : i32}
        %async_token_39 = air.execute [%90] {
          memref.dealloc %results_31 : memref<64x1xbf16, 2 : i32>
        } {id = 91 : i32}
        %async_token_40 = air.execute [%90] {
          memref.dealloc %results_33 : memref<64x1xbf16, 2 : i32>
        } {id = 92 : i32}
        %async_token_41 = air.execute [%66] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        } {id = 93 : i32}
        %async_token_42 = air.execute [%80] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        } {id = 94 : i32}
        %async_token_43 = air.execute [%70] {
          memref.dealloc %results_5 : memref<64x64xbf16, 1 : i32>
        } {id = 95 : i32}
        %async_token_44 = air.execute [%82] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        } {id = 96 : i32}
        %async_token_45 = air.execute [%74] {
          memref.dealloc %results_7 : memref<64x64xbf16, 1 : i32>
        } {id = 97 : i32}
        %async_token_46 = air.execute [%84] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        } {id = 98 : i32}
        %async_token_47 = air.execute [%78] {
          memref.dealloc %results_9 : memref<64x64xbf16, 1 : i32>
        } {id = 99 : i32}
        %async_token_48 = air.execute [%86] {
          memref.dealloc %results_17 : memref<64x64xbf16, 1 : i32>
        } {id = 100 : i32}
        %async_token_49 = air.execute [%89] {
          memref.dealloc %results_19 : memref<256x64xbf16, 1 : i32>
        } {id = 101 : i32}
      }
    }
    return
  }
}
