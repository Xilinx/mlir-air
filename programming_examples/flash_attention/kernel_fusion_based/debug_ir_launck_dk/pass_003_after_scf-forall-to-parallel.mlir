#map = affine_map<()[s0] -> (s0 * 32768)>
#map1 = affine_map<()[s0] -> (s0 * 2)>
#map2 = affine_map<()[s0] -> (s0 * 16384)>
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
  func.func @attention_bf16(%arg0: memref<2x256x128xbf16>, %arg1: memref<2x256x128xbf16>, %arg2: memref<2x256x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x128xbf16>, memref<2x256x128xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16> {
      %c2 = arith.constant 2 : index
      %c32768 = arith.constant 32768 : index
      %c12288 = arith.constant 12288 : index
      %c4096 = arith.constant 4096 : index
      %c24640 = arith.constant 24640 : index
      %c24576 = arith.constant 24576 : index
      %c16448 = arith.constant 16448 : index
      %c16384 = arith.constant 16384 : index
      %c8256 = arith.constant 8256 : index
      %c8192 = arith.constant 8192 : index
      %c1_0 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map()[%1]
      %3 = affine.apply #map()[%1]
      %4 = affine.apply #map2()[%1]
      %5 = affine.apply #map3()[%2, %0]
      %6 = affine.apply #map3()[%5, %c0]
      air.channel.put  @QKIn_0[%c0] (%arg8[%c0, %6] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %7 = affine.apply #map3()[%5, %c64]
      air.channel.put  @QKIn_0[%c0] (%arg8[%c0, %7] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %8 = affine.apply #map3()[%5, %c0]
      air.channel.put  @QKIn_1[%c0] (%arg8[%c0, %8] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %9 = affine.apply #map3()[%5, %c64]
      air.channel.put  @QKIn_1[%c0] (%arg8[%c0, %9] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %10 = affine.apply #map3()[%5, %c0]
      air.channel.put  @QKIn_2[%c0] (%arg8[%c0, %10] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %11 = affine.apply #map3()[%5, %c64]
      air.channel.put  @QKIn_2[%c0] (%arg8[%c0, %11] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %12 = affine.apply #map3()[%5, %c0]
      air.channel.put  @QKIn_3[%c0] (%arg8[%c0, %12] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %13 = affine.apply #map3()[%5, %c64]
      air.channel.put  @QKIn_3[%c0] (%arg8[%c0, %13] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %14 = affine.apply #map3()[%3, %c0]
      air.channel.put  @QKIn_0[%c0] (%arg9[%c0, %14] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %15 = affine.apply #map3()[%3, %c64]
      air.channel.put  @QKIn_0[%c0] (%arg9[%c0, %15] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %16 = affine.apply #map3()[%3, %c8192]
      air.channel.put  @QKIn_1[%c0] (%arg9[%c0, %16] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %17 = affine.apply #map3()[%3, %c8256]
      air.channel.put  @QKIn_1[%c0] (%arg9[%c0, %17] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %18 = affine.apply #map3()[%3, %c16384]
      air.channel.put  @QKIn_2[%c0] (%arg9[%c0, %18] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %19 = affine.apply #map3()[%3, %c16448]
      air.channel.put  @QKIn_2[%c0] (%arg9[%c0, %19] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %20 = affine.apply #map3()[%3, %c24576]
      air.channel.put  @QKIn_3[%c0] (%arg9[%c0, %20] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %21 = affine.apply #map3()[%3, %c24640]
      air.channel.put  @QKIn_3[%c0] (%arg9[%c0, %21] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %22 = affine.apply #map3()[%4, %c0]
      air.channel.put  @VIn_0[%c0] (%arg10[%c0, %c0, %22] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %23 = affine.apply #map3()[%4, %c4096]
      air.channel.put  @VIn_1[%c0] (%arg10[%c0, %c0, %23] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %24 = affine.apply #map3()[%4, %c8192]
      air.channel.put  @VIn_2[%c0] (%arg10[%c0, %c0, %24] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %25 = affine.apply #map3()[%4, %c12288]
      air.channel.put  @VIn_3[%c0] (%arg10[%c0, %c0, %25] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      air.channel.get  @GpOut[%c0] (%arg11[%5] [%c32768] [%c1_0]) : (memref<2x256x64xbf16>)
      %26 = affine.apply #map4()[%1]
      %27 = affine.apply #map()[%26]
      %28 = affine.apply #map()[%26]
      %29 = affine.apply #map2()[%26]
      %30 = affine.apply #map3()[%27, %0]
      %31 = affine.apply #map3()[%30, %c0]
      air.channel.put  @QKIn_0[%c1_0] (%arg8[%c0, %31] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %32 = affine.apply #map3()[%30, %c64]
      air.channel.put  @QKIn_0[%c1_0] (%arg8[%c0, %32] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %33 = affine.apply #map3()[%30, %c0]
      air.channel.put  @QKIn_1[%c1_0] (%arg8[%c0, %33] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %34 = affine.apply #map3()[%30, %c64]
      air.channel.put  @QKIn_1[%c1_0] (%arg8[%c0, %34] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %35 = affine.apply #map3()[%30, %c0]
      air.channel.put  @QKIn_2[%c1_0] (%arg8[%c0, %35] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %36 = affine.apply #map3()[%30, %c64]
      air.channel.put  @QKIn_2[%c1_0] (%arg8[%c0, %36] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %37 = affine.apply #map3()[%30, %c0]
      air.channel.put  @QKIn_3[%c1_0] (%arg8[%c0, %37] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %38 = affine.apply #map3()[%30, %c64]
      air.channel.put  @QKIn_3[%c1_0] (%arg8[%c0, %38] [%c256, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %39 = affine.apply #map3()[%28, %c0]
      air.channel.put  @QKIn_0[%c1_0] (%arg9[%c0, %39] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %40 = affine.apply #map3()[%28, %c64]
      air.channel.put  @QKIn_0[%c1_0] (%arg9[%c0, %40] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %41 = affine.apply #map3()[%28, %c8192]
      air.channel.put  @QKIn_1[%c1_0] (%arg9[%c0, %41] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %42 = affine.apply #map3()[%28, %c8256]
      air.channel.put  @QKIn_1[%c1_0] (%arg9[%c0, %42] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %43 = affine.apply #map3()[%28, %c16384]
      air.channel.put  @QKIn_2[%c1_0] (%arg9[%c0, %43] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %44 = affine.apply #map3()[%28, %c16448]
      air.channel.put  @QKIn_2[%c1_0] (%arg9[%c0, %44] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %45 = affine.apply #map3()[%28, %c24576]
      air.channel.put  @QKIn_3[%c1_0] (%arg9[%c0, %45] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %46 = affine.apply #map3()[%28, %c24640]
      air.channel.put  @QKIn_3[%c1_0] (%arg9[%c0, %46] [%c64, %c64] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %47 = affine.apply #map3()[%29, %c0]
      air.channel.put  @VIn_0[%c1_0] (%arg10[%c0, %c0, %47] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %48 = affine.apply #map3()[%29, %c4096]
      air.channel.put  @VIn_1[%c1_0] (%arg10[%c0, %c0, %48] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %49 = affine.apply #map3()[%29, %c8192]
      air.channel.put  @VIn_2[%c1_0] (%arg10[%c0, %c0, %49] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %50 = affine.apply #map3()[%29, %c12288]
      air.channel.put  @VIn_3[%c1_0] (%arg10[%c0, %c0, %50] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      air.channel.get  @GpOut[%c1_0] (%arg11[%30] [%c32768] [%c1_0]) : (memref<2x256x64xbf16>)
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) {
        %c64_1 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c8 = arith.constant 8 : index
        %c1_2 = arith.constant 1 : index
        %c0_3 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_4 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_5 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_6 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_7 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_8 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_9 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_10 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_11 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_12 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_13 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_14 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_15 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_16 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_17 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_18 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_19 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_3, %c0_3] (%alloc[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_3, %c0_3] (%alloc[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_3, %c0_3] (%alloc[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_3, %c0_3] (%alloc[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_4[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_3, %c0_3] (%alloc_4[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_4[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_3, %c0_3] (%alloc_4[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_4[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_3, %c0_3] (%alloc_4[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_1[%arg12] (%alloc_4[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_3, %c0_3] (%alloc_4[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_5[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_3, %c0_3] (%alloc_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_5[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_3, %c0_3] (%alloc_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_5[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_3, %c0_3] (%alloc_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_2[%arg12] (%alloc_5[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_3, %c0_3] (%alloc_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_6[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg12, %c0_3, %c0_3] (%alloc_6[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_6[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg12, %c0_3, %c0_3] (%alloc_6[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_6[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg12, %c0_3, %c0_3] (%alloc_6[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_3[%arg12] (%alloc_6[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg12, %c0_3, %c0_3] (%alloc_6[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @VIn_0[%arg12] (%alloc_7[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_0[%arg12, %c0_3, %c0_3] (%alloc_7[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @VIn_1[%arg12] (%alloc_8[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_1[%arg12, %c0_3, %c0_3] (%alloc_8[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @VIn_2[%arg12] (%alloc_9[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_2[%arg12, %c0_3, %c0_3] (%alloc_9[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @VIn_3[%arg12] (%alloc_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_3[%arg12, %c0_3, %c0_3] (%alloc_10[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_20 = arith.constant 0 : index
        %c4_21 = arith.constant 4 : index
        %c1_22 = arith.constant 1 : index
        scf.parallel (%arg16) = (%c0_20) to (%c4_21) step (%c1_22) {
          %51 = affine.apply #map5()[%arg16]
          air.channel.get  @Gp2L2[%arg16, %c0_3] (%alloc_11[%51, %c0_3] [%c64_1, %c64_1] [%c64_1, %c1_2]) : (memref<256x64xbf16, 1 : i32>)
          scf.reduce 
        }
        air.channel.put  @GpOut[%arg12] (%alloc_11[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%alloc_12, %arg21=%alloc_13, %arg22=%alloc_14, %arg23=%alloc_15, %arg24=%alloc_16, %arg25=%alloc_17, %arg26=%alloc_18, %arg27=%alloc_19, %arg28=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn.o"} {
          %c512_23 = arith.constant 512 : index
          %c64_24 = arith.constant 64 : index
          %c8_25 = arith.constant 8 : index
          %c0_26 = arith.constant 0 : index
          %c1_27 = arith.constant 1 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          func.call @zero_fill_gp_bf16(%arg25) : (memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_sp_bf16(%arg27) : (memref<64x1xbf16, 2 : i32>) -> ()
          func.call @neg_inf_fill_up_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %51 = arith.index_cast %arg16 : index to i32
          %52 = arith.cmpi eq, %51, %c0_i32 : i32
          scf.if %52 {
            func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %53 = arith.index_cast %arg16 : index to i32
          %54 = arith.cmpi eq, %53, %c1_i32 : i32
          scf.if %54 {
            func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %55 = arith.index_cast %arg16 : index to i32
          %56 = arith.cmpi eq, %55, %c2_i32 : i32
          scf.if %56 {
            func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %57 = arith.index_cast %arg16 : index to i32
          %58 = arith.cmpi eq, %57, %c3_i32 : i32
          scf.if %58 {
            func.call @copy_tile(%arg22, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %59 = arith.index_cast %arg16 : index to i32
          %60 = arith.cmpi eq, %59, %c0_i32 : i32
          scf.if %60 {
            func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %61 = arith.index_cast %arg16 : index to i32
          %62 = arith.cmpi eq, %61, %c1_i32 : i32
          scf.if %62 {
            func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %63 = arith.index_cast %arg16 : index to i32
          %64 = arith.cmpi eq, %63, %c2_i32 : i32
          scf.if %64 {
            func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %65 = arith.index_cast %arg16 : index to i32
          %66 = arith.cmpi eq, %65, %c3_i32 : i32
          scf.if %66 {
            func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          scf.for %arg29 = %c0_26 to %c1_27 step %c1_27 {
            %collapse_shape = memref.collapse_shape %arg24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            affine.if #set()[%arg16, %arg17] {
              air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg16, %arg17] {
              air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg16, %arg17] {
              air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            func.call @matmul_a_b_bf16(%arg20, %arg22, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            affine.if #set()[%arg16, %arg17] {
              air.channel.get  @QK2L1_0[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              air.channel.get  @QK2L1_1[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg16, %arg17] {
              air.channel.get  @QK2L1_2[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg16, %arg17] {
              air.channel.get  @QK2L1_3[%arg28, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            func.call @matmul_a_b_bf16(%arg21, %arg22, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            affine.if #set()[%arg16, %arg17] {
              air.channel.get  @V2L1_0[%arg28, %arg17, %arg16] (%arg23[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              air.channel.get  @V2L1_1[%arg28, %arg17, %arg16] (%arg23[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg16, %arg17] {
              air.channel.get  @V2L1_2[%arg28, %arg17, %arg16] (%arg23[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg16, %arg17] {
              air.channel.get  @V2L1_3[%arg28, %arg17, %arg16] (%arg23[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            %alloc_28 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_29 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg26, %alloc_28, %alloc_29) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_29, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg23, %arg25) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @accum_sp_r_s(%arg27, %alloc_29, %alloc_28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32, %alloc_28, %arg27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_28 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_29 : memref<64x1xbf16, 2 : i32>
          }
          affine.if #set3()[%arg16, %arg17] {
            %67 = arith.subi %arg17, %c1_27 : index
            air.channel.put  @cascade_gp[%arg16, %67] (%arg25[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %67] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %67] (%arg27[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_28 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_29 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_30 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_28[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_29[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_30[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_31 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @vector_copy_32elems(%c0_i32, %arg26, %alloc_31) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_29, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_32 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_29, %arg26, %alloc_32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_33 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_31, %arg26, %alloc_33) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_32, %alloc_28) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_33, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_28) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_34 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_34) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_30, %alloc_32, %alloc_34) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_33, %alloc_34) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_34, %alloc_30) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %67 = arith.subi %arg17, %c1_27 : index
              air.channel.put  @cascade_gp[%arg16, %67] (%alloc_28[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %67] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %67] (%alloc_30[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_28 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_29 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_30 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_31 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_32 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_33 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_34 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_28 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_29 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_30 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_28[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_29[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_30[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_31 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @vector_copy_32elems(%c0_i32, %arg26, %alloc_31) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_29, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_32 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_29, %arg26, %alloc_32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_33 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_31, %arg26, %alloc_33) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_32, %alloc_28) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_33, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_28) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_34 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_34) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_30, %alloc_32, %alloc_34) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_33, %alloc_34) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_34, %alloc_30) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_30, %alloc_28) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              air.channel.put  @Gp2L2[%arg16, %c0_26] (%alloc_28[%c0_26, %c0_26, %c0_26, %c0_26] [%c8_25, %c8_25, %c8_25, %c8_25] [%c64_24, %c8_25, %c512_23, %c1_27]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_28 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_29 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_30 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_31 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_32 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_33 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_34 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_12 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_13 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_14 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_15 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_16 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_17 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_18 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_19 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_7 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_4 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_8 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_5 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_9 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_6 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_10 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_11 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}
