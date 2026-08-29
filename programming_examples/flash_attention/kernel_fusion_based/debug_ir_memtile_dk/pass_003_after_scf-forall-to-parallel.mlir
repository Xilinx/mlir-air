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
    air.launch (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x128xbf16>, memref<2x512x128xbf16>, memref<2x512x64xbf16>, memref<2x256x64xbf16> {
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
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map()[%1]
      %3 = affine.apply #map2()[%1]
      %4 = affine.apply #map()[%1]
      %5 = affine.apply #map3()[%2, %0]
      air.channel.put  @QKIn_0[%c0] (%arg8[%c0, %5] [%c256, %c128] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      air.channel.put  @QKIn_1[%c0] (%arg8[%c0, %5] [%c256, %c128] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      air.channel.put  @QKIn_2[%c0] (%arg8[%c0, %5] [%c256, %c128] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      air.channel.put  @QKIn_3[%c0] (%arg8[%c0, %5] [%c256, %c128] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %6 = affine.apply #map3()[%3, %c0]
      air.channel.put  @QKIn_0[%c0] (%arg9[%c0, %6] [%c128, %c128] [%c128, %c1_0]) : (memref<2x512x128xbf16>)
      %7 = affine.apply #map3()[%3, %c16384]
      air.channel.put  @QKIn_1[%c0] (%arg9[%c0, %7] [%c128, %c128] [%c128, %c1_0]) : (memref<2x512x128xbf16>)
      %8 = affine.apply #map3()[%3, %c32768]
      air.channel.put  @QKIn_2[%c0] (%arg9[%c0, %8] [%c128, %c128] [%c128, %c1_0]) : (memref<2x512x128xbf16>)
      %9 = affine.apply #map3()[%3, %c49152]
      air.channel.put  @QKIn_3[%c0] (%arg9[%c0, %9] [%c128, %c128] [%c128, %c1_0]) : (memref<2x512x128xbf16>)
      %10 = affine.apply #map3()[%4, %c0]
      air.channel.put  @VIn_0[%c0] (%arg10[%c0, %c0, %10] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x512x64xbf16>)
      %11 = affine.apply #map3()[%4, %c8192]
      air.channel.put  @VIn_1[%c0] (%arg10[%c0, %c0, %11] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x512x64xbf16>)
      %12 = affine.apply #map3()[%4, %c16384]
      air.channel.put  @VIn_2[%c0] (%arg10[%c0, %c0, %12] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x512x64xbf16>)
      %13 = affine.apply #map3()[%4, %c24576]
      air.channel.put  @VIn_3[%c0] (%arg10[%c0, %c0, %13] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x512x64xbf16>)
      air.channel.get  @GpOut[%c0] (%arg11[%5] [%c32768] [%c1_0]) : (memref<2x256x64xbf16>)
      %14 = affine.apply #map4()[%1]
      %15 = affine.apply #map()[%14]
      %16 = affine.apply #map2()[%14]
      %17 = affine.apply #map()[%14]
      %18 = affine.apply #map3()[%15, %0]
      air.channel.put  @QKIn_0[%c1_0] (%arg8[%c0, %18] [%c256, %c128] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      air.channel.put  @QKIn_1[%c1_0] (%arg8[%c0, %18] [%c256, %c128] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      air.channel.put  @QKIn_2[%c1_0] (%arg8[%c0, %18] [%c256, %c128] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      air.channel.put  @QKIn_3[%c1_0] (%arg8[%c0, %18] [%c256, %c128] [%c128, %c1_0]) : (memref<2x256x128xbf16>)
      %19 = affine.apply #map3()[%16, %c0]
      air.channel.put  @QKIn_0[%c1_0] (%arg9[%c0, %19] [%c128, %c128] [%c128, %c1_0]) : (memref<2x512x128xbf16>)
      %20 = affine.apply #map3()[%16, %c16384]
      air.channel.put  @QKIn_1[%c1_0] (%arg9[%c0, %20] [%c128, %c128] [%c128, %c1_0]) : (memref<2x512x128xbf16>)
      %21 = affine.apply #map3()[%16, %c32768]
      air.channel.put  @QKIn_2[%c1_0] (%arg9[%c0, %21] [%c128, %c128] [%c128, %c1_0]) : (memref<2x512x128xbf16>)
      %22 = affine.apply #map3()[%16, %c49152]
      air.channel.put  @QKIn_3[%c1_0] (%arg9[%c0, %22] [%c128, %c128] [%c128, %c1_0]) : (memref<2x512x128xbf16>)
      %23 = affine.apply #map3()[%17, %c0]
      air.channel.put  @VIn_0[%c1_0] (%arg10[%c0, %c0, %23] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x512x64xbf16>)
      %24 = affine.apply #map3()[%17, %c8192]
      air.channel.put  @VIn_1[%c1_0] (%arg10[%c0, %c0, %24] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x512x64xbf16>)
      %25 = affine.apply #map3()[%17, %c16384]
      air.channel.put  @VIn_2[%c1_0] (%arg10[%c0, %c0, %25] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x512x64xbf16>)
      %26 = affine.apply #map3()[%17, %c24576]
      air.channel.put  @VIn_3[%c1_0] (%arg10[%c0, %c0, %26] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x512x64xbf16>)
      air.channel.get  @GpOut[%c1_0] (%arg11[%18] [%c32768] [%c1_0]) : (memref<2x256x64xbf16>)
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) {
        %c64_1 = arith.constant 64 : index
        %c128_2 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c8 = arith.constant 8 : index
        %c1_3 = arith.constant 1 : index
        %c2_4 = arith.constant 2 : index
        %c0_5 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %alloc = memref.alloc() : memref<64x128xbf16, 1 : i32>
        %alloc_6 = memref.alloc() : memref<64x128xbf16, 1 : i32>
        %alloc_7 = memref.alloc() : memref<64x128xbf16, 1 : i32>
        %alloc_8 = memref.alloc() : memref<64x128xbf16, 1 : i32>
        %alloc_9 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_10 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_11 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_12 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_13 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_14 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_15 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_16 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_17 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_18 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_19 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_20 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_21 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        scf.for %arg16 = %c0_5 to %c4 step %c1_3 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_5, %c0_5] (%alloc[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_5, %c0_5] (%alloc[%c0_5, %c0_5, %c64_1, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_5, %c0_5] (%alloc[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_5, %c0_5] (%alloc[%c0_5, %c0_5, %c64_1, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c4 step %c1_3 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_6[] [] []) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_5, %c0_5] (%alloc_6[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_5, %c0_5] (%alloc_6[%c0_5, %c0_5, %c64_1, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_6[] [] []) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_5, %c0_5] (%alloc_6[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_5, %c0_5] (%alloc_6[%c0_5, %c0_5, %c64_1, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c4 step %c1_3 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_7[] [] []) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_5, %c0_5] (%alloc_7[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_5, %c0_5] (%alloc_7[%c0_5, %c0_5, %c64_1, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_7[] [] []) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_5, %c0_5] (%alloc_7[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_5, %c0_5] (%alloc_7[%c0_5, %c0_5, %c64_1, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c4 step %c1_3 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_8[] [] []) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg12, %c0_5, %c0_5] (%alloc_8[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg12, %c0_5, %c0_5] (%alloc_8[%c0_5, %c0_5, %c64_1, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_8[] [] []) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg12, %c0_5, %c0_5] (%alloc_8[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg12, %c0_5, %c0_5] (%alloc_8[%c0_5, %c0_5, %c64_1, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c128_2, %c1_3]) : (memref<64x128xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 {
          air.channel.get  @VIn_0[%arg12] (%alloc_9[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_0[%arg12, %c0_5, %c0_5] (%alloc_9[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_3]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 {
          air.channel.get  @VIn_1[%arg12] (%alloc_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_1[%arg12, %c0_5, %c0_5] (%alloc_10[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_3]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 {
          air.channel.get  @VIn_2[%arg12] (%alloc_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_2[%arg12, %c0_5, %c0_5] (%alloc_11[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_3]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 {
          air.channel.get  @VIn_3[%arg12] (%alloc_12[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_3[%arg12, %c0_5, %c0_5] (%alloc_12[%c0_5, %c0_5, %c0_5, %c0_5] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_3]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_22 = arith.constant 0 : index
        %c4_23 = arith.constant 4 : index
        %c1_24 = arith.constant 1 : index
        scf.parallel (%arg16) = (%c0_22) to (%c4_23) step (%c1_24) {
          %27 = affine.apply #map5()[%arg16]
          air.channel.get  @Gp2L2[%arg16, %c0_5] (%alloc_13[%27, %c0_5] [%c64_1, %c64_1] [%c64_1, %c1_3]) : (memref<256x64xbf16, 1 : i32>)
          scf.reduce 
        }
        air.channel.put  @GpOut[%arg12] (%alloc_13[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%alloc_14, %arg21=%alloc_15, %arg22=%alloc_16, %arg23=%alloc_17, %arg24=%alloc_18, %arg25=%alloc_19, %arg26=%alloc_20, %arg27=%alloc_21, %arg28=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn.o"} {
          %c512_25 = arith.constant 512 : index
          %c64_26 = arith.constant 64 : index
          %c8_27 = arith.constant 8 : index
          %c1_28 = arith.constant 1 : index
          %c0_29 = arith.constant 0 : index
          %c2_30 = arith.constant 2 : index
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
          %27 = arith.index_cast %arg16 : index to i32
          %28 = arith.cmpi eq, %27, %c0_i32 : i32
          scf.if %28 {
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
          %29 = arith.index_cast %arg16 : index to i32
          %30 = arith.cmpi eq, %29, %c1_i32 : i32
          scf.if %30 {
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
          %31 = arith.index_cast %arg16 : index to i32
          %32 = arith.cmpi eq, %31, %c2_i32 : i32
          scf.if %32 {
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
          %33 = arith.index_cast %arg16 : index to i32
          %34 = arith.cmpi eq, %33, %c3_i32 : i32
          scf.if %34 {
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
          %35 = arith.index_cast %arg16 : index to i32
          %36 = arith.cmpi eq, %35, %c0_i32 : i32
          scf.if %36 {
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
          %37 = arith.index_cast %arg16 : index to i32
          %38 = arith.cmpi eq, %37, %c1_i32 : i32
          scf.if %38 {
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
          %39 = arith.index_cast %arg16 : index to i32
          %40 = arith.cmpi eq, %39, %c2_i32 : i32
          scf.if %40 {
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
          %41 = arith.index_cast %arg16 : index to i32
          %42 = arith.cmpi eq, %41, %c3_i32 : i32
          scf.if %42 {
            func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          scf.for %arg29 = %c0_29 to %c2_30 step %c1_28 {
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
            %alloc_31 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_32 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg26, %alloc_31, %alloc_32) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_32, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg23, %arg25) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @accum_sp_r_s(%arg27, %alloc_32, %alloc_31) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32, %alloc_31, %arg27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_31 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_32 : memref<64x1xbf16, 2 : i32>
          }
          affine.if #set3()[%arg16, %arg17] {
            %43 = arith.subi %arg17, %c1_28 : index
            air.channel.put  @cascade_gp[%arg16, %43] (%arg25[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %43] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %43] (%arg27[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_31 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_32 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_33 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_31[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_32[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_33[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_34 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @vector_copy_32elems(%c0_i32, %arg26, %alloc_34) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_32, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_35 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_32, %arg26, %alloc_35) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_36 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_34, %arg26, %alloc_36) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_35, %alloc_31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_36, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_31) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_37 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_37) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_33, %alloc_35, %alloc_37) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_36, %alloc_37) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_37, %alloc_33) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %43 = arith.subi %arg17, %c1_28 : index
              air.channel.put  @cascade_gp[%arg16, %43] (%alloc_31[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %43] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %43] (%alloc_33[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_31 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_32 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_33 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_34 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_35 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_36 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_37 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_31 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_32 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_33 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_31[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_32[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_33[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_34 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @vector_copy_32elems(%c0_i32, %arg26, %alloc_34) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_32, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_35 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_32, %arg26, %alloc_35) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_36 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_34, %arg26, %alloc_36) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_35, %alloc_31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_36, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_31) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_37 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_37) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_33, %alloc_35, %alloc_37) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_36, %alloc_37) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_37, %alloc_33) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_33, %alloc_31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              air.channel.put  @Gp2L2[%arg16, %c0_29] (%alloc_31[%c0_29, %c0_29, %c0_29, %c0_29] [%c8_27, %c8_27, %c8_27, %c8_27] [%c64_26, %c8_27, %c512_25, %c1_28]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_31 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_32 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_33 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_34 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_35 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_36 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_37 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_14 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_15 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_16 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_17 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_18 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_19 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_20 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_21 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x128xbf16, 1 : i32>
        memref.dealloc %alloc_9 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_6 : memref<64x128xbf16, 1 : i32>
        memref.dealloc %alloc_10 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_7 : memref<64x128xbf16, 1 : i32>
        memref.dealloc %alloc_11 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_8 : memref<64x128xbf16, 1 : i32>
        memref.dealloc %alloc_12 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_13 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}
