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
    air.launch (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16> {
      %c2 = arith.constant 2 : index
      %c16384 = arith.constant 16384 : index
      %c12288 = arith.constant 12288 : index
      %c8192 = arith.constant 8192 : index
      %c4096 = arith.constant 4096 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map()[%1]
      %3 = affine.apply #map()[%1]
      %4 = affine.apply #map()[%1]
      %5 = affine.apply #map2()[%2, %0]
      %6 = affine.apply #map2()[%5, %c0]
      air.channel.put  @QKIn_0[%c0] (%arg8[%c0, %6] [%c256, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %7 = affine.apply #map2()[%5, %c0]
      air.channel.put  @QKIn_1[%c0] (%arg8[%c0, %7] [%c256, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %8 = affine.apply #map2()[%5, %c0]
      air.channel.put  @QKIn_2[%c0] (%arg8[%c0, %8] [%c256, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %9 = affine.apply #map2()[%5, %c0]
      air.channel.put  @QKIn_3[%c0] (%arg8[%c0, %9] [%c256, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %10 = affine.apply #map2()[%3, %c0]
      air.channel.put  @QKIn_0[%c0] (%arg9[%c0, %10] [%c64, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %11 = affine.apply #map2()[%3, %c4096]
      air.channel.put  @QKIn_1[%c0] (%arg9[%c0, %11] [%c64, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %12 = affine.apply #map2()[%3, %c8192]
      air.channel.put  @QKIn_2[%c0] (%arg9[%c0, %12] [%c64, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %13 = affine.apply #map2()[%3, %c12288]
      air.channel.put  @QKIn_3[%c0] (%arg9[%c0, %13] [%c64, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %14 = affine.apply #map2()[%4, %c0]
      air.channel.put  @VIn_0[%c0] (%arg10[%c0, %c0, %14] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %15 = affine.apply #map2()[%4, %c4096]
      air.channel.put  @VIn_1[%c0] (%arg10[%c0, %c0, %15] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %16 = affine.apply #map2()[%4, %c8192]
      air.channel.put  @VIn_2[%c0] (%arg10[%c0, %c0, %16] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %17 = affine.apply #map2()[%4, %c12288]
      air.channel.put  @VIn_3[%c0] (%arg10[%c0, %c0, %17] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      air.channel.get  @GpOut[%c0] (%arg11[%5] [%c16384] [%c1_0]) : (memref<2x256x64xbf16>)
      %18 = affine.apply #map3()[%1]
      %19 = affine.apply #map()[%18]
      %20 = affine.apply #map()[%18]
      %21 = affine.apply #map()[%18]
      %22 = affine.apply #map2()[%19, %0]
      %23 = affine.apply #map2()[%22, %c0]
      air.channel.put  @QKIn_0[%c1_0] (%arg8[%c0, %23] [%c256, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %24 = affine.apply #map2()[%22, %c0]
      air.channel.put  @QKIn_1[%c1_0] (%arg8[%c0, %24] [%c256, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %25 = affine.apply #map2()[%22, %c0]
      air.channel.put  @QKIn_2[%c1_0] (%arg8[%c0, %25] [%c256, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %26 = affine.apply #map2()[%22, %c0]
      air.channel.put  @QKIn_3[%c1_0] (%arg8[%c0, %26] [%c256, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %27 = affine.apply #map2()[%20, %c0]
      air.channel.put  @QKIn_0[%c1_0] (%arg9[%c0, %27] [%c64, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %28 = affine.apply #map2()[%20, %c4096]
      air.channel.put  @QKIn_1[%c1_0] (%arg9[%c0, %28] [%c64, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %29 = affine.apply #map2()[%20, %c8192]
      air.channel.put  @QKIn_2[%c1_0] (%arg9[%c0, %29] [%c64, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %30 = affine.apply #map2()[%20, %c12288]
      air.channel.put  @QKIn_3[%c1_0] (%arg9[%c0, %30] [%c64, %c64] [%c64, %c1_0]) : (memref<2x256x64xbf16>)
      %31 = affine.apply #map2()[%21, %c0]
      air.channel.put  @VIn_0[%c1_0] (%arg10[%c0, %c0, %31] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %32 = affine.apply #map2()[%21, %c4096]
      air.channel.put  @VIn_1[%c1_0] (%arg10[%c0, %c0, %32] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %33 = affine.apply #map2()[%21, %c8192]
      air.channel.put  @VIn_2[%c1_0] (%arg10[%c0, %c0, %33] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      %34 = affine.apply #map2()[%21, %c12288]
      air.channel.put  @VIn_3[%c1_0] (%arg10[%c0, %c0, %34] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) : (memref<2x256x64xbf16>)
      air.channel.get  @GpOut[%c1_0] (%arg11[%22] [%c16384] [%c1_0]) : (memref<2x256x64xbf16>)
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
        %alloc_17 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_18 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_3, %c0_3] (%alloc[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg12, %c0_3, %c0_3] (%alloc[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_4[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_3, %c0_3] (%alloc_4[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_4[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg12, %c0_3, %c0_3] (%alloc_4[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_5[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_3, %c0_3] (%alloc_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_5[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg12, %c0_3, %c0_3] (%alloc_5[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c4 step %c1_2 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_6[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg12, %c0_3, %c0_3] (%alloc_6[%c0_3, %c0_3, %c0_3, %c0_3] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_1, %c1_2]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_3 to %c1_2 step %c1_2 {
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
        scf.forall (%arg16) in (4) {
          %35 = affine.apply #map4()[%arg16]
          air.channel.get  @Gp2L2[%arg16, %c0_3] (%alloc_11[%35, %c0_3] [%c64_1, %c64_1] [%c64_1, %c1_2]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_11[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%alloc_12, %arg21=%alloc_13, %arg22=%alloc_14, %arg23=%alloc_15, %arg24=%alloc_16, %arg25=%alloc_17, %arg26=%alloc_18, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn.o"} {
          %c512_19 = arith.constant 512 : index
          %c64_20 = arith.constant 64 : index
          %c8_21 = arith.constant 8 : index
          %c0_22 = arith.constant 0 : index
          %c1_23 = arith.constant 1 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %35 = arith.index_cast %arg16 : index to i32
          %36 = arith.cmpi eq, %35, %c0_i32 : i32
          scf.if %36 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %37 = arith.index_cast %arg16 : index to i32
          %38 = arith.cmpi eq, %37, %c1_i32 : i32
          scf.if %38 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %39 = arith.index_cast %arg16 : index to i32
          %40 = arith.cmpi eq, %39, %c2_i32 : i32
          scf.if %40 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg16, %arg17] {
            air.channel.get  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            air.channel.get  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.get  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %41 = arith.index_cast %arg16 : index to i32
          %42 = arith.cmpi eq, %41, %c3_i32 : i32
          scf.if %42 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          scf.for %arg28 = %c0_22 to %c1_23 step %c1_23 {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            affine.if #set()[%arg16, %arg17] {
              air.channel.get  @QK2L1_0[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              air.channel.get  @QK2L1_1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg16, %arg17] {
              air.channel.get  @QK2L1_2[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg16, %arg17] {
              air.channel.get  @QK2L1_3[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            affine.if #set()[%arg16, %arg17] {
              air.channel.get  @V2L1_0[%arg27, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              air.channel.get  @V2L1_1[%arg27, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg16, %arg17] {
              air.channel.get  @V2L1_2[%arg27, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg16, %arg17] {
              air.channel.get  @V2L1_3[%arg27, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            %alloc_24 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_25 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %alloc_24, %alloc_25) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_25, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @accum_sp_r_s(%arg26, %alloc_25, %alloc_24) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32, %alloc_24, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_24 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_25 : memref<64x1xbf16, 2 : i32>
          }
          affine.if #set3()[%arg16, %arg17] {
            %43 = arith.subi %arg17, %c1_23 : index
            air.channel.put  @cascade_gp[%arg16, %43] (%arg24[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %43] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %43] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_24 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_25 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_26 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_24[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_25[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_26[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_27 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @vector_copy_32elems(%c0_i32, %arg25, %alloc_27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_25, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_28 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_25, %arg25, %alloc_28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_29 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_27, %arg25, %alloc_29) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_28, %alloc_24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_29, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_24) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_30 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_30) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_26, %alloc_28, %alloc_30) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_29, %alloc_30) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_30, %alloc_26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %43 = arith.subi %arg17, %c1_23 : index
              air.channel.put  @cascade_gp[%arg16, %43] (%alloc_24[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %43] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %43] (%alloc_26[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_24 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_25 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_26 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_27 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_28 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_29 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_30 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_24 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_25 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_26 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_24[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_25[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_26[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_27 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @vector_copy_32elems(%c0_i32, %arg25, %alloc_27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_25, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_28 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_25, %arg25, %alloc_28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_29 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_27, %arg25, %alloc_29) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_28, %alloc_24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_29, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_24) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_30 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_30) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_26, %alloc_28, %alloc_30) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_29, %alloc_30) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_30, %alloc_26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_26, %alloc_24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              air.channel.put  @Gp2L2[%arg16, %c0_22] (%alloc_24[%c0_22, %c0_22, %c0_22, %c0_22] [%c8_21, %c8_21, %c8_21, %c8_21] [%c64_20, %c8_21, %c512_19, %c1_23]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_24 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_25 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_26 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_27 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_28 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_29 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_30 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_12 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_13 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_14 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_15 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_16 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_17 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_18 : memref<64x1xbf16, 2 : i32>
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
