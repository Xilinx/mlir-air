#map = affine_map<()[s0, s1] -> (s0 * 256 + s1 * 1024)>
#map1 = affine_map<()[s0] -> (s0 * 2)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<()[s0] -> (s0 * 64 + 768)>
#map4 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 64 + 1536)>
#map5 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 64)>
#map6 = affine_map<()[s0] -> (s0 * 1024)>
#map7 = affine_map<()[s0] -> (s0 * 1024 + 256)>
#map8 = affine_map<()[s0] -> (s0 * 1024 + 512)>
#map9 = affine_map<()[s0] -> (s0 * 1024 + 768)>
#map10 = affine_map<()[s0] -> (s0 + 1)>
#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set4 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
  func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @maximum_up_u_bf16(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @exp_up_minus_u(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @add_gp_g(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
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
  air.channel @cascade_gp [4, 3] {channel_type = "npu_cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "npu_cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "npu_cascade"}
  air.channel @Gp2L2 [4, 1]
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<3072x2304xbf16>, %arg1: memref<3072x768xbf16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    air.launch (%arg2, %arg3, %arg4) in (%arg5=%c4, %arg6=%c6, %arg7=%c3) args(%arg8=%arg0, %arg9=%arg1) : memref<3072x2304xbf16>, memref<3072x768xbf16> {
      %c0 = arith.constant 0 : index
      %0 = affine.apply #map()[%arg2, %arg4]
      %1 = affine.apply #map1()[%arg3]
      %2 = affine.apply #map2()[%1]
      %3 = affine.apply #map3()[%1]
      %4 = affine.apply #map4()[%1, %c0]
      %5 = affine.apply #map5()[%1, %c0]
      %c0_0 = arith.constant 0 : index
      air.channel.put  @QKIn_0[%c0_0] (%arg8[%0, %2] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      air.channel.put  @QKIn_1[%c0_0] (%arg8[%0, %2] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      air.channel.put  @QKIn_2[%c0_0] (%arg8[%0, %2] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      air.channel.put  @QKIn_3[%c0_0] (%arg8[%0, %2] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %6 = affine.apply #map6()[%arg4]
      air.channel.put  @QKIn_0[%c0_0] (%arg8[%6, %3] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %7 = affine.apply #map7()[%arg4]
      air.channel.put  @QKIn_1[%c0_0] (%arg8[%7, %3] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %8 = affine.apply #map8()[%arg4]
      air.channel.put  @QKIn_2[%c0_0] (%arg8[%8, %3] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %9 = affine.apply #map9()[%arg4]
      air.channel.put  @QKIn_3[%c0_0] (%arg8[%9, %3] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %10 = affine.apply #map6()[%arg4]
      air.channel.put  @VIn_0[%c0_0] (%arg8[%10, %4] [4, 64, 64] [147456, 2304, 1]) : (memref<3072x2304xbf16>)
      %11 = affine.apply #map7()[%arg4]
      air.channel.put  @VIn_1[%c0_0] (%arg8[%11, %4] [4, 64, 64] [147456, 2304, 1]) : (memref<3072x2304xbf16>)
      %12 = affine.apply #map8()[%arg4]
      air.channel.put  @VIn_2[%c0_0] (%arg8[%12, %4] [4, 64, 64] [147456, 2304, 1]) : (memref<3072x2304xbf16>)
      %13 = affine.apply #map9()[%arg4]
      air.channel.put  @VIn_3[%c0_0] (%arg8[%13, %4] [4, 64, 64] [147456, 2304, 1]) : (memref<3072x2304xbf16>)
      %14 = affine.apply #map10()[%1]
      %15 = affine.apply #map2()[%14]
      %16 = affine.apply #map3()[%14]
      %17 = affine.apply #map4()[%14, %c0]
      %18 = affine.apply #map5()[%14, %c0]
      %c1_1 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_1] (%arg8[%0, %15] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      air.channel.put  @QKIn_1[%c1_1] (%arg8[%0, %15] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      air.channel.put  @QKIn_2[%c1_1] (%arg8[%0, %15] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      air.channel.put  @QKIn_3[%c1_1] (%arg8[%0, %15] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %19 = affine.apply #map6()[%arg4]
      air.channel.put  @QKIn_0[%c1_1] (%arg8[%19, %16] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %20 = affine.apply #map7()[%arg4]
      air.channel.put  @QKIn_1[%c1_1] (%arg8[%20, %16] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %21 = affine.apply #map8()[%arg4]
      air.channel.put  @QKIn_2[%c1_1] (%arg8[%21, %16] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %22 = affine.apply #map9()[%arg4]
      air.channel.put  @QKIn_3[%c1_1] (%arg8[%22, %16] [4, 1, 64, 64] [147456, 64, 2304, 1]) : (memref<3072x2304xbf16>)
      %23 = affine.apply #map6()[%arg4]
      air.channel.put  @VIn_0[%c1_1] (%arg8[%23, %17] [4, 64, 64] [147456, 2304, 1]) : (memref<3072x2304xbf16>)
      %24 = affine.apply #map7()[%arg4]
      air.channel.put  @VIn_1[%c1_1] (%arg8[%24, %17] [4, 64, 64] [147456, 2304, 1]) : (memref<3072x2304xbf16>)
      %25 = affine.apply #map8()[%arg4]
      air.channel.put  @VIn_2[%c1_1] (%arg8[%25, %17] [4, 64, 64] [147456, 2304, 1]) : (memref<3072x2304xbf16>)
      %26 = affine.apply #map9()[%arg4]
      air.channel.put  @VIn_3[%c1_1] (%arg8[%26, %17] [4, 64, 64] [147456, 2304, 1]) : (memref<3072x2304xbf16>)
      %c2 = arith.constant 2 : index
      %c1_2 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg10, %arg11) in (%arg12=%c2, %arg13=%c1_2) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_5 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_6 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_7 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_8 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_9 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_10 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_11 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_12 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_13 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_14 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_15 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_16 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_17 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_18 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_19 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %c4_20 = arith.constant 4 : index
        %c4_21 = arith.constant 4 : index
        %c0_22 = arith.constant 0 : index
        %c4_23 = arith.constant 4 : index
        %c4_24 = arith.constant 4 : index
        %c4_25 = arith.constant 4 : index
        %c0_26 = arith.constant 0 : index
        %c1_27 = arith.constant 1 : index
        scf.for %arg14 = %c0_26 to %c4_24 step %c1_27 {
          air.channel.get  @QKIn_0[%arg10] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg10, %c0_22, %c0_22] (%alloc[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_28 = arith.constant 0 : index
        %c1_29 = arith.constant 1 : index
        scf.for %arg14 = %c0_28 to %c4_25 step %c1_29 {
          air.channel.get  @QKIn_0[%arg10] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_0[%arg10, %c0_22, %c0_22] (%alloc[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_30 = arith.constant 0 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg14 = %c0_30 to %c4_24 step %c1_31 {
          air.channel.get  @QKIn_1[%arg10] (%alloc_5[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg10, %c0_22, %c0_22] (%alloc_5[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_32 = arith.constant 0 : index
        %c1_33 = arith.constant 1 : index
        scf.for %arg14 = %c0_32 to %c4_25 step %c1_33 {
          air.channel.get  @QKIn_1[%arg10] (%alloc_5[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_1[%arg10, %c0_22, %c0_22] (%alloc_5[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_34 = arith.constant 0 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg14 = %c0_34 to %c4_24 step %c1_35 {
          air.channel.get  @QKIn_2[%arg10] (%alloc_6[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg10, %c0_22, %c0_22] (%alloc_6[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_36 = arith.constant 0 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg14 = %c0_36 to %c4_25 step %c1_37 {
          air.channel.get  @QKIn_2[%arg10] (%alloc_6[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_2[%arg10, %c0_22, %c0_22] (%alloc_6[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_38 = arith.constant 0 : index
        %c1_39 = arith.constant 1 : index
        scf.for %arg14 = %c0_38 to %c4_24 step %c1_39 {
          air.channel.get  @QKIn_3[%arg10] (%alloc_7[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg10, %c0_22, %c0_22] (%alloc_7[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_40 = arith.constant 0 : index
        %c1_41 = arith.constant 1 : index
        scf.for %arg14 = %c0_40 to %c4_25 step %c1_41 {
          air.channel.get  @QKIn_3[%arg10] (%alloc_7[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @QK2L1_3[%arg10, %c0_22, %c0_22] (%alloc_7[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_42 = arith.constant 0 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg14 = %c0_42 to %c4_23 step %c1_43 {
          air.channel.get  @VIn_0[%arg10] (%alloc_8[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_0[%arg10, %c0_22, %c0_22] (%alloc_8[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_44 = arith.constant 0 : index
        %c1_45 = arith.constant 1 : index
        scf.for %arg14 = %c0_44 to %c4_23 step %c1_45 {
          air.channel.get  @VIn_1[%arg10] (%alloc_9[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_1[%arg10, %c0_22, %c0_22] (%alloc_9[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_46 = arith.constant 0 : index
        %c1_47 = arith.constant 1 : index
        scf.for %arg14 = %c0_46 to %c4_23 step %c1_47 {
          air.channel.get  @VIn_2[%arg10] (%alloc_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_2[%arg10, %c0_22, %c0_22] (%alloc_10[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_48 = arith.constant 0 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg14 = %c0_48 to %c4_23 step %c1_49 {
          air.channel.get  @VIn_3[%arg10] (%alloc_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_3[%arg10, %c0_22, %c0_22] (%alloc_11[0, 0, 0, 0] [8, 8, 8, 8] [8, 512, 64, 1]) : (memref<64x64xbf16, 1 : i32>)
        }
        air.herd @herd_0  tile (%arg14, %arg15) in (%arg16=%c4_20, %arg17=%c4_21) args(%arg18=%alloc_13, %arg19=%alloc_14, %arg20=%alloc_15, %arg21=%alloc_16, %arg22=%alloc_17, %arg23=%alloc_18, %arg24=%alloc_19, %arg25=%arg10) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn_npu2.o"} {
          func.call @zero_fill_gp_bf16(%arg22) : (memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_sp_bf16(%arg24) : (memref<64x1xbf16, 2 : i32>) -> ()
          func.call @neg_inf_fill_up_bf16(%arg23) : (memref<64x1xbf16, 2 : i32>) -> ()
          affine.if #set()[%arg14, %arg15] {
            air.channel.get  @QK2L1_0[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg14, %arg15] {
            air.channel.get  @QK2L1_1[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg14, %arg15] {
            air.channel.get  @QK2L1_2[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg14, %arg15] {
            air.channel.get  @QK2L1_3[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %30 = arith.index_cast %arg14 : index to i32
          %c0_i32 = arith.constant 0 : i32
          %31 = arith.cmpi eq, %30, %c0_i32 : i32
          scf.if %31 {
            func.call @copy_tile(%arg19, %arg18) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg14, %arg15] {
            air.channel.get  @QK2L1_0[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg14, %arg15] {
            air.channel.get  @QK2L1_1[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg14, %arg15] {
            air.channel.get  @QK2L1_2[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg14, %arg15] {
            air.channel.get  @QK2L1_3[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %32 = arith.index_cast %arg14 : index to i32
          %c1_i32 = arith.constant 1 : i32
          %33 = arith.cmpi eq, %32, %c1_i32 : i32
          scf.if %33 {
            func.call @copy_tile(%arg19, %arg18) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg14, %arg15] {
            air.channel.get  @QK2L1_0[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg14, %arg15] {
            air.channel.get  @QK2L1_1[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg14, %arg15] {
            air.channel.get  @QK2L1_2[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg14, %arg15] {
            air.channel.get  @QK2L1_3[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %34 = arith.index_cast %arg14 : index to i32
          %c2_i32 = arith.constant 2 : i32
          %35 = arith.cmpi eq, %34, %c2_i32 : i32
          scf.if %35 {
            func.call @copy_tile(%arg19, %arg18) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          affine.if #set()[%arg14, %arg15] {
            air.channel.get  @QK2L1_0[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg14, %arg15] {
            air.channel.get  @QK2L1_1[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set2()[%arg14, %arg15] {
            air.channel.get  @QK2L1_2[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set3()[%arg14, %arg15] {
            air.channel.get  @QK2L1_3[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          %36 = arith.index_cast %arg14 : index to i32
          %c3_i32 = arith.constant 3 : i32
          %37 = arith.cmpi eq, %36, %c3_i32 : i32
          scf.if %37 {
            func.call @copy_tile(%arg19, %arg18) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %c4_50 = arith.constant 4 : index
          %c0_51 = arith.constant 0 : index
          %c1_52 = arith.constant 1 : index
          scf.for %arg26 = %c0_51 to %c4_50 step %c1_52 {
            %collapse_shape = memref.collapse_shape %arg21 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            affine.if #set()[%arg14, %arg15] {
              air.channel.get  @QK2L1_0[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg14, %arg15] {
              air.channel.get  @QK2L1_1[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg14, %arg15] {
              air.channel.get  @QK2L1_2[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg14, %arg15] {
              air.channel.get  @QK2L1_3[%arg25, %arg15, %arg14] (%arg19[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            func.call @matmul_a_b_bf16(%arg18, %arg19, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            affine.if #set()[%arg14, %arg15] {
              air.channel.get  @V2L1_0[%arg25, %arg15, %arg14] (%arg20[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg14, %arg15] {
              air.channel.get  @V2L1_1[%arg25, %arg15, %arg14] (%arg20[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg14, %arg15] {
              air.channel.get  @V2L1_2[%arg25, %arg15, %arg14] (%arg20[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg14, %arg15] {
              air.channel.get  @V2L1_3[%arg25, %arg15, %arg14] (%arg20[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            %alloc_54 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_55 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg23, %alloc_54, %alloc_55) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_55, %arg22) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg20, %arg22) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32_56 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg24, %alloc_55, %alloc_54) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32_56, %alloc_54, %arg24) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_54 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_55 : memref<64x1xbf16, 2 : i32>
          }
          %c1_53 = arith.constant 1 : index
          affine.if #set3()[%arg14, %arg15] {
            %38 = arith.subi %arg15, %c1_53 : index
            air.channel.put  @cascade_gp[%arg14, %38] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg14, %38] (%arg23[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg14, %38] (%arg24[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg14, %arg15] {
              %alloc_54 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_55 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_56 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg14, %arg15] (%alloc_54[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg14, %arg15] (%alloc_55[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg14, %arg15] (%alloc_56[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_57 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_58 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_58, %arg23, %alloc_57) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_55, %arg23) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_59 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_55, %arg23, %alloc_59) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_60 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_57, %arg23, %alloc_60) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_59, %alloc_54) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_60, %arg22) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg22, %alloc_54) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_61 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_61) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_56, %alloc_59, %alloc_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg24, %alloc_60, %alloc_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_58, %alloc_61, %alloc_56) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %38 = arith.subi %arg15, %c1_53 : index
              air.channel.put  @cascade_gp[%arg14, %38] (%alloc_54[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg14, %38] (%arg23[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg14, %38] (%alloc_56[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_54 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_55 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_56 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_57 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_59 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_60 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_61 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_54 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_55 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_56 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg14, %arg15] (%alloc_54[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg14, %arg15] (%alloc_55[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg14, %arg15] (%alloc_56[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_57 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_58 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_58, %arg23, %alloc_57) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_55, %arg23) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_59 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_55, %arg23, %alloc_59) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_60 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_57, %arg23, %alloc_60) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_59, %alloc_54) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_60, %arg22) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg22, %alloc_54) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_61 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_61) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_56, %alloc_59, %alloc_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg24, %alloc_60, %alloc_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_58, %alloc_61, %alloc_56) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_56, %alloc_54) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %c0_62 = arith.constant 0 : index
              air.channel.put  @Gp2L2[%arg14, %c0_62] (%alloc_54[0, 0, 0, 0] [8, 8, 8, 8] [64, 8, 512, 1]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_54 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_55 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_56 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_57 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_59 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_60 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_61 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        scf.forall (%arg14) in (4) {
          %30 = affine.apply #map2()[%arg14]
          %c0_50 = arith.constant 0 : index
          air.channel.get  @Gp2L2[%arg14, %c0_50] (%alloc_12[%30, 0] [64, 64] [64, 1]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg10] (%alloc_12[] [] []) : (memref<256x64xbf16, 1 : i32>)
        memref.dealloc %alloc_13 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_14 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_15 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_16 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_17 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_18 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_19 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_8 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_9 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_10 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_11 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_5 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_6 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_7 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_12 : memref<256x64xbf16, 1 : i32>
      }
      %27 = affine.apply #map5()[%1, %c0]
      %c0_3 = arith.constant 0 : index
      air.channel.get  @GpOut[%c0_3] (%arg9[%0, %27] [256, 64] [768, 1]) : (memref<3072x768xbf16>)
      %28 = affine.apply #map10()[%1]
      %29 = affine.apply #map5()[%28, %c0]
      %c1_4 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_4] (%arg9[%0, %29] [256, 64] [768, 1]) : (memref<3072x768xbf16>)
    }
    return
  }
}
