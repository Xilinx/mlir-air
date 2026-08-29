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
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    air.launch (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1_1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x128xbf16>, memref<2x256x128xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16> {
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map()[%1]
      %3 = affine.apply #map()[%1]
      %4 = affine.apply #map2()[%1]
      %c0 = arith.constant 0 : index
      %5 = affine.apply #map3()[%2, %0]
      %c0_2 = arith.constant 0 : index
      %6 = affine.apply #map3()[%5, %c0_2]
      %c0_3 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c1_4 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c0] (%arg8[%c0_3, %6] [%c256, %c64] [%c128, %c1_4]) : (memref<2x256x128xbf16>)
      %c64_5 = arith.constant 64 : index
      %7 = affine.apply #map3()[%5, %c64_5]
      %c0_6 = arith.constant 0 : index
      %c256_7 = arith.constant 256 : index
      %c64_8 = arith.constant 64 : index
      %c128_9 = arith.constant 128 : index
      %c1_10 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c0] (%arg8[%c0_6, %7] [%c256_7, %c64_8] [%c128_9, %c1_10]) : (memref<2x256x128xbf16>)
      %c0_11 = arith.constant 0 : index
      %8 = affine.apply #map3()[%5, %c0_11]
      %c0_12 = arith.constant 0 : index
      %c256_13 = arith.constant 256 : index
      %c64_14 = arith.constant 64 : index
      %c128_15 = arith.constant 128 : index
      %c1_16 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c0] (%arg8[%c0_12, %8] [%c256_13, %c64_14] [%c128_15, %c1_16]) : (memref<2x256x128xbf16>)
      %c64_17 = arith.constant 64 : index
      %9 = affine.apply #map3()[%5, %c64_17]
      %c0_18 = arith.constant 0 : index
      %c256_19 = arith.constant 256 : index
      %c64_20 = arith.constant 64 : index
      %c128_21 = arith.constant 128 : index
      %c1_22 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c0] (%arg8[%c0_18, %9] [%c256_19, %c64_20] [%c128_21, %c1_22]) : (memref<2x256x128xbf16>)
      %c0_23 = arith.constant 0 : index
      %10 = affine.apply #map3()[%5, %c0_23]
      %c0_24 = arith.constant 0 : index
      %c256_25 = arith.constant 256 : index
      %c64_26 = arith.constant 64 : index
      %c128_27 = arith.constant 128 : index
      %c1_28 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c0] (%arg8[%c0_24, %10] [%c256_25, %c64_26] [%c128_27, %c1_28]) : (memref<2x256x128xbf16>)
      %c64_29 = arith.constant 64 : index
      %11 = affine.apply #map3()[%5, %c64_29]
      %c0_30 = arith.constant 0 : index
      %c256_31 = arith.constant 256 : index
      %c64_32 = arith.constant 64 : index
      %c128_33 = arith.constant 128 : index
      %c1_34 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c0] (%arg8[%c0_30, %11] [%c256_31, %c64_32] [%c128_33, %c1_34]) : (memref<2x256x128xbf16>)
      %c0_35 = arith.constant 0 : index
      %12 = affine.apply #map3()[%5, %c0_35]
      %c0_36 = arith.constant 0 : index
      %c256_37 = arith.constant 256 : index
      %c64_38 = arith.constant 64 : index
      %c128_39 = arith.constant 128 : index
      %c1_40 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c0] (%arg8[%c0_36, %12] [%c256_37, %c64_38] [%c128_39, %c1_40]) : (memref<2x256x128xbf16>)
      %c64_41 = arith.constant 64 : index
      %13 = affine.apply #map3()[%5, %c64_41]
      %c0_42 = arith.constant 0 : index
      %c256_43 = arith.constant 256 : index
      %c64_44 = arith.constant 64 : index
      %c128_45 = arith.constant 128 : index
      %c1_46 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c0] (%arg8[%c0_42, %13] [%c256_43, %c64_44] [%c128_45, %c1_46]) : (memref<2x256x128xbf16>)
      %c0_47 = arith.constant 0 : index
      %14 = affine.apply #map3()[%3, %c0_47]
      %c0_48 = arith.constant 0 : index
      %c64_49 = arith.constant 64 : index
      %c64_50 = arith.constant 64 : index
      %c128_51 = arith.constant 128 : index
      %c1_52 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c0] (%arg9[%c0_48, %14] [%c64_49, %c64_50] [%c128_51, %c1_52]) : (memref<2x256x128xbf16>)
      %c64_53 = arith.constant 64 : index
      %15 = affine.apply #map3()[%3, %c64_53]
      %c0_54 = arith.constant 0 : index
      %c64_55 = arith.constant 64 : index
      %c64_56 = arith.constant 64 : index
      %c128_57 = arith.constant 128 : index
      %c1_58 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c0] (%arg9[%c0_54, %15] [%c64_55, %c64_56] [%c128_57, %c1_58]) : (memref<2x256x128xbf16>)
      %c8192 = arith.constant 8192 : index
      %16 = affine.apply #map3()[%3, %c8192]
      %c0_59 = arith.constant 0 : index
      %c64_60 = arith.constant 64 : index
      %c64_61 = arith.constant 64 : index
      %c128_62 = arith.constant 128 : index
      %c1_63 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c0] (%arg9[%c0_59, %16] [%c64_60, %c64_61] [%c128_62, %c1_63]) : (memref<2x256x128xbf16>)
      %c8256 = arith.constant 8256 : index
      %17 = affine.apply #map3()[%3, %c8256]
      %c0_64 = arith.constant 0 : index
      %c64_65 = arith.constant 64 : index
      %c64_66 = arith.constant 64 : index
      %c128_67 = arith.constant 128 : index
      %c1_68 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c0] (%arg9[%c0_64, %17] [%c64_65, %c64_66] [%c128_67, %c1_68]) : (memref<2x256x128xbf16>)
      %c16384 = arith.constant 16384 : index
      %18 = affine.apply #map3()[%3, %c16384]
      %c0_69 = arith.constant 0 : index
      %c64_70 = arith.constant 64 : index
      %c64_71 = arith.constant 64 : index
      %c128_72 = arith.constant 128 : index
      %c1_73 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c0] (%arg9[%c0_69, %18] [%c64_70, %c64_71] [%c128_72, %c1_73]) : (memref<2x256x128xbf16>)
      %c16448 = arith.constant 16448 : index
      %19 = affine.apply #map3()[%3, %c16448]
      %c0_74 = arith.constant 0 : index
      %c64_75 = arith.constant 64 : index
      %c64_76 = arith.constant 64 : index
      %c128_77 = arith.constant 128 : index
      %c1_78 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c0] (%arg9[%c0_74, %19] [%c64_75, %c64_76] [%c128_77, %c1_78]) : (memref<2x256x128xbf16>)
      %c24576 = arith.constant 24576 : index
      %20 = affine.apply #map3()[%3, %c24576]
      %c0_79 = arith.constant 0 : index
      %c64_80 = arith.constant 64 : index
      %c64_81 = arith.constant 64 : index
      %c128_82 = arith.constant 128 : index
      %c1_83 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c0] (%arg9[%c0_79, %20] [%c64_80, %c64_81] [%c128_82, %c1_83]) : (memref<2x256x128xbf16>)
      %c24640 = arith.constant 24640 : index
      %21 = affine.apply #map3()[%3, %c24640]
      %c0_84 = arith.constant 0 : index
      %c64_85 = arith.constant 64 : index
      %c64_86 = arith.constant 64 : index
      %c128_87 = arith.constant 128 : index
      %c1_88 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c0] (%arg9[%c0_84, %21] [%c64_85, %c64_86] [%c128_87, %c1_88]) : (memref<2x256x128xbf16>)
      %c0_89 = arith.constant 0 : index
      %22 = affine.apply #map3()[%4, %c0_89]
      %c0_90 = arith.constant 0 : index
      %c0_91 = arith.constant 0 : index
      %c1_92 = arith.constant 1 : index
      %c64_93 = arith.constant 64 : index
      %c64_94 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c64_95 = arith.constant 64 : index
      %c1_96 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c0] (%arg10[%c0_90, %c0_91, %22] [%c1_92, %c64_93, %c64_94] [%c4096, %c64_95, %c1_96]) : (memref<2x256x64xbf16>)
      %c4096_97 = arith.constant 4096 : index
      %23 = affine.apply #map3()[%4, %c4096_97]
      %c0_98 = arith.constant 0 : index
      %c0_99 = arith.constant 0 : index
      %c1_100 = arith.constant 1 : index
      %c64_101 = arith.constant 64 : index
      %c64_102 = arith.constant 64 : index
      %c4096_103 = arith.constant 4096 : index
      %c64_104 = arith.constant 64 : index
      %c1_105 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c0] (%arg10[%c0_98, %c0_99, %23] [%c1_100, %c64_101, %c64_102] [%c4096_103, %c64_104, %c1_105]) : (memref<2x256x64xbf16>)
      %c8192_106 = arith.constant 8192 : index
      %24 = affine.apply #map3()[%4, %c8192_106]
      %c0_107 = arith.constant 0 : index
      %c0_108 = arith.constant 0 : index
      %c1_109 = arith.constant 1 : index
      %c64_110 = arith.constant 64 : index
      %c64_111 = arith.constant 64 : index
      %c4096_112 = arith.constant 4096 : index
      %c64_113 = arith.constant 64 : index
      %c1_114 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c0] (%arg10[%c0_107, %c0_108, %24] [%c1_109, %c64_110, %c64_111] [%c4096_112, %c64_113, %c1_114]) : (memref<2x256x64xbf16>)
      %c12288 = arith.constant 12288 : index
      %25 = affine.apply #map3()[%4, %c12288]
      %c0_115 = arith.constant 0 : index
      %c0_116 = arith.constant 0 : index
      %c1_117 = arith.constant 1 : index
      %c64_118 = arith.constant 64 : index
      %c64_119 = arith.constant 64 : index
      %c4096_120 = arith.constant 4096 : index
      %c64_121 = arith.constant 64 : index
      %c1_122 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c0] (%arg10[%c0_115, %c0_116, %25] [%c1_117, %c64_118, %c64_119] [%c4096_120, %c64_121, %c1_122]) : (memref<2x256x64xbf16>)
      %c32768 = arith.constant 32768 : index
      %c1_123 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0] (%arg11[%5] [%c32768] [%c1_123]) : (memref<2x256x64xbf16>)
      %26 = affine.apply #map4()[%1]
      %27 = affine.apply #map()[%26]
      %28 = affine.apply #map()[%26]
      %29 = affine.apply #map2()[%26]
      %c1_124 = arith.constant 1 : index
      %30 = affine.apply #map3()[%27, %0]
      %c0_125 = arith.constant 0 : index
      %31 = affine.apply #map3()[%30, %c0_125]
      %c0_126 = arith.constant 0 : index
      %c256_127 = arith.constant 256 : index
      %c64_128 = arith.constant 64 : index
      %c128_129 = arith.constant 128 : index
      %c1_130 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_124] (%arg8[%c0_126, %31] [%c256_127, %c64_128] [%c128_129, %c1_130]) : (memref<2x256x128xbf16>)
      %c64_131 = arith.constant 64 : index
      %32 = affine.apply #map3()[%30, %c64_131]
      %c0_132 = arith.constant 0 : index
      %c256_133 = arith.constant 256 : index
      %c64_134 = arith.constant 64 : index
      %c128_135 = arith.constant 128 : index
      %c1_136 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_124] (%arg8[%c0_132, %32] [%c256_133, %c64_134] [%c128_135, %c1_136]) : (memref<2x256x128xbf16>)
      %c0_137 = arith.constant 0 : index
      %33 = affine.apply #map3()[%30, %c0_137]
      %c0_138 = arith.constant 0 : index
      %c256_139 = arith.constant 256 : index
      %c64_140 = arith.constant 64 : index
      %c128_141 = arith.constant 128 : index
      %c1_142 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_124] (%arg8[%c0_138, %33] [%c256_139, %c64_140] [%c128_141, %c1_142]) : (memref<2x256x128xbf16>)
      %c64_143 = arith.constant 64 : index
      %34 = affine.apply #map3()[%30, %c64_143]
      %c0_144 = arith.constant 0 : index
      %c256_145 = arith.constant 256 : index
      %c64_146 = arith.constant 64 : index
      %c128_147 = arith.constant 128 : index
      %c1_148 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_124] (%arg8[%c0_144, %34] [%c256_145, %c64_146] [%c128_147, %c1_148]) : (memref<2x256x128xbf16>)
      %c0_149 = arith.constant 0 : index
      %35 = affine.apply #map3()[%30, %c0_149]
      %c0_150 = arith.constant 0 : index
      %c256_151 = arith.constant 256 : index
      %c64_152 = arith.constant 64 : index
      %c128_153 = arith.constant 128 : index
      %c1_154 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_124] (%arg8[%c0_150, %35] [%c256_151, %c64_152] [%c128_153, %c1_154]) : (memref<2x256x128xbf16>)
      %c64_155 = arith.constant 64 : index
      %36 = affine.apply #map3()[%30, %c64_155]
      %c0_156 = arith.constant 0 : index
      %c256_157 = arith.constant 256 : index
      %c64_158 = arith.constant 64 : index
      %c128_159 = arith.constant 128 : index
      %c1_160 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_124] (%arg8[%c0_156, %36] [%c256_157, %c64_158] [%c128_159, %c1_160]) : (memref<2x256x128xbf16>)
      %c0_161 = arith.constant 0 : index
      %37 = affine.apply #map3()[%30, %c0_161]
      %c0_162 = arith.constant 0 : index
      %c256_163 = arith.constant 256 : index
      %c64_164 = arith.constant 64 : index
      %c128_165 = arith.constant 128 : index
      %c1_166 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_124] (%arg8[%c0_162, %37] [%c256_163, %c64_164] [%c128_165, %c1_166]) : (memref<2x256x128xbf16>)
      %c64_167 = arith.constant 64 : index
      %38 = affine.apply #map3()[%30, %c64_167]
      %c0_168 = arith.constant 0 : index
      %c256_169 = arith.constant 256 : index
      %c64_170 = arith.constant 64 : index
      %c128_171 = arith.constant 128 : index
      %c1_172 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_124] (%arg8[%c0_168, %38] [%c256_169, %c64_170] [%c128_171, %c1_172]) : (memref<2x256x128xbf16>)
      %c0_173 = arith.constant 0 : index
      %39 = affine.apply #map3()[%28, %c0_173]
      %c0_174 = arith.constant 0 : index
      %c64_175 = arith.constant 64 : index
      %c64_176 = arith.constant 64 : index
      %c128_177 = arith.constant 128 : index
      %c1_178 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_124] (%arg9[%c0_174, %39] [%c64_175, %c64_176] [%c128_177, %c1_178]) : (memref<2x256x128xbf16>)
      %c64_179 = arith.constant 64 : index
      %40 = affine.apply #map3()[%28, %c64_179]
      %c0_180 = arith.constant 0 : index
      %c64_181 = arith.constant 64 : index
      %c64_182 = arith.constant 64 : index
      %c128_183 = arith.constant 128 : index
      %c1_184 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_124] (%arg9[%c0_180, %40] [%c64_181, %c64_182] [%c128_183, %c1_184]) : (memref<2x256x128xbf16>)
      %c8192_185 = arith.constant 8192 : index
      %41 = affine.apply #map3()[%28, %c8192_185]
      %c0_186 = arith.constant 0 : index
      %c64_187 = arith.constant 64 : index
      %c64_188 = arith.constant 64 : index
      %c128_189 = arith.constant 128 : index
      %c1_190 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_124] (%arg9[%c0_186, %41] [%c64_187, %c64_188] [%c128_189, %c1_190]) : (memref<2x256x128xbf16>)
      %c8256_191 = arith.constant 8256 : index
      %42 = affine.apply #map3()[%28, %c8256_191]
      %c0_192 = arith.constant 0 : index
      %c64_193 = arith.constant 64 : index
      %c64_194 = arith.constant 64 : index
      %c128_195 = arith.constant 128 : index
      %c1_196 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_124] (%arg9[%c0_192, %42] [%c64_193, %c64_194] [%c128_195, %c1_196]) : (memref<2x256x128xbf16>)
      %c16384_197 = arith.constant 16384 : index
      %43 = affine.apply #map3()[%28, %c16384_197]
      %c0_198 = arith.constant 0 : index
      %c64_199 = arith.constant 64 : index
      %c64_200 = arith.constant 64 : index
      %c128_201 = arith.constant 128 : index
      %c1_202 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_124] (%arg9[%c0_198, %43] [%c64_199, %c64_200] [%c128_201, %c1_202]) : (memref<2x256x128xbf16>)
      %c16448_203 = arith.constant 16448 : index
      %44 = affine.apply #map3()[%28, %c16448_203]
      %c0_204 = arith.constant 0 : index
      %c64_205 = arith.constant 64 : index
      %c64_206 = arith.constant 64 : index
      %c128_207 = arith.constant 128 : index
      %c1_208 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_124] (%arg9[%c0_204, %44] [%c64_205, %c64_206] [%c128_207, %c1_208]) : (memref<2x256x128xbf16>)
      %c24576_209 = arith.constant 24576 : index
      %45 = affine.apply #map3()[%28, %c24576_209]
      %c0_210 = arith.constant 0 : index
      %c64_211 = arith.constant 64 : index
      %c64_212 = arith.constant 64 : index
      %c128_213 = arith.constant 128 : index
      %c1_214 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_124] (%arg9[%c0_210, %45] [%c64_211, %c64_212] [%c128_213, %c1_214]) : (memref<2x256x128xbf16>)
      %c24640_215 = arith.constant 24640 : index
      %46 = affine.apply #map3()[%28, %c24640_215]
      %c0_216 = arith.constant 0 : index
      %c64_217 = arith.constant 64 : index
      %c64_218 = arith.constant 64 : index
      %c128_219 = arith.constant 128 : index
      %c1_220 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_124] (%arg9[%c0_216, %46] [%c64_217, %c64_218] [%c128_219, %c1_220]) : (memref<2x256x128xbf16>)
      %c0_221 = arith.constant 0 : index
      %47 = affine.apply #map3()[%29, %c0_221]
      %c0_222 = arith.constant 0 : index
      %c0_223 = arith.constant 0 : index
      %c1_224 = arith.constant 1 : index
      %c64_225 = arith.constant 64 : index
      %c64_226 = arith.constant 64 : index
      %c4096_227 = arith.constant 4096 : index
      %c64_228 = arith.constant 64 : index
      %c1_229 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c1_124] (%arg10[%c0_222, %c0_223, %47] [%c1_224, %c64_225, %c64_226] [%c4096_227, %c64_228, %c1_229]) : (memref<2x256x64xbf16>)
      %c4096_230 = arith.constant 4096 : index
      %48 = affine.apply #map3()[%29, %c4096_230]
      %c0_231 = arith.constant 0 : index
      %c0_232 = arith.constant 0 : index
      %c1_233 = arith.constant 1 : index
      %c64_234 = arith.constant 64 : index
      %c64_235 = arith.constant 64 : index
      %c4096_236 = arith.constant 4096 : index
      %c64_237 = arith.constant 64 : index
      %c1_238 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c1_124] (%arg10[%c0_231, %c0_232, %48] [%c1_233, %c64_234, %c64_235] [%c4096_236, %c64_237, %c1_238]) : (memref<2x256x64xbf16>)
      %c8192_239 = arith.constant 8192 : index
      %49 = affine.apply #map3()[%29, %c8192_239]
      %c0_240 = arith.constant 0 : index
      %c0_241 = arith.constant 0 : index
      %c1_242 = arith.constant 1 : index
      %c64_243 = arith.constant 64 : index
      %c64_244 = arith.constant 64 : index
      %c4096_245 = arith.constant 4096 : index
      %c64_246 = arith.constant 64 : index
      %c1_247 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c1_124] (%arg10[%c0_240, %c0_241, %49] [%c1_242, %c64_243, %c64_244] [%c4096_245, %c64_246, %c1_247]) : (memref<2x256x64xbf16>)
      %c12288_248 = arith.constant 12288 : index
      %50 = affine.apply #map3()[%29, %c12288_248]
      %c0_249 = arith.constant 0 : index
      %c0_250 = arith.constant 0 : index
      %c1_251 = arith.constant 1 : index
      %c64_252 = arith.constant 64 : index
      %c64_253 = arith.constant 64 : index
      %c4096_254 = arith.constant 4096 : index
      %c64_255 = arith.constant 64 : index
      %c1_256 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c1_124] (%arg10[%c0_249, %c0_250, %50] [%c1_251, %c64_252, %c64_253] [%c4096_254, %c64_255, %c1_256]) : (memref<2x256x64xbf16>)
      %c32768_257 = arith.constant 32768 : index
      %c1_258 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_124] (%arg11[%30] [%c32768_257] [%c1_258]) : (memref<2x256x64xbf16>)
      %c2 = arith.constant 2 : index
      %c1_259 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_259) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_260 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_261 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_262 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_263 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_264 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_265 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_266 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_267 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_268 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_269 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_270 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_271 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_272 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_273 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_274 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_275 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %c4 = arith.constant 4 : index
        %c4_276 = arith.constant 4 : index
        %c0_277 = arith.constant 0 : index
        %c1_278 = arith.constant 1 : index
        %c0_279 = arith.constant 0 : index
        %c1_280 = arith.constant 1 : index
        scf.for %arg16 = %c0_279 to %c4 step %c1_280 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_277, %c0_277] (%alloc[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_281 = arith.constant 0 : index
        %c1_282 = arith.constant 1 : index
        scf.for %arg16 = %c0_281 to %c4 step %c1_282 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_277, %c0_277] (%alloc[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_283 = arith.constant 0 : index
        %c1_284 = arith.constant 1 : index
        scf.for %arg16 = %c0_283 to %c1_278 step %c1_284 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_277, %c0_277] (%alloc[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_321 = arith.constant 0 : index
          %c0_322 = arith.constant 0 : index
          %c0_323 = arith.constant 0 : index
          %c0_324 = arith.constant 0 : index
          %c8_325 = arith.constant 8 : index
          %c8_326 = arith.constant 8 : index
          %c8_327 = arith.constant 8 : index
          %c8_328 = arith.constant 8 : index
          %c8_329 = arith.constant 8 : index
          %c512_330 = arith.constant 512 : index
          %c64_331 = arith.constant 64 : index
          %c1_332 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_277, %c0_277] (%alloc[%c0_321, %c0_322, %c0_323, %c0_324] [%c8_325, %c8_326, %c8_327, %c8_328] [%c8_329, %c512_330, %c64_331, %c1_332]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_285 = arith.constant 0 : index
        %c1_286 = arith.constant 1 : index
        scf.for %arg16 = %c0_285 to %c4 step %c1_286 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_260[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_277, %c0_277] (%alloc_260[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_287 = arith.constant 0 : index
        %c1_288 = arith.constant 1 : index
        scf.for %arg16 = %c0_287 to %c4 step %c1_288 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_260[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_277, %c0_277] (%alloc_260[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_289 = arith.constant 0 : index
        %c1_290 = arith.constant 1 : index
        scf.for %arg16 = %c0_289 to %c1_278 step %c1_290 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_260[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_277, %c0_277] (%alloc_260[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_1[%arg12] (%alloc_260[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_321 = arith.constant 0 : index
          %c0_322 = arith.constant 0 : index
          %c0_323 = arith.constant 0 : index
          %c0_324 = arith.constant 0 : index
          %c8_325 = arith.constant 8 : index
          %c8_326 = arith.constant 8 : index
          %c8_327 = arith.constant 8 : index
          %c8_328 = arith.constant 8 : index
          %c8_329 = arith.constant 8 : index
          %c512_330 = arith.constant 512 : index
          %c64_331 = arith.constant 64 : index
          %c1_332 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_277, %c0_277] (%alloc_260[%c0_321, %c0_322, %c0_323, %c0_324] [%c8_325, %c8_326, %c8_327, %c8_328] [%c8_329, %c512_330, %c64_331, %c1_332]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_291 = arith.constant 0 : index
        %c1_292 = arith.constant 1 : index
        scf.for %arg16 = %c0_291 to %c4 step %c1_292 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_261[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_277, %c0_277] (%alloc_261[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_293 = arith.constant 0 : index
        %c1_294 = arith.constant 1 : index
        scf.for %arg16 = %c0_293 to %c4 step %c1_294 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_261[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_277, %c0_277] (%alloc_261[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_295 = arith.constant 0 : index
        %c1_296 = arith.constant 1 : index
        scf.for %arg16 = %c0_295 to %c1_278 step %c1_296 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_261[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_277, %c0_277] (%alloc_261[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_2[%arg12] (%alloc_261[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_321 = arith.constant 0 : index
          %c0_322 = arith.constant 0 : index
          %c0_323 = arith.constant 0 : index
          %c0_324 = arith.constant 0 : index
          %c8_325 = arith.constant 8 : index
          %c8_326 = arith.constant 8 : index
          %c8_327 = arith.constant 8 : index
          %c8_328 = arith.constant 8 : index
          %c8_329 = arith.constant 8 : index
          %c512_330 = arith.constant 512 : index
          %c64_331 = arith.constant 64 : index
          %c1_332 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_277, %c0_277] (%alloc_261[%c0_321, %c0_322, %c0_323, %c0_324] [%c8_325, %c8_326, %c8_327, %c8_328] [%c8_329, %c512_330, %c64_331, %c1_332]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_297 = arith.constant 0 : index
        %c1_298 = arith.constant 1 : index
        scf.for %arg16 = %c0_297 to %c4 step %c1_298 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_262[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_277, %c0_277] (%alloc_262[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_299 = arith.constant 0 : index
        %c1_300 = arith.constant 1 : index
        scf.for %arg16 = %c0_299 to %c4 step %c1_300 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_262[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_277, %c0_277] (%alloc_262[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_301 = arith.constant 0 : index
        %c1_302 = arith.constant 1 : index
        scf.for %arg16 = %c0_301 to %c1_278 step %c1_302 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_262[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_277, %c0_277] (%alloc_262[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_3[%arg12] (%alloc_262[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_321 = arith.constant 0 : index
          %c0_322 = arith.constant 0 : index
          %c0_323 = arith.constant 0 : index
          %c0_324 = arith.constant 0 : index
          %c8_325 = arith.constant 8 : index
          %c8_326 = arith.constant 8 : index
          %c8_327 = arith.constant 8 : index
          %c8_328 = arith.constant 8 : index
          %c8_329 = arith.constant 8 : index
          %c512_330 = arith.constant 512 : index
          %c64_331 = arith.constant 64 : index
          %c1_332 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_277, %c0_277] (%alloc_262[%c0_321, %c0_322, %c0_323, %c0_324] [%c8_325, %c8_326, %c8_327, %c8_328] [%c8_329, %c512_330, %c64_331, %c1_332]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_303 = arith.constant 0 : index
        %c1_304 = arith.constant 1 : index
        scf.for %arg16 = %c0_303 to %c1_278 step %c1_304 {
          air.channel.get  @VIn_0[%arg12] (%alloc_263[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @V2L1_0[%arg12, %c0_277, %c0_277] (%alloc_263[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_305 = arith.constant 0 : index
        %c1_306 = arith.constant 1 : index
        scf.for %arg16 = %c0_305 to %c1_278 step %c1_306 {
          air.channel.get  @VIn_1[%arg12] (%alloc_264[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @V2L1_1[%arg12, %c0_277, %c0_277] (%alloc_264[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_307 = arith.constant 0 : index
        %c1_308 = arith.constant 1 : index
        scf.for %arg16 = %c0_307 to %c1_278 step %c1_308 {
          air.channel.get  @VIn_2[%arg12] (%alloc_265[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @V2L1_2[%arg12, %c0_277, %c0_277] (%alloc_265[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_309 = arith.constant 0 : index
        %c1_310 = arith.constant 1 : index
        scf.for %arg16 = %c0_309 to %c1_278 step %c1_310 {
          air.channel.get  @VIn_3[%arg12] (%alloc_266[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c0_313 = arith.constant 0 : index
          %c0_314 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_315 = arith.constant 8 : index
          %c8_316 = arith.constant 8 : index
          %c8_317 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_319 = arith.constant 64 : index
          %c1_320 = arith.constant 1 : index
          air.channel.put  @V2L1_3[%arg12, %c0_277, %c0_277] (%alloc_266[%c0_311, %c0_312, %c0_313, %c0_314] [%c8, %c8_315, %c8_316, %c8_317] [%c8_318, %c512, %c64_319, %c1_320]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %51 = affine.apply #map5()[%arg16]
          %c0_311 = arith.constant 0 : index
          %c0_312 = arith.constant 0 : index
          %c64_313 = arith.constant 64 : index
          %c64_314 = arith.constant 64 : index
          %c64_315 = arith.constant 64 : index
          %c1_316 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%arg16, %c0_311] (%alloc_267[%51, %c0_312] [%c64_313, %c64_314] [%c64_315, %c1_316]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_267[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4_276) args(%arg20=%alloc_268, %arg21=%alloc_269, %arg22=%alloc_270, %arg23=%alloc_271, %arg24=%alloc_272, %arg25=%alloc_273, %arg26=%alloc_274, %arg27=%alloc_275, %arg28=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn.o"} {
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
          %c0_i32 = arith.constant 0 : i32
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
          %c1_i32 = arith.constant 1 : i32
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
          %c2_i32 = arith.constant 2 : i32
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
          %c3_i32 = arith.constant 3 : i32
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
          %c0_i32_311 = arith.constant 0 : i32
          %60 = arith.cmpi eq, %59, %c0_i32_311 : i32
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
          %c1_i32_312 = arith.constant 1 : i32
          %62 = arith.cmpi eq, %61, %c1_i32_312 : i32
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
          %c2_i32_313 = arith.constant 2 : i32
          %64 = arith.cmpi eq, %63, %c2_i32_313 : i32
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
          %c3_i32_314 = arith.constant 3 : i32
          %66 = arith.cmpi eq, %65, %c3_i32_314 : i32
          scf.if %66 {
            func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %c1_315 = arith.constant 1 : index
          %c0_316 = arith.constant 0 : index
          %c1_317 = arith.constant 1 : index
          scf.for %arg29 = %c0_316 to %c1_315 step %c1_317 {
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
            %alloc_319 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_320 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg26, %alloc_319, %alloc_320) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_320, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg23, %arg25) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32_321 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg27, %alloc_320, %alloc_319) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32_321, %alloc_319, %arg27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_319 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_320 : memref<64x1xbf16, 2 : i32>
          }
          %c1_318 = arith.constant 1 : index
          affine.if #set3()[%arg16, %arg17] {
            %67 = arith.subi %arg17, %c1_318 : index
            air.channel.put  @cascade_gp[%arg16, %67] (%arg25[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %67] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %67] (%arg27[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_319 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_320 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_321 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_319[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_320[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_321[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_322 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_323 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_323, %arg26, %alloc_322) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_320, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_324 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_320, %arg26, %alloc_324) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_325 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_322, %arg26, %alloc_325) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_324, %alloc_319) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_325, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_319) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_326 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_326) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_321, %alloc_324, %alloc_326) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_325, %alloc_326) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_323, %alloc_326, %alloc_321) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %67 = arith.subi %arg17, %c1_318 : index
              air.channel.put  @cascade_gp[%arg16, %67] (%alloc_319[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %67] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %67] (%alloc_321[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_319 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_320 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_321 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_322 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_324 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_325 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_326 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_319 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_320 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_321 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_319[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_320[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_321[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_322 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_323 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_323, %arg26, %alloc_322) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_320, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_324 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_320, %arg26, %alloc_324) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_325 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_322, %arg26, %alloc_325) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_324, %alloc_319) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_325, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_319) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_326 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_326) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_321, %alloc_324, %alloc_326) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_325, %alloc_326) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_323, %alloc_326, %alloc_321) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_321, %alloc_319) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %c0_327 = arith.constant 0 : index
              %c0_328 = arith.constant 0 : index
              %c0_329 = arith.constant 0 : index
              %c0_330 = arith.constant 0 : index
              %c0_331 = arith.constant 0 : index
              %c8 = arith.constant 8 : index
              %c8_332 = arith.constant 8 : index
              %c8_333 = arith.constant 8 : index
              %c8_334 = arith.constant 8 : index
              %c64_335 = arith.constant 64 : index
              %c8_336 = arith.constant 8 : index
              %c512 = arith.constant 512 : index
              %c1_337 = arith.constant 1 : index
              air.channel.put  @Gp2L2[%arg16, %c0_327] (%alloc_319[%c0_328, %c0_329, %c0_330, %c0_331] [%c8, %c8_332, %c8_333, %c8_334] [%c64_335, %c8_336, %c512, %c1_337]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_319 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_320 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_321 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_322 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_324 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_325 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_326 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_268 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_269 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_270 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_271 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_272 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_273 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_274 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_275 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_263 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_260 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_264 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_261 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_265 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_262 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_266 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_267 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}
