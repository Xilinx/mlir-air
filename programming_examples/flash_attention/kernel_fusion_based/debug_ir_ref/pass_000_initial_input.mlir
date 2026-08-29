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
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    air.launch (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1_1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16> {
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map()[%1]
      %3 = affine.apply #map()[%1]
      %4 = affine.apply #map()[%1]
      %c0 = arith.constant 0 : index
      %5 = affine.apply #map2()[%2, %0]
      %c0_2 = arith.constant 0 : index
      %6 = affine.apply #map2()[%5, %c0_2]
      %c0_3 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %c64_4 = arith.constant 64 : index
      %c1_5 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c0] (%arg8[%c0_3, %6] [%c256, %c64] [%c64_4, %c1_5]) : (memref<2x256x64xbf16>)
      %c0_6 = arith.constant 0 : index
      %7 = affine.apply #map2()[%5, %c0_6]
      %c0_7 = arith.constant 0 : index
      %c256_8 = arith.constant 256 : index
      %c64_9 = arith.constant 64 : index
      %c64_10 = arith.constant 64 : index
      %c1_11 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c0] (%arg8[%c0_7, %7] [%c256_8, %c64_9] [%c64_10, %c1_11]) : (memref<2x256x64xbf16>)
      %c0_12 = arith.constant 0 : index
      %8 = affine.apply #map2()[%5, %c0_12]
      %c0_13 = arith.constant 0 : index
      %c256_14 = arith.constant 256 : index
      %c64_15 = arith.constant 64 : index
      %c64_16 = arith.constant 64 : index
      %c1_17 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c0] (%arg8[%c0_13, %8] [%c256_14, %c64_15] [%c64_16, %c1_17]) : (memref<2x256x64xbf16>)
      %c0_18 = arith.constant 0 : index
      %9 = affine.apply #map2()[%5, %c0_18]
      %c0_19 = arith.constant 0 : index
      %c256_20 = arith.constant 256 : index
      %c64_21 = arith.constant 64 : index
      %c64_22 = arith.constant 64 : index
      %c1_23 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c0] (%arg8[%c0_19, %9] [%c256_20, %c64_21] [%c64_22, %c1_23]) : (memref<2x256x64xbf16>)
      %c0_24 = arith.constant 0 : index
      %10 = affine.apply #map2()[%3, %c0_24]
      %c0_25 = arith.constant 0 : index
      %c64_26 = arith.constant 64 : index
      %c64_27 = arith.constant 64 : index
      %c64_28 = arith.constant 64 : index
      %c1_29 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c0] (%arg9[%c0_25, %10] [%c64_26, %c64_27] [%c64_28, %c1_29]) : (memref<2x256x64xbf16>)
      %c4096 = arith.constant 4096 : index
      %11 = affine.apply #map2()[%3, %c4096]
      %c0_30 = arith.constant 0 : index
      %c64_31 = arith.constant 64 : index
      %c64_32 = arith.constant 64 : index
      %c64_33 = arith.constant 64 : index
      %c1_34 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c0] (%arg9[%c0_30, %11] [%c64_31, %c64_32] [%c64_33, %c1_34]) : (memref<2x256x64xbf16>)
      %c8192 = arith.constant 8192 : index
      %12 = affine.apply #map2()[%3, %c8192]
      %c0_35 = arith.constant 0 : index
      %c64_36 = arith.constant 64 : index
      %c64_37 = arith.constant 64 : index
      %c64_38 = arith.constant 64 : index
      %c1_39 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c0] (%arg9[%c0_35, %12] [%c64_36, %c64_37] [%c64_38, %c1_39]) : (memref<2x256x64xbf16>)
      %c12288 = arith.constant 12288 : index
      %13 = affine.apply #map2()[%3, %c12288]
      %c0_40 = arith.constant 0 : index
      %c64_41 = arith.constant 64 : index
      %c64_42 = arith.constant 64 : index
      %c64_43 = arith.constant 64 : index
      %c1_44 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c0] (%arg9[%c0_40, %13] [%c64_41, %c64_42] [%c64_43, %c1_44]) : (memref<2x256x64xbf16>)
      %c0_45 = arith.constant 0 : index
      %14 = affine.apply #map2()[%4, %c0_45]
      %c0_46 = arith.constant 0 : index
      %c0_47 = arith.constant 0 : index
      %c1_48 = arith.constant 1 : index
      %c64_49 = arith.constant 64 : index
      %c64_50 = arith.constant 64 : index
      %c4096_51 = arith.constant 4096 : index
      %c64_52 = arith.constant 64 : index
      %c1_53 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c0] (%arg10[%c0_46, %c0_47, %14] [%c1_48, %c64_49, %c64_50] [%c4096_51, %c64_52, %c1_53]) : (memref<2x256x64xbf16>)
      %c4096_54 = arith.constant 4096 : index
      %15 = affine.apply #map2()[%4, %c4096_54]
      %c0_55 = arith.constant 0 : index
      %c0_56 = arith.constant 0 : index
      %c1_57 = arith.constant 1 : index
      %c64_58 = arith.constant 64 : index
      %c64_59 = arith.constant 64 : index
      %c4096_60 = arith.constant 4096 : index
      %c64_61 = arith.constant 64 : index
      %c1_62 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c0] (%arg10[%c0_55, %c0_56, %15] [%c1_57, %c64_58, %c64_59] [%c4096_60, %c64_61, %c1_62]) : (memref<2x256x64xbf16>)
      %c8192_63 = arith.constant 8192 : index
      %16 = affine.apply #map2()[%4, %c8192_63]
      %c0_64 = arith.constant 0 : index
      %c0_65 = arith.constant 0 : index
      %c1_66 = arith.constant 1 : index
      %c64_67 = arith.constant 64 : index
      %c64_68 = arith.constant 64 : index
      %c4096_69 = arith.constant 4096 : index
      %c64_70 = arith.constant 64 : index
      %c1_71 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c0] (%arg10[%c0_64, %c0_65, %16] [%c1_66, %c64_67, %c64_68] [%c4096_69, %c64_70, %c1_71]) : (memref<2x256x64xbf16>)
      %c12288_72 = arith.constant 12288 : index
      %17 = affine.apply #map2()[%4, %c12288_72]
      %c0_73 = arith.constant 0 : index
      %c0_74 = arith.constant 0 : index
      %c1_75 = arith.constant 1 : index
      %c64_76 = arith.constant 64 : index
      %c64_77 = arith.constant 64 : index
      %c4096_78 = arith.constant 4096 : index
      %c64_79 = arith.constant 64 : index
      %c1_80 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c0] (%arg10[%c0_73, %c0_74, %17] [%c1_75, %c64_76, %c64_77] [%c4096_78, %c64_79, %c1_80]) : (memref<2x256x64xbf16>)
      %c16384 = arith.constant 16384 : index
      %c1_81 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0] (%arg11[%5] [%c16384] [%c1_81]) : (memref<2x256x64xbf16>)
      %18 = affine.apply #map3()[%1]
      %19 = affine.apply #map()[%18]
      %20 = affine.apply #map()[%18]
      %21 = affine.apply #map()[%18]
      %c1_82 = arith.constant 1 : index
      %22 = affine.apply #map2()[%19, %0]
      %c0_83 = arith.constant 0 : index
      %23 = affine.apply #map2()[%22, %c0_83]
      %c0_84 = arith.constant 0 : index
      %c256_85 = arith.constant 256 : index
      %c64_86 = arith.constant 64 : index
      %c64_87 = arith.constant 64 : index
      %c1_88 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_82] (%arg8[%c0_84, %23] [%c256_85, %c64_86] [%c64_87, %c1_88]) : (memref<2x256x64xbf16>)
      %c0_89 = arith.constant 0 : index
      %24 = affine.apply #map2()[%22, %c0_89]
      %c0_90 = arith.constant 0 : index
      %c256_91 = arith.constant 256 : index
      %c64_92 = arith.constant 64 : index
      %c64_93 = arith.constant 64 : index
      %c1_94 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_82] (%arg8[%c0_90, %24] [%c256_91, %c64_92] [%c64_93, %c1_94]) : (memref<2x256x64xbf16>)
      %c0_95 = arith.constant 0 : index
      %25 = affine.apply #map2()[%22, %c0_95]
      %c0_96 = arith.constant 0 : index
      %c256_97 = arith.constant 256 : index
      %c64_98 = arith.constant 64 : index
      %c64_99 = arith.constant 64 : index
      %c1_100 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_82] (%arg8[%c0_96, %25] [%c256_97, %c64_98] [%c64_99, %c1_100]) : (memref<2x256x64xbf16>)
      %c0_101 = arith.constant 0 : index
      %26 = affine.apply #map2()[%22, %c0_101]
      %c0_102 = arith.constant 0 : index
      %c256_103 = arith.constant 256 : index
      %c64_104 = arith.constant 64 : index
      %c64_105 = arith.constant 64 : index
      %c1_106 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_82] (%arg8[%c0_102, %26] [%c256_103, %c64_104] [%c64_105, %c1_106]) : (memref<2x256x64xbf16>)
      %c0_107 = arith.constant 0 : index
      %27 = affine.apply #map2()[%20, %c0_107]
      %c0_108 = arith.constant 0 : index
      %c64_109 = arith.constant 64 : index
      %c64_110 = arith.constant 64 : index
      %c64_111 = arith.constant 64 : index
      %c1_112 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_82] (%arg9[%c0_108, %27] [%c64_109, %c64_110] [%c64_111, %c1_112]) : (memref<2x256x64xbf16>)
      %c4096_113 = arith.constant 4096 : index
      %28 = affine.apply #map2()[%20, %c4096_113]
      %c0_114 = arith.constant 0 : index
      %c64_115 = arith.constant 64 : index
      %c64_116 = arith.constant 64 : index
      %c64_117 = arith.constant 64 : index
      %c1_118 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_82] (%arg9[%c0_114, %28] [%c64_115, %c64_116] [%c64_117, %c1_118]) : (memref<2x256x64xbf16>)
      %c8192_119 = arith.constant 8192 : index
      %29 = affine.apply #map2()[%20, %c8192_119]
      %c0_120 = arith.constant 0 : index
      %c64_121 = arith.constant 64 : index
      %c64_122 = arith.constant 64 : index
      %c64_123 = arith.constant 64 : index
      %c1_124 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_82] (%arg9[%c0_120, %29] [%c64_121, %c64_122] [%c64_123, %c1_124]) : (memref<2x256x64xbf16>)
      %c12288_125 = arith.constant 12288 : index
      %30 = affine.apply #map2()[%20, %c12288_125]
      %c0_126 = arith.constant 0 : index
      %c64_127 = arith.constant 64 : index
      %c64_128 = arith.constant 64 : index
      %c64_129 = arith.constant 64 : index
      %c1_130 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_82] (%arg9[%c0_126, %30] [%c64_127, %c64_128] [%c64_129, %c1_130]) : (memref<2x256x64xbf16>)
      %c0_131 = arith.constant 0 : index
      %31 = affine.apply #map2()[%21, %c0_131]
      %c0_132 = arith.constant 0 : index
      %c0_133 = arith.constant 0 : index
      %c1_134 = arith.constant 1 : index
      %c64_135 = arith.constant 64 : index
      %c64_136 = arith.constant 64 : index
      %c4096_137 = arith.constant 4096 : index
      %c64_138 = arith.constant 64 : index
      %c1_139 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c1_82] (%arg10[%c0_132, %c0_133, %31] [%c1_134, %c64_135, %c64_136] [%c4096_137, %c64_138, %c1_139]) : (memref<2x256x64xbf16>)
      %c4096_140 = arith.constant 4096 : index
      %32 = affine.apply #map2()[%21, %c4096_140]
      %c0_141 = arith.constant 0 : index
      %c0_142 = arith.constant 0 : index
      %c1_143 = arith.constant 1 : index
      %c64_144 = arith.constant 64 : index
      %c64_145 = arith.constant 64 : index
      %c4096_146 = arith.constant 4096 : index
      %c64_147 = arith.constant 64 : index
      %c1_148 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c1_82] (%arg10[%c0_141, %c0_142, %32] [%c1_143, %c64_144, %c64_145] [%c4096_146, %c64_147, %c1_148]) : (memref<2x256x64xbf16>)
      %c8192_149 = arith.constant 8192 : index
      %33 = affine.apply #map2()[%21, %c8192_149]
      %c0_150 = arith.constant 0 : index
      %c0_151 = arith.constant 0 : index
      %c1_152 = arith.constant 1 : index
      %c64_153 = arith.constant 64 : index
      %c64_154 = arith.constant 64 : index
      %c4096_155 = arith.constant 4096 : index
      %c64_156 = arith.constant 64 : index
      %c1_157 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c1_82] (%arg10[%c0_150, %c0_151, %33] [%c1_152, %c64_153, %c64_154] [%c4096_155, %c64_156, %c1_157]) : (memref<2x256x64xbf16>)
      %c12288_158 = arith.constant 12288 : index
      %34 = affine.apply #map2()[%21, %c12288_158]
      %c0_159 = arith.constant 0 : index
      %c0_160 = arith.constant 0 : index
      %c1_161 = arith.constant 1 : index
      %c64_162 = arith.constant 64 : index
      %c64_163 = arith.constant 64 : index
      %c4096_164 = arith.constant 4096 : index
      %c64_165 = arith.constant 64 : index
      %c1_166 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c1_82] (%arg10[%c0_159, %c0_160, %34] [%c1_161, %c64_162, %c64_163] [%c4096_164, %c64_165, %c1_166]) : (memref<2x256x64xbf16>)
      %c16384_167 = arith.constant 16384 : index
      %c1_168 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_82] (%arg11[%22] [%c16384_167] [%c1_168]) : (memref<2x256x64xbf16>)
      %c2 = arith.constant 2 : index
      %c1_169 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_169) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_170 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_171 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_172 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_173 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_174 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_175 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_176 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_177 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_178 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_179 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_180 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_181 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_182 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_183 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_184 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %c4 = arith.constant 4 : index
        %c4_185 = arith.constant 4 : index
        %c0_186 = arith.constant 0 : index
        %c1_187 = arith.constant 1 : index
        %c0_188 = arith.constant 0 : index
        %c1_189 = arith.constant 1 : index
        scf.for %arg16 = %c0_188 to %c4 step %c1_189 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_186, %c0_186] (%alloc[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_190 = arith.constant 0 : index
        %c1_191 = arith.constant 1 : index
        scf.for %arg16 = %c0_190 to %c1_187 step %c1_191 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_186, %c0_186] (%alloc[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_192 = arith.constant 0 : index
        %c1_193 = arith.constant 1 : index
        scf.for %arg16 = %c0_192 to %c4 step %c1_193 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_170[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_186, %c0_186] (%alloc_170[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_194 = arith.constant 0 : index
        %c1_195 = arith.constant 1 : index
        scf.for %arg16 = %c0_194 to %c1_187 step %c1_195 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_170[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_186, %c0_186] (%alloc_170[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_196 = arith.constant 0 : index
        %c1_197 = arith.constant 1 : index
        scf.for %arg16 = %c0_196 to %c4 step %c1_197 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_171[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_186, %c0_186] (%alloc_171[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_198 = arith.constant 0 : index
        %c1_199 = arith.constant 1 : index
        scf.for %arg16 = %c0_198 to %c1_187 step %c1_199 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_171[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_186, %c0_186] (%alloc_171[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_200 = arith.constant 0 : index
        %c1_201 = arith.constant 1 : index
        scf.for %arg16 = %c0_200 to %c4 step %c1_201 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_172[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_186, %c0_186] (%alloc_172[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_202 = arith.constant 0 : index
        %c1_203 = arith.constant 1 : index
        scf.for %arg16 = %c0_202 to %c1_187 step %c1_203 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_172[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_186, %c0_186] (%alloc_172[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_204 = arith.constant 0 : index
        %c1_205 = arith.constant 1 : index
        scf.for %arg16 = %c0_204 to %c1_187 step %c1_205 {
          air.channel.get  @VIn_0[%arg12] (%alloc_173[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @V2L1_0[%arg12, %c0_186, %c0_186] (%alloc_173[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_206 = arith.constant 0 : index
        %c1_207 = arith.constant 1 : index
        scf.for %arg16 = %c0_206 to %c1_187 step %c1_207 {
          air.channel.get  @VIn_1[%arg12] (%alloc_174[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @V2L1_1[%arg12, %c0_186, %c0_186] (%alloc_174[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_208 = arith.constant 0 : index
        %c1_209 = arith.constant 1 : index
        scf.for %arg16 = %c0_208 to %c1_187 step %c1_209 {
          air.channel.get  @VIn_2[%arg12] (%alloc_175[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @V2L1_2[%arg12, %c0_186, %c0_186] (%alloc_175[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_210 = arith.constant 0 : index
        %c1_211 = arith.constant 1 : index
        scf.for %arg16 = %c0_210 to %c1_187 step %c1_211 {
          air.channel.get  @VIn_3[%arg12] (%alloc_176[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c0_214 = arith.constant 0 : index
          %c0_215 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_220 = arith.constant 64 : index
          %c1_221 = arith.constant 1 : index
          air.channel.put  @V2L1_3[%arg12, %c0_186, %c0_186] (%alloc_176[%c0_212, %c0_213, %c0_214, %c0_215] [%c8, %c8_216, %c8_217, %c8_218] [%c8_219, %c512, %c64_220, %c1_221]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %35 = affine.apply #map4()[%arg16]
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c64_214 = arith.constant 64 : index
          %c64_215 = arith.constant 64 : index
          %c64_216 = arith.constant 64 : index
          %c1_217 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%arg16, %c0_212] (%alloc_177[%35, %c0_213] [%c64_214, %c64_215] [%c64_216, %c1_217]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_177[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4_185) args(%arg20=%alloc_178, %arg21=%alloc_179, %arg22=%alloc_180, %arg23=%alloc_181, %arg24=%alloc_182, %arg25=%alloc_183, %arg26=%alloc_184, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn.o"} {
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
          %c0_i32 = arith.constant 0 : i32
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
          %c1_i32 = arith.constant 1 : i32
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
          %c2_i32 = arith.constant 2 : i32
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
          %c3_i32 = arith.constant 3 : i32
          %42 = arith.cmpi eq, %41, %c3_i32 : i32
          scf.if %42 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %c1_212 = arith.constant 1 : index
          %c0_213 = arith.constant 0 : index
          %c1_214 = arith.constant 1 : index
          scf.for %arg28 = %c0_213 to %c1_212 step %c1_214 {
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
            %alloc_216 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_217 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %alloc_216, %alloc_217) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_217, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32_218 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg26, %alloc_217, %alloc_216) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32_218, %alloc_216, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_216 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_217 : memref<64x1xbf16, 2 : i32>
          }
          %c1_215 = arith.constant 1 : index
          affine.if #set3()[%arg16, %arg17] {
            %43 = arith.subi %arg17, %c1_215 : index
            air.channel.put  @cascade_gp[%arg16, %43] (%arg24[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %43] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %43] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_216 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_217 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_218 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_216[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_217[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_218[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_219 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_220 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_220, %arg25, %alloc_219) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_217, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_221 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_217, %arg25, %alloc_221) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_222 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_219, %arg25, %alloc_222) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_221, %alloc_216) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_222, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_216) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_223 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_223) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_218, %alloc_221, %alloc_223) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_222, %alloc_223) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_220, %alloc_223, %alloc_218) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %43 = arith.subi %arg17, %c1_215 : index
              air.channel.put  @cascade_gp[%arg16, %43] (%alloc_216[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %43] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %43] (%alloc_218[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_216 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_217 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_218 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_219 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_221 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_222 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_223 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_216 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_217 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_218 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_216[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_217[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_218[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_219 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_220 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_220, %arg25, %alloc_219) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_217, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_221 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_217, %arg25, %alloc_221) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_222 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_219, %arg25, %alloc_222) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_221, %alloc_216) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_222, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_216) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_223 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_223) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_218, %alloc_221, %alloc_223) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_222, %alloc_223) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_220, %alloc_223, %alloc_218) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_218, %alloc_216) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %c0_224 = arith.constant 0 : index
              %c0_225 = arith.constant 0 : index
              %c0_226 = arith.constant 0 : index
              %c0_227 = arith.constant 0 : index
              %c0_228 = arith.constant 0 : index
              %c8 = arith.constant 8 : index
              %c8_229 = arith.constant 8 : index
              %c8_230 = arith.constant 8 : index
              %c8_231 = arith.constant 8 : index
              %c64_232 = arith.constant 64 : index
              %c8_233 = arith.constant 8 : index
              %c512 = arith.constant 512 : index
              %c1_234 = arith.constant 1 : index
              air.channel.put  @Gp2L2[%arg16, %c0_224] (%alloc_216[%c0_225, %c0_226, %c0_227, %c0_228] [%c8, %c8_229, %c8_230, %c8_231] [%c64_232, %c8_233, %c512, %c1_234]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_216 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_217 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_218 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_219 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_221 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_222 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_223 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_178 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_179 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_180 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_181 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_182 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_183 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_184 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_173 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_170 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_174 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_171 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_175 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_172 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_176 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_177 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}
