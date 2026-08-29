#map = affine_map<()[s0] -> (s0 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 2)>
#map2 = affine_map<()[s0] -> (s0 * 32768)>
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
  air.channel @QK2L1 [2, 4, 1] {broadcast_shape = [2 : index, 4 : index, 4 : index]}
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
  func.func @attention_bf16(%arg0: memref<2x256x64xbf16>, %arg1: memref<2x512x64xbf16>, %arg2: memref<2x512x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    air.launch (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1_1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x256x64xbf16> {
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map()[%1]
      %3 = affine.apply #map2()[%1]
      %4 = affine.apply #map2()[%1]
      %c0 = arith.constant 0 : index
      %5 = affine.apply #map3()[%2, %0]
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %c0_5 = arith.constant 0 : index
      %c0_6 = arith.constant 0 : index
      %c0_7 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c8_8 = arith.constant 8 : index
      %c8_9 = arith.constant 8 : index
      %c8_10 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c8_11 = arith.constant 8 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c1_12 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c0_2, %c0_3] (%arg8[%c0_4, %c0_5, %c0_6, %c0_7, %5] [%c4, %c8, %c8_8, %c8_9, %c8_10] [%c4096, %c8_11, %c512, %c64, %c1_12]) : (memref<2x256x64xbf16>)
      %c1_13 = arith.constant 1 : index
      %c0_14 = arith.constant 0 : index
      %c0_15 = arith.constant 0 : index
      %c0_16 = arith.constant 0 : index
      %c0_17 = arith.constant 0 : index
      %c0_18 = arith.constant 0 : index
      %c4_19 = arith.constant 4 : index
      %c8_20 = arith.constant 8 : index
      %c8_21 = arith.constant 8 : index
      %c8_22 = arith.constant 8 : index
      %c8_23 = arith.constant 8 : index
      %c4096_24 = arith.constant 4096 : index
      %c8_25 = arith.constant 8 : index
      %c512_26 = arith.constant 512 : index
      %c64_27 = arith.constant 64 : index
      %c1_28 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c1_13, %c0_14] (%arg8[%c0_15, %c0_16, %c0_17, %c0_18, %5] [%c4_19, %c8_20, %c8_21, %c8_22, %c8_23] [%c4096_24, %c8_25, %c512_26, %c64_27, %c1_28]) : (memref<2x256x64xbf16>)
      %c2 = arith.constant 2 : index
      %c0_29 = arith.constant 0 : index
      %c0_30 = arith.constant 0 : index
      %c0_31 = arith.constant 0 : index
      %c0_32 = arith.constant 0 : index
      %c0_33 = arith.constant 0 : index
      %c4_34 = arith.constant 4 : index
      %c8_35 = arith.constant 8 : index
      %c8_36 = arith.constant 8 : index
      %c8_37 = arith.constant 8 : index
      %c8_38 = arith.constant 8 : index
      %c4096_39 = arith.constant 4096 : index
      %c8_40 = arith.constant 8 : index
      %c512_41 = arith.constant 512 : index
      %c64_42 = arith.constant 64 : index
      %c1_43 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c2, %c0_29] (%arg8[%c0_30, %c0_31, %c0_32, %c0_33, %5] [%c4_34, %c8_35, %c8_36, %c8_37, %c8_38] [%c4096_39, %c8_40, %c512_41, %c64_42, %c1_43]) : (memref<2x256x64xbf16>)
      %c3 = arith.constant 3 : index
      %c0_44 = arith.constant 0 : index
      %c0_45 = arith.constant 0 : index
      %c0_46 = arith.constant 0 : index
      %c0_47 = arith.constant 0 : index
      %c0_48 = arith.constant 0 : index
      %c4_49 = arith.constant 4 : index
      %c8_50 = arith.constant 8 : index
      %c8_51 = arith.constant 8 : index
      %c8_52 = arith.constant 8 : index
      %c8_53 = arith.constant 8 : index
      %c4096_54 = arith.constant 4096 : index
      %c8_55 = arith.constant 8 : index
      %c512_56 = arith.constant 512 : index
      %c64_57 = arith.constant 64 : index
      %c1_58 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c3, %c0_44] (%arg8[%c0_45, %c0_46, %c0_47, %c0_48, %5] [%c4_49, %c8_50, %c8_51, %c8_52, %c8_53] [%c4096_54, %c8_55, %c512_56, %c64_57, %c1_58]) : (memref<2x256x64xbf16>)
      %c0_59 = arith.constant 0 : index
      %6 = affine.apply #map3()[%3, %c0_59]
      %c0_60 = arith.constant 0 : index
      %c0_61 = arith.constant 0 : index
      %c0_62 = arith.constant 0 : index
      %c0_63 = arith.constant 0 : index
      %c0_64 = arith.constant 0 : index
      %c0_65 = arith.constant 0 : index
      %c2_66 = arith.constant 2 : index
      %c8_67 = arith.constant 8 : index
      %c8_68 = arith.constant 8 : index
      %c8_69 = arith.constant 8 : index
      %c8_70 = arith.constant 8 : index
      %c4096_71 = arith.constant 4096 : index
      %c8_72 = arith.constant 8 : index
      %c512_73 = arith.constant 512 : index
      %c64_74 = arith.constant 64 : index
      %c1_75 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c0_60, %c0_61] (%arg9[%c0_62, %c0_63, %c0_64, %c0_65, %6] [%c2_66, %c8_67, %c8_68, %c8_69, %c8_70] [%c4096_71, %c8_72, %c512_73, %c64_74, %c1_75]) : (memref<2x512x64xbf16>)
      %c8192 = arith.constant 8192 : index
      %7 = affine.apply #map3()[%3, %c8192]
      %c1_76 = arith.constant 1 : index
      %c0_77 = arith.constant 0 : index
      %c0_78 = arith.constant 0 : index
      %c0_79 = arith.constant 0 : index
      %c0_80 = arith.constant 0 : index
      %c0_81 = arith.constant 0 : index
      %c2_82 = arith.constant 2 : index
      %c8_83 = arith.constant 8 : index
      %c8_84 = arith.constant 8 : index
      %c8_85 = arith.constant 8 : index
      %c8_86 = arith.constant 8 : index
      %c4096_87 = arith.constant 4096 : index
      %c8_88 = arith.constant 8 : index
      %c512_89 = arith.constant 512 : index
      %c64_90 = arith.constant 64 : index
      %c1_91 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c1_76, %c0_77] (%arg9[%c0_78, %c0_79, %c0_80, %c0_81, %7] [%c2_82, %c8_83, %c8_84, %c8_85, %c8_86] [%c4096_87, %c8_88, %c512_89, %c64_90, %c1_91]) : (memref<2x512x64xbf16>)
      %c16384 = arith.constant 16384 : index
      %8 = affine.apply #map3()[%3, %c16384]
      %c2_92 = arith.constant 2 : index
      %c0_93 = arith.constant 0 : index
      %c0_94 = arith.constant 0 : index
      %c0_95 = arith.constant 0 : index
      %c0_96 = arith.constant 0 : index
      %c0_97 = arith.constant 0 : index
      %c2_98 = arith.constant 2 : index
      %c8_99 = arith.constant 8 : index
      %c8_100 = arith.constant 8 : index
      %c8_101 = arith.constant 8 : index
      %c8_102 = arith.constant 8 : index
      %c4096_103 = arith.constant 4096 : index
      %c8_104 = arith.constant 8 : index
      %c512_105 = arith.constant 512 : index
      %c64_106 = arith.constant 64 : index
      %c1_107 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c2_92, %c0_93] (%arg9[%c0_94, %c0_95, %c0_96, %c0_97, %8] [%c2_98, %c8_99, %c8_100, %c8_101, %c8_102] [%c4096_103, %c8_104, %c512_105, %c64_106, %c1_107]) : (memref<2x512x64xbf16>)
      %c24576 = arith.constant 24576 : index
      %9 = affine.apply #map3()[%3, %c24576]
      %c3_108 = arith.constant 3 : index
      %c0_109 = arith.constant 0 : index
      %c0_110 = arith.constant 0 : index
      %c0_111 = arith.constant 0 : index
      %c0_112 = arith.constant 0 : index
      %c0_113 = arith.constant 0 : index
      %c2_114 = arith.constant 2 : index
      %c8_115 = arith.constant 8 : index
      %c8_116 = arith.constant 8 : index
      %c8_117 = arith.constant 8 : index
      %c8_118 = arith.constant 8 : index
      %c4096_119 = arith.constant 4096 : index
      %c8_120 = arith.constant 8 : index
      %c512_121 = arith.constant 512 : index
      %c64_122 = arith.constant 64 : index
      %c1_123 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c3_108, %c0_109] (%arg9[%c0_110, %c0_111, %c0_112, %c0_113, %9] [%c2_114, %c8_115, %c8_116, %c8_117, %c8_118] [%c4096_119, %c8_120, %c512_121, %c64_122, %c1_123]) : (memref<2x512x64xbf16>)
      %c0_124 = arith.constant 0 : index
      %10 = affine.apply #map3()[%4, %c0_124]
      %c0_125 = arith.constant 0 : index
      %c0_126 = arith.constant 0 : index
      %c2_127 = arith.constant 2 : index
      %c64_128 = arith.constant 64 : index
      %c64_129 = arith.constant 64 : index
      %c4096_130 = arith.constant 4096 : index
      %c64_131 = arith.constant 64 : index
      %c1_132 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c0] (%arg10[%c0_125, %c0_126, %10] [%c2_127, %c64_128, %c64_129] [%c4096_130, %c64_131, %c1_132]) : (memref<2x512x64xbf16>)
      %c8192_133 = arith.constant 8192 : index
      %11 = affine.apply #map3()[%4, %c8192_133]
      %c0_134 = arith.constant 0 : index
      %c0_135 = arith.constant 0 : index
      %c2_136 = arith.constant 2 : index
      %c64_137 = arith.constant 64 : index
      %c64_138 = arith.constant 64 : index
      %c4096_139 = arith.constant 4096 : index
      %c64_140 = arith.constant 64 : index
      %c1_141 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c0] (%arg10[%c0_134, %c0_135, %11] [%c2_136, %c64_137, %c64_138] [%c4096_139, %c64_140, %c1_141]) : (memref<2x512x64xbf16>)
      %c16384_142 = arith.constant 16384 : index
      %12 = affine.apply #map3()[%4, %c16384_142]
      %c0_143 = arith.constant 0 : index
      %c0_144 = arith.constant 0 : index
      %c2_145 = arith.constant 2 : index
      %c64_146 = arith.constant 64 : index
      %c64_147 = arith.constant 64 : index
      %c4096_148 = arith.constant 4096 : index
      %c64_149 = arith.constant 64 : index
      %c1_150 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c0] (%arg10[%c0_143, %c0_144, %12] [%c2_145, %c64_146, %c64_147] [%c4096_148, %c64_149, %c1_150]) : (memref<2x512x64xbf16>)
      %c24576_151 = arith.constant 24576 : index
      %13 = affine.apply #map3()[%4, %c24576_151]
      %c0_152 = arith.constant 0 : index
      %c0_153 = arith.constant 0 : index
      %c2_154 = arith.constant 2 : index
      %c64_155 = arith.constant 64 : index
      %c64_156 = arith.constant 64 : index
      %c4096_157 = arith.constant 4096 : index
      %c64_158 = arith.constant 64 : index
      %c1_159 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c0] (%arg10[%c0_152, %c0_153, %13] [%c2_154, %c64_155, %c64_156] [%c4096_157, %c64_158, %c1_159]) : (memref<2x512x64xbf16>)
      %c16384_160 = arith.constant 16384 : index
      %c1_161 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0] (%arg11[%5] [%c16384_160] [%c1_161]) : (memref<2x256x64xbf16>)
      %14 = affine.apply #map4()[%1]
      %15 = affine.apply #map()[%14]
      %16 = affine.apply #map2()[%14]
      %17 = affine.apply #map2()[%14]
      %c1_162 = arith.constant 1 : index
      %18 = affine.apply #map3()[%15, %0]
      %c0_163 = arith.constant 0 : index
      %c0_164 = arith.constant 0 : index
      %c0_165 = arith.constant 0 : index
      %c0_166 = arith.constant 0 : index
      %c0_167 = arith.constant 0 : index
      %c0_168 = arith.constant 0 : index
      %c4_169 = arith.constant 4 : index
      %c8_170 = arith.constant 8 : index
      %c8_171 = arith.constant 8 : index
      %c8_172 = arith.constant 8 : index
      %c8_173 = arith.constant 8 : index
      %c4096_174 = arith.constant 4096 : index
      %c8_175 = arith.constant 8 : index
      %c512_176 = arith.constant 512 : index
      %c64_177 = arith.constant 64 : index
      %c1_178 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_162, %c0_163, %c0_164] (%arg8[%c0_165, %c0_166, %c0_167, %c0_168, %18] [%c4_169, %c8_170, %c8_171, %c8_172, %c8_173] [%c4096_174, %c8_175, %c512_176, %c64_177, %c1_178]) : (memref<2x256x64xbf16>)
      %c1_179 = arith.constant 1 : index
      %c0_180 = arith.constant 0 : index
      %c0_181 = arith.constant 0 : index
      %c0_182 = arith.constant 0 : index
      %c0_183 = arith.constant 0 : index
      %c0_184 = arith.constant 0 : index
      %c4_185 = arith.constant 4 : index
      %c8_186 = arith.constant 8 : index
      %c8_187 = arith.constant 8 : index
      %c8_188 = arith.constant 8 : index
      %c8_189 = arith.constant 8 : index
      %c4096_190 = arith.constant 4096 : index
      %c8_191 = arith.constant 8 : index
      %c512_192 = arith.constant 512 : index
      %c64_193 = arith.constant 64 : index
      %c1_194 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_162, %c1_179, %c0_180] (%arg8[%c0_181, %c0_182, %c0_183, %c0_184, %18] [%c4_185, %c8_186, %c8_187, %c8_188, %c8_189] [%c4096_190, %c8_191, %c512_192, %c64_193, %c1_194]) : (memref<2x256x64xbf16>)
      %c2_195 = arith.constant 2 : index
      %c0_196 = arith.constant 0 : index
      %c0_197 = arith.constant 0 : index
      %c0_198 = arith.constant 0 : index
      %c0_199 = arith.constant 0 : index
      %c0_200 = arith.constant 0 : index
      %c4_201 = arith.constant 4 : index
      %c8_202 = arith.constant 8 : index
      %c8_203 = arith.constant 8 : index
      %c8_204 = arith.constant 8 : index
      %c8_205 = arith.constant 8 : index
      %c4096_206 = arith.constant 4096 : index
      %c8_207 = arith.constant 8 : index
      %c512_208 = arith.constant 512 : index
      %c64_209 = arith.constant 64 : index
      %c1_210 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_162, %c2_195, %c0_196] (%arg8[%c0_197, %c0_198, %c0_199, %c0_200, %18] [%c4_201, %c8_202, %c8_203, %c8_204, %c8_205] [%c4096_206, %c8_207, %c512_208, %c64_209, %c1_210]) : (memref<2x256x64xbf16>)
      %c3_211 = arith.constant 3 : index
      %c0_212 = arith.constant 0 : index
      %c0_213 = arith.constant 0 : index
      %c0_214 = arith.constant 0 : index
      %c0_215 = arith.constant 0 : index
      %c0_216 = arith.constant 0 : index
      %c4_217 = arith.constant 4 : index
      %c8_218 = arith.constant 8 : index
      %c8_219 = arith.constant 8 : index
      %c8_220 = arith.constant 8 : index
      %c8_221 = arith.constant 8 : index
      %c4096_222 = arith.constant 4096 : index
      %c8_223 = arith.constant 8 : index
      %c512_224 = arith.constant 512 : index
      %c64_225 = arith.constant 64 : index
      %c1_226 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_162, %c3_211, %c0_212] (%arg8[%c0_213, %c0_214, %c0_215, %c0_216, %18] [%c4_217, %c8_218, %c8_219, %c8_220, %c8_221] [%c4096_222, %c8_223, %c512_224, %c64_225, %c1_226]) : (memref<2x256x64xbf16>)
      %c0_227 = arith.constant 0 : index
      %19 = affine.apply #map3()[%16, %c0_227]
      %c0_228 = arith.constant 0 : index
      %c0_229 = arith.constant 0 : index
      %c0_230 = arith.constant 0 : index
      %c0_231 = arith.constant 0 : index
      %c0_232 = arith.constant 0 : index
      %c0_233 = arith.constant 0 : index
      %c2_234 = arith.constant 2 : index
      %c8_235 = arith.constant 8 : index
      %c8_236 = arith.constant 8 : index
      %c8_237 = arith.constant 8 : index
      %c8_238 = arith.constant 8 : index
      %c4096_239 = arith.constant 4096 : index
      %c8_240 = arith.constant 8 : index
      %c512_241 = arith.constant 512 : index
      %c64_242 = arith.constant 64 : index
      %c1_243 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_162, %c0_228, %c0_229] (%arg9[%c0_230, %c0_231, %c0_232, %c0_233, %19] [%c2_234, %c8_235, %c8_236, %c8_237, %c8_238] [%c4096_239, %c8_240, %c512_241, %c64_242, %c1_243]) : (memref<2x512x64xbf16>)
      %c8192_244 = arith.constant 8192 : index
      %20 = affine.apply #map3()[%16, %c8192_244]
      %c1_245 = arith.constant 1 : index
      %c0_246 = arith.constant 0 : index
      %c0_247 = arith.constant 0 : index
      %c0_248 = arith.constant 0 : index
      %c0_249 = arith.constant 0 : index
      %c0_250 = arith.constant 0 : index
      %c2_251 = arith.constant 2 : index
      %c8_252 = arith.constant 8 : index
      %c8_253 = arith.constant 8 : index
      %c8_254 = arith.constant 8 : index
      %c8_255 = arith.constant 8 : index
      %c4096_256 = arith.constant 4096 : index
      %c8_257 = arith.constant 8 : index
      %c512_258 = arith.constant 512 : index
      %c64_259 = arith.constant 64 : index
      %c1_260 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_162, %c1_245, %c0_246] (%arg9[%c0_247, %c0_248, %c0_249, %c0_250, %20] [%c2_251, %c8_252, %c8_253, %c8_254, %c8_255] [%c4096_256, %c8_257, %c512_258, %c64_259, %c1_260]) : (memref<2x512x64xbf16>)
      %c16384_261 = arith.constant 16384 : index
      %21 = affine.apply #map3()[%16, %c16384_261]
      %c2_262 = arith.constant 2 : index
      %c0_263 = arith.constant 0 : index
      %c0_264 = arith.constant 0 : index
      %c0_265 = arith.constant 0 : index
      %c0_266 = arith.constant 0 : index
      %c0_267 = arith.constant 0 : index
      %c2_268 = arith.constant 2 : index
      %c8_269 = arith.constant 8 : index
      %c8_270 = arith.constant 8 : index
      %c8_271 = arith.constant 8 : index
      %c8_272 = arith.constant 8 : index
      %c4096_273 = arith.constant 4096 : index
      %c8_274 = arith.constant 8 : index
      %c512_275 = arith.constant 512 : index
      %c64_276 = arith.constant 64 : index
      %c1_277 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_162, %c2_262, %c0_263] (%arg9[%c0_264, %c0_265, %c0_266, %c0_267, %21] [%c2_268, %c8_269, %c8_270, %c8_271, %c8_272] [%c4096_273, %c8_274, %c512_275, %c64_276, %c1_277]) : (memref<2x512x64xbf16>)
      %c24576_278 = arith.constant 24576 : index
      %22 = affine.apply #map3()[%16, %c24576_278]
      %c3_279 = arith.constant 3 : index
      %c0_280 = arith.constant 0 : index
      %c0_281 = arith.constant 0 : index
      %c0_282 = arith.constant 0 : index
      %c0_283 = arith.constant 0 : index
      %c0_284 = arith.constant 0 : index
      %c2_285 = arith.constant 2 : index
      %c8_286 = arith.constant 8 : index
      %c8_287 = arith.constant 8 : index
      %c8_288 = arith.constant 8 : index
      %c8_289 = arith.constant 8 : index
      %c4096_290 = arith.constant 4096 : index
      %c8_291 = arith.constant 8 : index
      %c512_292 = arith.constant 512 : index
      %c64_293 = arith.constant 64 : index
      %c1_294 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_162, %c3_279, %c0_280] (%arg9[%c0_281, %c0_282, %c0_283, %c0_284, %22] [%c2_285, %c8_286, %c8_287, %c8_288, %c8_289] [%c4096_290, %c8_291, %c512_292, %c64_293, %c1_294]) : (memref<2x512x64xbf16>)
      %c0_295 = arith.constant 0 : index
      %23 = affine.apply #map3()[%17, %c0_295]
      %c0_296 = arith.constant 0 : index
      %c0_297 = arith.constant 0 : index
      %c2_298 = arith.constant 2 : index
      %c64_299 = arith.constant 64 : index
      %c64_300 = arith.constant 64 : index
      %c4096_301 = arith.constant 4096 : index
      %c64_302 = arith.constant 64 : index
      %c1_303 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c1_162] (%arg10[%c0_296, %c0_297, %23] [%c2_298, %c64_299, %c64_300] [%c4096_301, %c64_302, %c1_303]) : (memref<2x512x64xbf16>)
      %c8192_304 = arith.constant 8192 : index
      %24 = affine.apply #map3()[%17, %c8192_304]
      %c0_305 = arith.constant 0 : index
      %c0_306 = arith.constant 0 : index
      %c2_307 = arith.constant 2 : index
      %c64_308 = arith.constant 64 : index
      %c64_309 = arith.constant 64 : index
      %c4096_310 = arith.constant 4096 : index
      %c64_311 = arith.constant 64 : index
      %c1_312 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c1_162] (%arg10[%c0_305, %c0_306, %24] [%c2_307, %c64_308, %c64_309] [%c4096_310, %c64_311, %c1_312]) : (memref<2x512x64xbf16>)
      %c16384_313 = arith.constant 16384 : index
      %25 = affine.apply #map3()[%17, %c16384_313]
      %c0_314 = arith.constant 0 : index
      %c0_315 = arith.constant 0 : index
      %c2_316 = arith.constant 2 : index
      %c64_317 = arith.constant 64 : index
      %c64_318 = arith.constant 64 : index
      %c4096_319 = arith.constant 4096 : index
      %c64_320 = arith.constant 64 : index
      %c1_321 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c1_162] (%arg10[%c0_314, %c0_315, %25] [%c2_316, %c64_317, %c64_318] [%c4096_319, %c64_320, %c1_321]) : (memref<2x512x64xbf16>)
      %c24576_322 = arith.constant 24576 : index
      %26 = affine.apply #map3()[%17, %c24576_322]
      %c0_323 = arith.constant 0 : index
      %c0_324 = arith.constant 0 : index
      %c2_325 = arith.constant 2 : index
      %c64_326 = arith.constant 64 : index
      %c64_327 = arith.constant 64 : index
      %c4096_328 = arith.constant 4096 : index
      %c64_329 = arith.constant 64 : index
      %c1_330 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c1_162] (%arg10[%c0_323, %c0_324, %26] [%c2_325, %c64_326, %c64_327] [%c4096_328, %c64_329, %c1_330]) : (memref<2x512x64xbf16>)
      %c16384_331 = arith.constant 16384 : index
      %c1_332 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_162] (%arg11[%18] [%c16384_331] [%c1_332]) : (memref<2x256x64xbf16>)
      %c2_333 = arith.constant 2 : index
      %c1_334 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2_333, %arg15=%c1_334) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_335 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_336 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_337 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_338 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_339 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_340 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_341 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_342 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_343 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_344 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_345 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %c4_346 = arith.constant 4 : index
        %c4_347 = arith.constant 4 : index
        %c0_348 = arith.constant 0 : index
        %c2_349 = arith.constant 2 : index
        %c0_350 = arith.constant 0 : index
        %c1_351 = arith.constant 1 : index
        scf.for %arg16 = %c0_350 to %c2_349 step %c1_351 {
          air.channel.get  @VIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_358 = arith.constant 0 : index
          %c0_359 = arith.constant 0 : index
          %c0_360 = arith.constant 0 : index
          %c0_361 = arith.constant 0 : index
          %c8_362 = arith.constant 8 : index
          %c8_363 = arith.constant 8 : index
          %c8_364 = arith.constant 8 : index
          %c8_365 = arith.constant 8 : index
          %c8_366 = arith.constant 8 : index
          %c512_367 = arith.constant 512 : index
          %c64_368 = arith.constant 64 : index
          %c1_369 = arith.constant 1 : index
          air.channel.put  @V2L1_0[%arg12, %c0_348, %c0_348] (%alloc[%c0_358, %c0_359, %c0_360, %c0_361] [%c8_362, %c8_363, %c8_364, %c8_365] [%c8_366, %c512_367, %c64_368, %c1_369]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_352 = arith.constant 0 : index
        %c1_353 = arith.constant 1 : index
        scf.for %arg16 = %c0_352 to %c2_349 step %c1_353 {
          air.channel.get  @VIn_1[%arg12] (%alloc_335[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_358 = arith.constant 0 : index
          %c0_359 = arith.constant 0 : index
          %c0_360 = arith.constant 0 : index
          %c0_361 = arith.constant 0 : index
          %c8_362 = arith.constant 8 : index
          %c8_363 = arith.constant 8 : index
          %c8_364 = arith.constant 8 : index
          %c8_365 = arith.constant 8 : index
          %c8_366 = arith.constant 8 : index
          %c512_367 = arith.constant 512 : index
          %c64_368 = arith.constant 64 : index
          %c1_369 = arith.constant 1 : index
          air.channel.put  @V2L1_1[%arg12, %c0_348, %c0_348] (%alloc_335[%c0_358, %c0_359, %c0_360, %c0_361] [%c8_362, %c8_363, %c8_364, %c8_365] [%c8_366, %c512_367, %c64_368, %c1_369]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_354 = arith.constant 0 : index
        %c1_355 = arith.constant 1 : index
        scf.for %arg16 = %c0_354 to %c2_349 step %c1_355 {
          air.channel.get  @VIn_2[%arg12] (%alloc_336[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_358 = arith.constant 0 : index
          %c0_359 = arith.constant 0 : index
          %c0_360 = arith.constant 0 : index
          %c0_361 = arith.constant 0 : index
          %c8_362 = arith.constant 8 : index
          %c8_363 = arith.constant 8 : index
          %c8_364 = arith.constant 8 : index
          %c8_365 = arith.constant 8 : index
          %c8_366 = arith.constant 8 : index
          %c512_367 = arith.constant 512 : index
          %c64_368 = arith.constant 64 : index
          %c1_369 = arith.constant 1 : index
          air.channel.put  @V2L1_2[%arg12, %c0_348, %c0_348] (%alloc_336[%c0_358, %c0_359, %c0_360, %c0_361] [%c8_362, %c8_363, %c8_364, %c8_365] [%c8_366, %c512_367, %c64_368, %c1_369]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_356 = arith.constant 0 : index
        %c1_357 = arith.constant 1 : index
        scf.for %arg16 = %c0_356 to %c2_349 step %c1_357 {
          air.channel.get  @VIn_3[%arg12] (%alloc_337[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_358 = arith.constant 0 : index
          %c0_359 = arith.constant 0 : index
          %c0_360 = arith.constant 0 : index
          %c0_361 = arith.constant 0 : index
          %c8_362 = arith.constant 8 : index
          %c8_363 = arith.constant 8 : index
          %c8_364 = arith.constant 8 : index
          %c8_365 = arith.constant 8 : index
          %c8_366 = arith.constant 8 : index
          %c512_367 = arith.constant 512 : index
          %c64_368 = arith.constant 64 : index
          %c1_369 = arith.constant 1 : index
          air.channel.put  @V2L1_3[%arg12, %c0_348, %c0_348] (%alloc_337[%c0_358, %c0_359, %c0_360, %c0_361] [%c8_362, %c8_363, %c8_364, %c8_365] [%c8_366, %c512_367, %c64_368, %c1_369]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %27 = affine.apply #map5()[%arg16]
          %c0_358 = arith.constant 0 : index
          %c0_359 = arith.constant 0 : index
          %c64_360 = arith.constant 64 : index
          %c64_361 = arith.constant 64 : index
          %c64_362 = arith.constant 64 : index
          %c1_363 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%arg16, %c0_358] (%alloc_338[%27, %c0_359] [%c64_360, %c64_361] [%c64_362, %c1_363]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_338[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4_346, %arg19=%c4_347) args(%arg20=%alloc_339, %arg21=%alloc_340, %arg22=%alloc_341, %arg23=%alloc_342, %arg24=%alloc_343, %arg25=%alloc_344, %arg26=%alloc_345, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn.o"} {
          func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          air.channel.get  @QK2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %27 = arith.index_cast %arg16 : index to i32
          %c0_i32 = arith.constant 0 : i32
          %28 = arith.cmpi eq, %27, %c0_i32 : i32
          scf.if %28 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @QK2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %29 = arith.index_cast %arg16 : index to i32
          %c1_i32 = arith.constant 1 : i32
          %30 = arith.cmpi eq, %29, %c1_i32 : i32
          scf.if %30 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @QK2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %31 = arith.index_cast %arg16 : index to i32
          %c2_i32 = arith.constant 2 : i32
          %32 = arith.cmpi eq, %31, %c2_i32 : i32
          scf.if %32 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @QK2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %33 = arith.index_cast %arg16 : index to i32
          %c3_i32 = arith.constant 3 : i32
          %34 = arith.cmpi eq, %33, %c3_i32 : i32
          scf.if %34 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %c2_358 = arith.constant 2 : index
          %c0_359 = arith.constant 0 : index
          %c1_360 = arith.constant 1 : index
          scf.for %arg28 = %c0_359 to %c2_358 step %c1_360 {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            air.channel.get  @QK2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
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
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            %alloc_362 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_363 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %alloc_362, %alloc_363) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_363, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32_364 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg26, %alloc_363, %alloc_362) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32_364, %alloc_362, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_362 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_363 : memref<64x1xbf16, 2 : i32>
          }
          %c1_361 = arith.constant 1 : index
          affine.if #set3()[%arg16, %arg17] {
            %35 = arith.subi %arg17, %c1_361 : index
            air.channel.put  @cascade_gp[%arg16, %35] (%arg24[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %35] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %35] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_362 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_363 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_364 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_362[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_363[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_364[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_365 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_366 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_366, %arg25, %alloc_365) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_363, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_367 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_363, %arg25, %alloc_367) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_368 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_365, %arg25, %alloc_368) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_367, %alloc_362) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_368, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_362) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_369 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_369) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_364, %alloc_367, %alloc_369) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_368, %alloc_369) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_366, %alloc_369, %alloc_364) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %35 = arith.subi %arg17, %c1_361 : index
              air.channel.put  @cascade_gp[%arg16, %35] (%alloc_362[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %35] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %35] (%alloc_364[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_362 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_363 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_364 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_365 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_367 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_368 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_369 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_362 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_363 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_364 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_362[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_363[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_364[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_365 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_366 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_366, %arg25, %alloc_365) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_363, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_367 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_363, %arg25, %alloc_367) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_368 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_365, %arg25, %alloc_368) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_367, %alloc_362) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_368, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_362) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_369 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_369) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_364, %alloc_367, %alloc_369) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_368, %alloc_369) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_366, %alloc_369, %alloc_364) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_364, %alloc_362) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %c0_370 = arith.constant 0 : index
              %c0_371 = arith.constant 0 : index
              %c0_372 = arith.constant 0 : index
              %c0_373 = arith.constant 0 : index
              %c0_374 = arith.constant 0 : index
              %c8_375 = arith.constant 8 : index
              %c8_376 = arith.constant 8 : index
              %c8_377 = arith.constant 8 : index
              %c8_378 = arith.constant 8 : index
              %c64_379 = arith.constant 64 : index
              %c8_380 = arith.constant 8 : index
              %c512_381 = arith.constant 512 : index
              %c1_382 = arith.constant 1 : index
              air.channel.put  @Gp2L2[%arg16, %c0_370] (%alloc_362[%c0_371, %c0_372, %c0_373, %c0_374] [%c8_375, %c8_376, %c8_377, %c8_378] [%c64_379, %c8_380, %c512_381, %c1_382]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_362 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_363 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_364 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_365 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_367 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_368 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_369 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_339 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_340 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_341 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_342 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_343 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_344 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_345 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_335 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_336 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_337 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_338 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}

