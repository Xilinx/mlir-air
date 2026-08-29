#map = affine_map<()[s0] -> (s0 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 2)>
#map2 = affine_map<()[s0] -> (s0 * 131072)>
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
  func.func @attention_bf16(%arg0: memref<12x2048x64xbf16>, %arg1: memref<12x2048x64xbf16>, %arg2: memref<12x2048x64xbf16>, %arg3: memref<12x2048x64xbf16>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c6 = arith.constant 6 : index
    air.launch (%arg4, %arg5) in (%arg6=%c8, %arg7=%c6) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<12x2048x64xbf16>, memref<12x2048x64xbf16>, memref<12x2048x64xbf16>, memref<12x2048x64xbf16> {
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map2()[%1]
      %3 = affine.apply #map2()[%1]
      %4 = affine.apply #map2()[%1]
      %c0 = arith.constant 0 : index
      %5 = affine.apply #map3()[%2, %0]
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %c0_5 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8_6 = arith.constant 8 : index
      %c8_7 = arith.constant 8 : index
      %c8_8 = arith.constant 8 : index
      %c8_9 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c8_10 = arith.constant 8 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c1_11 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c0_0, %c0_1] (%arg8[%c0_2, %c0_3, %c0_4, %c0_5, %5] [%c4, %c8_6, %c8_7, %c8_8, %c8_9] [%c4096, %c8_10, %c512, %c64, %c1_11]) : (memref<12x2048x64xbf16>)
      %c1_12 = arith.constant 1 : index
      %c0_13 = arith.constant 0 : index
      %c0_14 = arith.constant 0 : index
      %c0_15 = arith.constant 0 : index
      %c0_16 = arith.constant 0 : index
      %c0_17 = arith.constant 0 : index
      %c4_18 = arith.constant 4 : index
      %c8_19 = arith.constant 8 : index
      %c8_20 = arith.constant 8 : index
      %c8_21 = arith.constant 8 : index
      %c8_22 = arith.constant 8 : index
      %c4096_23 = arith.constant 4096 : index
      %c8_24 = arith.constant 8 : index
      %c512_25 = arith.constant 512 : index
      %c64_26 = arith.constant 64 : index
      %c1_27 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c1_12, %c0_13] (%arg8[%c0_14, %c0_15, %c0_16, %c0_17, %5] [%c4_18, %c8_19, %c8_20, %c8_21, %c8_22] [%c4096_23, %c8_24, %c512_25, %c64_26, %c1_27]) : (memref<12x2048x64xbf16>)
      %c2 = arith.constant 2 : index
      %c0_28 = arith.constant 0 : index
      %c0_29 = arith.constant 0 : index
      %c0_30 = arith.constant 0 : index
      %c0_31 = arith.constant 0 : index
      %c0_32 = arith.constant 0 : index
      %c4_33 = arith.constant 4 : index
      %c8_34 = arith.constant 8 : index
      %c8_35 = arith.constant 8 : index
      %c8_36 = arith.constant 8 : index
      %c8_37 = arith.constant 8 : index
      %c4096_38 = arith.constant 4096 : index
      %c8_39 = arith.constant 8 : index
      %c512_40 = arith.constant 512 : index
      %c64_41 = arith.constant 64 : index
      %c1_42 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c2, %c0_28] (%arg8[%c0_29, %c0_30, %c0_31, %c0_32, %5] [%c4_33, %c8_34, %c8_35, %c8_36, %c8_37] [%c4096_38, %c8_39, %c512_40, %c64_41, %c1_42]) : (memref<12x2048x64xbf16>)
      %c3 = arith.constant 3 : index
      %c0_43 = arith.constant 0 : index
      %c0_44 = arith.constant 0 : index
      %c0_45 = arith.constant 0 : index
      %c0_46 = arith.constant 0 : index
      %c0_47 = arith.constant 0 : index
      %c4_48 = arith.constant 4 : index
      %c8_49 = arith.constant 8 : index
      %c8_50 = arith.constant 8 : index
      %c8_51 = arith.constant 8 : index
      %c8_52 = arith.constant 8 : index
      %c4096_53 = arith.constant 4096 : index
      %c8_54 = arith.constant 8 : index
      %c512_55 = arith.constant 512 : index
      %c64_56 = arith.constant 64 : index
      %c1_57 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c3, %c0_43] (%arg8[%c0_44, %c0_45, %c0_46, %c0_47, %5] [%c4_48, %c8_49, %c8_50, %c8_51, %c8_52] [%c4096_53, %c8_54, %c512_55, %c64_56, %c1_57]) : (memref<12x2048x64xbf16>)
      %c0_58 = arith.constant 0 : index
      %6 = affine.apply #map3()[%3, %c0_58]
      %c0_59 = arith.constant 0 : index
      %c0_60 = arith.constant 0 : index
      %c0_61 = arith.constant 0 : index
      %c0_62 = arith.constant 0 : index
      %c0_63 = arith.constant 0 : index
      %c0_64 = arith.constant 0 : index
      %c8_65 = arith.constant 8 : index
      %c8_66 = arith.constant 8 : index
      %c8_67 = arith.constant 8 : index
      %c8_68 = arith.constant 8 : index
      %c8_69 = arith.constant 8 : index
      %c4096_70 = arith.constant 4096 : index
      %c8_71 = arith.constant 8 : index
      %c512_72 = arith.constant 512 : index
      %c64_73 = arith.constant 64 : index
      %c1_74 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c0_59, %c0_60] (%arg9[%c0_61, %c0_62, %c0_63, %c0_64, %6] [%c8_65, %c8_66, %c8_67, %c8_68, %c8_69] [%c4096_70, %c8_71, %c512_72, %c64_73, %c1_74]) : (memref<12x2048x64xbf16>)
      %c32768 = arith.constant 32768 : index
      %7 = affine.apply #map3()[%3, %c32768]
      %c1_75 = arith.constant 1 : index
      %c0_76 = arith.constant 0 : index
      %c0_77 = arith.constant 0 : index
      %c0_78 = arith.constant 0 : index
      %c0_79 = arith.constant 0 : index
      %c0_80 = arith.constant 0 : index
      %c8_81 = arith.constant 8 : index
      %c8_82 = arith.constant 8 : index
      %c8_83 = arith.constant 8 : index
      %c8_84 = arith.constant 8 : index
      %c8_85 = arith.constant 8 : index
      %c4096_86 = arith.constant 4096 : index
      %c8_87 = arith.constant 8 : index
      %c512_88 = arith.constant 512 : index
      %c64_89 = arith.constant 64 : index
      %c1_90 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c1_75, %c0_76] (%arg9[%c0_77, %c0_78, %c0_79, %c0_80, %7] [%c8_81, %c8_82, %c8_83, %c8_84, %c8_85] [%c4096_86, %c8_87, %c512_88, %c64_89, %c1_90]) : (memref<12x2048x64xbf16>)
      %c65536 = arith.constant 65536 : index
      %8 = affine.apply #map3()[%3, %c65536]
      %c2_91 = arith.constant 2 : index
      %c0_92 = arith.constant 0 : index
      %c0_93 = arith.constant 0 : index
      %c0_94 = arith.constant 0 : index
      %c0_95 = arith.constant 0 : index
      %c0_96 = arith.constant 0 : index
      %c8_97 = arith.constant 8 : index
      %c8_98 = arith.constant 8 : index
      %c8_99 = arith.constant 8 : index
      %c8_100 = arith.constant 8 : index
      %c8_101 = arith.constant 8 : index
      %c4096_102 = arith.constant 4096 : index
      %c8_103 = arith.constant 8 : index
      %c512_104 = arith.constant 512 : index
      %c64_105 = arith.constant 64 : index
      %c1_106 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c2_91, %c0_92] (%arg9[%c0_93, %c0_94, %c0_95, %c0_96, %8] [%c8_97, %c8_98, %c8_99, %c8_100, %c8_101] [%c4096_102, %c8_103, %c512_104, %c64_105, %c1_106]) : (memref<12x2048x64xbf16>)
      %c98304 = arith.constant 98304 : index
      %9 = affine.apply #map3()[%3, %c98304]
      %c3_107 = arith.constant 3 : index
      %c0_108 = arith.constant 0 : index
      %c0_109 = arith.constant 0 : index
      %c0_110 = arith.constant 0 : index
      %c0_111 = arith.constant 0 : index
      %c0_112 = arith.constant 0 : index
      %c8_113 = arith.constant 8 : index
      %c8_114 = arith.constant 8 : index
      %c8_115 = arith.constant 8 : index
      %c8_116 = arith.constant 8 : index
      %c8_117 = arith.constant 8 : index
      %c4096_118 = arith.constant 4096 : index
      %c8_119 = arith.constant 8 : index
      %c512_120 = arith.constant 512 : index
      %c64_121 = arith.constant 64 : index
      %c1_122 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c3_107, %c0_108] (%arg9[%c0_109, %c0_110, %c0_111, %c0_112, %9] [%c8_113, %c8_114, %c8_115, %c8_116, %c8_117] [%c4096_118, %c8_119, %c512_120, %c64_121, %c1_122]) : (memref<12x2048x64xbf16>)
      %c0_123 = arith.constant 0 : index
      %10 = affine.apply #map3()[%4, %c0_123]
      %c0_124 = arith.constant 0 : index
      %c0_125 = arith.constant 0 : index
      %c8_126 = arith.constant 8 : index
      %c64_127 = arith.constant 64 : index
      %c64_128 = arith.constant 64 : index
      %c4096_129 = arith.constant 4096 : index
      %c64_130 = arith.constant 64 : index
      %c1_131 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c0] (%arg10[%c0_124, %c0_125, %10] [%c8_126, %c64_127, %c64_128] [%c4096_129, %c64_130, %c1_131]) : (memref<12x2048x64xbf16>)
      %c32768_132 = arith.constant 32768 : index
      %11 = affine.apply #map3()[%4, %c32768_132]
      %c0_133 = arith.constant 0 : index
      %c0_134 = arith.constant 0 : index
      %c8_135 = arith.constant 8 : index
      %c64_136 = arith.constant 64 : index
      %c64_137 = arith.constant 64 : index
      %c4096_138 = arith.constant 4096 : index
      %c64_139 = arith.constant 64 : index
      %c1_140 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c0] (%arg10[%c0_133, %c0_134, %11] [%c8_135, %c64_136, %c64_137] [%c4096_138, %c64_139, %c1_140]) : (memref<12x2048x64xbf16>)
      %c65536_141 = arith.constant 65536 : index
      %12 = affine.apply #map3()[%4, %c65536_141]
      %c0_142 = arith.constant 0 : index
      %c0_143 = arith.constant 0 : index
      %c8_144 = arith.constant 8 : index
      %c64_145 = arith.constant 64 : index
      %c64_146 = arith.constant 64 : index
      %c4096_147 = arith.constant 4096 : index
      %c64_148 = arith.constant 64 : index
      %c1_149 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c0] (%arg10[%c0_142, %c0_143, %12] [%c8_144, %c64_145, %c64_146] [%c4096_147, %c64_148, %c1_149]) : (memref<12x2048x64xbf16>)
      %c98304_150 = arith.constant 98304 : index
      %13 = affine.apply #map3()[%4, %c98304_150]
      %c0_151 = arith.constant 0 : index
      %c0_152 = arith.constant 0 : index
      %c8_153 = arith.constant 8 : index
      %c64_154 = arith.constant 64 : index
      %c64_155 = arith.constant 64 : index
      %c4096_156 = arith.constant 4096 : index
      %c64_157 = arith.constant 64 : index
      %c1_158 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c0] (%arg10[%c0_151, %c0_152, %13] [%c8_153, %c64_154, %c64_155] [%c4096_156, %c64_157, %c1_158]) : (memref<12x2048x64xbf16>)
      %c16384 = arith.constant 16384 : index
      %c1_159 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0] (%arg11[%5] [%c16384] [%c1_159]) : (memref<12x2048x64xbf16>)
      %14 = affine.apply #map4()[%1]
      %15 = affine.apply #map2()[%14]
      %16 = affine.apply #map2()[%14]
      %17 = affine.apply #map2()[%14]
      %c1_160 = arith.constant 1 : index
      %18 = affine.apply #map3()[%15, %0]
      %c0_161 = arith.constant 0 : index
      %c0_162 = arith.constant 0 : index
      %c0_163 = arith.constant 0 : index
      %c0_164 = arith.constant 0 : index
      %c0_165 = arith.constant 0 : index
      %c0_166 = arith.constant 0 : index
      %c4_167 = arith.constant 4 : index
      %c8_168 = arith.constant 8 : index
      %c8_169 = arith.constant 8 : index
      %c8_170 = arith.constant 8 : index
      %c8_171 = arith.constant 8 : index
      %c4096_172 = arith.constant 4096 : index
      %c8_173 = arith.constant 8 : index
      %c512_174 = arith.constant 512 : index
      %c64_175 = arith.constant 64 : index
      %c1_176 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_160, %c0_161, %c0_162] (%arg8[%c0_163, %c0_164, %c0_165, %c0_166, %18] [%c4_167, %c8_168, %c8_169, %c8_170, %c8_171] [%c4096_172, %c8_173, %c512_174, %c64_175, %c1_176]) : (memref<12x2048x64xbf16>)
      %c1_177 = arith.constant 1 : index
      %c0_178 = arith.constant 0 : index
      %c0_179 = arith.constant 0 : index
      %c0_180 = arith.constant 0 : index
      %c0_181 = arith.constant 0 : index
      %c0_182 = arith.constant 0 : index
      %c4_183 = arith.constant 4 : index
      %c8_184 = arith.constant 8 : index
      %c8_185 = arith.constant 8 : index
      %c8_186 = arith.constant 8 : index
      %c8_187 = arith.constant 8 : index
      %c4096_188 = arith.constant 4096 : index
      %c8_189 = arith.constant 8 : index
      %c512_190 = arith.constant 512 : index
      %c64_191 = arith.constant 64 : index
      %c1_192 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_160, %c1_177, %c0_178] (%arg8[%c0_179, %c0_180, %c0_181, %c0_182, %18] [%c4_183, %c8_184, %c8_185, %c8_186, %c8_187] [%c4096_188, %c8_189, %c512_190, %c64_191, %c1_192]) : (memref<12x2048x64xbf16>)
      %c2_193 = arith.constant 2 : index
      %c0_194 = arith.constant 0 : index
      %c0_195 = arith.constant 0 : index
      %c0_196 = arith.constant 0 : index
      %c0_197 = arith.constant 0 : index
      %c0_198 = arith.constant 0 : index
      %c4_199 = arith.constant 4 : index
      %c8_200 = arith.constant 8 : index
      %c8_201 = arith.constant 8 : index
      %c8_202 = arith.constant 8 : index
      %c8_203 = arith.constant 8 : index
      %c4096_204 = arith.constant 4096 : index
      %c8_205 = arith.constant 8 : index
      %c512_206 = arith.constant 512 : index
      %c64_207 = arith.constant 64 : index
      %c1_208 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_160, %c2_193, %c0_194] (%arg8[%c0_195, %c0_196, %c0_197, %c0_198, %18] [%c4_199, %c8_200, %c8_201, %c8_202, %c8_203] [%c4096_204, %c8_205, %c512_206, %c64_207, %c1_208]) : (memref<12x2048x64xbf16>)
      %c3_209 = arith.constant 3 : index
      %c0_210 = arith.constant 0 : index
      %c0_211 = arith.constant 0 : index
      %c0_212 = arith.constant 0 : index
      %c0_213 = arith.constant 0 : index
      %c0_214 = arith.constant 0 : index
      %c4_215 = arith.constant 4 : index
      %c8_216 = arith.constant 8 : index
      %c8_217 = arith.constant 8 : index
      %c8_218 = arith.constant 8 : index
      %c8_219 = arith.constant 8 : index
      %c4096_220 = arith.constant 4096 : index
      %c8_221 = arith.constant 8 : index
      %c512_222 = arith.constant 512 : index
      %c64_223 = arith.constant 64 : index
      %c1_224 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_160, %c3_209, %c0_210] (%arg8[%c0_211, %c0_212, %c0_213, %c0_214, %18] [%c4_215, %c8_216, %c8_217, %c8_218, %c8_219] [%c4096_220, %c8_221, %c512_222, %c64_223, %c1_224]) : (memref<12x2048x64xbf16>)
      %c0_225 = arith.constant 0 : index
      %19 = affine.apply #map3()[%16, %c0_225]
      %c0_226 = arith.constant 0 : index
      %c0_227 = arith.constant 0 : index
      %c0_228 = arith.constant 0 : index
      %c0_229 = arith.constant 0 : index
      %c0_230 = arith.constant 0 : index
      %c0_231 = arith.constant 0 : index
      %c8_232 = arith.constant 8 : index
      %c8_233 = arith.constant 8 : index
      %c8_234 = arith.constant 8 : index
      %c8_235 = arith.constant 8 : index
      %c8_236 = arith.constant 8 : index
      %c4096_237 = arith.constant 4096 : index
      %c8_238 = arith.constant 8 : index
      %c512_239 = arith.constant 512 : index
      %c64_240 = arith.constant 64 : index
      %c1_241 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_160, %c0_226, %c0_227] (%arg9[%c0_228, %c0_229, %c0_230, %c0_231, %19] [%c8_232, %c8_233, %c8_234, %c8_235, %c8_236] [%c4096_237, %c8_238, %c512_239, %c64_240, %c1_241]) : (memref<12x2048x64xbf16>)
      %c32768_242 = arith.constant 32768 : index
      %20 = affine.apply #map3()[%16, %c32768_242]
      %c1_243 = arith.constant 1 : index
      %c0_244 = arith.constant 0 : index
      %c0_245 = arith.constant 0 : index
      %c0_246 = arith.constant 0 : index
      %c0_247 = arith.constant 0 : index
      %c0_248 = arith.constant 0 : index
      %c8_249 = arith.constant 8 : index
      %c8_250 = arith.constant 8 : index
      %c8_251 = arith.constant 8 : index
      %c8_252 = arith.constant 8 : index
      %c8_253 = arith.constant 8 : index
      %c4096_254 = arith.constant 4096 : index
      %c8_255 = arith.constant 8 : index
      %c512_256 = arith.constant 512 : index
      %c64_257 = arith.constant 64 : index
      %c1_258 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_160, %c1_243, %c0_244] (%arg9[%c0_245, %c0_246, %c0_247, %c0_248, %20] [%c8_249, %c8_250, %c8_251, %c8_252, %c8_253] [%c4096_254, %c8_255, %c512_256, %c64_257, %c1_258]) : (memref<12x2048x64xbf16>)
      %c65536_259 = arith.constant 65536 : index
      %21 = affine.apply #map3()[%16, %c65536_259]
      %c2_260 = arith.constant 2 : index
      %c0_261 = arith.constant 0 : index
      %c0_262 = arith.constant 0 : index
      %c0_263 = arith.constant 0 : index
      %c0_264 = arith.constant 0 : index
      %c0_265 = arith.constant 0 : index
      %c8_266 = arith.constant 8 : index
      %c8_267 = arith.constant 8 : index
      %c8_268 = arith.constant 8 : index
      %c8_269 = arith.constant 8 : index
      %c8_270 = arith.constant 8 : index
      %c4096_271 = arith.constant 4096 : index
      %c8_272 = arith.constant 8 : index
      %c512_273 = arith.constant 512 : index
      %c64_274 = arith.constant 64 : index
      %c1_275 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_160, %c2_260, %c0_261] (%arg9[%c0_262, %c0_263, %c0_264, %c0_265, %21] [%c8_266, %c8_267, %c8_268, %c8_269, %c8_270] [%c4096_271, %c8_272, %c512_273, %c64_274, %c1_275]) : (memref<12x2048x64xbf16>)
      %c98304_276 = arith.constant 98304 : index
      %22 = affine.apply #map3()[%16, %c98304_276]
      %c3_277 = arith.constant 3 : index
      %c0_278 = arith.constant 0 : index
      %c0_279 = arith.constant 0 : index
      %c0_280 = arith.constant 0 : index
      %c0_281 = arith.constant 0 : index
      %c0_282 = arith.constant 0 : index
      %c8_283 = arith.constant 8 : index
      %c8_284 = arith.constant 8 : index
      %c8_285 = arith.constant 8 : index
      %c8_286 = arith.constant 8 : index
      %c8_287 = arith.constant 8 : index
      %c4096_288 = arith.constant 4096 : index
      %c8_289 = arith.constant 8 : index
      %c512_290 = arith.constant 512 : index
      %c64_291 = arith.constant 64 : index
      %c1_292 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c1_160, %c3_277, %c0_278] (%arg9[%c0_279, %c0_280, %c0_281, %c0_282, %22] [%c8_283, %c8_284, %c8_285, %c8_286, %c8_287] [%c4096_288, %c8_289, %c512_290, %c64_291, %c1_292]) : (memref<12x2048x64xbf16>)
      %c0_293 = arith.constant 0 : index
      %23 = affine.apply #map3()[%17, %c0_293]
      %c0_294 = arith.constant 0 : index
      %c0_295 = arith.constant 0 : index
      %c8_296 = arith.constant 8 : index
      %c64_297 = arith.constant 64 : index
      %c64_298 = arith.constant 64 : index
      %c4096_299 = arith.constant 4096 : index
      %c64_300 = arith.constant 64 : index
      %c1_301 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c1_160] (%arg10[%c0_294, %c0_295, %23] [%c8_296, %c64_297, %c64_298] [%c4096_299, %c64_300, %c1_301]) : (memref<12x2048x64xbf16>)
      %c32768_302 = arith.constant 32768 : index
      %24 = affine.apply #map3()[%17, %c32768_302]
      %c0_303 = arith.constant 0 : index
      %c0_304 = arith.constant 0 : index
      %c8_305 = arith.constant 8 : index
      %c64_306 = arith.constant 64 : index
      %c64_307 = arith.constant 64 : index
      %c4096_308 = arith.constant 4096 : index
      %c64_309 = arith.constant 64 : index
      %c1_310 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c1_160] (%arg10[%c0_303, %c0_304, %24] [%c8_305, %c64_306, %c64_307] [%c4096_308, %c64_309, %c1_310]) : (memref<12x2048x64xbf16>)
      %c65536_311 = arith.constant 65536 : index
      %25 = affine.apply #map3()[%17, %c65536_311]
      %c0_312 = arith.constant 0 : index
      %c0_313 = arith.constant 0 : index
      %c8_314 = arith.constant 8 : index
      %c64_315 = arith.constant 64 : index
      %c64_316 = arith.constant 64 : index
      %c4096_317 = arith.constant 4096 : index
      %c64_318 = arith.constant 64 : index
      %c1_319 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c1_160] (%arg10[%c0_312, %c0_313, %25] [%c8_314, %c64_315, %c64_316] [%c4096_317, %c64_318, %c1_319]) : (memref<12x2048x64xbf16>)
      %c98304_320 = arith.constant 98304 : index
      %26 = affine.apply #map3()[%17, %c98304_320]
      %c0_321 = arith.constant 0 : index
      %c0_322 = arith.constant 0 : index
      %c8_323 = arith.constant 8 : index
      %c64_324 = arith.constant 64 : index
      %c64_325 = arith.constant 64 : index
      %c4096_326 = arith.constant 4096 : index
      %c64_327 = arith.constant 64 : index
      %c1_328 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c1_160] (%arg10[%c0_321, %c0_322, %26] [%c8_323, %c64_324, %c64_325] [%c4096_326, %c64_327, %c1_328]) : (memref<12x2048x64xbf16>)
      %c16384_329 = arith.constant 16384 : index
      %c1_330 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_160] (%arg11[%18] [%c16384_329] [%c1_330]) : (memref<12x2048x64xbf16>)
      %c2_331 = arith.constant 2 : index
      %c1_332 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2_331, %arg15=%c1_332) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_333 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_334 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_335 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_336 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_337 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_338 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_339 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_340 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_341 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_342 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_343 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %c4_344 = arith.constant 4 : index
        %c4_345 = arith.constant 4 : index
        %c0_346 = arith.constant 0 : index
        %c8_347 = arith.constant 8 : index
        %c0_348 = arith.constant 0 : index
        %c1_349 = arith.constant 1 : index
        scf.for %arg16 = %c0_348 to %c8_347 step %c1_349 {
          air.channel.get  @VIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_356 = arith.constant 0 : index
          %c0_357 = arith.constant 0 : index
          %c0_358 = arith.constant 0 : index
          %c0_359 = arith.constant 0 : index
          %c8_360 = arith.constant 8 : index
          %c8_361 = arith.constant 8 : index
          %c8_362 = arith.constant 8 : index
          %c8_363 = arith.constant 8 : index
          %c8_364 = arith.constant 8 : index
          %c512_365 = arith.constant 512 : index
          %c64_366 = arith.constant 64 : index
          %c1_367 = arith.constant 1 : index
          air.channel.put  @V2L1_0[%arg12, %c0_346, %c0_346] (%alloc[%c0_356, %c0_357, %c0_358, %c0_359] [%c8_360, %c8_361, %c8_362, %c8_363] [%c8_364, %c512_365, %c64_366, %c1_367]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_350 = arith.constant 0 : index
        %c1_351 = arith.constant 1 : index
        scf.for %arg16 = %c0_350 to %c8_347 step %c1_351 {
          air.channel.get  @VIn_1[%arg12] (%alloc_333[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_356 = arith.constant 0 : index
          %c0_357 = arith.constant 0 : index
          %c0_358 = arith.constant 0 : index
          %c0_359 = arith.constant 0 : index
          %c8_360 = arith.constant 8 : index
          %c8_361 = arith.constant 8 : index
          %c8_362 = arith.constant 8 : index
          %c8_363 = arith.constant 8 : index
          %c8_364 = arith.constant 8 : index
          %c512_365 = arith.constant 512 : index
          %c64_366 = arith.constant 64 : index
          %c1_367 = arith.constant 1 : index
          air.channel.put  @V2L1_1[%arg12, %c0_346, %c0_346] (%alloc_333[%c0_356, %c0_357, %c0_358, %c0_359] [%c8_360, %c8_361, %c8_362, %c8_363] [%c8_364, %c512_365, %c64_366, %c1_367]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_352 = arith.constant 0 : index
        %c1_353 = arith.constant 1 : index
        scf.for %arg16 = %c0_352 to %c8_347 step %c1_353 {
          air.channel.get  @VIn_2[%arg12] (%alloc_334[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_356 = arith.constant 0 : index
          %c0_357 = arith.constant 0 : index
          %c0_358 = arith.constant 0 : index
          %c0_359 = arith.constant 0 : index
          %c8_360 = arith.constant 8 : index
          %c8_361 = arith.constant 8 : index
          %c8_362 = arith.constant 8 : index
          %c8_363 = arith.constant 8 : index
          %c8_364 = arith.constant 8 : index
          %c512_365 = arith.constant 512 : index
          %c64_366 = arith.constant 64 : index
          %c1_367 = arith.constant 1 : index
          air.channel.put  @V2L1_2[%arg12, %c0_346, %c0_346] (%alloc_334[%c0_356, %c0_357, %c0_358, %c0_359] [%c8_360, %c8_361, %c8_362, %c8_363] [%c8_364, %c512_365, %c64_366, %c1_367]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_354 = arith.constant 0 : index
        %c1_355 = arith.constant 1 : index
        scf.for %arg16 = %c0_354 to %c8_347 step %c1_355 {
          air.channel.get  @VIn_3[%arg12] (%alloc_335[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_356 = arith.constant 0 : index
          %c0_357 = arith.constant 0 : index
          %c0_358 = arith.constant 0 : index
          %c0_359 = arith.constant 0 : index
          %c8_360 = arith.constant 8 : index
          %c8_361 = arith.constant 8 : index
          %c8_362 = arith.constant 8 : index
          %c8_363 = arith.constant 8 : index
          %c8_364 = arith.constant 8 : index
          %c512_365 = arith.constant 512 : index
          %c64_366 = arith.constant 64 : index
          %c1_367 = arith.constant 1 : index
          air.channel.put  @V2L1_3[%arg12, %c0_346, %c0_346] (%alloc_335[%c0_356, %c0_357, %c0_358, %c0_359] [%c8_360, %c8_361, %c8_362, %c8_363] [%c8_364, %c512_365, %c64_366, %c1_367]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %27 = affine.apply #map5()[%arg16]
          %c0_356 = arith.constant 0 : index
          %c0_357 = arith.constant 0 : index
          %c64_358 = arith.constant 64 : index
          %c64_359 = arith.constant 64 : index
          %c64_360 = arith.constant 64 : index
          %c1_361 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%arg16, %c0_356] (%alloc_336[%27, %c0_357] [%c64_358, %c64_359] [%c64_360, %c1_361]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_336[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4_344, %arg19=%c4_345) args(%arg20=%alloc_337, %arg21=%alloc_338, %arg22=%alloc_339, %arg23=%alloc_340, %arg24=%alloc_341, %arg25=%alloc_342, %arg26=%alloc_343, %arg27=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn.o"} {
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
          %c8_356 = arith.constant 8 : index
          %c0_357 = arith.constant 0 : index
          %c1_358 = arith.constant 1 : index
          scf.for %arg28 = %c0_357 to %c8_356 step %c1_358 {
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
            %alloc_360 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_361 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %alloc_360, %alloc_361) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_361, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32_362 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg26, %alloc_361, %alloc_360) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32_362, %alloc_360, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_360 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_361 : memref<64x1xbf16, 2 : i32>
          }
          %c1_359 = arith.constant 1 : index
          affine.if #set3()[%arg16, %arg17] {
            %35 = arith.subi %arg17, %c1_359 : index
            air.channel.put  @cascade_gp[%arg16, %35] (%arg24[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %35] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %35] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_360 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_361 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_362 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_360[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_361[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_362[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_363 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_364 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_364, %arg25, %alloc_363) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_361, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_365 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_361, %arg25, %alloc_365) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_366 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_363, %arg25, %alloc_366) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_365, %alloc_360) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_366, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_360) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_367 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_367) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_362, %alloc_365, %alloc_367) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_366, %alloc_367) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_364, %alloc_367, %alloc_362) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %35 = arith.subi %arg17, %c1_359 : index
              air.channel.put  @cascade_gp[%arg16, %35] (%alloc_360[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %35] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %35] (%alloc_362[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_360 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_361 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_362 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_363 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_365 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_366 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_367 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_360 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_361 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_362 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_360[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_361[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_362[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_363 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_364 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_364, %arg25, %alloc_363) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_361, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_365 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_361, %arg25, %alloc_365) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_366 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_363, %arg25, %alloc_366) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_365, %alloc_360) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_366, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_360) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_367 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_367) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_362, %alloc_365, %alloc_367) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_366, %alloc_367) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_364, %alloc_367, %alloc_362) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_362, %alloc_360) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %c0_368 = arith.constant 0 : index
              %c0_369 = arith.constant 0 : index
              %c0_370 = arith.constant 0 : index
              %c0_371 = arith.constant 0 : index
              %c0_372 = arith.constant 0 : index
              %c8_373 = arith.constant 8 : index
              %c8_374 = arith.constant 8 : index
              %c8_375 = arith.constant 8 : index
              %c8_376 = arith.constant 8 : index
              %c64_377 = arith.constant 64 : index
              %c8_378 = arith.constant 8 : index
              %c512_379 = arith.constant 512 : index
              %c1_380 = arith.constant 1 : index
              air.channel.put  @Gp2L2[%arg16, %c0_368] (%alloc_360[%c0_369, %c0_370, %c0_371, %c0_372] [%c8_373, %c8_374, %c8_375, %c8_376] [%c64_377, %c8_378, %c512_379, %c1_380]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_360 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_361 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_362 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_363 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_365 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_366 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_367 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_337 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_338 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_339 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_340 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_341 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_342 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_343 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_333 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_334 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_335 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_336 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}
