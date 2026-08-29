#map = affine_map<()[s0] -> (s0 * 64)>
#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s1 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 >= 0)>
module {
  func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  air.channel @QK_0 [4, 1] {channel_type = "dma_packet"}
  air.channel @QK_1 [4, 1] {channel_type = "dma_packet"}
  air.channel @V2L1_0 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_0 [1]
  air.channel @V2L1_1 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_1 [1]
  air.channel @cascade_gp [4] {channel_type = "cascade"}
  air.channel @cascade_up [4] {channel_type = "cascade"}
  air.channel @cascade_sp [4] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  air.channel @GpOut [1]
  func.func @full_4x2_direct(%arg0: memref<256x64xbf16>, %arg1: memref<256x64xbf16>, %arg2: memref<256x64xbf16>, %arg3: memref<256x64xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<256x64xbf16>, memref<256x64xbf16>, memref<256x64xbf16>, memref<256x64xbf16> {
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c8_5 = arith.constant 8 : index
      %c8_6 = arith.constant 8 : index
      %c8_7 = arith.constant 8 : index
      %c8_8 = arith.constant 8 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c1_9 = arith.constant 1 : index
      air.channel.put  @QK_0[%c0, %c0_0] (%arg8[%c0_1, %c0_2, %c0_3, %c0_4] [%c8, %c8_5, %c8_6, %c8_7] [%c8_8, %c512, %c64, %c1_9]) : (memref<256x64xbf16>)
      %c1_10 = arith.constant 1 : index
      %c0_11 = arith.constant 0 : index
      %c0_12 = arith.constant 0 : index
      %c0_13 = arith.constant 0 : index
      %c0_14 = arith.constant 0 : index
      %c4096 = arith.constant 4096 : index
      %c8_15 = arith.constant 8 : index
      %c8_16 = arith.constant 8 : index
      %c8_17 = arith.constant 8 : index
      %c8_18 = arith.constant 8 : index
      %c8_19 = arith.constant 8 : index
      %c512_20 = arith.constant 512 : index
      %c64_21 = arith.constant 64 : index
      %c1_22 = arith.constant 1 : index
      air.channel.put  @QK_0[%c1_10, %c0_11] (%arg8[%c0_12, %c0_13, %c0_14, %c4096] [%c8_15, %c8_16, %c8_17, %c8_18] [%c8_19, %c512_20, %c64_21, %c1_22]) : (memref<256x64xbf16>)
      %c2 = arith.constant 2 : index
      %c0_23 = arith.constant 0 : index
      %c0_24 = arith.constant 0 : index
      %c0_25 = arith.constant 0 : index
      %c0_26 = arith.constant 0 : index
      %c8192 = arith.constant 8192 : index
      %c8_27 = arith.constant 8 : index
      %c8_28 = arith.constant 8 : index
      %c8_29 = arith.constant 8 : index
      %c8_30 = arith.constant 8 : index
      %c8_31 = arith.constant 8 : index
      %c512_32 = arith.constant 512 : index
      %c64_33 = arith.constant 64 : index
      %c1_34 = arith.constant 1 : index
      air.channel.put  @QK_0[%c2, %c0_23] (%arg8[%c0_24, %c0_25, %c0_26, %c8192] [%c8_27, %c8_28, %c8_29, %c8_30] [%c8_31, %c512_32, %c64_33, %c1_34]) : (memref<256x64xbf16>)
      %c3 = arith.constant 3 : index
      %c0_35 = arith.constant 0 : index
      %c0_36 = arith.constant 0 : index
      %c0_37 = arith.constant 0 : index
      %c0_38 = arith.constant 0 : index
      %c12288 = arith.constant 12288 : index
      %c8_39 = arith.constant 8 : index
      %c8_40 = arith.constant 8 : index
      %c8_41 = arith.constant 8 : index
      %c8_42 = arith.constant 8 : index
      %c8_43 = arith.constant 8 : index
      %c512_44 = arith.constant 512 : index
      %c64_45 = arith.constant 64 : index
      %c1_46 = arith.constant 1 : index
      air.channel.put  @QK_0[%c3, %c0_35] (%arg8[%c0_36, %c0_37, %c0_38, %c12288] [%c8_39, %c8_40, %c8_41, %c8_42] [%c8_43, %c512_44, %c64_45, %c1_46]) : (memref<256x64xbf16>)
      %c0_47 = arith.constant 0 : index
      %c0_48 = arith.constant 0 : index
      %c0_49 = arith.constant 0 : index
      %c0_50 = arith.constant 0 : index
      %c0_51 = arith.constant 0 : index
      %c0_52 = arith.constant 0 : index
      %c8_53 = arith.constant 8 : index
      %c8_54 = arith.constant 8 : index
      %c8_55 = arith.constant 8 : index
      %c8_56 = arith.constant 8 : index
      %c512_57 = arith.constant 512 : index
      %c8_58 = arith.constant 8 : index
      %c64_59 = arith.constant 64 : index
      %c1_60 = arith.constant 1 : index
      air.channel.put  @QK_0[%c0_47, %c0_48] (%arg9[%c0_49, %c0_50, %c0_51, %c0_52] [%c8_53, %c8_54, %c8_55, %c8_56] [%c512_57, %c8_58, %c64_59, %c1_60]) : (memref<256x64xbf16>)
      %c1_61 = arith.constant 1 : index
      %c0_62 = arith.constant 0 : index
      %c0_63 = arith.constant 0 : index
      %c0_64 = arith.constant 0 : index
      %c0_65 = arith.constant 0 : index
      %c0_66 = arith.constant 0 : index
      %c8_67 = arith.constant 8 : index
      %c8_68 = arith.constant 8 : index
      %c8_69 = arith.constant 8 : index
      %c8_70 = arith.constant 8 : index
      %c512_71 = arith.constant 512 : index
      %c8_72 = arith.constant 8 : index
      %c64_73 = arith.constant 64 : index
      %c1_74 = arith.constant 1 : index
      air.channel.put  @QK_0[%c1_61, %c0_62] (%arg9[%c0_63, %c0_64, %c0_65, %c0_66] [%c8_67, %c8_68, %c8_69, %c8_70] [%c512_71, %c8_72, %c64_73, %c1_74]) : (memref<256x64xbf16>)
      %c2_75 = arith.constant 2 : index
      %c0_76 = arith.constant 0 : index
      %c0_77 = arith.constant 0 : index
      %c0_78 = arith.constant 0 : index
      %c0_79 = arith.constant 0 : index
      %c0_80 = arith.constant 0 : index
      %c8_81 = arith.constant 8 : index
      %c8_82 = arith.constant 8 : index
      %c8_83 = arith.constant 8 : index
      %c8_84 = arith.constant 8 : index
      %c512_85 = arith.constant 512 : index
      %c8_86 = arith.constant 8 : index
      %c64_87 = arith.constant 64 : index
      %c1_88 = arith.constant 1 : index
      air.channel.put  @QK_0[%c2_75, %c0_76] (%arg9[%c0_77, %c0_78, %c0_79, %c0_80] [%c8_81, %c8_82, %c8_83, %c8_84] [%c512_85, %c8_86, %c64_87, %c1_88]) : (memref<256x64xbf16>)
      %c3_89 = arith.constant 3 : index
      %c0_90 = arith.constant 0 : index
      %c0_91 = arith.constant 0 : index
      %c0_92 = arith.constant 0 : index
      %c0_93 = arith.constant 0 : index
      %c0_94 = arith.constant 0 : index
      %c8_95 = arith.constant 8 : index
      %c8_96 = arith.constant 8 : index
      %c8_97 = arith.constant 8 : index
      %c8_98 = arith.constant 8 : index
      %c512_99 = arith.constant 512 : index
      %c8_100 = arith.constant 8 : index
      %c64_101 = arith.constant 64 : index
      %c1_102 = arith.constant 1 : index
      air.channel.put  @QK_0[%c3_89, %c0_90] (%arg9[%c0_91, %c0_92, %c0_93, %c0_94] [%c8_95, %c8_96, %c8_97, %c8_98] [%c512_99, %c8_100, %c64_101, %c1_102]) : (memref<256x64xbf16>)
      %c0_103 = arith.constant 0 : index
      %c0_104 = arith.constant 0 : index
      %c0_105 = arith.constant 0 : index
      %c0_106 = arith.constant 0 : index
      %c0_107 = arith.constant 0 : index
      %c4096_108 = arith.constant 4096 : index
      %c8_109 = arith.constant 8 : index
      %c8_110 = arith.constant 8 : index
      %c8_111 = arith.constant 8 : index
      %c8_112 = arith.constant 8 : index
      %c512_113 = arith.constant 512 : index
      %c8_114 = arith.constant 8 : index
      %c64_115 = arith.constant 64 : index
      %c1_116 = arith.constant 1 : index
      air.channel.put  @QK_0[%c0_103, %c0_104] (%arg9[%c0_105, %c0_106, %c0_107, %c4096_108] [%c8_109, %c8_110, %c8_111, %c8_112] [%c512_113, %c8_114, %c64_115, %c1_116]) : (memref<256x64xbf16>)
      %c1_117 = arith.constant 1 : index
      %c0_118 = arith.constant 0 : index
      %c0_119 = arith.constant 0 : index
      %c0_120 = arith.constant 0 : index
      %c0_121 = arith.constant 0 : index
      %c4096_122 = arith.constant 4096 : index
      %c8_123 = arith.constant 8 : index
      %c8_124 = arith.constant 8 : index
      %c8_125 = arith.constant 8 : index
      %c8_126 = arith.constant 8 : index
      %c512_127 = arith.constant 512 : index
      %c8_128 = arith.constant 8 : index
      %c64_129 = arith.constant 64 : index
      %c1_130 = arith.constant 1 : index
      air.channel.put  @QK_0[%c1_117, %c0_118] (%arg9[%c0_119, %c0_120, %c0_121, %c4096_122] [%c8_123, %c8_124, %c8_125, %c8_126] [%c512_127, %c8_128, %c64_129, %c1_130]) : (memref<256x64xbf16>)
      %c2_131 = arith.constant 2 : index
      %c0_132 = arith.constant 0 : index
      %c0_133 = arith.constant 0 : index
      %c0_134 = arith.constant 0 : index
      %c0_135 = arith.constant 0 : index
      %c4096_136 = arith.constant 4096 : index
      %c8_137 = arith.constant 8 : index
      %c8_138 = arith.constant 8 : index
      %c8_139 = arith.constant 8 : index
      %c8_140 = arith.constant 8 : index
      %c512_141 = arith.constant 512 : index
      %c8_142 = arith.constant 8 : index
      %c64_143 = arith.constant 64 : index
      %c1_144 = arith.constant 1 : index
      air.channel.put  @QK_0[%c2_131, %c0_132] (%arg9[%c0_133, %c0_134, %c0_135, %c4096_136] [%c8_137, %c8_138, %c8_139, %c8_140] [%c512_141, %c8_142, %c64_143, %c1_144]) : (memref<256x64xbf16>)
      %c3_145 = arith.constant 3 : index
      %c0_146 = arith.constant 0 : index
      %c0_147 = arith.constant 0 : index
      %c0_148 = arith.constant 0 : index
      %c0_149 = arith.constant 0 : index
      %c4096_150 = arith.constant 4096 : index
      %c8_151 = arith.constant 8 : index
      %c8_152 = arith.constant 8 : index
      %c8_153 = arith.constant 8 : index
      %c8_154 = arith.constant 8 : index
      %c512_155 = arith.constant 512 : index
      %c8_156 = arith.constant 8 : index
      %c64_157 = arith.constant 64 : index
      %c1_158 = arith.constant 1 : index
      air.channel.put  @QK_0[%c3_145, %c0_146] (%arg9[%c0_147, %c0_148, %c0_149, %c4096_150] [%c8_151, %c8_152, %c8_153, %c8_154] [%c512_155, %c8_156, %c64_157, %c1_158]) : (memref<256x64xbf16>)
      %c0_159 = arith.constant 0 : index
      %c0_160 = arith.constant 0 : index
      %c0_161 = arith.constant 0 : index
      %c0_162 = arith.constant 0 : index
      %c0_163 = arith.constant 0 : index
      %c0_164 = arith.constant 0 : index
      %c8_165 = arith.constant 8 : index
      %c8_166 = arith.constant 8 : index
      %c8_167 = arith.constant 8 : index
      %c8_168 = arith.constant 8 : index
      %c8_169 = arith.constant 8 : index
      %c512_170 = arith.constant 512 : index
      %c64_171 = arith.constant 64 : index
      %c1_172 = arith.constant 1 : index
      air.channel.put  @QK_1[%c0_159, %c0_160] (%arg8[%c0_161, %c0_162, %c0_163, %c0_164] [%c8_165, %c8_166, %c8_167, %c8_168] [%c8_169, %c512_170, %c64_171, %c1_172]) : (memref<256x64xbf16>)
      %c1_173 = arith.constant 1 : index
      %c0_174 = arith.constant 0 : index
      %c0_175 = arith.constant 0 : index
      %c0_176 = arith.constant 0 : index
      %c0_177 = arith.constant 0 : index
      %c4096_178 = arith.constant 4096 : index
      %c8_179 = arith.constant 8 : index
      %c8_180 = arith.constant 8 : index
      %c8_181 = arith.constant 8 : index
      %c8_182 = arith.constant 8 : index
      %c8_183 = arith.constant 8 : index
      %c512_184 = arith.constant 512 : index
      %c64_185 = arith.constant 64 : index
      %c1_186 = arith.constant 1 : index
      air.channel.put  @QK_1[%c1_173, %c0_174] (%arg8[%c0_175, %c0_176, %c0_177, %c4096_178] [%c8_179, %c8_180, %c8_181, %c8_182] [%c8_183, %c512_184, %c64_185, %c1_186]) : (memref<256x64xbf16>)
      %c2_187 = arith.constant 2 : index
      %c0_188 = arith.constant 0 : index
      %c0_189 = arith.constant 0 : index
      %c0_190 = arith.constant 0 : index
      %c0_191 = arith.constant 0 : index
      %c8192_192 = arith.constant 8192 : index
      %c8_193 = arith.constant 8 : index
      %c8_194 = arith.constant 8 : index
      %c8_195 = arith.constant 8 : index
      %c8_196 = arith.constant 8 : index
      %c8_197 = arith.constant 8 : index
      %c512_198 = arith.constant 512 : index
      %c64_199 = arith.constant 64 : index
      %c1_200 = arith.constant 1 : index
      air.channel.put  @QK_1[%c2_187, %c0_188] (%arg8[%c0_189, %c0_190, %c0_191, %c8192_192] [%c8_193, %c8_194, %c8_195, %c8_196] [%c8_197, %c512_198, %c64_199, %c1_200]) : (memref<256x64xbf16>)
      %c3_201 = arith.constant 3 : index
      %c0_202 = arith.constant 0 : index
      %c0_203 = arith.constant 0 : index
      %c0_204 = arith.constant 0 : index
      %c0_205 = arith.constant 0 : index
      %c12288_206 = arith.constant 12288 : index
      %c8_207 = arith.constant 8 : index
      %c8_208 = arith.constant 8 : index
      %c8_209 = arith.constant 8 : index
      %c8_210 = arith.constant 8 : index
      %c8_211 = arith.constant 8 : index
      %c512_212 = arith.constant 512 : index
      %c64_213 = arith.constant 64 : index
      %c1_214 = arith.constant 1 : index
      air.channel.put  @QK_1[%c3_201, %c0_202] (%arg8[%c0_203, %c0_204, %c0_205, %c12288_206] [%c8_207, %c8_208, %c8_209, %c8_210] [%c8_211, %c512_212, %c64_213, %c1_214]) : (memref<256x64xbf16>)
      %c0_215 = arith.constant 0 : index
      %c0_216 = arith.constant 0 : index
      %c0_217 = arith.constant 0 : index
      %c0_218 = arith.constant 0 : index
      %c0_219 = arith.constant 0 : index
      %c8192_220 = arith.constant 8192 : index
      %c8_221 = arith.constant 8 : index
      %c8_222 = arith.constant 8 : index
      %c8_223 = arith.constant 8 : index
      %c8_224 = arith.constant 8 : index
      %c512_225 = arith.constant 512 : index
      %c8_226 = arith.constant 8 : index
      %c64_227 = arith.constant 64 : index
      %c1_228 = arith.constant 1 : index
      air.channel.put  @QK_1[%c0_215, %c0_216] (%arg9[%c0_217, %c0_218, %c0_219, %c8192_220] [%c8_221, %c8_222, %c8_223, %c8_224] [%c512_225, %c8_226, %c64_227, %c1_228]) : (memref<256x64xbf16>)
      %c1_229 = arith.constant 1 : index
      %c0_230 = arith.constant 0 : index
      %c0_231 = arith.constant 0 : index
      %c0_232 = arith.constant 0 : index
      %c0_233 = arith.constant 0 : index
      %c8192_234 = arith.constant 8192 : index
      %c8_235 = arith.constant 8 : index
      %c8_236 = arith.constant 8 : index
      %c8_237 = arith.constant 8 : index
      %c8_238 = arith.constant 8 : index
      %c512_239 = arith.constant 512 : index
      %c8_240 = arith.constant 8 : index
      %c64_241 = arith.constant 64 : index
      %c1_242 = arith.constant 1 : index
      air.channel.put  @QK_1[%c1_229, %c0_230] (%arg9[%c0_231, %c0_232, %c0_233, %c8192_234] [%c8_235, %c8_236, %c8_237, %c8_238] [%c512_239, %c8_240, %c64_241, %c1_242]) : (memref<256x64xbf16>)
      %c2_243 = arith.constant 2 : index
      %c0_244 = arith.constant 0 : index
      %c0_245 = arith.constant 0 : index
      %c0_246 = arith.constant 0 : index
      %c0_247 = arith.constant 0 : index
      %c8192_248 = arith.constant 8192 : index
      %c8_249 = arith.constant 8 : index
      %c8_250 = arith.constant 8 : index
      %c8_251 = arith.constant 8 : index
      %c8_252 = arith.constant 8 : index
      %c512_253 = arith.constant 512 : index
      %c8_254 = arith.constant 8 : index
      %c64_255 = arith.constant 64 : index
      %c1_256 = arith.constant 1 : index
      air.channel.put  @QK_1[%c2_243, %c0_244] (%arg9[%c0_245, %c0_246, %c0_247, %c8192_248] [%c8_249, %c8_250, %c8_251, %c8_252] [%c512_253, %c8_254, %c64_255, %c1_256]) : (memref<256x64xbf16>)
      %c3_257 = arith.constant 3 : index
      %c0_258 = arith.constant 0 : index
      %c0_259 = arith.constant 0 : index
      %c0_260 = arith.constant 0 : index
      %c0_261 = arith.constant 0 : index
      %c8192_262 = arith.constant 8192 : index
      %c8_263 = arith.constant 8 : index
      %c8_264 = arith.constant 8 : index
      %c8_265 = arith.constant 8 : index
      %c8_266 = arith.constant 8 : index
      %c512_267 = arith.constant 512 : index
      %c8_268 = arith.constant 8 : index
      %c64_269 = arith.constant 64 : index
      %c1_270 = arith.constant 1 : index
      air.channel.put  @QK_1[%c3_257, %c0_258] (%arg9[%c0_259, %c0_260, %c0_261, %c8192_262] [%c8_263, %c8_264, %c8_265, %c8_266] [%c512_267, %c8_268, %c64_269, %c1_270]) : (memref<256x64xbf16>)
      %c0_271 = arith.constant 0 : index
      %c0_272 = arith.constant 0 : index
      %c0_273 = arith.constant 0 : index
      %c0_274 = arith.constant 0 : index
      %c0_275 = arith.constant 0 : index
      %c12288_276 = arith.constant 12288 : index
      %c8_277 = arith.constant 8 : index
      %c8_278 = arith.constant 8 : index
      %c8_279 = arith.constant 8 : index
      %c8_280 = arith.constant 8 : index
      %c512_281 = arith.constant 512 : index
      %c8_282 = arith.constant 8 : index
      %c64_283 = arith.constant 64 : index
      %c1_284 = arith.constant 1 : index
      air.channel.put  @QK_1[%c0_271, %c0_272] (%arg9[%c0_273, %c0_274, %c0_275, %c12288_276] [%c8_277, %c8_278, %c8_279, %c8_280] [%c512_281, %c8_282, %c64_283, %c1_284]) : (memref<256x64xbf16>)
      %c1_285 = arith.constant 1 : index
      %c0_286 = arith.constant 0 : index
      %c0_287 = arith.constant 0 : index
      %c0_288 = arith.constant 0 : index
      %c0_289 = arith.constant 0 : index
      %c12288_290 = arith.constant 12288 : index
      %c8_291 = arith.constant 8 : index
      %c8_292 = arith.constant 8 : index
      %c8_293 = arith.constant 8 : index
      %c8_294 = arith.constant 8 : index
      %c512_295 = arith.constant 512 : index
      %c8_296 = arith.constant 8 : index
      %c64_297 = arith.constant 64 : index
      %c1_298 = arith.constant 1 : index
      air.channel.put  @QK_1[%c1_285, %c0_286] (%arg9[%c0_287, %c0_288, %c0_289, %c12288_290] [%c8_291, %c8_292, %c8_293, %c8_294] [%c512_295, %c8_296, %c64_297, %c1_298]) : (memref<256x64xbf16>)
      %c2_299 = arith.constant 2 : index
      %c0_300 = arith.constant 0 : index
      %c0_301 = arith.constant 0 : index
      %c0_302 = arith.constant 0 : index
      %c0_303 = arith.constant 0 : index
      %c12288_304 = arith.constant 12288 : index
      %c8_305 = arith.constant 8 : index
      %c8_306 = arith.constant 8 : index
      %c8_307 = arith.constant 8 : index
      %c8_308 = arith.constant 8 : index
      %c512_309 = arith.constant 512 : index
      %c8_310 = arith.constant 8 : index
      %c64_311 = arith.constant 64 : index
      %c1_312 = arith.constant 1 : index
      air.channel.put  @QK_1[%c2_299, %c0_300] (%arg9[%c0_301, %c0_302, %c0_303, %c12288_304] [%c8_305, %c8_306, %c8_307, %c8_308] [%c512_309, %c8_310, %c64_311, %c1_312]) : (memref<256x64xbf16>)
      %c3_313 = arith.constant 3 : index
      %c0_314 = arith.constant 0 : index
      %c0_315 = arith.constant 0 : index
      %c0_316 = arith.constant 0 : index
      %c0_317 = arith.constant 0 : index
      %c12288_318 = arith.constant 12288 : index
      %c8_319 = arith.constant 8 : index
      %c8_320 = arith.constant 8 : index
      %c8_321 = arith.constant 8 : index
      %c8_322 = arith.constant 8 : index
      %c512_323 = arith.constant 512 : index
      %c8_324 = arith.constant 8 : index
      %c64_325 = arith.constant 64 : index
      %c1_326 = arith.constant 1 : index
      air.channel.put  @QK_1[%c3_313, %c0_314] (%arg9[%c0_315, %c0_316, %c0_317, %c12288_318] [%c8_319, %c8_320, %c8_321, %c8_322] [%c512_323, %c8_324, %c64_325, %c1_326]) : (memref<256x64xbf16>)
      %c0_327 = arith.constant 0 : index
      %c0_328 = arith.constant 0 : index
      %c0_329 = arith.constant 0 : index
      %c0_330 = arith.constant 0 : index
      %c2_331 = arith.constant 2 : index
      %c64_332 = arith.constant 64 : index
      %c64_333 = arith.constant 64 : index
      %c4096_334 = arith.constant 4096 : index
      %c64_335 = arith.constant 64 : index
      %c1_336 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c0_327] (%arg10[%c0_328, %c0_329, %c0_330] [%c2_331, %c64_332, %c64_333] [%c4096_334, %c64_335, %c1_336]) : (memref<256x64xbf16>)
      %c0_337 = arith.constant 0 : index
      %c0_338 = arith.constant 0 : index
      %c0_339 = arith.constant 0 : index
      %c8192_340 = arith.constant 8192 : index
      %c2_341 = arith.constant 2 : index
      %c64_342 = arith.constant 64 : index
      %c64_343 = arith.constant 64 : index
      %c4096_344 = arith.constant 4096 : index
      %c64_345 = arith.constant 64 : index
      %c1_346 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c0_337] (%arg10[%c0_338, %c0_339, %c8192_340] [%c2_341, %c64_342, %c64_343] [%c4096_344, %c64_345, %c1_346]) : (memref<256x64xbf16>)
      %c0_347 = arith.constant 0 : index
      air.channel.get  @GpOut[%c0_347] (%arg11[] [] []) : (memref<256x64xbf16>)
      %c1_348 = arith.constant 1 : index
      air.segment @s  unroll(%arg12, %arg13) in (%arg14=%c1_348, %arg15=%c1_348) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_349 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_350 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_351 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_352 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_353 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_354 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_355 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_356 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_357 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %c4 = arith.constant 4 : index
        %c2_358 = arith.constant 2 : index
        %c0_359 = arith.constant 0 : index
        %c2_360 = arith.constant 2 : index
        %c0_361 = arith.constant 0 : index
        %c1_362 = arith.constant 1 : index
        scf.for %arg16 = %c0_361 to %c2_360 step %c1_362 {
          %c0_366 = arith.constant 0 : index
          air.channel.get  @VIn_0[%c0_366] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_367 = arith.constant 0 : index
          %c0_368 = arith.constant 0 : index
          %c0_369 = arith.constant 0 : index
          %c0_370 = arith.constant 0 : index
          %c8_371 = arith.constant 8 : index
          %c8_372 = arith.constant 8 : index
          %c8_373 = arith.constant 8 : index
          %c8_374 = arith.constant 8 : index
          %c8_375 = arith.constant 8 : index
          %c512_376 = arith.constant 512 : index
          %c64_377 = arith.constant 64 : index
          %c1_378 = arith.constant 1 : index
          air.channel.put  @V2L1_0[%c0_359, %c0_359] (%alloc[%c0_367, %c0_368, %c0_369, %c0_370] [%c8_371, %c8_372, %c8_373, %c8_374] [%c8_375, %c512_376, %c64_377, %c1_378]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_363 = arith.constant 0 : index
        %c1_364 = arith.constant 1 : index
        scf.for %arg16 = %c0_363 to %c2_360 step %c1_364 {
          %c0_366 = arith.constant 0 : index
          air.channel.get  @VIn_1[%c0_366] (%alloc_349[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_367 = arith.constant 0 : index
          %c0_368 = arith.constant 0 : index
          %c0_369 = arith.constant 0 : index
          %c0_370 = arith.constant 0 : index
          %c8_371 = arith.constant 8 : index
          %c8_372 = arith.constant 8 : index
          %c8_373 = arith.constant 8 : index
          %c8_374 = arith.constant 8 : index
          %c8_375 = arith.constant 8 : index
          %c512_376 = arith.constant 512 : index
          %c64_377 = arith.constant 64 : index
          %c1_378 = arith.constant 1 : index
          air.channel.put  @V2L1_1[%c0_359, %c0_359] (%alloc_349[%c0_367, %c0_368, %c0_369, %c0_370] [%c8_371, %c8_372, %c8_373, %c8_374] [%c8_375, %c512_376, %c64_377, %c1_378]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %0 = affine.apply #map()[%arg16]
          %c0_366 = arith.constant 0 : index
          %c0_367 = arith.constant 0 : index
          %c64_368 = arith.constant 64 : index
          %c64_369 = arith.constant 64 : index
          %c64_370 = arith.constant 64 : index
          %c1_371 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%arg16, %c0_366] (%alloc_350[%0, %c0_367] [%c64_368, %c64_369] [%c64_370, %c1_371]) : (memref<256x64xbf16, 1 : i32>)
        }
        %c0_365 = arith.constant 0 : index
        air.channel.put  @GpOut[%c0_365] (%alloc_350[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @h  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c2_358) args(%arg20=%alloc_351, %arg21=%alloc_352, %arg22=%alloc_353, %arg23=%alloc_354, %arg24=%alloc_355, %arg25=%alloc_356, %arg26=%alloc_357) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32> attributes {link_with = "attn.o"} {
          affine.if #set()[%arg16, %arg17] {
            %c0_369 = arith.constant 0 : index
            air.channel.get  @QK_0[%arg16, %c0_369] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          affine.if #set1()[%arg16, %arg17] {
            %c0_369 = arith.constant 0 : index
            air.channel.get  @QK_1[%arg16, %c0_369] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          }
          func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          affine.if #set1()[%arg16, %arg17] {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          affine.if #set2()[%arg16, %arg17] {
            air.channel.get  @cascade_gp[%arg16] (%arg24[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.get  @cascade_up[%arg16] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.get  @cascade_sp[%arg16] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
          }
          %c2_366 = arith.constant 2 : index
          %c0_367 = arith.constant 0 : index
          %c1_368 = arith.constant 1 : index
          scf.for %arg27 = %c0_367 to %c2_366 step %c1_368 {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            affine.if #set()[%arg16, %arg17] {
              %c0_371 = arith.constant 0 : index
              air.channel.get  @QK_0[%arg16, %c0_371] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              %c0_371 = arith.constant 0 : index
              air.channel.get  @QK_1[%arg16, %c0_371] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set()[%arg16, %arg17] {
              air.channel.get  @V2L1_0[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              air.channel.get  @V2L1_1[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            %alloc_369 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_370 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %alloc_369, %alloc_370) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_370, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg26, %alloc_370, %alloc_369) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32, %alloc_369, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_369 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_370 : memref<64x1xbf16, 2 : i32>
          }
          affine.if #set3()[%arg16, %arg17] {
            air.channel.put  @cascade_gp[%arg16] (%arg24[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
          }
          affine.if #set()[%arg16, %arg17] {
            func.call @div_gp_sp(%arg26, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_369 = arith.constant 0 : index
            %c0_370 = arith.constant 0 : index
            %c0_371 = arith.constant 0 : index
            %c0_372 = arith.constant 0 : index
            %c0_373 = arith.constant 0 : index
            %c8_374 = arith.constant 8 : index
            %c8_375 = arith.constant 8 : index
            %c8_376 = arith.constant 8 : index
            %c8_377 = arith.constant 8 : index
            %c64_378 = arith.constant 64 : index
            %c8_379 = arith.constant 8 : index
            %c512_380 = arith.constant 512 : index
            %c1_381 = arith.constant 1 : index
            air.channel.put  @Gp2L2[%arg16, %c0_369] (%arg24[%c0_370, %c0_371, %c0_372, %c0_373] [%c8_374, %c8_375, %c8_376, %c8_377] [%c64_378, %c8_379, %c512_380, %c1_381]) : (memref<64x64xbf16, 2 : i32>)
          }
        }
        memref.dealloc %alloc_351 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_352 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_353 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_354 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_355 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_356 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_357 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_349 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_350 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}
