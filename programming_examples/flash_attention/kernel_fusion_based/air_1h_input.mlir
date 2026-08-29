#map = affine_map<()[s0] -> (s0 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
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
  air.channel @QK2L1 [1, 4] {broadcast_shape = [4 : index, 4 : index]}
  air.channel @V2L1_0 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_0 [1]
  air.channel @V2L1_1 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_1 [1]
  air.channel @V2L1_2 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_2 [1]
  air.channel @V2L1_3 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_3 [1]
  air.channel @cascade_gp [4, 3] {channel_type = "cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  air.channel @GpOut [1]
  func.func @attention_bf16(%arg0: memref<256x64xbf16>, %arg1: memref<512x64xbf16>, %arg2: memref<512x64xbf16>, %arg3: memref<256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<256x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16>, memref<256x64xbf16> {
      %0 = affine.apply #map()[%arg4]
      %c0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %c0_5 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c8_6 = arith.constant 8 : index
      %c8_7 = arith.constant 8 : index
      %c8_8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c8_9 = arith.constant 8 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c1_10 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0, %c0_1] (%arg8[%c0_2, %c0_3, %c0_4, %c0_5, %0] [%c4, %c8, %c8_6, %c8_7, %c8_8] [%c4096, %c8_9, %c512, %c64, %c1_10]) : (memref<256x64xbf16>)
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c0_13 = arith.constant 0 : index
      %c0_14 = arith.constant 0 : index
      %c0_15 = arith.constant 0 : index
      %c0_16 = arith.constant 0 : index
      %c4_17 = arith.constant 4 : index
      %c8_18 = arith.constant 8 : index
      %c8_19 = arith.constant 8 : index
      %c8_20 = arith.constant 8 : index
      %c8_21 = arith.constant 8 : index
      %c4096_22 = arith.constant 4096 : index
      %c8_23 = arith.constant 8 : index
      %c512_24 = arith.constant 512 : index
      %c64_25 = arith.constant 64 : index
      %c1_26 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0_11, %c1_12] (%arg8[%c0_13, %c0_14, %c0_15, %c0_16, %0] [%c4_17, %c8_18, %c8_19, %c8_20, %c8_21] [%c4096_22, %c8_23, %c512_24, %c64_25, %c1_26]) : (memref<256x64xbf16>)
      %c0_27 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c0_28 = arith.constant 0 : index
      %c0_29 = arith.constant 0 : index
      %c0_30 = arith.constant 0 : index
      %c0_31 = arith.constant 0 : index
      %c4_32 = arith.constant 4 : index
      %c8_33 = arith.constant 8 : index
      %c8_34 = arith.constant 8 : index
      %c8_35 = arith.constant 8 : index
      %c8_36 = arith.constant 8 : index
      %c4096_37 = arith.constant 4096 : index
      %c8_38 = arith.constant 8 : index
      %c512_39 = arith.constant 512 : index
      %c64_40 = arith.constant 64 : index
      %c1_41 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0_27, %c2] (%arg8[%c0_28, %c0_29, %c0_30, %c0_31, %0] [%c4_32, %c8_33, %c8_34, %c8_35, %c8_36] [%c4096_37, %c8_38, %c512_39, %c64_40, %c1_41]) : (memref<256x64xbf16>)
      %c0_42 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c0_43 = arith.constant 0 : index
      %c0_44 = arith.constant 0 : index
      %c0_45 = arith.constant 0 : index
      %c0_46 = arith.constant 0 : index
      %c4_47 = arith.constant 4 : index
      %c8_48 = arith.constant 8 : index
      %c8_49 = arith.constant 8 : index
      %c8_50 = arith.constant 8 : index
      %c8_51 = arith.constant 8 : index
      %c4096_52 = arith.constant 4096 : index
      %c8_53 = arith.constant 8 : index
      %c512_54 = arith.constant 512 : index
      %c64_55 = arith.constant 64 : index
      %c1_56 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0_42, %c3] (%arg8[%c0_43, %c0_44, %c0_45, %c0_46, %0] [%c4_47, %c8_48, %c8_49, %c8_50, %c8_51] [%c4096_52, %c8_53, %c512_54, %c64_55, %c1_56]) : (memref<256x64xbf16>)
      %c0_57 = arith.constant 0 : index
      %c0_58 = arith.constant 0 : index
      %c0_59 = arith.constant 0 : index
      %c0_60 = arith.constant 0 : index
      %c0_61 = arith.constant 0 : index
      %c0_62 = arith.constant 0 : index
      %c0_63 = arith.constant 0 : index
      %c2_64 = arith.constant 2 : index
      %c8_65 = arith.constant 8 : index
      %c8_66 = arith.constant 8 : index
      %c8_67 = arith.constant 8 : index
      %c8_68 = arith.constant 8 : index
      %c4096_69 = arith.constant 4096 : index
      %c8_70 = arith.constant 8 : index
      %c512_71 = arith.constant 512 : index
      %c64_72 = arith.constant 64 : index
      %c1_73 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0_57, %c0_58] (%arg9[%c0_59, %c0_60, %c0_61, %c0_62, %c0_63] [%c2_64, %c8_65, %c8_66, %c8_67, %c8_68] [%c4096_69, %c8_70, %c512_71, %c64_72, %c1_73]) : (memref<512x64xbf16>)
      %c0_74 = arith.constant 0 : index
      %c1_75 = arith.constant 1 : index
      %c0_76 = arith.constant 0 : index
      %c0_77 = arith.constant 0 : index
      %c0_78 = arith.constant 0 : index
      %c0_79 = arith.constant 0 : index
      %c8192 = arith.constant 8192 : index
      %c2_80 = arith.constant 2 : index
      %c8_81 = arith.constant 8 : index
      %c8_82 = arith.constant 8 : index
      %c8_83 = arith.constant 8 : index
      %c8_84 = arith.constant 8 : index
      %c4096_85 = arith.constant 4096 : index
      %c8_86 = arith.constant 8 : index
      %c512_87 = arith.constant 512 : index
      %c64_88 = arith.constant 64 : index
      %c1_89 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0_74, %c1_75] (%arg9[%c0_76, %c0_77, %c0_78, %c0_79, %c8192] [%c2_80, %c8_81, %c8_82, %c8_83, %c8_84] [%c4096_85, %c8_86, %c512_87, %c64_88, %c1_89]) : (memref<512x64xbf16>)
      %c0_90 = arith.constant 0 : index
      %c2_91 = arith.constant 2 : index
      %c0_92 = arith.constant 0 : index
      %c0_93 = arith.constant 0 : index
      %c0_94 = arith.constant 0 : index
      %c0_95 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %c2_96 = arith.constant 2 : index
      %c8_97 = arith.constant 8 : index
      %c8_98 = arith.constant 8 : index
      %c8_99 = arith.constant 8 : index
      %c8_100 = arith.constant 8 : index
      %c4096_101 = arith.constant 4096 : index
      %c8_102 = arith.constant 8 : index
      %c512_103 = arith.constant 512 : index
      %c64_104 = arith.constant 64 : index
      %c1_105 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0_90, %c2_91] (%arg9[%c0_92, %c0_93, %c0_94, %c0_95, %c16384] [%c2_96, %c8_97, %c8_98, %c8_99, %c8_100] [%c4096_101, %c8_102, %c512_103, %c64_104, %c1_105]) : (memref<512x64xbf16>)
      %c0_106 = arith.constant 0 : index
      %c3_107 = arith.constant 3 : index
      %c0_108 = arith.constant 0 : index
      %c0_109 = arith.constant 0 : index
      %c0_110 = arith.constant 0 : index
      %c0_111 = arith.constant 0 : index
      %c24576 = arith.constant 24576 : index
      %c2_112 = arith.constant 2 : index
      %c8_113 = arith.constant 8 : index
      %c8_114 = arith.constant 8 : index
      %c8_115 = arith.constant 8 : index
      %c8_116 = arith.constant 8 : index
      %c4096_117 = arith.constant 4096 : index
      %c8_118 = arith.constant 8 : index
      %c512_119 = arith.constant 512 : index
      %c64_120 = arith.constant 64 : index
      %c1_121 = arith.constant 1 : index
      air.channel.put  @QK2L1[%c0_106, %c3_107] (%arg9[%c0_108, %c0_109, %c0_110, %c0_111, %c24576] [%c2_112, %c8_113, %c8_114, %c8_115, %c8_116] [%c4096_117, %c8_118, %c512_119, %c64_120, %c1_121]) : (memref<512x64xbf16>)
      %c0_122 = arith.constant 0 : index
      %c0_123 = arith.constant 0 : index
      %c0_124 = arith.constant 0 : index
      %c0_125 = arith.constant 0 : index
      %c2_126 = arith.constant 2 : index
      %c64_127 = arith.constant 64 : index
      %c64_128 = arith.constant 64 : index
      %c4096_129 = arith.constant 4096 : index
      %c64_130 = arith.constant 64 : index
      %c1_131 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c0_122] (%arg10[%c0_123, %c0_124, %c0_125] [%c2_126, %c64_127, %c64_128] [%c4096_129, %c64_130, %c1_131]) : (memref<512x64xbf16>)
      %c0_132 = arith.constant 0 : index
      %c0_133 = arith.constant 0 : index
      %c0_134 = arith.constant 0 : index
      %c8192_135 = arith.constant 8192 : index
      %c2_136 = arith.constant 2 : index
      %c64_137 = arith.constant 64 : index
      %c64_138 = arith.constant 64 : index
      %c4096_139 = arith.constant 4096 : index
      %c64_140 = arith.constant 64 : index
      %c1_141 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c0_132] (%arg10[%c0_133, %c0_134, %c8192_135] [%c2_136, %c64_137, %c64_138] [%c4096_139, %c64_140, %c1_141]) : (memref<512x64xbf16>)
      %c0_142 = arith.constant 0 : index
      %c0_143 = arith.constant 0 : index
      %c0_144 = arith.constant 0 : index
      %c16384_145 = arith.constant 16384 : index
      %c2_146 = arith.constant 2 : index
      %c64_147 = arith.constant 64 : index
      %c64_148 = arith.constant 64 : index
      %c4096_149 = arith.constant 4096 : index
      %c64_150 = arith.constant 64 : index
      %c1_151 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c0_142] (%arg10[%c0_143, %c0_144, %c16384_145] [%c2_146, %c64_147, %c64_148] [%c4096_149, %c64_150, %c1_151]) : (memref<512x64xbf16>)
      %c0_152 = arith.constant 0 : index
      %c0_153 = arith.constant 0 : index
      %c0_154 = arith.constant 0 : index
      %c24576_155 = arith.constant 24576 : index
      %c2_156 = arith.constant 2 : index
      %c64_157 = arith.constant 64 : index
      %c64_158 = arith.constant 64 : index
      %c4096_159 = arith.constant 4096 : index
      %c64_160 = arith.constant 64 : index
      %c1_161 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c0_152] (%arg10[%c0_153, %c0_154, %c24576_155] [%c2_156, %c64_157, %c64_158] [%c4096_159, %c64_160, %c1_161]) : (memref<512x64xbf16>)
      %c0_162 = arith.constant 0 : index
      %c16384_163 = arith.constant 16384 : index
      %c1_164 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0_162] (%arg11[%0] [%c16384_163] [%c1_164]) : (memref<256x64xbf16>)
      %c1_165 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c1_165, %arg15=%c1_165) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_166 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_167 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_168 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_169 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_170 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_171 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_172 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_173 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_174 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_175 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_176 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %c4_177 = arith.constant 4 : index
        %c4_178 = arith.constant 4 : index
        %c0_179 = arith.constant 0 : index
        %c2_180 = arith.constant 2 : index
        %c0_181 = arith.constant 0 : index
        %c1_182 = arith.constant 1 : index
        scf.for %arg16 = %c0_181 to %c2_180 step %c1_182 {
          %c0_190 = arith.constant 0 : index
          air.channel.get  @VIn_0[%c0_190] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_191 = arith.constant 0 : index
          %c0_192 = arith.constant 0 : index
          %c0_193 = arith.constant 0 : index
          %c0_194 = arith.constant 0 : index
          %c8_195 = arith.constant 8 : index
          %c8_196 = arith.constant 8 : index
          %c8_197 = arith.constant 8 : index
          %c8_198 = arith.constant 8 : index
          %c8_199 = arith.constant 8 : index
          %c512_200 = arith.constant 512 : index
          %c64_201 = arith.constant 64 : index
          %c1_202 = arith.constant 1 : index
          air.channel.put  @V2L1_0[%c0_179, %c0_179] (%alloc[%c0_191, %c0_192, %c0_193, %c0_194] [%c8_195, %c8_196, %c8_197, %c8_198] [%c8_199, %c512_200, %c64_201, %c1_202]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_183 = arith.constant 0 : index
        %c1_184 = arith.constant 1 : index
        scf.for %arg16 = %c0_183 to %c2_180 step %c1_184 {
          %c0_190 = arith.constant 0 : index
          air.channel.get  @VIn_1[%c0_190] (%alloc_166[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_191 = arith.constant 0 : index
          %c0_192 = arith.constant 0 : index
          %c0_193 = arith.constant 0 : index
          %c0_194 = arith.constant 0 : index
          %c8_195 = arith.constant 8 : index
          %c8_196 = arith.constant 8 : index
          %c8_197 = arith.constant 8 : index
          %c8_198 = arith.constant 8 : index
          %c8_199 = arith.constant 8 : index
          %c512_200 = arith.constant 512 : index
          %c64_201 = arith.constant 64 : index
          %c1_202 = arith.constant 1 : index
          air.channel.put  @V2L1_1[%c0_179, %c0_179] (%alloc_166[%c0_191, %c0_192, %c0_193, %c0_194] [%c8_195, %c8_196, %c8_197, %c8_198] [%c8_199, %c512_200, %c64_201, %c1_202]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_185 = arith.constant 0 : index
        %c1_186 = arith.constant 1 : index
        scf.for %arg16 = %c0_185 to %c2_180 step %c1_186 {
          %c0_190 = arith.constant 0 : index
          air.channel.get  @VIn_2[%c0_190] (%alloc_167[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_191 = arith.constant 0 : index
          %c0_192 = arith.constant 0 : index
          %c0_193 = arith.constant 0 : index
          %c0_194 = arith.constant 0 : index
          %c8_195 = arith.constant 8 : index
          %c8_196 = arith.constant 8 : index
          %c8_197 = arith.constant 8 : index
          %c8_198 = arith.constant 8 : index
          %c8_199 = arith.constant 8 : index
          %c512_200 = arith.constant 512 : index
          %c64_201 = arith.constant 64 : index
          %c1_202 = arith.constant 1 : index
          air.channel.put  @V2L1_2[%c0_179, %c0_179] (%alloc_167[%c0_191, %c0_192, %c0_193, %c0_194] [%c8_195, %c8_196, %c8_197, %c8_198] [%c8_199, %c512_200, %c64_201, %c1_202]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_187 = arith.constant 0 : index
        %c1_188 = arith.constant 1 : index
        scf.for %arg16 = %c0_187 to %c2_180 step %c1_188 {
          %c0_190 = arith.constant 0 : index
          air.channel.get  @VIn_3[%c0_190] (%alloc_168[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_191 = arith.constant 0 : index
          %c0_192 = arith.constant 0 : index
          %c0_193 = arith.constant 0 : index
          %c0_194 = arith.constant 0 : index
          %c8_195 = arith.constant 8 : index
          %c8_196 = arith.constant 8 : index
          %c8_197 = arith.constant 8 : index
          %c8_198 = arith.constant 8 : index
          %c8_199 = arith.constant 8 : index
          %c512_200 = arith.constant 512 : index
          %c64_201 = arith.constant 64 : index
          %c1_202 = arith.constant 1 : index
          air.channel.put  @V2L1_3[%c0_179, %c0_179] (%alloc_168[%c0_191, %c0_192, %c0_193, %c0_194] [%c8_195, %c8_196, %c8_197, %c8_198] [%c8_199, %c512_200, %c64_201, %c1_202]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %1 = affine.apply #map1()[%arg16]
          %c0_190 = arith.constant 0 : index
          %c0_191 = arith.constant 0 : index
          %c64_192 = arith.constant 64 : index
          %c64_193 = arith.constant 64 : index
          %c64_194 = arith.constant 64 : index
          %c1_195 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%arg16, %c0_190] (%alloc_169[%1, %c0_191] [%c64_192, %c64_193] [%c64_194, %c1_195]) : (memref<256x64xbf16, 1 : i32>)
        }
        %c0_189 = arith.constant 0 : index
        air.channel.put  @GpOut[%c0_189] (%alloc_169[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4_177, %arg19=%c4_178) args(%arg20=%alloc_170, %arg21=%alloc_171, %arg22=%alloc_172, %arg23=%alloc_173, %arg24=%alloc_174, %arg25=%alloc_175, %arg26=%alloc_176) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32> attributes {link_with = "attn.o"} {
          func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %1 = arith.index_cast %arg16 : index to i32
          %c0_i32 = arith.constant 0 : i32
          %2 = arith.cmpi eq, %1, %c0_i32 : i32
          scf.if %2 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %3 = arith.index_cast %arg16 : index to i32
          %c1_i32 = arith.constant 1 : i32
          %4 = arith.cmpi eq, %3, %c1_i32 : i32
          scf.if %4 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %5 = arith.index_cast %arg16 : index to i32
          %c2_i32 = arith.constant 2 : i32
          %6 = arith.cmpi eq, %5, %c2_i32 : i32
          scf.if %6 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %7 = arith.index_cast %arg16 : index to i32
          %c3_i32 = arith.constant 3 : i32
          %8 = arith.cmpi eq, %7, %c3_i32 : i32
          scf.if %8 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %c2_190 = arith.constant 2 : index
          %c0_191 = arith.constant 0 : index
          %c1_192 = arith.constant 1 : index
          scf.for %arg27 = %c0_191 to %c2_190 step %c1_192 {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            affine.if #set()[%arg16, %arg17] {
              air.channel.get  @V2L1_0[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              air.channel.get  @V2L1_1[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg16, %arg17] {
              air.channel.get  @V2L1_2[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg16, %arg17] {
              air.channel.get  @V2L1_3[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            %alloc_194 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_195 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %alloc_194, %alloc_195) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_195, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32_196 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg26, %alloc_195, %alloc_194) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32_196, %alloc_194, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_194 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_195 : memref<64x1xbf16, 2 : i32>
          }
          %c1_193 = arith.constant 1 : index
          affine.if #set3()[%arg16, %arg17] {
            %9 = arith.subi %arg17, %c1_193 : index
            air.channel.put  @cascade_gp[%arg16, %9] (%arg24[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %9] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %9] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_194 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_195 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_196 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_194[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_195[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_196[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_197 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_198 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_198, %arg25, %alloc_197) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_195, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_199 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_195, %arg25, %alloc_199) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_200 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_197, %arg25, %alloc_200) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_199, %alloc_194) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_200, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_194) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_201 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_201) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_196, %alloc_199, %alloc_201) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_200, %alloc_201) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_198, %alloc_201, %alloc_196) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %9 = arith.subi %arg17, %c1_193 : index
              air.channel.put  @cascade_gp[%arg16, %9] (%alloc_194[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %9] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %9] (%alloc_196[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_194 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_195 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_196 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_197 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_199 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_200 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_201 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_194 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_195 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_196 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_194[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_195[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_196[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_197 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_198 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_198, %arg25, %alloc_197) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_195, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_199 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_195, %arg25, %alloc_199) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_200 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_197, %arg25, %alloc_200) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_199, %alloc_194) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_200, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_194) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_201 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_201) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_196, %alloc_199, %alloc_201) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_200, %alloc_201) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_198, %alloc_201, %alloc_196) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_196, %alloc_194) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %c0_202 = arith.constant 0 : index
              %c0_203 = arith.constant 0 : index
              %c0_204 = arith.constant 0 : index
              %c0_205 = arith.constant 0 : index
              %c0_206 = arith.constant 0 : index
              %c8_207 = arith.constant 8 : index
              %c8_208 = arith.constant 8 : index
              %c8_209 = arith.constant 8 : index
              %c8_210 = arith.constant 8 : index
              %c64_211 = arith.constant 64 : index
              %c8_212 = arith.constant 8 : index
              %c512_213 = arith.constant 512 : index
              %c1_214 = arith.constant 1 : index
              air.channel.put  @Gp2L2[%arg16, %c0_202] (%alloc_194[%c0_203, %c0_204, %c0_205, %c0_206] [%c8_207, %c8_208, %c8_209, %c8_210] [%c64_211, %c8_212, %c512_213, %c1_214]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_194 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_195 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_196 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_197 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_199 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_200 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_201 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_170 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_171 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_172 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_173 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_174 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_175 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_176 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_166 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_167 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_168 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_169 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}

