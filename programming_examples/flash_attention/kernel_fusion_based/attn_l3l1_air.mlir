#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 16384)>
#map2 = affine_map<()[s0, s1] -> (s0 * 131072 + s1)>
#map3 = affine_map<()[s0, s1] -> (s0 * 131072 + s1 * 64)>
#map4 = affine_map<()[s0] -> (s0 * 131072)>
#map5 = affine_map<()[s0] -> (s0 * 64)>
#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0, -s1 + 1 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0, -s1 + 2 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set4 = affine_set<()[s0, s1] : (s1 - 3 == 0, s0 >= 0, -s0 + 3 >= 0)>
#set5 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
  func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
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
  air.channel @Q2L1 [4, 1] {broadcast_shape = [4 : index, 4 : index], channel_type = "dma_packet"}
  air.channel @K2L1 [1, 4] {broadcast_shape = [4 : index, 4 : index], channel_type = "dma_packet"}
  air.channel @VIn_0 [1]
  air.channel @V2L1_0 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_1 [1]
  air.channel @V2L1_1 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_2 [1]
  air.channel @V2L1_2 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_3 [1]
  air.channel @V2L1_3 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @cascade [4, 3] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  air.channel @GpOut [1]
  func.func @attention_bf16(%arg0: memref<1x2048x64xbf16>, %arg1: memref<1x2048x64xbf16>, %arg2: memref<1x2048x64xbf16>, %arg3: memref<1x2048x64xbf16>) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    air.launch (%arg4, %arg5) in (%arg6=%c8, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<1x2048x64xbf16>, memref<1x2048x64xbf16>, memref<1x2048x64xbf16>, memref<1x2048x64xbf16> {
      %c0 = arith.constant 0 : index
      %0 = affine.apply #map()[%arg5]
      %1 = affine.apply #map1()[%arg4]
      %2 = affine.apply #map2()[%0, %1]
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %c0_5 = arith.constant 0 : index
      %c8_6 = arith.constant 8 : index
      %c8_7 = arith.constant 8 : index
      %c8_8 = arith.constant 8 : index
      %c8_9 = arith.constant 8 : index
      %c8_10 = arith.constant 8 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c1_11 = arith.constant 1 : index
      air.channel.put  @Q2L1[%c0_0, %c0_1] (%arg8[%c0_2, %c0_3, %c0_4, %c0_5, %2] [%c8_6, %c8_7, %c8_8, %c8_9] [%c8_10, %c512, %c64, %c1_11]) : (memref<1x2048x64xbf16>)
      %3 = affine.apply #map2()[%0, %1]
      %c1_12 = arith.constant 1 : index
      %c0_13 = arith.constant 0 : index
      %c0_14 = arith.constant 0 : index
      %c0_15 = arith.constant 0 : index
      %c0_16 = arith.constant 0 : index
      %c4096 = arith.constant 4096 : index
      %c8_17 = arith.constant 8 : index
      %c8_18 = arith.constant 8 : index
      %c8_19 = arith.constant 8 : index
      %c8_20 = arith.constant 8 : index
      %c8_21 = arith.constant 8 : index
      %c512_22 = arith.constant 512 : index
      %c64_23 = arith.constant 64 : index
      %c1_24 = arith.constant 1 : index
      air.channel.put  @Q2L1[%c1_12, %c0_13] (%arg8[%c0_14, %c0_15, %c0_16, %c4096, %3] [%c8_17, %c8_18, %c8_19, %c8_20] [%c8_21, %c512_22, %c64_23, %c1_24]) : (memref<1x2048x64xbf16>)
      %4 = affine.apply #map2()[%0, %1]
      %c2 = arith.constant 2 : index
      %c0_25 = arith.constant 0 : index
      %c0_26 = arith.constant 0 : index
      %c0_27 = arith.constant 0 : index
      %c0_28 = arith.constant 0 : index
      %c8192 = arith.constant 8192 : index
      %c8_29 = arith.constant 8 : index
      %c8_30 = arith.constant 8 : index
      %c8_31 = arith.constant 8 : index
      %c8_32 = arith.constant 8 : index
      %c8_33 = arith.constant 8 : index
      %c512_34 = arith.constant 512 : index
      %c64_35 = arith.constant 64 : index
      %c1_36 = arith.constant 1 : index
      air.channel.put  @Q2L1[%c2, %c0_25] (%arg8[%c0_26, %c0_27, %c0_28, %c8192, %4] [%c8_29, %c8_30, %c8_31, %c8_32] [%c8_33, %c512_34, %c64_35, %c1_36]) : (memref<1x2048x64xbf16>)
      %5 = affine.apply #map2()[%0, %1]
      %c3 = arith.constant 3 : index
      %c0_37 = arith.constant 0 : index
      %c0_38 = arith.constant 0 : index
      %c0_39 = arith.constant 0 : index
      %c0_40 = arith.constant 0 : index
      %c12288 = arith.constant 12288 : index
      %c8_41 = arith.constant 8 : index
      %c8_42 = arith.constant 8 : index
      %c8_43 = arith.constant 8 : index
      %c8_44 = arith.constant 8 : index
      %c8_45 = arith.constant 8 : index
      %c512_46 = arith.constant 512 : index
      %c64_47 = arith.constant 64 : index
      %c1_48 = arith.constant 1 : index
      air.channel.put  @Q2L1[%c3, %c0_37] (%arg8[%c0_38, %c0_39, %c0_40, %c12288, %5] [%c8_41, %c8_42, %c8_43, %c8_44] [%c8_45, %c512_46, %c64_47, %c1_48]) : (memref<1x2048x64xbf16>)
      %c0_49 = arith.constant 0 : index
      %6 = affine.apply #map3()[%0, %c0_49]
      %c0_50 = arith.constant 0 : index
      %c0_51 = arith.constant 0 : index
      %c0_52 = arith.constant 0 : index
      %c0_53 = arith.constant 0 : index
      %c0_54 = arith.constant 0 : index
      %c8_55 = arith.constant 8 : index
      %c8_56 = arith.constant 8 : index
      %c8_57 = arith.constant 8 : index
      %c8_58 = arith.constant 8 : index
      %c512_59 = arith.constant 512 : index
      %c8_60 = arith.constant 8 : index
      %c64_61 = arith.constant 64 : index
      %c1_62 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_50, %c0_51] (%arg9[%c0_52, %c0_53, %c0_54, %6] [%c8_55, %c8_56, %c8_57, %c8_58] [%c512_59, %c8_60, %c64_61, %c1_62]) : (memref<1x2048x64xbf16>)
      %c512_63 = arith.constant 512 : index
      %7 = affine.apply #map3()[%0, %c512_63]
      %c0_64 = arith.constant 0 : index
      %c1_65 = arith.constant 1 : index
      %c0_66 = arith.constant 0 : index
      %c0_67 = arith.constant 0 : index
      %c0_68 = arith.constant 0 : index
      %c8_69 = arith.constant 8 : index
      %c8_70 = arith.constant 8 : index
      %c8_71 = arith.constant 8 : index
      %c8_72 = arith.constant 8 : index
      %c512_73 = arith.constant 512 : index
      %c8_74 = arith.constant 8 : index
      %c64_75 = arith.constant 64 : index
      %c1_76 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_64, %c1_65] (%arg9[%c0_66, %c0_67, %c0_68, %7] [%c8_69, %c8_70, %c8_71, %c8_72] [%c512_73, %c8_74, %c64_75, %c1_76]) : (memref<1x2048x64xbf16>)
      %c1024 = arith.constant 1024 : index
      %8 = affine.apply #map3()[%0, %c1024]
      %c0_77 = arith.constant 0 : index
      %c2_78 = arith.constant 2 : index
      %c0_79 = arith.constant 0 : index
      %c0_80 = arith.constant 0 : index
      %c0_81 = arith.constant 0 : index
      %c8_82 = arith.constant 8 : index
      %c8_83 = arith.constant 8 : index
      %c8_84 = arith.constant 8 : index
      %c8_85 = arith.constant 8 : index
      %c512_86 = arith.constant 512 : index
      %c8_87 = arith.constant 8 : index
      %c64_88 = arith.constant 64 : index
      %c1_89 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_77, %c2_78] (%arg9[%c0_79, %c0_80, %c0_81, %8] [%c8_82, %c8_83, %c8_84, %c8_85] [%c512_86, %c8_87, %c64_88, %c1_89]) : (memref<1x2048x64xbf16>)
      %c1536 = arith.constant 1536 : index
      %9 = affine.apply #map3()[%0, %c1536]
      %c0_90 = arith.constant 0 : index
      %c3_91 = arith.constant 3 : index
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
      air.channel.put  @K2L1[%c0_90, %c3_91] (%arg9[%c0_92, %c0_93, %c0_94, %9] [%c8_95, %c8_96, %c8_97, %c8_98] [%c512_99, %c8_100, %c64_101, %c1_102]) : (memref<1x2048x64xbf16>)
      %c64_103 = arith.constant 64 : index
      %10 = affine.apply #map3()[%0, %c64_103]
      %c0_104 = arith.constant 0 : index
      %c0_105 = arith.constant 0 : index
      %c0_106 = arith.constant 0 : index
      %c0_107 = arith.constant 0 : index
      %c0_108 = arith.constant 0 : index
      %c8_109 = arith.constant 8 : index
      %c8_110 = arith.constant 8 : index
      %c8_111 = arith.constant 8 : index
      %c8_112 = arith.constant 8 : index
      %c512_113 = arith.constant 512 : index
      %c8_114 = arith.constant 8 : index
      %c64_115 = arith.constant 64 : index
      %c1_116 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_104, %c0_105] (%arg9[%c0_106, %c0_107, %c0_108, %10] [%c8_109, %c8_110, %c8_111, %c8_112] [%c512_113, %c8_114, %c64_115, %c1_116]) : (memref<1x2048x64xbf16>)
      %c576 = arith.constant 576 : index
      %11 = affine.apply #map3()[%0, %c576]
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      %c0_119 = arith.constant 0 : index
      %c0_120 = arith.constant 0 : index
      %c0_121 = arith.constant 0 : index
      %c8_122 = arith.constant 8 : index
      %c8_123 = arith.constant 8 : index
      %c8_124 = arith.constant 8 : index
      %c8_125 = arith.constant 8 : index
      %c512_126 = arith.constant 512 : index
      %c8_127 = arith.constant 8 : index
      %c64_128 = arith.constant 64 : index
      %c1_129 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_117, %c1_118] (%arg9[%c0_119, %c0_120, %c0_121, %11] [%c8_122, %c8_123, %c8_124, %c8_125] [%c512_126, %c8_127, %c64_128, %c1_129]) : (memref<1x2048x64xbf16>)
      %c1088 = arith.constant 1088 : index
      %12 = affine.apply #map3()[%0, %c1088]
      %c0_130 = arith.constant 0 : index
      %c2_131 = arith.constant 2 : index
      %c0_132 = arith.constant 0 : index
      %c0_133 = arith.constant 0 : index
      %c0_134 = arith.constant 0 : index
      %c8_135 = arith.constant 8 : index
      %c8_136 = arith.constant 8 : index
      %c8_137 = arith.constant 8 : index
      %c8_138 = arith.constant 8 : index
      %c512_139 = arith.constant 512 : index
      %c8_140 = arith.constant 8 : index
      %c64_141 = arith.constant 64 : index
      %c1_142 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_130, %c2_131] (%arg9[%c0_132, %c0_133, %c0_134, %12] [%c8_135, %c8_136, %c8_137, %c8_138] [%c512_139, %c8_140, %c64_141, %c1_142]) : (memref<1x2048x64xbf16>)
      %c1600 = arith.constant 1600 : index
      %13 = affine.apply #map3()[%0, %c1600]
      %c0_143 = arith.constant 0 : index
      %c3_144 = arith.constant 3 : index
      %c0_145 = arith.constant 0 : index
      %c0_146 = arith.constant 0 : index
      %c0_147 = arith.constant 0 : index
      %c8_148 = arith.constant 8 : index
      %c8_149 = arith.constant 8 : index
      %c8_150 = arith.constant 8 : index
      %c8_151 = arith.constant 8 : index
      %c512_152 = arith.constant 512 : index
      %c8_153 = arith.constant 8 : index
      %c64_154 = arith.constant 64 : index
      %c1_155 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_143, %c3_144] (%arg9[%c0_145, %c0_146, %c0_147, %13] [%c8_148, %c8_149, %c8_150, %c8_151] [%c512_152, %c8_153, %c64_154, %c1_155]) : (memref<1x2048x64xbf16>)
      %c128 = arith.constant 128 : index
      %14 = affine.apply #map3()[%0, %c128]
      %c0_156 = arith.constant 0 : index
      %c0_157 = arith.constant 0 : index
      %c0_158 = arith.constant 0 : index
      %c0_159 = arith.constant 0 : index
      %c0_160 = arith.constant 0 : index
      %c8_161 = arith.constant 8 : index
      %c8_162 = arith.constant 8 : index
      %c8_163 = arith.constant 8 : index
      %c8_164 = arith.constant 8 : index
      %c512_165 = arith.constant 512 : index
      %c8_166 = arith.constant 8 : index
      %c64_167 = arith.constant 64 : index
      %c1_168 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_156, %c0_157] (%arg9[%c0_158, %c0_159, %c0_160, %14] [%c8_161, %c8_162, %c8_163, %c8_164] [%c512_165, %c8_166, %c64_167, %c1_168]) : (memref<1x2048x64xbf16>)
      %c640 = arith.constant 640 : index
      %15 = affine.apply #map3()[%0, %c640]
      %c0_169 = arith.constant 0 : index
      %c1_170 = arith.constant 1 : index
      %c0_171 = arith.constant 0 : index
      %c0_172 = arith.constant 0 : index
      %c0_173 = arith.constant 0 : index
      %c8_174 = arith.constant 8 : index
      %c8_175 = arith.constant 8 : index
      %c8_176 = arith.constant 8 : index
      %c8_177 = arith.constant 8 : index
      %c512_178 = arith.constant 512 : index
      %c8_179 = arith.constant 8 : index
      %c64_180 = arith.constant 64 : index
      %c1_181 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_169, %c1_170] (%arg9[%c0_171, %c0_172, %c0_173, %15] [%c8_174, %c8_175, %c8_176, %c8_177] [%c512_178, %c8_179, %c64_180, %c1_181]) : (memref<1x2048x64xbf16>)
      %c1152 = arith.constant 1152 : index
      %16 = affine.apply #map3()[%0, %c1152]
      %c0_182 = arith.constant 0 : index
      %c2_183 = arith.constant 2 : index
      %c0_184 = arith.constant 0 : index
      %c0_185 = arith.constant 0 : index
      %c0_186 = arith.constant 0 : index
      %c8_187 = arith.constant 8 : index
      %c8_188 = arith.constant 8 : index
      %c8_189 = arith.constant 8 : index
      %c8_190 = arith.constant 8 : index
      %c512_191 = arith.constant 512 : index
      %c8_192 = arith.constant 8 : index
      %c64_193 = arith.constant 64 : index
      %c1_194 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_182, %c2_183] (%arg9[%c0_184, %c0_185, %c0_186, %16] [%c8_187, %c8_188, %c8_189, %c8_190] [%c512_191, %c8_192, %c64_193, %c1_194]) : (memref<1x2048x64xbf16>)
      %c1664 = arith.constant 1664 : index
      %17 = affine.apply #map3()[%0, %c1664]
      %c0_195 = arith.constant 0 : index
      %c3_196 = arith.constant 3 : index
      %c0_197 = arith.constant 0 : index
      %c0_198 = arith.constant 0 : index
      %c0_199 = arith.constant 0 : index
      %c8_200 = arith.constant 8 : index
      %c8_201 = arith.constant 8 : index
      %c8_202 = arith.constant 8 : index
      %c8_203 = arith.constant 8 : index
      %c512_204 = arith.constant 512 : index
      %c8_205 = arith.constant 8 : index
      %c64_206 = arith.constant 64 : index
      %c1_207 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_195, %c3_196] (%arg9[%c0_197, %c0_198, %c0_199, %17] [%c8_200, %c8_201, %c8_202, %c8_203] [%c512_204, %c8_205, %c64_206, %c1_207]) : (memref<1x2048x64xbf16>)
      %c192 = arith.constant 192 : index
      %18 = affine.apply #map3()[%0, %c192]
      %c0_208 = arith.constant 0 : index
      %c0_209 = arith.constant 0 : index
      %c0_210 = arith.constant 0 : index
      %c0_211 = arith.constant 0 : index
      %c0_212 = arith.constant 0 : index
      %c8_213 = arith.constant 8 : index
      %c8_214 = arith.constant 8 : index
      %c8_215 = arith.constant 8 : index
      %c8_216 = arith.constant 8 : index
      %c512_217 = arith.constant 512 : index
      %c8_218 = arith.constant 8 : index
      %c64_219 = arith.constant 64 : index
      %c1_220 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_208, %c0_209] (%arg9[%c0_210, %c0_211, %c0_212, %18] [%c8_213, %c8_214, %c8_215, %c8_216] [%c512_217, %c8_218, %c64_219, %c1_220]) : (memref<1x2048x64xbf16>)
      %c704 = arith.constant 704 : index
      %19 = affine.apply #map3()[%0, %c704]
      %c0_221 = arith.constant 0 : index
      %c1_222 = arith.constant 1 : index
      %c0_223 = arith.constant 0 : index
      %c0_224 = arith.constant 0 : index
      %c0_225 = arith.constant 0 : index
      %c8_226 = arith.constant 8 : index
      %c8_227 = arith.constant 8 : index
      %c8_228 = arith.constant 8 : index
      %c8_229 = arith.constant 8 : index
      %c512_230 = arith.constant 512 : index
      %c8_231 = arith.constant 8 : index
      %c64_232 = arith.constant 64 : index
      %c1_233 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_221, %c1_222] (%arg9[%c0_223, %c0_224, %c0_225, %19] [%c8_226, %c8_227, %c8_228, %c8_229] [%c512_230, %c8_231, %c64_232, %c1_233]) : (memref<1x2048x64xbf16>)
      %c1216 = arith.constant 1216 : index
      %20 = affine.apply #map3()[%0, %c1216]
      %c0_234 = arith.constant 0 : index
      %c2_235 = arith.constant 2 : index
      %c0_236 = arith.constant 0 : index
      %c0_237 = arith.constant 0 : index
      %c0_238 = arith.constant 0 : index
      %c8_239 = arith.constant 8 : index
      %c8_240 = arith.constant 8 : index
      %c8_241 = arith.constant 8 : index
      %c8_242 = arith.constant 8 : index
      %c512_243 = arith.constant 512 : index
      %c8_244 = arith.constant 8 : index
      %c64_245 = arith.constant 64 : index
      %c1_246 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_234, %c2_235] (%arg9[%c0_236, %c0_237, %c0_238, %20] [%c8_239, %c8_240, %c8_241, %c8_242] [%c512_243, %c8_244, %c64_245, %c1_246]) : (memref<1x2048x64xbf16>)
      %c1728 = arith.constant 1728 : index
      %21 = affine.apply #map3()[%0, %c1728]
      %c0_247 = arith.constant 0 : index
      %c3_248 = arith.constant 3 : index
      %c0_249 = arith.constant 0 : index
      %c0_250 = arith.constant 0 : index
      %c0_251 = arith.constant 0 : index
      %c8_252 = arith.constant 8 : index
      %c8_253 = arith.constant 8 : index
      %c8_254 = arith.constant 8 : index
      %c8_255 = arith.constant 8 : index
      %c512_256 = arith.constant 512 : index
      %c8_257 = arith.constant 8 : index
      %c64_258 = arith.constant 64 : index
      %c1_259 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_247, %c3_248] (%arg9[%c0_249, %c0_250, %c0_251, %21] [%c8_252, %c8_253, %c8_254, %c8_255] [%c512_256, %c8_257, %c64_258, %c1_259]) : (memref<1x2048x64xbf16>)
      %c256 = arith.constant 256 : index
      %22 = affine.apply #map3()[%0, %c256]
      %c0_260 = arith.constant 0 : index
      %c0_261 = arith.constant 0 : index
      %c0_262 = arith.constant 0 : index
      %c0_263 = arith.constant 0 : index
      %c0_264 = arith.constant 0 : index
      %c8_265 = arith.constant 8 : index
      %c8_266 = arith.constant 8 : index
      %c8_267 = arith.constant 8 : index
      %c8_268 = arith.constant 8 : index
      %c512_269 = arith.constant 512 : index
      %c8_270 = arith.constant 8 : index
      %c64_271 = arith.constant 64 : index
      %c1_272 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_260, %c0_261] (%arg9[%c0_262, %c0_263, %c0_264, %22] [%c8_265, %c8_266, %c8_267, %c8_268] [%c512_269, %c8_270, %c64_271, %c1_272]) : (memref<1x2048x64xbf16>)
      %c768 = arith.constant 768 : index
      %23 = affine.apply #map3()[%0, %c768]
      %c0_273 = arith.constant 0 : index
      %c1_274 = arith.constant 1 : index
      %c0_275 = arith.constant 0 : index
      %c0_276 = arith.constant 0 : index
      %c0_277 = arith.constant 0 : index
      %c8_278 = arith.constant 8 : index
      %c8_279 = arith.constant 8 : index
      %c8_280 = arith.constant 8 : index
      %c8_281 = arith.constant 8 : index
      %c512_282 = arith.constant 512 : index
      %c8_283 = arith.constant 8 : index
      %c64_284 = arith.constant 64 : index
      %c1_285 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_273, %c1_274] (%arg9[%c0_275, %c0_276, %c0_277, %23] [%c8_278, %c8_279, %c8_280, %c8_281] [%c512_282, %c8_283, %c64_284, %c1_285]) : (memref<1x2048x64xbf16>)
      %c1280 = arith.constant 1280 : index
      %24 = affine.apply #map3()[%0, %c1280]
      %c0_286 = arith.constant 0 : index
      %c2_287 = arith.constant 2 : index
      %c0_288 = arith.constant 0 : index
      %c0_289 = arith.constant 0 : index
      %c0_290 = arith.constant 0 : index
      %c8_291 = arith.constant 8 : index
      %c8_292 = arith.constant 8 : index
      %c8_293 = arith.constant 8 : index
      %c8_294 = arith.constant 8 : index
      %c512_295 = arith.constant 512 : index
      %c8_296 = arith.constant 8 : index
      %c64_297 = arith.constant 64 : index
      %c1_298 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_286, %c2_287] (%arg9[%c0_288, %c0_289, %c0_290, %24] [%c8_291, %c8_292, %c8_293, %c8_294] [%c512_295, %c8_296, %c64_297, %c1_298]) : (memref<1x2048x64xbf16>)
      %c1792 = arith.constant 1792 : index
      %25 = affine.apply #map3()[%0, %c1792]
      %c0_299 = arith.constant 0 : index
      %c3_300 = arith.constant 3 : index
      %c0_301 = arith.constant 0 : index
      %c0_302 = arith.constant 0 : index
      %c0_303 = arith.constant 0 : index
      %c8_304 = arith.constant 8 : index
      %c8_305 = arith.constant 8 : index
      %c8_306 = arith.constant 8 : index
      %c8_307 = arith.constant 8 : index
      %c512_308 = arith.constant 512 : index
      %c8_309 = arith.constant 8 : index
      %c64_310 = arith.constant 64 : index
      %c1_311 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_299, %c3_300] (%arg9[%c0_301, %c0_302, %c0_303, %25] [%c8_304, %c8_305, %c8_306, %c8_307] [%c512_308, %c8_309, %c64_310, %c1_311]) : (memref<1x2048x64xbf16>)
      %c320 = arith.constant 320 : index
      %26 = affine.apply #map3()[%0, %c320]
      %c0_312 = arith.constant 0 : index
      %c0_313 = arith.constant 0 : index
      %c0_314 = arith.constant 0 : index
      %c0_315 = arith.constant 0 : index
      %c0_316 = arith.constant 0 : index
      %c8_317 = arith.constant 8 : index
      %c8_318 = arith.constant 8 : index
      %c8_319 = arith.constant 8 : index
      %c8_320 = arith.constant 8 : index
      %c512_321 = arith.constant 512 : index
      %c8_322 = arith.constant 8 : index
      %c64_323 = arith.constant 64 : index
      %c1_324 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_312, %c0_313] (%arg9[%c0_314, %c0_315, %c0_316, %26] [%c8_317, %c8_318, %c8_319, %c8_320] [%c512_321, %c8_322, %c64_323, %c1_324]) : (memref<1x2048x64xbf16>)
      %c832 = arith.constant 832 : index
      %27 = affine.apply #map3()[%0, %c832]
      %c0_325 = arith.constant 0 : index
      %c1_326 = arith.constant 1 : index
      %c0_327 = arith.constant 0 : index
      %c0_328 = arith.constant 0 : index
      %c0_329 = arith.constant 0 : index
      %c8_330 = arith.constant 8 : index
      %c8_331 = arith.constant 8 : index
      %c8_332 = arith.constant 8 : index
      %c8_333 = arith.constant 8 : index
      %c512_334 = arith.constant 512 : index
      %c8_335 = arith.constant 8 : index
      %c64_336 = arith.constant 64 : index
      %c1_337 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_325, %c1_326] (%arg9[%c0_327, %c0_328, %c0_329, %27] [%c8_330, %c8_331, %c8_332, %c8_333] [%c512_334, %c8_335, %c64_336, %c1_337]) : (memref<1x2048x64xbf16>)
      %c1344 = arith.constant 1344 : index
      %28 = affine.apply #map3()[%0, %c1344]
      %c0_338 = arith.constant 0 : index
      %c2_339 = arith.constant 2 : index
      %c0_340 = arith.constant 0 : index
      %c0_341 = arith.constant 0 : index
      %c0_342 = arith.constant 0 : index
      %c8_343 = arith.constant 8 : index
      %c8_344 = arith.constant 8 : index
      %c8_345 = arith.constant 8 : index
      %c8_346 = arith.constant 8 : index
      %c512_347 = arith.constant 512 : index
      %c8_348 = arith.constant 8 : index
      %c64_349 = arith.constant 64 : index
      %c1_350 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_338, %c2_339] (%arg9[%c0_340, %c0_341, %c0_342, %28] [%c8_343, %c8_344, %c8_345, %c8_346] [%c512_347, %c8_348, %c64_349, %c1_350]) : (memref<1x2048x64xbf16>)
      %c1856 = arith.constant 1856 : index
      %29 = affine.apply #map3()[%0, %c1856]
      %c0_351 = arith.constant 0 : index
      %c3_352 = arith.constant 3 : index
      %c0_353 = arith.constant 0 : index
      %c0_354 = arith.constant 0 : index
      %c0_355 = arith.constant 0 : index
      %c8_356 = arith.constant 8 : index
      %c8_357 = arith.constant 8 : index
      %c8_358 = arith.constant 8 : index
      %c8_359 = arith.constant 8 : index
      %c512_360 = arith.constant 512 : index
      %c8_361 = arith.constant 8 : index
      %c64_362 = arith.constant 64 : index
      %c1_363 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_351, %c3_352] (%arg9[%c0_353, %c0_354, %c0_355, %29] [%c8_356, %c8_357, %c8_358, %c8_359] [%c512_360, %c8_361, %c64_362, %c1_363]) : (memref<1x2048x64xbf16>)
      %c384 = arith.constant 384 : index
      %30 = affine.apply #map3()[%0, %c384]
      %c0_364 = arith.constant 0 : index
      %c0_365 = arith.constant 0 : index
      %c0_366 = arith.constant 0 : index
      %c0_367 = arith.constant 0 : index
      %c0_368 = arith.constant 0 : index
      %c8_369 = arith.constant 8 : index
      %c8_370 = arith.constant 8 : index
      %c8_371 = arith.constant 8 : index
      %c8_372 = arith.constant 8 : index
      %c512_373 = arith.constant 512 : index
      %c8_374 = arith.constant 8 : index
      %c64_375 = arith.constant 64 : index
      %c1_376 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_364, %c0_365] (%arg9[%c0_366, %c0_367, %c0_368, %30] [%c8_369, %c8_370, %c8_371, %c8_372] [%c512_373, %c8_374, %c64_375, %c1_376]) : (memref<1x2048x64xbf16>)
      %c896 = arith.constant 896 : index
      %31 = affine.apply #map3()[%0, %c896]
      %c0_377 = arith.constant 0 : index
      %c1_378 = arith.constant 1 : index
      %c0_379 = arith.constant 0 : index
      %c0_380 = arith.constant 0 : index
      %c0_381 = arith.constant 0 : index
      %c8_382 = arith.constant 8 : index
      %c8_383 = arith.constant 8 : index
      %c8_384 = arith.constant 8 : index
      %c8_385 = arith.constant 8 : index
      %c512_386 = arith.constant 512 : index
      %c8_387 = arith.constant 8 : index
      %c64_388 = arith.constant 64 : index
      %c1_389 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_377, %c1_378] (%arg9[%c0_379, %c0_380, %c0_381, %31] [%c8_382, %c8_383, %c8_384, %c8_385] [%c512_386, %c8_387, %c64_388, %c1_389]) : (memref<1x2048x64xbf16>)
      %c1408 = arith.constant 1408 : index
      %32 = affine.apply #map3()[%0, %c1408]
      %c0_390 = arith.constant 0 : index
      %c2_391 = arith.constant 2 : index
      %c0_392 = arith.constant 0 : index
      %c0_393 = arith.constant 0 : index
      %c0_394 = arith.constant 0 : index
      %c8_395 = arith.constant 8 : index
      %c8_396 = arith.constant 8 : index
      %c8_397 = arith.constant 8 : index
      %c8_398 = arith.constant 8 : index
      %c512_399 = arith.constant 512 : index
      %c8_400 = arith.constant 8 : index
      %c64_401 = arith.constant 64 : index
      %c1_402 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_390, %c2_391] (%arg9[%c0_392, %c0_393, %c0_394, %32] [%c8_395, %c8_396, %c8_397, %c8_398] [%c512_399, %c8_400, %c64_401, %c1_402]) : (memref<1x2048x64xbf16>)
      %c1920 = arith.constant 1920 : index
      %33 = affine.apply #map3()[%0, %c1920]
      %c0_403 = arith.constant 0 : index
      %c3_404 = arith.constant 3 : index
      %c0_405 = arith.constant 0 : index
      %c0_406 = arith.constant 0 : index
      %c0_407 = arith.constant 0 : index
      %c8_408 = arith.constant 8 : index
      %c8_409 = arith.constant 8 : index
      %c8_410 = arith.constant 8 : index
      %c8_411 = arith.constant 8 : index
      %c512_412 = arith.constant 512 : index
      %c8_413 = arith.constant 8 : index
      %c64_414 = arith.constant 64 : index
      %c1_415 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_403, %c3_404] (%arg9[%c0_405, %c0_406, %c0_407, %33] [%c8_408, %c8_409, %c8_410, %c8_411] [%c512_412, %c8_413, %c64_414, %c1_415]) : (memref<1x2048x64xbf16>)
      %c448 = arith.constant 448 : index
      %34 = affine.apply #map3()[%0, %c448]
      %c0_416 = arith.constant 0 : index
      %c0_417 = arith.constant 0 : index
      %c0_418 = arith.constant 0 : index
      %c0_419 = arith.constant 0 : index
      %c0_420 = arith.constant 0 : index
      %c8_421 = arith.constant 8 : index
      %c8_422 = arith.constant 8 : index
      %c8_423 = arith.constant 8 : index
      %c8_424 = arith.constant 8 : index
      %c512_425 = arith.constant 512 : index
      %c8_426 = arith.constant 8 : index
      %c64_427 = arith.constant 64 : index
      %c1_428 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_416, %c0_417] (%arg9[%c0_418, %c0_419, %c0_420, %34] [%c8_421, %c8_422, %c8_423, %c8_424] [%c512_425, %c8_426, %c64_427, %c1_428]) : (memref<1x2048x64xbf16>)
      %c960 = arith.constant 960 : index
      %35 = affine.apply #map3()[%0, %c960]
      %c0_429 = arith.constant 0 : index
      %c1_430 = arith.constant 1 : index
      %c0_431 = arith.constant 0 : index
      %c0_432 = arith.constant 0 : index
      %c0_433 = arith.constant 0 : index
      %c8_434 = arith.constant 8 : index
      %c8_435 = arith.constant 8 : index
      %c8_436 = arith.constant 8 : index
      %c8_437 = arith.constant 8 : index
      %c512_438 = arith.constant 512 : index
      %c8_439 = arith.constant 8 : index
      %c64_440 = arith.constant 64 : index
      %c1_441 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_429, %c1_430] (%arg9[%c0_431, %c0_432, %c0_433, %35] [%c8_434, %c8_435, %c8_436, %c8_437] [%c512_438, %c8_439, %c64_440, %c1_441]) : (memref<1x2048x64xbf16>)
      %c1472 = arith.constant 1472 : index
      %36 = affine.apply #map3()[%0, %c1472]
      %c0_442 = arith.constant 0 : index
      %c2_443 = arith.constant 2 : index
      %c0_444 = arith.constant 0 : index
      %c0_445 = arith.constant 0 : index
      %c0_446 = arith.constant 0 : index
      %c8_447 = arith.constant 8 : index
      %c8_448 = arith.constant 8 : index
      %c8_449 = arith.constant 8 : index
      %c8_450 = arith.constant 8 : index
      %c512_451 = arith.constant 512 : index
      %c8_452 = arith.constant 8 : index
      %c64_453 = arith.constant 64 : index
      %c1_454 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_442, %c2_443] (%arg9[%c0_444, %c0_445, %c0_446, %36] [%c8_447, %c8_448, %c8_449, %c8_450] [%c512_451, %c8_452, %c64_453, %c1_454]) : (memref<1x2048x64xbf16>)
      %c1984 = arith.constant 1984 : index
      %37 = affine.apply #map3()[%0, %c1984]
      %c0_455 = arith.constant 0 : index
      %c3_456 = arith.constant 3 : index
      %c0_457 = arith.constant 0 : index
      %c0_458 = arith.constant 0 : index
      %c0_459 = arith.constant 0 : index
      %c8_460 = arith.constant 8 : index
      %c8_461 = arith.constant 8 : index
      %c8_462 = arith.constant 8 : index
      %c8_463 = arith.constant 8 : index
      %c512_464 = arith.constant 512 : index
      %c8_465 = arith.constant 8 : index
      %c64_466 = arith.constant 64 : index
      %c1_467 = arith.constant 1 : index
      air.channel.put  @K2L1[%c0_455, %c3_456] (%arg9[%c0_457, %c0_458, %c0_459, %37] [%c8_460, %c8_461, %c8_462, %c8_463] [%c512_464, %c8_465, %c64_466, %c1_467]) : (memref<1x2048x64xbf16>)
      %38 = affine.apply #map4()[%0]
      %c0_468 = arith.constant 0 : index
      %c0_469 = arith.constant 0 : index
      %c0_470 = arith.constant 0 : index
      %c0_471 = arith.constant 0 : index
      %c8_472 = arith.constant 8 : index
      %c64_473 = arith.constant 64 : index
      %c64_474 = arith.constant 64 : index
      %c4096_475 = arith.constant 4096 : index
      %c64_476 = arith.constant 64 : index
      %c1_477 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c0_468] (%arg10[%c0_469, %c0_470, %c0_471, %38] [%c8_472, %c64_473, %c64_474] [%c4096_475, %c64_476, %c1_477]) : (memref<1x2048x64xbf16>)
      %39 = affine.apply #map4()[%0]
      %c0_478 = arith.constant 0 : index
      %c0_479 = arith.constant 0 : index
      %c0_480 = arith.constant 0 : index
      %c512_481 = arith.constant 512 : index
      %c8_482 = arith.constant 8 : index
      %c64_483 = arith.constant 64 : index
      %c64_484 = arith.constant 64 : index
      %c4096_485 = arith.constant 4096 : index
      %c64_486 = arith.constant 64 : index
      %c1_487 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c0_478] (%arg10[%c0_479, %c0_480, %c512_481, %39] [%c8_482, %c64_483, %c64_484] [%c4096_485, %c64_486, %c1_487]) : (memref<1x2048x64xbf16>)
      %40 = affine.apply #map4()[%0]
      %c0_488 = arith.constant 0 : index
      %c0_489 = arith.constant 0 : index
      %c0_490 = arith.constant 0 : index
      %c1024_491 = arith.constant 1024 : index
      %c8_492 = arith.constant 8 : index
      %c64_493 = arith.constant 64 : index
      %c64_494 = arith.constant 64 : index
      %c4096_495 = arith.constant 4096 : index
      %c64_496 = arith.constant 64 : index
      %c1_497 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c0_488] (%arg10[%c0_489, %c0_490, %c1024_491, %40] [%c8_492, %c64_493, %c64_494] [%c4096_495, %c64_496, %c1_497]) : (memref<1x2048x64xbf16>)
      %41 = affine.apply #map4()[%0]
      %c0_498 = arith.constant 0 : index
      %c0_499 = arith.constant 0 : index
      %c0_500 = arith.constant 0 : index
      %c1536_501 = arith.constant 1536 : index
      %c8_502 = arith.constant 8 : index
      %c64_503 = arith.constant 64 : index
      %c64_504 = arith.constant 64 : index
      %c4096_505 = arith.constant 4096 : index
      %c64_506 = arith.constant 64 : index
      %c1_507 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c0_498] (%arg10[%c0_499, %c0_500, %c1536_501, %41] [%c8_502, %c64_503, %c64_504] [%c4096_505, %c64_506, %c1_507]) : (memref<1x2048x64xbf16>)
      %42 = affine.apply #map2()[%0, %1]
      %c0_508 = arith.constant 0 : index
      %c0_509 = arith.constant 0 : index
      %c256_510 = arith.constant 256 : index
      %c64_511 = arith.constant 64 : index
      %c64_512 = arith.constant 64 : index
      %c1_513 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0_508] (%arg11[%c0_509, %42] [%c256_510, %c64_511] [%c64_512, %c1_513]) : (memref<1x2048x64xbf16>)
      %c1_514 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c1_514, %arg15=%c1_514) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_515 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_516 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_517 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_518 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_519 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_520 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_521 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_522 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_523 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_524 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %c4 = arith.constant 4 : index
        %c4_525 = arith.constant 4 : index
        %c0_526 = arith.constant 0 : index
        %c8_527 = arith.constant 8 : index
        %c0_528 = arith.constant 0 : index
        %c1_529 = arith.constant 1 : index
        scf.for %arg16 = %c0_528 to %c8_527 step %c1_529 {
          %c0_531 = arith.constant 0 : index
          air.channel.get  @VIn_0[%c0_531] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_532 = arith.constant 0 : index
          %c0_533 = arith.constant 0 : index
          %c0_534 = arith.constant 0 : index
          %c0_535 = arith.constant 0 : index
          %c8_536 = arith.constant 8 : index
          %c8_537 = arith.constant 8 : index
          %c8_538 = arith.constant 8 : index
          %c8_539 = arith.constant 8 : index
          %c8_540 = arith.constant 8 : index
          %c512_541 = arith.constant 512 : index
          %c64_542 = arith.constant 64 : index
          %c1_543 = arith.constant 1 : index
          air.channel.put  @V2L1_0[%c0_526, %c0_526] (%alloc[%c0_532, %c0_533, %c0_534, %c0_535] [%c8_536, %c8_537, %c8_538, %c8_539] [%c8_540, %c512_541, %c64_542, %c1_543]) : (memref<64x64xbf16, 1 : i32>)
          %c0_544 = arith.constant 0 : index
          air.channel.get  @VIn_1[%c0_544] (%alloc_515[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_545 = arith.constant 0 : index
          %c0_546 = arith.constant 0 : index
          %c0_547 = arith.constant 0 : index
          %c0_548 = arith.constant 0 : index
          %c8_549 = arith.constant 8 : index
          %c8_550 = arith.constant 8 : index
          %c8_551 = arith.constant 8 : index
          %c8_552 = arith.constant 8 : index
          %c8_553 = arith.constant 8 : index
          %c512_554 = arith.constant 512 : index
          %c64_555 = arith.constant 64 : index
          %c1_556 = arith.constant 1 : index
          air.channel.put  @V2L1_1[%c0_526, %c0_526] (%alloc_515[%c0_545, %c0_546, %c0_547, %c0_548] [%c8_549, %c8_550, %c8_551, %c8_552] [%c8_553, %c512_554, %c64_555, %c1_556]) : (memref<64x64xbf16, 1 : i32>)
          %c0_557 = arith.constant 0 : index
          air.channel.get  @VIn_2[%c0_557] (%alloc_516[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_558 = arith.constant 0 : index
          %c0_559 = arith.constant 0 : index
          %c0_560 = arith.constant 0 : index
          %c0_561 = arith.constant 0 : index
          %c8_562 = arith.constant 8 : index
          %c8_563 = arith.constant 8 : index
          %c8_564 = arith.constant 8 : index
          %c8_565 = arith.constant 8 : index
          %c8_566 = arith.constant 8 : index
          %c512_567 = arith.constant 512 : index
          %c64_568 = arith.constant 64 : index
          %c1_569 = arith.constant 1 : index
          air.channel.put  @V2L1_2[%c0_526, %c0_526] (%alloc_516[%c0_558, %c0_559, %c0_560, %c0_561] [%c8_562, %c8_563, %c8_564, %c8_565] [%c8_566, %c512_567, %c64_568, %c1_569]) : (memref<64x64xbf16, 1 : i32>)
          %c0_570 = arith.constant 0 : index
          air.channel.get  @VIn_3[%c0_570] (%alloc_517[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_571 = arith.constant 0 : index
          %c0_572 = arith.constant 0 : index
          %c0_573 = arith.constant 0 : index
          %c0_574 = arith.constant 0 : index
          %c8_575 = arith.constant 8 : index
          %c8_576 = arith.constant 8 : index
          %c8_577 = arith.constant 8 : index
          %c8_578 = arith.constant 8 : index
          %c8_579 = arith.constant 8 : index
          %c512_580 = arith.constant 512 : index
          %c64_581 = arith.constant 64 : index
          %c1_582 = arith.constant 1 : index
          air.channel.put  @V2L1_3[%c0_526, %c0_526] (%alloc_517[%c0_571, %c0_572, %c0_573, %c0_574] [%c8_575, %c8_576, %c8_577, %c8_578] [%c8_579, %c512_580, %c64_581, %c1_582]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %43 = affine.apply #map5()[%arg16]
          %c0_531 = arith.constant 0 : index
          %c0_532 = arith.constant 0 : index
          %c64_533 = arith.constant 64 : index
          %c64_534 = arith.constant 64 : index
          %c64_535 = arith.constant 64 : index
          %c1_536 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%arg16, %c0_531] (%alloc_518[%43, %c0_532] [%c64_533, %c64_534] [%c64_535, %c1_536]) : (memref<256x64xbf16, 1 : i32>)
        }
        %c0_530 = arith.constant 0 : index
        air.channel.put  @GpOut[%c0_530] (%alloc_518[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4_525) args(%arg20=%alloc_519, %arg21=%alloc_520, %arg22=%alloc_521, %arg23=%alloc_522, %arg24=%alloc_523, %arg25=%alloc_524) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32> attributes {link_with = "attn.o"} {
          air.channel.get  @Q2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_sp_bf16(%arg23) : (memref<64x1xbf16, 2 : i32>) -> ()
          func.call @neg_inf_fill_up_bf16(%arg22) : (memref<64x1xbf16, 2 : i32>) -> ()
          %c8_531 = arith.constant 8 : index
          %c0_532 = arith.constant 0 : index
          %c1_533 = arith.constant 1 : index
          scf.for %arg26 = %c0_532 to %c8_531 step %c1_533 {
            air.channel.get  @K2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            %collapse_shape = memref.collapse_shape %arg25 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            %alloc_536 = memref.alloc() : memref<64x64xbf16, 2 : i32>
            affine.if #set()[%arg16, %arg17] {
              air.channel.get  @V2L1_0[%arg16, %c0_532] (%alloc_536[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              air.channel.get  @V2L1_1[%arg16, %c0_532] (%alloc_536[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg16, %arg17] {
              air.channel.get  @V2L1_2[%arg16, %c0_532] (%alloc_536[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg16, %arg17] {
              air.channel.get  @V2L1_3[%arg16, %c0_532] (%alloc_536[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            %c0_i32 = arith.constant 0 : i32
            %alloc_537 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_538 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg22, %alloc_537, %alloc_538) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_538, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %alloc_536, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @accum_sp_r_s(%arg23, %alloc_538, %alloc_537) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32, %alloc_537, %arg23) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_537 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_538 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_536 : memref<64x64xbf16, 2 : i32>
          }
          %c1_534 = arith.constant 1 : index
          %alloc_535 = memref.alloc() : memref<64x1xbf16, 2 : i32>
          affine.if #set4()[%arg16, %arg17] {
            %43 = arith.subi %arg17, %c1_534 : index
            air.channel.put  @cascade[%arg16, %43] (%arg24[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade[%arg16, %43] (%arg22[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade[%arg16, %43] (%arg23[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set5()[%arg16, %arg17] {
              %alloc_536 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_537 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade[%arg16, %arg17] (%arg25[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade[%arg16, %arg17] (%alloc_536[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade[%arg16, %arg17] (%alloc_537[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_538 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32, %arg22, %alloc_538) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_536, %arg22) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @exp_up_minus_u(%alloc_536, %arg22, %alloc_535) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_539 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_538, %arg22, %alloc_539) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_535, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_539, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %arg25) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_540 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_540) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_537, %alloc_535, %alloc_540) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg23, %alloc_539, %alloc_540) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_540, %alloc_537) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %43 = arith.subi %arg17, %c1_534 : index
              air.channel.put  @cascade[%arg16, %43] (%arg25[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade[%arg16, %43] (%arg22[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade[%arg16, %43] (%alloc_537[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_538 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_539 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_540 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_536 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_537 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade[%arg16, %arg17] (%arg25[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade[%arg16, %arg17] (%alloc_536[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade[%arg16, %arg17] (%alloc_537[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_538 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32, %arg22, %alloc_538) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_536, %arg22) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @exp_up_minus_u(%alloc_536, %arg22, %alloc_535) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_539 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_538, %arg22, %alloc_539) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_535, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_539, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %arg25) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_540 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_540) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_537, %alloc_535, %alloc_540) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg23, %alloc_539, %alloc_540) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_540, %alloc_537) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_537, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              memref.dealloc %alloc_538 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_539 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_540 : memref<64x1xbf16, 2 : i32>
              %c0_541 = arith.constant 0 : index
              %c0_542 = arith.constant 0 : index
              %c0_543 = arith.constant 0 : index
              %c0_544 = arith.constant 0 : index
              %c0_545 = arith.constant 0 : index
              %c8_546 = arith.constant 8 : index
              %c8_547 = arith.constant 8 : index
              %c8_548 = arith.constant 8 : index
              %c8_549 = arith.constant 8 : index
              %c64_550 = arith.constant 64 : index
              %c8_551 = arith.constant 8 : index
              %c512_552 = arith.constant 512 : index
              %c1_553 = arith.constant 1 : index
              air.channel.put  @Gp2L2[%arg16, %c0_541] (%arg25[%c0_542, %c0_543, %c0_544, %c0_545] [%c8_546, %c8_547, %c8_548, %c8_549] [%c64_550, %c8_551, %c512_552, %c1_553]) : (memref<64x64xbf16, 2 : i32>)
            }
          }
        }
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_515 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_516 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_517 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_518 : memref<256x64xbf16, 1 : i32>
        memref.dealloc %alloc_519 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_520 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_521 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_522 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_523 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_524 : memref<64x64xbf16, 2 : i32>
      }
    }
    return
  }
}

