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
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    air.launch (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1_1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x128xbf16>, memref<2x512x128xbf16>, memref<2x512x64xbf16>, memref<2x256x64xbf16> {
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map()[%1]
      %3 = affine.apply #map2()[%1]
      %4 = affine.apply #map()[%1]
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
      %c0_49 = arith.constant 0 : index
      %c0_50 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c2_51 = arith.constant 2 : index
      %c64_52 = arith.constant 64 : index
      %c64_53 = arith.constant 64 : index
      %c8192 = arith.constant 8192 : index
      %c64_54 = arith.constant 64 : index
      %c128_55 = arith.constant 128 : index
      %c1_56 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c0] (%arg9[%c0_48, %c0_49, %c0_50, %14] [%c2, %c2_51, %c64_52, %c64_53] [%c8192, %c64_54, %c128_55, %c1_56]) : (memref<2x512x128xbf16>)
      %c16384 = arith.constant 16384 : index
      %15 = affine.apply #map3()[%3, %c16384]
      %c0_57 = arith.constant 0 : index
      %c0_58 = arith.constant 0 : index
      %c0_59 = arith.constant 0 : index
      %c2_60 = arith.constant 2 : index
      %c2_61 = arith.constant 2 : index
      %c64_62 = arith.constant 64 : index
      %c64_63 = arith.constant 64 : index
      %c8192_64 = arith.constant 8192 : index
      %c64_65 = arith.constant 64 : index
      %c128_66 = arith.constant 128 : index
      %c1_67 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c0] (%arg9[%c0_57, %c0_58, %c0_59, %15] [%c2_60, %c2_61, %c64_62, %c64_63] [%c8192_64, %c64_65, %c128_66, %c1_67]) : (memref<2x512x128xbf16>)
      %c32768 = arith.constant 32768 : index
      %16 = affine.apply #map3()[%3, %c32768]
      %c0_68 = arith.constant 0 : index
      %c0_69 = arith.constant 0 : index
      %c0_70 = arith.constant 0 : index
      %c2_71 = arith.constant 2 : index
      %c2_72 = arith.constant 2 : index
      %c64_73 = arith.constant 64 : index
      %c64_74 = arith.constant 64 : index
      %c8192_75 = arith.constant 8192 : index
      %c64_76 = arith.constant 64 : index
      %c128_77 = arith.constant 128 : index
      %c1_78 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c0] (%arg9[%c0_68, %c0_69, %c0_70, %16] [%c2_71, %c2_72, %c64_73, %c64_74] [%c8192_75, %c64_76, %c128_77, %c1_78]) : (memref<2x512x128xbf16>)
      %c49152 = arith.constant 49152 : index
      %17 = affine.apply #map3()[%3, %c49152]
      %c0_79 = arith.constant 0 : index
      %c0_80 = arith.constant 0 : index
      %c0_81 = arith.constant 0 : index
      %c2_82 = arith.constant 2 : index
      %c2_83 = arith.constant 2 : index
      %c64_84 = arith.constant 64 : index
      %c64_85 = arith.constant 64 : index
      %c8192_86 = arith.constant 8192 : index
      %c64_87 = arith.constant 64 : index
      %c128_88 = arith.constant 128 : index
      %c1_89 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c0] (%arg9[%c0_79, %c0_80, %c0_81, %17] [%c2_82, %c2_83, %c64_84, %c64_85] [%c8192_86, %c64_87, %c128_88, %c1_89]) : (memref<2x512x128xbf16>)
      %c0_90 = arith.constant 0 : index
      %18 = affine.apply #map3()[%4, %c0_90]
      %c0_91 = arith.constant 0 : index
      %c0_92 = arith.constant 0 : index
      %c2_93 = arith.constant 2 : index
      %c64_94 = arith.constant 64 : index
      %c64_95 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c64_96 = arith.constant 64 : index
      %c1_97 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c0] (%arg10[%c0_91, %c0_92, %18] [%c2_93, %c64_94, %c64_95] [%c4096, %c64_96, %c1_97]) : (memref<2x512x64xbf16>)
      %c8192_98 = arith.constant 8192 : index
      %19 = affine.apply #map3()[%4, %c8192_98]
      %c0_99 = arith.constant 0 : index
      %c0_100 = arith.constant 0 : index
      %c2_101 = arith.constant 2 : index
      %c64_102 = arith.constant 64 : index
      %c64_103 = arith.constant 64 : index
      %c4096_104 = arith.constant 4096 : index
      %c64_105 = arith.constant 64 : index
      %c1_106 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c0] (%arg10[%c0_99, %c0_100, %19] [%c2_101, %c64_102, %c64_103] [%c4096_104, %c64_105, %c1_106]) : (memref<2x512x64xbf16>)
      %c16384_107 = arith.constant 16384 : index
      %20 = affine.apply #map3()[%4, %c16384_107]
      %c0_108 = arith.constant 0 : index
      %c0_109 = arith.constant 0 : index
      %c2_110 = arith.constant 2 : index
      %c64_111 = arith.constant 64 : index
      %c64_112 = arith.constant 64 : index
      %c4096_113 = arith.constant 4096 : index
      %c64_114 = arith.constant 64 : index
      %c1_115 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c0] (%arg10[%c0_108, %c0_109, %20] [%c2_110, %c64_111, %c64_112] [%c4096_113, %c64_114, %c1_115]) : (memref<2x512x64xbf16>)
      %c24576 = arith.constant 24576 : index
      %21 = affine.apply #map3()[%4, %c24576]
      %c0_116 = arith.constant 0 : index
      %c0_117 = arith.constant 0 : index
      %c2_118 = arith.constant 2 : index
      %c64_119 = arith.constant 64 : index
      %c64_120 = arith.constant 64 : index
      %c4096_121 = arith.constant 4096 : index
      %c64_122 = arith.constant 64 : index
      %c1_123 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c0] (%arg10[%c0_116, %c0_117, %21] [%c2_118, %c64_119, %c64_120] [%c4096_121, %c64_122, %c1_123]) : (memref<2x512x64xbf16>)
      %c32768_124 = arith.constant 32768 : index
      %c1_125 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0] (%arg11[%5] [%c32768_124] [%c1_125]) : (memref<2x256x64xbf16>)
      %22 = affine.apply #map4()[%1]
      %23 = affine.apply #map()[%22]
      %24 = affine.apply #map2()[%22]
      %25 = affine.apply #map()[%22]
      %c1_126 = arith.constant 1 : index
      %26 = affine.apply #map3()[%23, %0]
      %c0_127 = arith.constant 0 : index
      %27 = affine.apply #map3()[%26, %c0_127]
      %c0_128 = arith.constant 0 : index
      %c256_129 = arith.constant 256 : index
      %c64_130 = arith.constant 64 : index
      %c128_131 = arith.constant 128 : index
      %c1_132 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_126] (%arg8[%c0_128, %27] [%c256_129, %c64_130] [%c128_131, %c1_132]) : (memref<2x256x128xbf16>)
      %c64_133 = arith.constant 64 : index
      %28 = affine.apply #map3()[%26, %c64_133]
      %c0_134 = arith.constant 0 : index
      %c256_135 = arith.constant 256 : index
      %c64_136 = arith.constant 64 : index
      %c128_137 = arith.constant 128 : index
      %c1_138 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_126] (%arg8[%c0_134, %28] [%c256_135, %c64_136] [%c128_137, %c1_138]) : (memref<2x256x128xbf16>)
      %c0_139 = arith.constant 0 : index
      %29 = affine.apply #map3()[%26, %c0_139]
      %c0_140 = arith.constant 0 : index
      %c256_141 = arith.constant 256 : index
      %c64_142 = arith.constant 64 : index
      %c128_143 = arith.constant 128 : index
      %c1_144 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_126] (%arg8[%c0_140, %29] [%c256_141, %c64_142] [%c128_143, %c1_144]) : (memref<2x256x128xbf16>)
      %c64_145 = arith.constant 64 : index
      %30 = affine.apply #map3()[%26, %c64_145]
      %c0_146 = arith.constant 0 : index
      %c256_147 = arith.constant 256 : index
      %c64_148 = arith.constant 64 : index
      %c128_149 = arith.constant 128 : index
      %c1_150 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_126] (%arg8[%c0_146, %30] [%c256_147, %c64_148] [%c128_149, %c1_150]) : (memref<2x256x128xbf16>)
      %c0_151 = arith.constant 0 : index
      %31 = affine.apply #map3()[%26, %c0_151]
      %c0_152 = arith.constant 0 : index
      %c256_153 = arith.constant 256 : index
      %c64_154 = arith.constant 64 : index
      %c128_155 = arith.constant 128 : index
      %c1_156 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_126] (%arg8[%c0_152, %31] [%c256_153, %c64_154] [%c128_155, %c1_156]) : (memref<2x256x128xbf16>)
      %c64_157 = arith.constant 64 : index
      %32 = affine.apply #map3()[%26, %c64_157]
      %c0_158 = arith.constant 0 : index
      %c256_159 = arith.constant 256 : index
      %c64_160 = arith.constant 64 : index
      %c128_161 = arith.constant 128 : index
      %c1_162 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_126] (%arg8[%c0_158, %32] [%c256_159, %c64_160] [%c128_161, %c1_162]) : (memref<2x256x128xbf16>)
      %c0_163 = arith.constant 0 : index
      %33 = affine.apply #map3()[%26, %c0_163]
      %c0_164 = arith.constant 0 : index
      %c256_165 = arith.constant 256 : index
      %c64_166 = arith.constant 64 : index
      %c128_167 = arith.constant 128 : index
      %c1_168 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_126] (%arg8[%c0_164, %33] [%c256_165, %c64_166] [%c128_167, %c1_168]) : (memref<2x256x128xbf16>)
      %c64_169 = arith.constant 64 : index
      %34 = affine.apply #map3()[%26, %c64_169]
      %c0_170 = arith.constant 0 : index
      %c256_171 = arith.constant 256 : index
      %c64_172 = arith.constant 64 : index
      %c128_173 = arith.constant 128 : index
      %c1_174 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_126] (%arg8[%c0_170, %34] [%c256_171, %c64_172] [%c128_173, %c1_174]) : (memref<2x256x128xbf16>)
      %c0_175 = arith.constant 0 : index
      %35 = affine.apply #map3()[%24, %c0_175]
      %c0_176 = arith.constant 0 : index
      %c0_177 = arith.constant 0 : index
      %c0_178 = arith.constant 0 : index
      %c2_179 = arith.constant 2 : index
      %c2_180 = arith.constant 2 : index
      %c64_181 = arith.constant 64 : index
      %c64_182 = arith.constant 64 : index
      %c8192_183 = arith.constant 8192 : index
      %c64_184 = arith.constant 64 : index
      %c128_185 = arith.constant 128 : index
      %c1_186 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_126] (%arg9[%c0_176, %c0_177, %c0_178, %35] [%c2_179, %c2_180, %c64_181, %c64_182] [%c8192_183, %c64_184, %c128_185, %c1_186]) : (memref<2x512x128xbf16>)
      %c16384_187 = arith.constant 16384 : index
      %36 = affine.apply #map3()[%24, %c16384_187]
      %c0_188 = arith.constant 0 : index
      %c0_189 = arith.constant 0 : index
      %c0_190 = arith.constant 0 : index
      %c2_191 = arith.constant 2 : index
      %c2_192 = arith.constant 2 : index
      %c64_193 = arith.constant 64 : index
      %c64_194 = arith.constant 64 : index
      %c8192_195 = arith.constant 8192 : index
      %c64_196 = arith.constant 64 : index
      %c128_197 = arith.constant 128 : index
      %c1_198 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_126] (%arg9[%c0_188, %c0_189, %c0_190, %36] [%c2_191, %c2_192, %c64_193, %c64_194] [%c8192_195, %c64_196, %c128_197, %c1_198]) : (memref<2x512x128xbf16>)
      %c32768_199 = arith.constant 32768 : index
      %37 = affine.apply #map3()[%24, %c32768_199]
      %c0_200 = arith.constant 0 : index
      %c0_201 = arith.constant 0 : index
      %c0_202 = arith.constant 0 : index
      %c2_203 = arith.constant 2 : index
      %c2_204 = arith.constant 2 : index
      %c64_205 = arith.constant 64 : index
      %c64_206 = arith.constant 64 : index
      %c8192_207 = arith.constant 8192 : index
      %c64_208 = arith.constant 64 : index
      %c128_209 = arith.constant 128 : index
      %c1_210 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_126] (%arg9[%c0_200, %c0_201, %c0_202, %37] [%c2_203, %c2_204, %c64_205, %c64_206] [%c8192_207, %c64_208, %c128_209, %c1_210]) : (memref<2x512x128xbf16>)
      %c49152_211 = arith.constant 49152 : index
      %38 = affine.apply #map3()[%24, %c49152_211]
      %c0_212 = arith.constant 0 : index
      %c0_213 = arith.constant 0 : index
      %c0_214 = arith.constant 0 : index
      %c2_215 = arith.constant 2 : index
      %c2_216 = arith.constant 2 : index
      %c64_217 = arith.constant 64 : index
      %c64_218 = arith.constant 64 : index
      %c8192_219 = arith.constant 8192 : index
      %c64_220 = arith.constant 64 : index
      %c128_221 = arith.constant 128 : index
      %c1_222 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_126] (%arg9[%c0_212, %c0_213, %c0_214, %38] [%c2_215, %c2_216, %c64_217, %c64_218] [%c8192_219, %c64_220, %c128_221, %c1_222]) : (memref<2x512x128xbf16>)
      %c0_223 = arith.constant 0 : index
      %39 = affine.apply #map3()[%25, %c0_223]
      %c0_224 = arith.constant 0 : index
      %c0_225 = arith.constant 0 : index
      %c2_226 = arith.constant 2 : index
      %c64_227 = arith.constant 64 : index
      %c64_228 = arith.constant 64 : index
      %c4096_229 = arith.constant 4096 : index
      %c64_230 = arith.constant 64 : index
      %c1_231 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c1_126] (%arg10[%c0_224, %c0_225, %39] [%c2_226, %c64_227, %c64_228] [%c4096_229, %c64_230, %c1_231]) : (memref<2x512x64xbf16>)
      %c8192_232 = arith.constant 8192 : index
      %40 = affine.apply #map3()[%25, %c8192_232]
      %c0_233 = arith.constant 0 : index
      %c0_234 = arith.constant 0 : index
      %c2_235 = arith.constant 2 : index
      %c64_236 = arith.constant 64 : index
      %c64_237 = arith.constant 64 : index
      %c4096_238 = arith.constant 4096 : index
      %c64_239 = arith.constant 64 : index
      %c1_240 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c1_126] (%arg10[%c0_233, %c0_234, %40] [%c2_235, %c64_236, %c64_237] [%c4096_238, %c64_239, %c1_240]) : (memref<2x512x64xbf16>)
      %c16384_241 = arith.constant 16384 : index
      %41 = affine.apply #map3()[%25, %c16384_241]
      %c0_242 = arith.constant 0 : index
      %c0_243 = arith.constant 0 : index
      %c2_244 = arith.constant 2 : index
      %c64_245 = arith.constant 64 : index
      %c64_246 = arith.constant 64 : index
      %c4096_247 = arith.constant 4096 : index
      %c64_248 = arith.constant 64 : index
      %c1_249 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c1_126] (%arg10[%c0_242, %c0_243, %41] [%c2_244, %c64_245, %c64_246] [%c4096_247, %c64_248, %c1_249]) : (memref<2x512x64xbf16>)
      %c24576_250 = arith.constant 24576 : index
      %42 = affine.apply #map3()[%25, %c24576_250]
      %c0_251 = arith.constant 0 : index
      %c0_252 = arith.constant 0 : index
      %c2_253 = arith.constant 2 : index
      %c64_254 = arith.constant 64 : index
      %c64_255 = arith.constant 64 : index
      %c4096_256 = arith.constant 4096 : index
      %c64_257 = arith.constant 64 : index
      %c1_258 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c1_126] (%arg10[%c0_251, %c0_252, %42] [%c2_253, %c64_254, %c64_255] [%c4096_256, %c64_257, %c1_258]) : (memref<2x512x64xbf16>)
      %c32768_259 = arith.constant 32768 : index
      %c1_260 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_126] (%arg11[%26] [%c32768_259] [%c1_260]) : (memref<2x256x64xbf16>)
      %c2_261 = arith.constant 2 : index
      %c1_262 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2_261, %arg15=%c1_262) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_263 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_264 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_265 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_266 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_267 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_268 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_269 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_270 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_271 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_272 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_273 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_274 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_275 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_276 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_277 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_278 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %c4 = arith.constant 4 : index
        %c4_279 = arith.constant 4 : index
        %c0_280 = arith.constant 0 : index
        %c2_281 = arith.constant 2 : index
        %c0_282 = arith.constant 0 : index
        %c1_283 = arith.constant 1 : index
        scf.for %arg16 = %c0_282 to %c4 step %c1_283 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_280, %c0_280] (%alloc[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_284 = arith.constant 0 : index
        %c1_285 = arith.constant 1 : index
        scf.for %arg16 = %c0_284 to %c4 step %c1_285 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_280, %c0_280] (%alloc[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_286 = arith.constant 0 : index
        %c1_287 = arith.constant 1 : index
        scf.for %arg16 = %c0_286 to %c2_281 step %c1_287 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_280, %c0_280] (%alloc[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_324 = arith.constant 0 : index
          %c0_325 = arith.constant 0 : index
          %c0_326 = arith.constant 0 : index
          %c0_327 = arith.constant 0 : index
          %c8_328 = arith.constant 8 : index
          %c8_329 = arith.constant 8 : index
          %c8_330 = arith.constant 8 : index
          %c8_331 = arith.constant 8 : index
          %c8_332 = arith.constant 8 : index
          %c512_333 = arith.constant 512 : index
          %c64_334 = arith.constant 64 : index
          %c1_335 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_280, %c0_280] (%alloc[%c0_324, %c0_325, %c0_326, %c0_327] [%c8_328, %c8_329, %c8_330, %c8_331] [%c8_332, %c512_333, %c64_334, %c1_335]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_288 = arith.constant 0 : index
        %c1_289 = arith.constant 1 : index
        scf.for %arg16 = %c0_288 to %c4 step %c1_289 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_263[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_280, %c0_280] (%alloc_263[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_290 = arith.constant 0 : index
        %c1_291 = arith.constant 1 : index
        scf.for %arg16 = %c0_290 to %c4 step %c1_291 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_263[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_280, %c0_280] (%alloc_263[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_292 = arith.constant 0 : index
        %c1_293 = arith.constant 1 : index
        scf.for %arg16 = %c0_292 to %c2_281 step %c1_293 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_263[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_280, %c0_280] (%alloc_263[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_1[%arg12] (%alloc_263[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_324 = arith.constant 0 : index
          %c0_325 = arith.constant 0 : index
          %c0_326 = arith.constant 0 : index
          %c0_327 = arith.constant 0 : index
          %c8_328 = arith.constant 8 : index
          %c8_329 = arith.constant 8 : index
          %c8_330 = arith.constant 8 : index
          %c8_331 = arith.constant 8 : index
          %c8_332 = arith.constant 8 : index
          %c512_333 = arith.constant 512 : index
          %c64_334 = arith.constant 64 : index
          %c1_335 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_280, %c0_280] (%alloc_263[%c0_324, %c0_325, %c0_326, %c0_327] [%c8_328, %c8_329, %c8_330, %c8_331] [%c8_332, %c512_333, %c64_334, %c1_335]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_294 = arith.constant 0 : index
        %c1_295 = arith.constant 1 : index
        scf.for %arg16 = %c0_294 to %c4 step %c1_295 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_264[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_280, %c0_280] (%alloc_264[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_296 = arith.constant 0 : index
        %c1_297 = arith.constant 1 : index
        scf.for %arg16 = %c0_296 to %c4 step %c1_297 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_264[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_280, %c0_280] (%alloc_264[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_298 = arith.constant 0 : index
        %c1_299 = arith.constant 1 : index
        scf.for %arg16 = %c0_298 to %c2_281 step %c1_299 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_264[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_280, %c0_280] (%alloc_264[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_2[%arg12] (%alloc_264[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_324 = arith.constant 0 : index
          %c0_325 = arith.constant 0 : index
          %c0_326 = arith.constant 0 : index
          %c0_327 = arith.constant 0 : index
          %c8_328 = arith.constant 8 : index
          %c8_329 = arith.constant 8 : index
          %c8_330 = arith.constant 8 : index
          %c8_331 = arith.constant 8 : index
          %c8_332 = arith.constant 8 : index
          %c512_333 = arith.constant 512 : index
          %c64_334 = arith.constant 64 : index
          %c1_335 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_280, %c0_280] (%alloc_264[%c0_324, %c0_325, %c0_326, %c0_327] [%c8_328, %c8_329, %c8_330, %c8_331] [%c8_332, %c512_333, %c64_334, %c1_335]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_300 = arith.constant 0 : index
        %c1_301 = arith.constant 1 : index
        scf.for %arg16 = %c0_300 to %c4 step %c1_301 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_265[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_280, %c0_280] (%alloc_265[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_302 = arith.constant 0 : index
        %c1_303 = arith.constant 1 : index
        scf.for %arg16 = %c0_302 to %c4 step %c1_303 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_265[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_280, %c0_280] (%alloc_265[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_304 = arith.constant 0 : index
        %c1_305 = arith.constant 1 : index
        scf.for %arg16 = %c0_304 to %c2_281 step %c1_305 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_265[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_280, %c0_280] (%alloc_265[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.get  @QKIn_3[%arg12] (%alloc_265[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_324 = arith.constant 0 : index
          %c0_325 = arith.constant 0 : index
          %c0_326 = arith.constant 0 : index
          %c0_327 = arith.constant 0 : index
          %c8_328 = arith.constant 8 : index
          %c8_329 = arith.constant 8 : index
          %c8_330 = arith.constant 8 : index
          %c8_331 = arith.constant 8 : index
          %c8_332 = arith.constant 8 : index
          %c512_333 = arith.constant 512 : index
          %c64_334 = arith.constant 64 : index
          %c1_335 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_280, %c0_280] (%alloc_265[%c0_324, %c0_325, %c0_326, %c0_327] [%c8_328, %c8_329, %c8_330, %c8_331] [%c8_332, %c512_333, %c64_334, %c1_335]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_306 = arith.constant 0 : index
        %c1_307 = arith.constant 1 : index
        scf.for %arg16 = %c0_306 to %c2_281 step %c1_307 {
          air.channel.get  @VIn_0[%arg12] (%alloc_266[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @V2L1_0[%arg12, %c0_280, %c0_280] (%alloc_266[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_308 = arith.constant 0 : index
        %c1_309 = arith.constant 1 : index
        scf.for %arg16 = %c0_308 to %c2_281 step %c1_309 {
          air.channel.get  @VIn_1[%arg12] (%alloc_267[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @V2L1_1[%arg12, %c0_280, %c0_280] (%alloc_267[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_310 = arith.constant 0 : index
        %c1_311 = arith.constant 1 : index
        scf.for %arg16 = %c0_310 to %c2_281 step %c1_311 {
          air.channel.get  @VIn_2[%arg12] (%alloc_268[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @V2L1_2[%arg12, %c0_280, %c0_280] (%alloc_268[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_312 = arith.constant 0 : index
        %c1_313 = arith.constant 1 : index
        scf.for %arg16 = %c0_312 to %c2_281 step %c1_313 {
          air.channel.get  @VIn_3[%arg12] (%alloc_269[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c0_316 = arith.constant 0 : index
          %c0_317 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_318 = arith.constant 8 : index
          %c8_319 = arith.constant 8 : index
          %c8_320 = arith.constant 8 : index
          %c8_321 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_322 = arith.constant 64 : index
          %c1_323 = arith.constant 1 : index
          air.channel.put  @V2L1_3[%arg12, %c0_280, %c0_280] (%alloc_269[%c0_314, %c0_315, %c0_316, %c0_317] [%c8, %c8_318, %c8_319, %c8_320] [%c8_321, %c512, %c64_322, %c1_323]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %43 = affine.apply #map5()[%arg16]
          %c0_314 = arith.constant 0 : index
          %c0_315 = arith.constant 0 : index
          %c64_316 = arith.constant 64 : index
          %c64_317 = arith.constant 64 : index
          %c64_318 = arith.constant 64 : index
          %c1_319 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%arg16, %c0_314] (%alloc_270[%43, %c0_315] [%c64_316, %c64_317] [%c64_318, %c1_319]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_270[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4_279) args(%arg20=%alloc_271, %arg21=%alloc_272, %arg22=%alloc_273, %arg23=%alloc_274, %arg24=%alloc_275, %arg25=%alloc_276, %arg26=%alloc_277, %arg27=%alloc_278, %arg28=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn.o"} {
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
          %43 = arith.index_cast %arg16 : index to i32
          %c0_i32 = arith.constant 0 : i32
          %44 = arith.cmpi eq, %43, %c0_i32 : i32
          scf.if %44 {
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
          %45 = arith.index_cast %arg16 : index to i32
          %c1_i32 = arith.constant 1 : i32
          %46 = arith.cmpi eq, %45, %c1_i32 : i32
          scf.if %46 {
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
          %47 = arith.index_cast %arg16 : index to i32
          %c2_i32 = arith.constant 2 : i32
          %48 = arith.cmpi eq, %47, %c2_i32 : i32
          scf.if %48 {
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
          %49 = arith.index_cast %arg16 : index to i32
          %c3_i32 = arith.constant 3 : i32
          %50 = arith.cmpi eq, %49, %c3_i32 : i32
          scf.if %50 {
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
          %51 = arith.index_cast %arg16 : index to i32
          %c0_i32_314 = arith.constant 0 : i32
          %52 = arith.cmpi eq, %51, %c0_i32_314 : i32
          scf.if %52 {
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
          %53 = arith.index_cast %arg16 : index to i32
          %c1_i32_315 = arith.constant 1 : i32
          %54 = arith.cmpi eq, %53, %c1_i32_315 : i32
          scf.if %54 {
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
          %55 = arith.index_cast %arg16 : index to i32
          %c2_i32_316 = arith.constant 2 : i32
          %56 = arith.cmpi eq, %55, %c2_i32_316 : i32
          scf.if %56 {
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
          %57 = arith.index_cast %arg16 : index to i32
          %c3_i32_317 = arith.constant 3 : i32
          %58 = arith.cmpi eq, %57, %c3_i32_317 : i32
          scf.if %58 {
            func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %c2_318 = arith.constant 2 : index
          %c0_319 = arith.constant 0 : index
          %c1_320 = arith.constant 1 : index
          scf.for %arg29 = %c0_319 to %c2_318 step %c1_320 {
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
            %alloc_322 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_323 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg26, %alloc_322, %alloc_323) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_323, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg23, %arg25) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32_324 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg27, %alloc_323, %alloc_322) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32_324, %alloc_322, %arg27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_322 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_323 : memref<64x1xbf16, 2 : i32>
          }
          %c1_321 = arith.constant 1 : index
          affine.if #set3()[%arg16, %arg17] {
            %59 = arith.subi %arg17, %c1_321 : index
            air.channel.put  @cascade_gp[%arg16, %59] (%arg25[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %59] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %59] (%arg27[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_322 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_323 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_324 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_322[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_323[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_324[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_325 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_326 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_326, %arg26, %alloc_325) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_323, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_327 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_323, %arg26, %alloc_327) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_328 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_325, %arg26, %alloc_328) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_327, %alloc_322) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_328, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_322) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_329 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_329) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_324, %alloc_327, %alloc_329) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_328, %alloc_329) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_326, %alloc_329, %alloc_324) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %59 = arith.subi %arg17, %c1_321 : index
              air.channel.put  @cascade_gp[%arg16, %59] (%alloc_322[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %59] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %59] (%alloc_324[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_322 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_323 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_324 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_325 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_327 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_328 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_329 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_322 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_323 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_324 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_322[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_323[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_324[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_325 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_326 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_326, %arg26, %alloc_325) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_323, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_327 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_323, %arg26, %alloc_327) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_328 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_325, %arg26, %alloc_328) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_327, %alloc_322) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_328, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_322) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_329 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_329) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_324, %alloc_327, %alloc_329) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_328, %alloc_329) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_326, %alloc_329, %alloc_324) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_324, %alloc_322) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %c0_330 = arith.constant 0 : index
              %c0_331 = arith.constant 0 : index
              %c0_332 = arith.constant 0 : index
              %c0_333 = arith.constant 0 : index
              %c0_334 = arith.constant 0 : index
              %c8 = arith.constant 8 : index
              %c8_335 = arith.constant 8 : index
              %c8_336 = arith.constant 8 : index
              %c8_337 = arith.constant 8 : index
              %c64_338 = arith.constant 64 : index
              %c8_339 = arith.constant 8 : index
              %c512 = arith.constant 512 : index
              %c1_340 = arith.constant 1 : index
              air.channel.put  @Gp2L2[%arg16, %c0_330] (%alloc_322[%c0_331, %c0_332, %c0_333, %c0_334] [%c8, %c8_335, %c8_336, %c8_337] [%c64_338, %c8_339, %c512, %c1_340]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_322 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_323 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_324 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_325 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_327 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_328 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_329 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_271 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_272 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_273 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_274 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_275 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_276 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_277 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_278 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_266 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_263 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_267 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_264 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_268 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_265 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_269 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_270 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}
