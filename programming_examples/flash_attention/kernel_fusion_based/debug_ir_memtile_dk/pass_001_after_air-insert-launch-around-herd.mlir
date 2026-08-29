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
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c128_3 = arith.constant 128 : index
      %c1_4 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c0] (%arg8[%c0_2, %5] [%c256, %c128] [%c128_3, %c1_4]) : (memref<2x256x128xbf16>)
      %c0_5 = arith.constant 0 : index
      %c256_6 = arith.constant 256 : index
      %c128_7 = arith.constant 128 : index
      %c128_8 = arith.constant 128 : index
      %c1_9 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c0] (%arg8[%c0_5, %5] [%c256_6, %c128_7] [%c128_8, %c1_9]) : (memref<2x256x128xbf16>)
      %c0_10 = arith.constant 0 : index
      %c256_11 = arith.constant 256 : index
      %c128_12 = arith.constant 128 : index
      %c128_13 = arith.constant 128 : index
      %c1_14 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c0] (%arg8[%c0_10, %5] [%c256_11, %c128_12] [%c128_13, %c1_14]) : (memref<2x256x128xbf16>)
      %c0_15 = arith.constant 0 : index
      %c256_16 = arith.constant 256 : index
      %c128_17 = arith.constant 128 : index
      %c128_18 = arith.constant 128 : index
      %c1_19 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c0] (%arg8[%c0_15, %5] [%c256_16, %c128_17] [%c128_18, %c1_19]) : (memref<2x256x128xbf16>)
      %c0_20 = arith.constant 0 : index
      %6 = affine.apply #map3()[%3, %c0_20]
      %c0_21 = arith.constant 0 : index
      %c128_22 = arith.constant 128 : index
      %c128_23 = arith.constant 128 : index
      %c128_24 = arith.constant 128 : index
      %c1_25 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c0] (%arg9[%c0_21, %6] [%c128_22, %c128_23] [%c128_24, %c1_25]) : (memref<2x512x128xbf16>)
      %c16384 = arith.constant 16384 : index
      %7 = affine.apply #map3()[%3, %c16384]
      %c0_26 = arith.constant 0 : index
      %c128_27 = arith.constant 128 : index
      %c128_28 = arith.constant 128 : index
      %c128_29 = arith.constant 128 : index
      %c1_30 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c0] (%arg9[%c0_26, %7] [%c128_27, %c128_28] [%c128_29, %c1_30]) : (memref<2x512x128xbf16>)
      %c32768 = arith.constant 32768 : index
      %8 = affine.apply #map3()[%3, %c32768]
      %c0_31 = arith.constant 0 : index
      %c128_32 = arith.constant 128 : index
      %c128_33 = arith.constant 128 : index
      %c128_34 = arith.constant 128 : index
      %c1_35 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c0] (%arg9[%c0_31, %8] [%c128_32, %c128_33] [%c128_34, %c1_35]) : (memref<2x512x128xbf16>)
      %c49152 = arith.constant 49152 : index
      %9 = affine.apply #map3()[%3, %c49152]
      %c0_36 = arith.constant 0 : index
      %c128_37 = arith.constant 128 : index
      %c128_38 = arith.constant 128 : index
      %c128_39 = arith.constant 128 : index
      %c1_40 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c0] (%arg9[%c0_36, %9] [%c128_37, %c128_38] [%c128_39, %c1_40]) : (memref<2x512x128xbf16>)
      %c0_41 = arith.constant 0 : index
      %10 = affine.apply #map3()[%4, %c0_41]
      %c0_42 = arith.constant 0 : index
      %c0_43 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c64_44 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c64_45 = arith.constant 64 : index
      %c1_46 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c0] (%arg10[%c0_42, %c0_43, %10] [%c2, %c64, %c64_44] [%c4096, %c64_45, %c1_46]) : (memref<2x512x64xbf16>)
      %c8192 = arith.constant 8192 : index
      %11 = affine.apply #map3()[%4, %c8192]
      %c0_47 = arith.constant 0 : index
      %c0_48 = arith.constant 0 : index
      %c2_49 = arith.constant 2 : index
      %c64_50 = arith.constant 64 : index
      %c64_51 = arith.constant 64 : index
      %c4096_52 = arith.constant 4096 : index
      %c64_53 = arith.constant 64 : index
      %c1_54 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c0] (%arg10[%c0_47, %c0_48, %11] [%c2_49, %c64_50, %c64_51] [%c4096_52, %c64_53, %c1_54]) : (memref<2x512x64xbf16>)
      %c16384_55 = arith.constant 16384 : index
      %12 = affine.apply #map3()[%4, %c16384_55]
      %c0_56 = arith.constant 0 : index
      %c0_57 = arith.constant 0 : index
      %c2_58 = arith.constant 2 : index
      %c64_59 = arith.constant 64 : index
      %c64_60 = arith.constant 64 : index
      %c4096_61 = arith.constant 4096 : index
      %c64_62 = arith.constant 64 : index
      %c1_63 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c0] (%arg10[%c0_56, %c0_57, %12] [%c2_58, %c64_59, %c64_60] [%c4096_61, %c64_62, %c1_63]) : (memref<2x512x64xbf16>)
      %c24576 = arith.constant 24576 : index
      %13 = affine.apply #map3()[%4, %c24576]
      %c0_64 = arith.constant 0 : index
      %c0_65 = arith.constant 0 : index
      %c2_66 = arith.constant 2 : index
      %c64_67 = arith.constant 64 : index
      %c64_68 = arith.constant 64 : index
      %c4096_69 = arith.constant 4096 : index
      %c64_70 = arith.constant 64 : index
      %c1_71 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c0] (%arg10[%c0_64, %c0_65, %13] [%c2_66, %c64_67, %c64_68] [%c4096_69, %c64_70, %c1_71]) : (memref<2x512x64xbf16>)
      %c32768_72 = arith.constant 32768 : index
      %c1_73 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0] (%arg11[%5] [%c32768_72] [%c1_73]) : (memref<2x256x64xbf16>)
      %14 = affine.apply #map4()[%1]
      %15 = affine.apply #map()[%14]
      %16 = affine.apply #map2()[%14]
      %17 = affine.apply #map()[%14]
      %c1_74 = arith.constant 1 : index
      %18 = affine.apply #map3()[%15, %0]
      %c0_75 = arith.constant 0 : index
      %c256_76 = arith.constant 256 : index
      %c128_77 = arith.constant 128 : index
      %c128_78 = arith.constant 128 : index
      %c1_79 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_74] (%arg8[%c0_75, %18] [%c256_76, %c128_77] [%c128_78, %c1_79]) : (memref<2x256x128xbf16>)
      %c0_80 = arith.constant 0 : index
      %c256_81 = arith.constant 256 : index
      %c128_82 = arith.constant 128 : index
      %c128_83 = arith.constant 128 : index
      %c1_84 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_74] (%arg8[%c0_80, %18] [%c256_81, %c128_82] [%c128_83, %c1_84]) : (memref<2x256x128xbf16>)
      %c0_85 = arith.constant 0 : index
      %c256_86 = arith.constant 256 : index
      %c128_87 = arith.constant 128 : index
      %c128_88 = arith.constant 128 : index
      %c1_89 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_74] (%arg8[%c0_85, %18] [%c256_86, %c128_87] [%c128_88, %c1_89]) : (memref<2x256x128xbf16>)
      %c0_90 = arith.constant 0 : index
      %c256_91 = arith.constant 256 : index
      %c128_92 = arith.constant 128 : index
      %c128_93 = arith.constant 128 : index
      %c1_94 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_74] (%arg8[%c0_90, %18] [%c256_91, %c128_92] [%c128_93, %c1_94]) : (memref<2x256x128xbf16>)
      %c0_95 = arith.constant 0 : index
      %19 = affine.apply #map3()[%16, %c0_95]
      %c0_96 = arith.constant 0 : index
      %c128_97 = arith.constant 128 : index
      %c128_98 = arith.constant 128 : index
      %c128_99 = arith.constant 128 : index
      %c1_100 = arith.constant 1 : index
      air.channel.put  @QKIn_0[%c1_74] (%arg9[%c0_96, %19] [%c128_97, %c128_98] [%c128_99, %c1_100]) : (memref<2x512x128xbf16>)
      %c16384_101 = arith.constant 16384 : index
      %20 = affine.apply #map3()[%16, %c16384_101]
      %c0_102 = arith.constant 0 : index
      %c128_103 = arith.constant 128 : index
      %c128_104 = arith.constant 128 : index
      %c128_105 = arith.constant 128 : index
      %c1_106 = arith.constant 1 : index
      air.channel.put  @QKIn_1[%c1_74] (%arg9[%c0_102, %20] [%c128_103, %c128_104] [%c128_105, %c1_106]) : (memref<2x512x128xbf16>)
      %c32768_107 = arith.constant 32768 : index
      %21 = affine.apply #map3()[%16, %c32768_107]
      %c0_108 = arith.constant 0 : index
      %c128_109 = arith.constant 128 : index
      %c128_110 = arith.constant 128 : index
      %c128_111 = arith.constant 128 : index
      %c1_112 = arith.constant 1 : index
      air.channel.put  @QKIn_2[%c1_74] (%arg9[%c0_108, %21] [%c128_109, %c128_110] [%c128_111, %c1_112]) : (memref<2x512x128xbf16>)
      %c49152_113 = arith.constant 49152 : index
      %22 = affine.apply #map3()[%16, %c49152_113]
      %c0_114 = arith.constant 0 : index
      %c128_115 = arith.constant 128 : index
      %c128_116 = arith.constant 128 : index
      %c128_117 = arith.constant 128 : index
      %c1_118 = arith.constant 1 : index
      air.channel.put  @QKIn_3[%c1_74] (%arg9[%c0_114, %22] [%c128_115, %c128_116] [%c128_117, %c1_118]) : (memref<2x512x128xbf16>)
      %c0_119 = arith.constant 0 : index
      %23 = affine.apply #map3()[%17, %c0_119]
      %c0_120 = arith.constant 0 : index
      %c0_121 = arith.constant 0 : index
      %c2_122 = arith.constant 2 : index
      %c64_123 = arith.constant 64 : index
      %c64_124 = arith.constant 64 : index
      %c4096_125 = arith.constant 4096 : index
      %c64_126 = arith.constant 64 : index
      %c1_127 = arith.constant 1 : index
      air.channel.put  @VIn_0[%c1_74] (%arg10[%c0_120, %c0_121, %23] [%c2_122, %c64_123, %c64_124] [%c4096_125, %c64_126, %c1_127]) : (memref<2x512x64xbf16>)
      %c8192_128 = arith.constant 8192 : index
      %24 = affine.apply #map3()[%17, %c8192_128]
      %c0_129 = arith.constant 0 : index
      %c0_130 = arith.constant 0 : index
      %c2_131 = arith.constant 2 : index
      %c64_132 = arith.constant 64 : index
      %c64_133 = arith.constant 64 : index
      %c4096_134 = arith.constant 4096 : index
      %c64_135 = arith.constant 64 : index
      %c1_136 = arith.constant 1 : index
      air.channel.put  @VIn_1[%c1_74] (%arg10[%c0_129, %c0_130, %24] [%c2_131, %c64_132, %c64_133] [%c4096_134, %c64_135, %c1_136]) : (memref<2x512x64xbf16>)
      %c16384_137 = arith.constant 16384 : index
      %25 = affine.apply #map3()[%17, %c16384_137]
      %c0_138 = arith.constant 0 : index
      %c0_139 = arith.constant 0 : index
      %c2_140 = arith.constant 2 : index
      %c64_141 = arith.constant 64 : index
      %c64_142 = arith.constant 64 : index
      %c4096_143 = arith.constant 4096 : index
      %c64_144 = arith.constant 64 : index
      %c1_145 = arith.constant 1 : index
      air.channel.put  @VIn_2[%c1_74] (%arg10[%c0_138, %c0_139, %25] [%c2_140, %c64_141, %c64_142] [%c4096_143, %c64_144, %c1_145]) : (memref<2x512x64xbf16>)
      %c24576_146 = arith.constant 24576 : index
      %26 = affine.apply #map3()[%17, %c24576_146]
      %c0_147 = arith.constant 0 : index
      %c0_148 = arith.constant 0 : index
      %c2_149 = arith.constant 2 : index
      %c64_150 = arith.constant 64 : index
      %c64_151 = arith.constant 64 : index
      %c4096_152 = arith.constant 4096 : index
      %c64_153 = arith.constant 64 : index
      %c1_154 = arith.constant 1 : index
      air.channel.put  @VIn_3[%c1_74] (%arg10[%c0_147, %c0_148, %26] [%c2_149, %c64_150, %c64_151] [%c4096_152, %c64_153, %c1_154]) : (memref<2x512x64xbf16>)
      %c32768_155 = arith.constant 32768 : index
      %c1_156 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_74] (%arg11[%18] [%c32768_155] [%c1_156]) : (memref<2x256x64xbf16>)
      %c2_157 = arith.constant 2 : index
      %c1_158 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2_157, %arg15=%c1_158) {
        %alloc = memref.alloc() : memref<64x128xbf16, 1 : i32>
        %alloc_159 = memref.alloc() : memref<64x128xbf16, 1 : i32>
        %alloc_160 = memref.alloc() : memref<64x128xbf16, 1 : i32>
        %alloc_161 = memref.alloc() : memref<64x128xbf16, 1 : i32>
        %alloc_162 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_163 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_164 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_165 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_166 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_167 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_168 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_169 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_170 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_171 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_172 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_173 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_174 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %c4 = arith.constant 4 : index
        %c4_175 = arith.constant 4 : index
        %c0_176 = arith.constant 0 : index
        %c2_177 = arith.constant 2 : index
        %c0_178 = arith.constant 0 : index
        %c1_179 = arith.constant 1 : index
        scf.for %arg16 = %c0_178 to %c4 step %c1_179 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x128xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c128_210 = arith.constant 128 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_176, %c0_176] (%alloc[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c128_210, %c1_211]) : (memref<64x128xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c64_214 = arith.constant 64 : index
          %c0_215 = arith.constant 0 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c8_220 = arith.constant 8 : index
          %c512_221 = arith.constant 512 : index
          %c128_222 = arith.constant 128 : index
          %c1_223 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_176, %c0_176] (%alloc[%c0_212, %c0_213, %c64_214, %c0_215] [%c8_216, %c8_217, %c8_218, %c8_219] [%c8_220, %c512_221, %c128_222, %c1_223]) : (memref<64x128xbf16, 1 : i32>)
        }
        %c0_180 = arith.constant 0 : index
        %c1_181 = arith.constant 1 : index
        scf.for %arg16 = %c0_180 to %c2_177 step %c1_181 {
          air.channel.get  @QKIn_0[%arg12] (%alloc[] [] []) : (memref<64x128xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c128_210 = arith.constant 128 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_176, %c0_176] (%alloc[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c128_210, %c1_211]) : (memref<64x128xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c64_214 = arith.constant 64 : index
          %c0_215 = arith.constant 0 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c8_220 = arith.constant 8 : index
          %c512_221 = arith.constant 512 : index
          %c128_222 = arith.constant 128 : index
          %c1_223 = arith.constant 1 : index
          air.channel.put  @QK2L1_0[%arg12, %c0_176, %c0_176] (%alloc[%c0_212, %c0_213, %c64_214, %c0_215] [%c8_216, %c8_217, %c8_218, %c8_219] [%c8_220, %c512_221, %c128_222, %c1_223]) : (memref<64x128xbf16, 1 : i32>)
        }
        %c0_182 = arith.constant 0 : index
        %c1_183 = arith.constant 1 : index
        scf.for %arg16 = %c0_182 to %c4 step %c1_183 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_159[] [] []) : (memref<64x128xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c128_210 = arith.constant 128 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_176, %c0_176] (%alloc_159[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c128_210, %c1_211]) : (memref<64x128xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c64_214 = arith.constant 64 : index
          %c0_215 = arith.constant 0 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c8_220 = arith.constant 8 : index
          %c512_221 = arith.constant 512 : index
          %c128_222 = arith.constant 128 : index
          %c1_223 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_176, %c0_176] (%alloc_159[%c0_212, %c0_213, %c64_214, %c0_215] [%c8_216, %c8_217, %c8_218, %c8_219] [%c8_220, %c512_221, %c128_222, %c1_223]) : (memref<64x128xbf16, 1 : i32>)
        }
        %c0_184 = arith.constant 0 : index
        %c1_185 = arith.constant 1 : index
        scf.for %arg16 = %c0_184 to %c2_177 step %c1_185 {
          air.channel.get  @QKIn_1[%arg12] (%alloc_159[] [] []) : (memref<64x128xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c128_210 = arith.constant 128 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_176, %c0_176] (%alloc_159[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c128_210, %c1_211]) : (memref<64x128xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c64_214 = arith.constant 64 : index
          %c0_215 = arith.constant 0 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c8_220 = arith.constant 8 : index
          %c512_221 = arith.constant 512 : index
          %c128_222 = arith.constant 128 : index
          %c1_223 = arith.constant 1 : index
          air.channel.put  @QK2L1_1[%arg12, %c0_176, %c0_176] (%alloc_159[%c0_212, %c0_213, %c64_214, %c0_215] [%c8_216, %c8_217, %c8_218, %c8_219] [%c8_220, %c512_221, %c128_222, %c1_223]) : (memref<64x128xbf16, 1 : i32>)
        }
        %c0_186 = arith.constant 0 : index
        %c1_187 = arith.constant 1 : index
        scf.for %arg16 = %c0_186 to %c4 step %c1_187 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_160[] [] []) : (memref<64x128xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c128_210 = arith.constant 128 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_176, %c0_176] (%alloc_160[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c128_210, %c1_211]) : (memref<64x128xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c64_214 = arith.constant 64 : index
          %c0_215 = arith.constant 0 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c8_220 = arith.constant 8 : index
          %c512_221 = arith.constant 512 : index
          %c128_222 = arith.constant 128 : index
          %c1_223 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_176, %c0_176] (%alloc_160[%c0_212, %c0_213, %c64_214, %c0_215] [%c8_216, %c8_217, %c8_218, %c8_219] [%c8_220, %c512_221, %c128_222, %c1_223]) : (memref<64x128xbf16, 1 : i32>)
        }
        %c0_188 = arith.constant 0 : index
        %c1_189 = arith.constant 1 : index
        scf.for %arg16 = %c0_188 to %c2_177 step %c1_189 {
          air.channel.get  @QKIn_2[%arg12] (%alloc_160[] [] []) : (memref<64x128xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c128_210 = arith.constant 128 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_176, %c0_176] (%alloc_160[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c128_210, %c1_211]) : (memref<64x128xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c64_214 = arith.constant 64 : index
          %c0_215 = arith.constant 0 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c8_220 = arith.constant 8 : index
          %c512_221 = arith.constant 512 : index
          %c128_222 = arith.constant 128 : index
          %c1_223 = arith.constant 1 : index
          air.channel.put  @QK2L1_2[%arg12, %c0_176, %c0_176] (%alloc_160[%c0_212, %c0_213, %c64_214, %c0_215] [%c8_216, %c8_217, %c8_218, %c8_219] [%c8_220, %c512_221, %c128_222, %c1_223]) : (memref<64x128xbf16, 1 : i32>)
        }
        %c0_190 = arith.constant 0 : index
        %c1_191 = arith.constant 1 : index
        scf.for %arg16 = %c0_190 to %c4 step %c1_191 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_161[] [] []) : (memref<64x128xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c128_210 = arith.constant 128 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_176, %c0_176] (%alloc_161[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c128_210, %c1_211]) : (memref<64x128xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c64_214 = arith.constant 64 : index
          %c0_215 = arith.constant 0 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c8_220 = arith.constant 8 : index
          %c512_221 = arith.constant 512 : index
          %c128_222 = arith.constant 128 : index
          %c1_223 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_176, %c0_176] (%alloc_161[%c0_212, %c0_213, %c64_214, %c0_215] [%c8_216, %c8_217, %c8_218, %c8_219] [%c8_220, %c512_221, %c128_222, %c1_223]) : (memref<64x128xbf16, 1 : i32>)
        }
        %c0_192 = arith.constant 0 : index
        %c1_193 = arith.constant 1 : index
        scf.for %arg16 = %c0_192 to %c2_177 step %c1_193 {
          air.channel.get  @QKIn_3[%arg12] (%alloc_161[] [] []) : (memref<64x128xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c128_210 = arith.constant 128 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_176, %c0_176] (%alloc_161[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c128_210, %c1_211]) : (memref<64x128xbf16, 1 : i32>)
          %c0_212 = arith.constant 0 : index
          %c0_213 = arith.constant 0 : index
          %c64_214 = arith.constant 64 : index
          %c0_215 = arith.constant 0 : index
          %c8_216 = arith.constant 8 : index
          %c8_217 = arith.constant 8 : index
          %c8_218 = arith.constant 8 : index
          %c8_219 = arith.constant 8 : index
          %c8_220 = arith.constant 8 : index
          %c512_221 = arith.constant 512 : index
          %c128_222 = arith.constant 128 : index
          %c1_223 = arith.constant 1 : index
          air.channel.put  @QK2L1_3[%arg12, %c0_176, %c0_176] (%alloc_161[%c0_212, %c0_213, %c64_214, %c0_215] [%c8_216, %c8_217, %c8_218, %c8_219] [%c8_220, %c512_221, %c128_222, %c1_223]) : (memref<64x128xbf16, 1 : i32>)
        }
        %c0_194 = arith.constant 0 : index
        %c1_195 = arith.constant 1 : index
        scf.for %arg16 = %c0_194 to %c2_177 step %c1_195 {
          air.channel.get  @VIn_0[%arg12] (%alloc_162[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_210 = arith.constant 64 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @V2L1_0[%arg12, %c0_176, %c0_176] (%alloc_162[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c64_210, %c1_211]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_196 = arith.constant 0 : index
        %c1_197 = arith.constant 1 : index
        scf.for %arg16 = %c0_196 to %c2_177 step %c1_197 {
          air.channel.get  @VIn_1[%arg12] (%alloc_163[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_210 = arith.constant 64 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @V2L1_1[%arg12, %c0_176, %c0_176] (%alloc_163[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c64_210, %c1_211]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_198 = arith.constant 0 : index
        %c1_199 = arith.constant 1 : index
        scf.for %arg16 = %c0_198 to %c2_177 step %c1_199 {
          air.channel.get  @VIn_2[%arg12] (%alloc_164[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_210 = arith.constant 64 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @V2L1_2[%arg12, %c0_176, %c0_176] (%alloc_164[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c64_210, %c1_211]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_200 = arith.constant 0 : index
        %c1_201 = arith.constant 1 : index
        scf.for %arg16 = %c0_200 to %c2_177 step %c1_201 {
          air.channel.get  @VIn_3[%arg12] (%alloc_165[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %c64_210 = arith.constant 64 : index
          %c1_211 = arith.constant 1 : index
          air.channel.put  @V2L1_3[%arg12, %c0_176, %c0_176] (%alloc_165[%c0_202, %c0_203, %c0_204, %c0_205] [%c8, %c8_206, %c8_207, %c8_208] [%c8_209, %c512, %c64_210, %c1_211]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %27 = affine.apply #map5()[%arg16]
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c64_204 = arith.constant 64 : index
          %c64_205 = arith.constant 64 : index
          %c64_206 = arith.constant 64 : index
          %c1_207 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%arg16, %c0_202] (%alloc_166[%27, %c0_203] [%c64_204, %c64_205] [%c64_206, %c1_207]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_166[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4_175) args(%arg20=%alloc_167, %arg21=%alloc_168, %arg22=%alloc_169, %arg23=%alloc_170, %arg24=%alloc_171, %arg25=%alloc_172, %arg26=%alloc_173, %arg27=%alloc_174, %arg28=%arg12) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index attributes {link_with = "attn.o"} {
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
          %27 = arith.index_cast %arg16 : index to i32
          %c0_i32 = arith.constant 0 : i32
          %28 = arith.cmpi eq, %27, %c0_i32 : i32
          scf.if %28 {
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
          %29 = arith.index_cast %arg16 : index to i32
          %c1_i32 = arith.constant 1 : i32
          %30 = arith.cmpi eq, %29, %c1_i32 : i32
          scf.if %30 {
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
          %31 = arith.index_cast %arg16 : index to i32
          %c2_i32 = arith.constant 2 : i32
          %32 = arith.cmpi eq, %31, %c2_i32 : i32
          scf.if %32 {
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
          %33 = arith.index_cast %arg16 : index to i32
          %c3_i32 = arith.constant 3 : i32
          %34 = arith.cmpi eq, %33, %c3_i32 : i32
          scf.if %34 {
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
          %35 = arith.index_cast %arg16 : index to i32
          %c0_i32_202 = arith.constant 0 : i32
          %36 = arith.cmpi eq, %35, %c0_i32_202 : i32
          scf.if %36 {
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
          %37 = arith.index_cast %arg16 : index to i32
          %c1_i32_203 = arith.constant 1 : i32
          %38 = arith.cmpi eq, %37, %c1_i32_203 : i32
          scf.if %38 {
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
          %39 = arith.index_cast %arg16 : index to i32
          %c2_i32_204 = arith.constant 2 : i32
          %40 = arith.cmpi eq, %39, %c2_i32_204 : i32
          scf.if %40 {
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
          %41 = arith.index_cast %arg16 : index to i32
          %c3_i32_205 = arith.constant 3 : i32
          %42 = arith.cmpi eq, %41, %c3_i32_205 : i32
          scf.if %42 {
            func.call @copy_tile(%arg22, %arg21) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %c2_206 = arith.constant 2 : index
          %c0_207 = arith.constant 0 : index
          %c1_208 = arith.constant 1 : index
          scf.for %arg29 = %c0_207 to %c2_206 step %c1_208 {
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
            %alloc_210 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_211 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg26, %alloc_210, %alloc_211) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_211, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg23, %arg25) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32_212 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg27, %alloc_211, %alloc_210) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32_212, %alloc_210, %arg27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_210 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_211 : memref<64x1xbf16, 2 : i32>
          }
          %c1_209 = arith.constant 1 : index
          affine.if #set3()[%arg16, %arg17] {
            %43 = arith.subi %arg17, %c1_209 : index
            air.channel.put  @cascade_gp[%arg16, %43] (%arg25[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %43] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %43] (%arg27[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_210 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_211 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_212 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_210[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_211[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_212[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_213 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_214 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_214, %arg26, %alloc_213) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_211, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_215 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_211, %arg26, %alloc_215) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_216 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_213, %arg26, %alloc_216) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_215, %alloc_210) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_216, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_210) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_217 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_217) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_212, %alloc_215, %alloc_217) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_216, %alloc_217) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_214, %alloc_217, %alloc_212) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %43 = arith.subi %arg17, %c1_209 : index
              air.channel.put  @cascade_gp[%arg16, %43] (%alloc_210[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %43] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %43] (%alloc_212[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_210 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_211 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_212 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_213 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_215 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_216 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_217 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_210 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_211 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_212 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_210[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_211[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_212[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_213 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %c0_i32_214 = arith.constant 0 : i32
              func.call @vector_copy_32elems(%c0_i32_214, %arg26, %alloc_213) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_211, %arg26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_215 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_211, %arg26, %alloc_215) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_216 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_213, %arg26, %alloc_216) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_215, %alloc_210) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_216, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg25, %alloc_210) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_217 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_217) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_212, %alloc_215, %alloc_217) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg27, %alloc_216, %alloc_217) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32_214, %alloc_217, %alloc_212) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_212, %alloc_210) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %c0_218 = arith.constant 0 : index
              %c0_219 = arith.constant 0 : index
              %c0_220 = arith.constant 0 : index
              %c0_221 = arith.constant 0 : index
              %c0_222 = arith.constant 0 : index
              %c8 = arith.constant 8 : index
              %c8_223 = arith.constant 8 : index
              %c8_224 = arith.constant 8 : index
              %c8_225 = arith.constant 8 : index
              %c64_226 = arith.constant 64 : index
              %c8_227 = arith.constant 8 : index
              %c512 = arith.constant 512 : index
              %c1_228 = arith.constant 1 : index
              air.channel.put  @Gp2L2[%arg16, %c0_218] (%alloc_210[%c0_219, %c0_220, %c0_221, %c0_222] [%c8, %c8_223, %c8_224, %c8_225] [%c64_226, %c8_227, %c512, %c1_228]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_210 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_211 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_212 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_213 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_215 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_216 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_217 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_167 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_168 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_169 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_170 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_171 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_172 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_173 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_174 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x128xbf16, 1 : i32>
        memref.dealloc %alloc_162 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_159 : memref<64x128xbf16, 1 : i32>
        memref.dealloc %alloc_163 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_160 : memref<64x128xbf16, 1 : i32>
        memref.dealloc %alloc_164 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_161 : memref<64x128xbf16, 1 : i32>
        memref.dealloc %alloc_165 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_166 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}
