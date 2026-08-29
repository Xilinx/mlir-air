#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0] -> (s0 * 512)>
#map3 = affine_map<()[s0] -> (s0 * 512 + 64)>
#map4 = affine_map<()[s0] -> (s0 * 512 + 128)>
#map5 = affine_map<()[s0] -> (s0 * 512 + 192)>
#map6 = affine_map<()[s0] -> (s0 * 128 + 64)>
#map7 = affine_map<()[s0] -> (s0 * 512 + 256)>
#map8 = affine_map<()[s0] -> (s0 * 512 + 320)>
#map9 = affine_map<()[s0] -> (s0 * 512 + 384)>
#map10 = affine_map<()[s0] -> (s0 * 512 + 448)>
#map11 = affine_map<()[s0] -> (s0 * 64)>
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
  func.func private @apply_causal_mask(memref<64x64xbf16, 2 : i32>, i32, i32) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  air.channel @Q2L1 [2, 4, 1] {broadcast_shape = [2 : index, 4 : index, 4 : index]}
  air.channel @QIn [2]
  air.channel @K2L1 [2, 4, 1] {broadcast_shape = [2 : index, 4 : index, 4 : index]}
  air.channel @KIn [2]
  air.channel @V2L1 [2, 4, 1] {broadcast_shape = [2 : index, 4 : index, 4 : index]}
  air.channel @VIn [2]
  air.channel @Gp2L2 [4, 4]
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<512x512xbf16>, %arg1: memref<512x128xbf16>, %arg2: memref<512x128xbf16>, %arg3: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1_0) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<512x512xbf16>, memref<512x128xbf16>, memref<512x128xbf16>, memref<512x512xbf16> {
      %c0 = arith.constant 0 : index
      %0 = affine.apply #map()[%arg4]
      %c0_1 = arith.constant 0 : index
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map1()[%arg5]
      %c0_2 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1_3 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c64_4 = arith.constant 64 : index
      %c8192 = arith.constant 8192 : index
      %c64_5 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c1_6 = arith.constant 1 : index
      air.channel.put  @KIn[%c0_1] (%arg9[%c0_2, %1] [%c8, %c1_3, %c64, %c64_4] [%c8192, %c64_5, %c128, %c1_6]) : (memref<512x128xbf16>)
      %c0_7 = arith.constant 0 : index
      %c8_8 = arith.constant 8 : index
      %c64_9 = arith.constant 64 : index
      %c64_10 = arith.constant 64 : index
      %c8192_11 = arith.constant 8192 : index
      %c128_12 = arith.constant 128 : index
      %c1_13 = arith.constant 1 : index
      air.channel.put  @VIn[%c0_1] (%arg10[%c0_7, %2] [%c8_8, %c64_9, %c64_10] [%c8192_11, %c128_12, %c1_13]) : (memref<512x128xbf16>)
      %3 = affine.apply #map2()[%arg5]
      %c4 = arith.constant 4 : index
      %c1_14 = arith.constant 1 : index
      %c64_15 = arith.constant 64 : index
      %c64_16 = arith.constant 64 : index
      %c32768 = arith.constant 32768 : index
      %c64_17 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c1_18 = arith.constant 1 : index
      air.channel.put  @QIn[%c0_1] (%arg8[%0, %3] [%c4, %c1_14, %c64_15, %c64_16] [%c32768, %c64_17, %c512, %c1_18]) : (memref<512x512xbf16>)
      %4 = affine.apply #map3()[%arg5]
      %c4_19 = arith.constant 4 : index
      %c1_20 = arith.constant 1 : index
      %c64_21 = arith.constant 64 : index
      %c64_22 = arith.constant 64 : index
      %c32768_23 = arith.constant 32768 : index
      %c64_24 = arith.constant 64 : index
      %c512_25 = arith.constant 512 : index
      %c1_26 = arith.constant 1 : index
      air.channel.put  @QIn[%c0_1] (%arg8[%0, %4] [%c4_19, %c1_20, %c64_21, %c64_22] [%c32768_23, %c64_24, %c512_25, %c1_26]) : (memref<512x512xbf16>)
      %5 = affine.apply #map4()[%arg5]
      %c4_27 = arith.constant 4 : index
      %c1_28 = arith.constant 1 : index
      %c64_29 = arith.constant 64 : index
      %c64_30 = arith.constant 64 : index
      %c32768_31 = arith.constant 32768 : index
      %c64_32 = arith.constant 64 : index
      %c512_33 = arith.constant 512 : index
      %c1_34 = arith.constant 1 : index
      air.channel.put  @QIn[%c0_1] (%arg8[%0, %5] [%c4_27, %c1_28, %c64_29, %c64_30] [%c32768_31, %c64_32, %c512_33, %c1_34]) : (memref<512x512xbf16>)
      %6 = affine.apply #map5()[%arg5]
      %c4_35 = arith.constant 4 : index
      %c1_36 = arith.constant 1 : index
      %c64_37 = arith.constant 64 : index
      %c64_38 = arith.constant 64 : index
      %c32768_39 = arith.constant 32768 : index
      %c64_40 = arith.constant 64 : index
      %c512_41 = arith.constant 512 : index
      %c1_42 = arith.constant 1 : index
      air.channel.put  @QIn[%c0_1] (%arg8[%0, %6] [%c4_35, %c1_36, %c64_37, %c64_38] [%c32768_39, %c64_40, %c512_41, %c1_42]) : (memref<512x512xbf16>)
      %c1_43 = arith.constant 1 : index
      %7 = affine.apply #map6()[%arg5]
      %8 = affine.apply #map6()[%arg5]
      %c0_44 = arith.constant 0 : index
      %c8_45 = arith.constant 8 : index
      %c1_46 = arith.constant 1 : index
      %c64_47 = arith.constant 64 : index
      %c64_48 = arith.constant 64 : index
      %c8192_49 = arith.constant 8192 : index
      %c64_50 = arith.constant 64 : index
      %c128_51 = arith.constant 128 : index
      %c1_52 = arith.constant 1 : index
      air.channel.put  @KIn[%c1_43] (%arg9[%c0_44, %7] [%c8_45, %c1_46, %c64_47, %c64_48] [%c8192_49, %c64_50, %c128_51, %c1_52]) : (memref<512x128xbf16>)
      %c0_53 = arith.constant 0 : index
      %c8_54 = arith.constant 8 : index
      %c64_55 = arith.constant 64 : index
      %c64_56 = arith.constant 64 : index
      %c8192_57 = arith.constant 8192 : index
      %c128_58 = arith.constant 128 : index
      %c1_59 = arith.constant 1 : index
      air.channel.put  @VIn[%c1_43] (%arg10[%c0_53, %8] [%c8_54, %c64_55, %c64_56] [%c8192_57, %c128_58, %c1_59]) : (memref<512x128xbf16>)
      %9 = affine.apply #map7()[%arg5]
      %c4_60 = arith.constant 4 : index
      %c1_61 = arith.constant 1 : index
      %c64_62 = arith.constant 64 : index
      %c64_63 = arith.constant 64 : index
      %c32768_64 = arith.constant 32768 : index
      %c64_65 = arith.constant 64 : index
      %c512_66 = arith.constant 512 : index
      %c1_67 = arith.constant 1 : index
      air.channel.put  @QIn[%c1_43] (%arg8[%0, %9] [%c4_60, %c1_61, %c64_62, %c64_63] [%c32768_64, %c64_65, %c512_66, %c1_67]) : (memref<512x512xbf16>)
      %10 = affine.apply #map8()[%arg5]
      %c4_68 = arith.constant 4 : index
      %c1_69 = arith.constant 1 : index
      %c64_70 = arith.constant 64 : index
      %c64_71 = arith.constant 64 : index
      %c32768_72 = arith.constant 32768 : index
      %c64_73 = arith.constant 64 : index
      %c512_74 = arith.constant 512 : index
      %c1_75 = arith.constant 1 : index
      air.channel.put  @QIn[%c1_43] (%arg8[%0, %10] [%c4_68, %c1_69, %c64_70, %c64_71] [%c32768_72, %c64_73, %c512_74, %c1_75]) : (memref<512x512xbf16>)
      %11 = affine.apply #map9()[%arg5]
      %c4_76 = arith.constant 4 : index
      %c1_77 = arith.constant 1 : index
      %c64_78 = arith.constant 64 : index
      %c64_79 = arith.constant 64 : index
      %c32768_80 = arith.constant 32768 : index
      %c64_81 = arith.constant 64 : index
      %c512_82 = arith.constant 512 : index
      %c1_83 = arith.constant 1 : index
      air.channel.put  @QIn[%c1_43] (%arg8[%0, %11] [%c4_76, %c1_77, %c64_78, %c64_79] [%c32768_80, %c64_81, %c512_82, %c1_83]) : (memref<512x512xbf16>)
      %12 = affine.apply #map10()[%arg5]
      %c4_84 = arith.constant 4 : index
      %c1_85 = arith.constant 1 : index
      %c64_86 = arith.constant 64 : index
      %c64_87 = arith.constant 64 : index
      %c32768_88 = arith.constant 32768 : index
      %c64_89 = arith.constant 64 : index
      %c512_90 = arith.constant 512 : index
      %c1_91 = arith.constant 1 : index
      air.channel.put  @QIn[%c1_43] (%arg8[%0, %12] [%c4_84, %c1_85, %c64_86, %c64_87] [%c32768_88, %c64_89, %c512_90, %c1_91]) : (memref<512x512xbf16>)
      %c2_92 = arith.constant 2 : index
      %c1_93 = arith.constant 1 : index
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2_92, %arg15=%c1_93) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_127 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_128 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_129 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_130 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_131 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_132 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_133 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_134 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_135 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_136 = memref.alloc() : memref<3xi32, 2 : i32>
        %c4_137 = arith.constant 4 : index
        %c4_138 = arith.constant 4 : index
        %c0_139 = arith.constant 0 : index
        %c8_140 = arith.constant 8 : index
        %c4_141 = arith.constant 4 : index
        %c8_142 = arith.constant 8 : index
        %c0_143 = arith.constant 0 : index
        %c0_144 = arith.constant 0 : index
        %c1_145 = arith.constant 1 : index
        scf.for %arg16 = %c0_144 to %c4_141 step %c1_145 {
          air.channel.get  @QIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_162 = arith.constant 0 : index
          %c0_163 = arith.constant 0 : index
          %c0_164 = arith.constant 0 : index
          %c0_165 = arith.constant 0 : index
          %c8_166 = arith.constant 8 : index
          %c8_167 = arith.constant 8 : index
          %c8_168 = arith.constant 8 : index
          %c8_169 = arith.constant 8 : index
          %c8_170 = arith.constant 8 : index
          %c512_171 = arith.constant 512 : index
          %c64_172 = arith.constant 64 : index
          %c1_173 = arith.constant 1 : index
          air.channel.put  @Q2L1[%arg12, %c0_143, %c0_139] (%alloc[%c0_162, %c0_163, %c0_164, %c0_165] [%c8_166, %c8_167, %c8_168, %c8_169] [%c8_170, %c512_171, %c64_172, %c1_173]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c1_146 = arith.constant 1 : index
        %c0_147 = arith.constant 0 : index
        %c1_148 = arith.constant 1 : index
        scf.for %arg16 = %c0_147 to %c4_141 step %c1_148 {
          air.channel.get  @QIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_162 = arith.constant 0 : index
          %c0_163 = arith.constant 0 : index
          %c0_164 = arith.constant 0 : index
          %c0_165 = arith.constant 0 : index
          %c8_166 = arith.constant 8 : index
          %c8_167 = arith.constant 8 : index
          %c8_168 = arith.constant 8 : index
          %c8_169 = arith.constant 8 : index
          %c8_170 = arith.constant 8 : index
          %c512_171 = arith.constant 512 : index
          %c64_172 = arith.constant 64 : index
          %c1_173 = arith.constant 1 : index
          air.channel.put  @Q2L1[%arg12, %c1_146, %c0_139] (%alloc[%c0_162, %c0_163, %c0_164, %c0_165] [%c8_166, %c8_167, %c8_168, %c8_169] [%c8_170, %c512_171, %c64_172, %c1_173]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c2_149 = arith.constant 2 : index
        %c0_150 = arith.constant 0 : index
        %c1_151 = arith.constant 1 : index
        scf.for %arg16 = %c0_150 to %c4_141 step %c1_151 {
          air.channel.get  @QIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_162 = arith.constant 0 : index
          %c0_163 = arith.constant 0 : index
          %c0_164 = arith.constant 0 : index
          %c0_165 = arith.constant 0 : index
          %c8_166 = arith.constant 8 : index
          %c8_167 = arith.constant 8 : index
          %c8_168 = arith.constant 8 : index
          %c8_169 = arith.constant 8 : index
          %c8_170 = arith.constant 8 : index
          %c512_171 = arith.constant 512 : index
          %c64_172 = arith.constant 64 : index
          %c1_173 = arith.constant 1 : index
          air.channel.put  @Q2L1[%arg12, %c2_149, %c0_139] (%alloc[%c0_162, %c0_163, %c0_164, %c0_165] [%c8_166, %c8_167, %c8_168, %c8_169] [%c8_170, %c512_171, %c64_172, %c1_173]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c3 = arith.constant 3 : index
        %c0_152 = arith.constant 0 : index
        %c1_153 = arith.constant 1 : index
        scf.for %arg16 = %c0_152 to %c4_141 step %c1_153 {
          air.channel.get  @QIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_162 = arith.constant 0 : index
          %c0_163 = arith.constant 0 : index
          %c0_164 = arith.constant 0 : index
          %c0_165 = arith.constant 0 : index
          %c8_166 = arith.constant 8 : index
          %c8_167 = arith.constant 8 : index
          %c8_168 = arith.constant 8 : index
          %c8_169 = arith.constant 8 : index
          %c8_170 = arith.constant 8 : index
          %c512_171 = arith.constant 512 : index
          %c64_172 = arith.constant 64 : index
          %c1_173 = arith.constant 1 : index
          air.channel.put  @Q2L1[%arg12, %c3, %c0_139] (%alloc[%c0_162, %c0_163, %c0_164, %c0_165] [%c8_166, %c8_167, %c8_168, %c8_169] [%c8_170, %c512_171, %c64_172, %c1_173]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_154 = arith.constant 0 : index
        %c1_155 = arith.constant 1 : index
        scf.for %arg16 = %c0_154 to %c8_142 step %c1_155 {
          air.channel.get  @KIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_162 = arith.constant 0 : index
          %c0_163 = arith.constant 0 : index
          %c0_164 = arith.constant 0 : index
          %c0_165 = arith.constant 0 : index
          %c0_166 = arith.constant 0 : index
          %c8_167 = arith.constant 8 : index
          %c8_168 = arith.constant 8 : index
          %c8_169 = arith.constant 8 : index
          %c8_170 = arith.constant 8 : index
          %c8_171 = arith.constant 8 : index
          %c512_172 = arith.constant 512 : index
          %c64_173 = arith.constant 64 : index
          %c1_174 = arith.constant 1 : index
          air.channel.put  @K2L1[%arg12, %c0_162, %c0_139] (%alloc[%c0_163, %c0_164, %c0_165, %c0_166] [%c8_167, %c8_168, %c8_169, %c8_170] [%c8_171, %c512_172, %c64_173, %c1_174]) : (memref<64x64xbf16, 1 : i32>)
          %c1_175 = arith.constant 1 : index
          %c0_176 = arith.constant 0 : index
          %c0_177 = arith.constant 0 : index
          %c0_178 = arith.constant 0 : index
          %c0_179 = arith.constant 0 : index
          %c8_180 = arith.constant 8 : index
          %c8_181 = arith.constant 8 : index
          %c8_182 = arith.constant 8 : index
          %c8_183 = arith.constant 8 : index
          %c8_184 = arith.constant 8 : index
          %c512_185 = arith.constant 512 : index
          %c64_186 = arith.constant 64 : index
          %c1_187 = arith.constant 1 : index
          air.channel.put  @K2L1[%arg12, %c1_175, %c0_139] (%alloc[%c0_176, %c0_177, %c0_178, %c0_179] [%c8_180, %c8_181, %c8_182, %c8_183] [%c8_184, %c512_185, %c64_186, %c1_187]) : (memref<64x64xbf16, 1 : i32>)
          %c2_188 = arith.constant 2 : index
          %c0_189 = arith.constant 0 : index
          %c0_190 = arith.constant 0 : index
          %c0_191 = arith.constant 0 : index
          %c0_192 = arith.constant 0 : index
          %c8_193 = arith.constant 8 : index
          %c8_194 = arith.constant 8 : index
          %c8_195 = arith.constant 8 : index
          %c8_196 = arith.constant 8 : index
          %c8_197 = arith.constant 8 : index
          %c512_198 = arith.constant 512 : index
          %c64_199 = arith.constant 64 : index
          %c1_200 = arith.constant 1 : index
          air.channel.put  @K2L1[%arg12, %c2_188, %c0_139] (%alloc[%c0_189, %c0_190, %c0_191, %c0_192] [%c8_193, %c8_194, %c8_195, %c8_196] [%c8_197, %c512_198, %c64_199, %c1_200]) : (memref<64x64xbf16, 1 : i32>)
          %c3_201 = arith.constant 3 : index
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c8_210 = arith.constant 8 : index
          %c512_211 = arith.constant 512 : index
          %c64_212 = arith.constant 64 : index
          %c1_213 = arith.constant 1 : index
          air.channel.put  @K2L1[%arg12, %c3_201, %c0_139] (%alloc[%c0_202, %c0_203, %c0_204, %c0_205] [%c8_206, %c8_207, %c8_208, %c8_209] [%c8_210, %c512_211, %c64_212, %c1_213]) : (memref<64x64xbf16, 1 : i32>)
        }
        %c0_156 = arith.constant 0 : index
        %c1_157 = arith.constant 1 : index
        scf.for %arg16 = %c0_156 to %c8_140 step %c1_157 {
          air.channel.get  @VIn[%arg12] (%alloc_127[] [] []) : (memref<64x64xbf16, 1 : i32>)
          %c0_162 = arith.constant 0 : index
          %c0_163 = arith.constant 0 : index
          %c0_164 = arith.constant 0 : index
          %c0_165 = arith.constant 0 : index
          %c0_166 = arith.constant 0 : index
          %c8_167 = arith.constant 8 : index
          %c8_168 = arith.constant 8 : index
          %c8_169 = arith.constant 8 : index
          %c8_170 = arith.constant 8 : index
          %c8_171 = arith.constant 8 : index
          %c512_172 = arith.constant 512 : index
          %c64_173 = arith.constant 64 : index
          %c1_174 = arith.constant 1 : index
          air.channel.put  @V2L1[%arg12, %c0_162, %c0_139] (%alloc_127[%c0_163, %c0_164, %c0_165, %c0_166] [%c8_167, %c8_168, %c8_169, %c8_170] [%c8_171, %c512_172, %c64_173, %c1_174]) : (memref<64x64xbf16, 1 : i32>)
          %c1_175 = arith.constant 1 : index
          %c0_176 = arith.constant 0 : index
          %c0_177 = arith.constant 0 : index
          %c0_178 = arith.constant 0 : index
          %c0_179 = arith.constant 0 : index
          %c8_180 = arith.constant 8 : index
          %c8_181 = arith.constant 8 : index
          %c8_182 = arith.constant 8 : index
          %c8_183 = arith.constant 8 : index
          %c8_184 = arith.constant 8 : index
          %c512_185 = arith.constant 512 : index
          %c64_186 = arith.constant 64 : index
          %c1_187 = arith.constant 1 : index
          air.channel.put  @V2L1[%arg12, %c1_175, %c0_139] (%alloc_127[%c0_176, %c0_177, %c0_178, %c0_179] [%c8_180, %c8_181, %c8_182, %c8_183] [%c8_184, %c512_185, %c64_186, %c1_187]) : (memref<64x64xbf16, 1 : i32>)
          %c2_188 = arith.constant 2 : index
          %c0_189 = arith.constant 0 : index
          %c0_190 = arith.constant 0 : index
          %c0_191 = arith.constant 0 : index
          %c0_192 = arith.constant 0 : index
          %c8_193 = arith.constant 8 : index
          %c8_194 = arith.constant 8 : index
          %c8_195 = arith.constant 8 : index
          %c8_196 = arith.constant 8 : index
          %c8_197 = arith.constant 8 : index
          %c512_198 = arith.constant 512 : index
          %c64_199 = arith.constant 64 : index
          %c1_200 = arith.constant 1 : index
          air.channel.put  @V2L1[%arg12, %c2_188, %c0_139] (%alloc_127[%c0_189, %c0_190, %c0_191, %c0_192] [%c8_193, %c8_194, %c8_195, %c8_196] [%c8_197, %c512_198, %c64_199, %c1_200]) : (memref<64x64xbf16, 1 : i32>)
          %c3_201 = arith.constant 3 : index
          %c0_202 = arith.constant 0 : index
          %c0_203 = arith.constant 0 : index
          %c0_204 = arith.constant 0 : index
          %c0_205 = arith.constant 0 : index
          %c8_206 = arith.constant 8 : index
          %c8_207 = arith.constant 8 : index
          %c8_208 = arith.constant 8 : index
          %c8_209 = arith.constant 8 : index
          %c8_210 = arith.constant 8 : index
          %c512_211 = arith.constant 512 : index
          %c64_212 = arith.constant 64 : index
          %c1_213 = arith.constant 1 : index
          air.channel.put  @V2L1[%arg12, %c3_201, %c0_139] (%alloc_127[%c0_202, %c0_203, %c0_204, %c0_205] [%c8_206, %c8_207, %c8_208, %c8_209] [%c8_210, %c512_211, %c64_212, %c1_213]) : (memref<64x64xbf16, 1 : i32>)
        }
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4_137, %arg19=%c4_138) args(%arg20=%alloc_129, %arg21=%alloc_130, %arg22=%alloc_131, %arg23=%alloc_132, %arg24=%alloc_133, %arg25=%alloc_134, %arg26=%alloc_135, %arg27=%arg12, %arg28=%alloc_136) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index, memref<3xi32, 2 : i32> attributes {link_with = "attn_npu2.o"} {
          func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          %c0_162 = arith.constant 0 : index
          %c1_163 = arith.constant 1 : index
          %c2_164 = arith.constant 2 : index
          %21 = memref.load %arg28[%c1_163] : memref<3xi32, 2 : i32>
          %c0_i32 = arith.constant 0 : i32
          %22 = arith.cmpi eq, %21, %c0_i32 : i32
          scf.if %22 {
            %c0_i32_183 = arith.constant 0 : i32
            memref.store %c0_i32_183, %arg28[%c0_162] : memref<3xi32, 2 : i32>
            %c1_i32_184 = arith.constant 1 : i32
            memref.store %c1_i32_184, %arg28[%c1_163] : memref<3xi32, 2 : i32>
            %c0_i32_185 = arith.constant 0 : i32
            memref.store %c0_i32_185, %arg28[%c2_164] : memref<3xi32, 2 : i32>
          }
          air.channel.get  @Q2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %23 = arith.index_cast %arg16 : index to i32
          %c0_i32_165 = arith.constant 0 : i32
          %24 = arith.cmpi eq, %23, %c0_i32_165 : i32
          scf.if %24 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @Q2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %25 = arith.index_cast %arg16 : index to i32
          %c1_i32 = arith.constant 1 : i32
          %26 = arith.cmpi eq, %25, %c1_i32 : i32
          scf.if %26 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @Q2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %27 = arith.index_cast %arg16 : index to i32
          %c2_i32 = arith.constant 2 : i32
          %28 = arith.cmpi eq, %27, %c2_i32 : i32
          scf.if %28 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @Q2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %29 = arith.index_cast %arg16 : index to i32
          %c3_i32 = arith.constant 3 : i32
          %30 = arith.cmpi eq, %29, %c3_i32 : i32
          scf.if %30 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %c8_166 = arith.constant 8 : index
          %c0_167 = arith.constant 0 : index
          %c1_168 = arith.constant 1 : index
          scf.for %arg29 = %c0_167 to %c8_166 step %c1_168 {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            air.channel.get  @K2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            air.channel.get  @V2L1[%arg27, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            %35 = arith.index_cast %arg29 : index to i32
            %36 = memref.load %arg28[%c0_162] : memref<3xi32, 2 : i32>
            %37 = arith.index_cast %arg16 : index to i32
            %38 = arith.addi %36, %37 : i32
            func.call @apply_causal_mask(%arg23, %38, %35) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
            %alloc_183 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_184 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %alloc_183, %alloc_184) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_184, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            %c0_i32_185 = arith.constant 0 : i32
            func.call @accum_sp_r_s(%arg26, %alloc_184, %alloc_183) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32_185, %alloc_183, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_183 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_184 : memref<64x1xbf16, 2 : i32>
          }
          func.call @div_gp_sp(%arg26, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          %c0_169 = arith.constant 0 : index
          %c0_170 = arith.constant 0 : index
          %c0_171 = arith.constant 0 : index
          %c0_172 = arith.constant 0 : index
          %c8_173 = arith.constant 8 : index
          %c8_174 = arith.constant 8 : index
          %c8_175 = arith.constant 8 : index
          %c8_176 = arith.constant 8 : index
          %c64_177 = arith.constant 64 : index
          %c8_178 = arith.constant 8 : index
          %c512_179 = arith.constant 512 : index
          %c1_180 = arith.constant 1 : index
          air.channel.put  @Gp2L2[%arg17, %arg16] (%arg24[%c0_169, %c0_170, %c0_171, %c0_172] [%c8_173, %c8_174, %c8_175, %c8_176] [%c64_177, %c8_178, %c512_179, %c1_180]) : (memref<64x64xbf16, 2 : i32>)
          %31 = memref.load %arg28[%c2_164] : memref<3xi32, 2 : i32>
          %c1_i32_181 = arith.constant 1 : i32
          %32 = arith.addi %31, %c1_i32_181 : i32
          %c1_i32_182 = arith.constant 1 : i32
          %33 = arith.cmpi sge, %32, %c1_i32_182 : i32
          scf.if %33 {
            %35 = memref.load %arg28[%c0_162] : memref<3xi32, 2 : i32>
            %c4_i32 = arith.constant 4 : i32
            %36 = arith.addi %35, %c4_i32 : i32
            memref.store %36, %arg28[%c0_162] : memref<3xi32, 2 : i32>
            %c0_i32_183 = arith.constant 0 : i32
            memref.store %c0_i32_183, %arg28[%c2_164] : memref<3xi32, 2 : i32>
          }
          %34 = arith.cmpi slt, %32, %c1_i32_182 : i32
          scf.if %34 {
            memref.store %32, %arg28[%c2_164] : memref<3xi32, 2 : i32>
          }
        }
        %c0_158 = arith.constant 0 : index
        scf.forall (%arg16) in (4) {
          %21 = affine.apply #map11()[%arg16]
          %c0_162 = arith.constant 0 : index
          %c64_163 = arith.constant 64 : index
          %c64_164 = arith.constant 64 : index
          %c64_165 = arith.constant 64 : index
          %c1_166 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%c0_158, %arg16] (%alloc_128[%21, %c0_162] [%c64_163, %c64_164] [%c64_165, %c1_166]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_128[] [] []) : (memref<256x64xbf16, 1 : i32>)
        %c1_159 = arith.constant 1 : index
        scf.forall (%arg16) in (4) {
          %21 = affine.apply #map11()[%arg16]
          %c0_162 = arith.constant 0 : index
          %c64_163 = arith.constant 64 : index
          %c64_164 = arith.constant 64 : index
          %c64_165 = arith.constant 64 : index
          %c1_166 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%c1_159, %arg16] (%alloc_128[%21, %c0_162] [%c64_163, %c64_164] [%c64_165, %c1_166]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_128[] [] []) : (memref<256x64xbf16, 1 : i32>)
        %c2_160 = arith.constant 2 : index
        scf.forall (%arg16) in (4) {
          %21 = affine.apply #map11()[%arg16]
          %c0_162 = arith.constant 0 : index
          %c64_163 = arith.constant 64 : index
          %c64_164 = arith.constant 64 : index
          %c64_165 = arith.constant 64 : index
          %c1_166 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%c2_160, %arg16] (%alloc_128[%21, %c0_162] [%c64_163, %c64_164] [%c64_165, %c1_166]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_128[] [] []) : (memref<256x64xbf16, 1 : i32>)
        %c3_161 = arith.constant 3 : index
        scf.forall (%arg16) in (4) {
          %21 = affine.apply #map11()[%arg16]
          %c0_162 = arith.constant 0 : index
          %c64_163 = arith.constant 64 : index
          %c64_164 = arith.constant 64 : index
          %c64_165 = arith.constant 64 : index
          %c1_166 = arith.constant 1 : index
          air.channel.get  @Gp2L2[%c3_161, %arg16] (%alloc_128[%21, %c0_162] [%c64_163, %c64_164] [%c64_165, %c1_166]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_128[] [] []) : (memref<256x64xbf16, 1 : i32>)
        memref.dealloc %alloc_129 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_130 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_131 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_132 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_133 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_134 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_135 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_127 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_128 : memref<256x64xbf16, 1 : i32>
        memref.dealloc %alloc_136 : memref<3xi32, 2 : i32>
      }
      %c0_94 = arith.constant 0 : index
      %13 = affine.apply #map2()[%arg5]
      %c256 = arith.constant 256 : index
      %c64_95 = arith.constant 64 : index
      %c512_96 = arith.constant 512 : index
      %c1_97 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0_94] (%arg11[%0, %13] [%c256, %c64_95] [%c512_96, %c1_97]) : (memref<512x512xbf16>)
      %14 = affine.apply #map3()[%arg5]
      %c256_98 = arith.constant 256 : index
      %c64_99 = arith.constant 64 : index
      %c512_100 = arith.constant 512 : index
      %c1_101 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0_94] (%arg11[%0, %14] [%c256_98, %c64_99] [%c512_100, %c1_101]) : (memref<512x512xbf16>)
      %15 = affine.apply #map4()[%arg5]
      %c256_102 = arith.constant 256 : index
      %c64_103 = arith.constant 64 : index
      %c512_104 = arith.constant 512 : index
      %c1_105 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0_94] (%arg11[%0, %15] [%c256_102, %c64_103] [%c512_104, %c1_105]) : (memref<512x512xbf16>)
      %16 = affine.apply #map5()[%arg5]
      %c256_106 = arith.constant 256 : index
      %c64_107 = arith.constant 64 : index
      %c512_108 = arith.constant 512 : index
      %c1_109 = arith.constant 1 : index
      air.channel.get  @GpOut[%c0_94] (%arg11[%0, %16] [%c256_106, %c64_107] [%c512_108, %c1_109]) : (memref<512x512xbf16>)
      %c1_110 = arith.constant 1 : index
      %17 = affine.apply #map7()[%arg5]
      %c256_111 = arith.constant 256 : index
      %c64_112 = arith.constant 64 : index
      %c512_113 = arith.constant 512 : index
      %c1_114 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_110] (%arg11[%0, %17] [%c256_111, %c64_112] [%c512_113, %c1_114]) : (memref<512x512xbf16>)
      %18 = affine.apply #map8()[%arg5]
      %c256_115 = arith.constant 256 : index
      %c64_116 = arith.constant 64 : index
      %c512_117 = arith.constant 512 : index
      %c1_118 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_110] (%arg11[%0, %18] [%c256_115, %c64_116] [%c512_117, %c1_118]) : (memref<512x512xbf16>)
      %19 = affine.apply #map9()[%arg5]
      %c256_119 = arith.constant 256 : index
      %c64_120 = arith.constant 64 : index
      %c512_121 = arith.constant 512 : index
      %c1_122 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_110] (%arg11[%0, %19] [%c256_119, %c64_120] [%c512_121, %c1_122]) : (memref<512x512xbf16>)
      %20 = affine.apply #map10()[%arg5]
      %c256_123 = arith.constant 256 : index
      %c64_124 = arith.constant 64 : index
      %c512_125 = arith.constant 512 : index
      %c1_126 = arith.constant 1 : index
      air.channel.get  @GpOut[%c1_110] (%arg11[%0, %20] [%c256_123, %c64_124] [%c512_125, %c1_126]) : (memref<512x512xbf16>)
    }
    return
  }
}

