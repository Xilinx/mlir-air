python3 /home/erweiw/mlir-air/programming_examples/vector_matrix_multiplication/block_quantized_i8/single_core/single_core.py --k 576 --n 576 -p
#map = affine_map<()[s0] -> (s0 * 48)>
module {
  air.channel @aL3ToL2 []
  air.channel @bL3ToL2 []
  air.channel @aL2ToL1 []
  air.channel @bL2ToL1 []
  air.channel @cL1ToL2 []
  air.channel @cL2ToL3 []
  func.func private @linalg_fill_i32_view16x8xi32as2(f32, memref<6x8xf32, 2 : i32>) attributes {link_with = "vm.o", llvm.emit_c_interface}
  func.func private @vecmat_i8_f32_i32_32(memref<6x16xi8, 2 : i32>, memref<3xf32, 2 : i32>, memref<6x6x16x8xi8, 2 : i32>, memref<6x3x8xf32, 2 : i32>, memref<6x8xf32, 2 : i32>) attributes {link_with = "vm.o", llvm.emit_c_interface}
  func.func @vecmat_i8(%arg0: memref<576xi8>, %arg1: memref<18xf32>, %arg2: memref<576x576xi8>, %arg3: memref<18x576xf32>, %arg4: memref<576xf32>) {
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    air.launch (%arg5, %arg6) in (%arg7=%c1, %arg8=%c12) args(%arg9=%arg0, %arg10=%arg1, %arg11=%arg2, %arg12=%arg3, %arg13=%arg4) : memref<576xi8>, memref<18xf32>, memref<576x576xi8>, memref<18x576xf32>, memref<576xf32> {
      air.channel.put  @aL3ToL2[] (%arg9[] [] []) : (memref<576xi8>)
      air.channel.put  @aL3ToL2[] (%arg10[] [] []) : (memref<18xf32>)
      %0 = affine.apply #map()[%arg6]
      %c0 = arith.constant 0 : index
      %c576 = arith.constant 576 : index
      %c48 = arith.constant 48 : index
      %c576_0 = arith.constant 576 : index
      %c1_1 = arith.constant 1 : index
      air.channel.put  @bL3ToL2[] (%arg11[%c0, %0] [%c576, %c48] [%c576_0, %c1_1]) : (memref<576x576xi8>)
      %c0_2 = arith.constant 0 : index
      %c18 = arith.constant 18 : index
      %c48_3 = arith.constant 48 : index
      %c576_4 = arith.constant 576 : index
      %c1_5 = arith.constant 1 : index
      air.channel.put  @bL3ToL2[] (%arg12[%c0_2, %0] [%c18, %c48_3] [%c576_4, %c1_5]) : (memref<18x576xf32>)
      %c48_6 = arith.constant 48 : index
      %c1_7 = arith.constant 1 : index
      air.channel.get  @cL2ToL3[] (%arg13[%0] [%c48_6] [%c1_7]) : (memref<576xf32>)
      air.segment @vecmat_i8_0  {
        %alloc = memref.alloc() : memref<576xi8, 1 : i32>
        %alloc_8 = memref.alloc() : memref<18xf32, 1 : i32>
        air.channel.get  @aL3ToL2[] (%alloc[] [] []) : (memref<576xi8, 1 : i32>)
        air.channel.get  @aL3ToL2[] (%alloc_8[] [] []) : (memref<18xf32, 1 : i32>)
        %alloc_9 = memref.alloc() : memref<576x48xi8, 1 : i32>
        %alloc_10 = memref.alloc() : memref<18x48xf32, 1 : i32>
        air.channel.get  @bL3ToL2[] (%alloc_9[] [] []) : (memref<576x48xi8, 1 : i32>)
        air.channel.get  @bL3ToL2[] (%alloc_10[] [] []) : (memref<18x48xf32, 1 : i32>)
        %c0_11 = arith.constant 0 : index
        %c96 = arith.constant 96 : index
        %c1_12 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc[%c0_11] [%c96] [%c1_12]) : (memref<576xi8, 1 : i32>)
        %c0_13 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_14 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc_8[%c0_13] [%c3] [%c1_14]) : (memref<18xf32, 1 : i32>)
        %c96_15 = arith.constant 96 : index
        %c96_16 = arith.constant 96 : index
        %c1_17 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc[%c96_15] [%c96_16] [%c1_17]) : (memref<576xi8, 1 : i32>)
        %c3_18 = arith.constant 3 : index
        %c3_19 = arith.constant 3 : index
        %c1_20 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc_8[%c3_18] [%c3_19] [%c1_20]) : (memref<18xf32, 1 : i32>)
        %c192 = arith.constant 192 : index
        %c96_21 = arith.constant 96 : index
        %c1_22 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc[%c192] [%c96_21] [%c1_22]) : (memref<576xi8, 1 : i32>)
        %c6 = arith.constant 6 : index
        %c3_23 = arith.constant 3 : index
        %c1_24 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc_8[%c6] [%c3_23] [%c1_24]) : (memref<18xf32, 1 : i32>)
        %c288 = arith.constant 288 : index
        %c96_25 = arith.constant 96 : index
        %c1_26 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc[%c288] [%c96_25] [%c1_26]) : (memref<576xi8, 1 : i32>)
        %c9 = arith.constant 9 : index
        %c3_27 = arith.constant 3 : index
        %c1_28 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc_8[%c9] [%c3_27] [%c1_28]) : (memref<18xf32, 1 : i32>)
        %c384 = arith.constant 384 : index
        %c96_29 = arith.constant 96 : index
        %c1_30 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc[%c384] [%c96_29] [%c1_30]) : (memref<576xi8, 1 : i32>)
        %c12_31 = arith.constant 12 : index
        %c3_32 = arith.constant 3 : index
        %c1_33 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc_8[%c12_31] [%c3_32] [%c1_33]) : (memref<18xf32, 1 : i32>)
        %c480 = arith.constant 480 : index
        %c96_34 = arith.constant 96 : index
        %c1_35 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc[%c480] [%c96_34] [%c1_35]) : (memref<576xi8, 1 : i32>)
        %c15 = arith.constant 15 : index
        %c3_36 = arith.constant 3 : index
        %c1_37 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc_8[%c15] [%c3_36] [%c1_37]) : (memref<18xf32, 1 : i32>)
        %c0_38 = arith.constant 0 : index
        %c0_39 = arith.constant 0 : index
        %c0_40 = arith.constant 0 : index
        %c6_41 = arith.constant 6 : index
        %c96_42 = arith.constant 96 : index
        %c8 = arith.constant 8 : index
        %c8_43 = arith.constant 8 : index
        %c48_44 = arith.constant 48 : index
        %c1_45 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_9[%c0_38, %c0_39, %c0_40] [%c6_41, %c96_42, %c8] [%c8_43, %c48_44, %c1_45]) : (memref<576x48xi8, 1 : i32>)
        %c0_46 = arith.constant 0 : index
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %c6_49 = arith.constant 6 : index
        %c3_50 = arith.constant 3 : index
        %c8_51 = arith.constant 8 : index
        %c8_52 = arith.constant 8 : index
        %c48_53 = arith.constant 48 : index
        %c1_54 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_10[%c0_46, %c0_47, %c0_48] [%c6_49, %c3_50, %c8_51] [%c8_52, %c48_53, %c1_54]) : (memref<18x48xf32, 1 : i32>)
        %c0_55 = arith.constant 0 : index
        %c96_56 = arith.constant 96 : index
        %c0_57 = arith.constant 0 : index
        %c6_58 = arith.constant 6 : index
        %c96_59 = arith.constant 96 : index
        %c8_60 = arith.constant 8 : index
        %c8_61 = arith.constant 8 : index
        %c48_62 = arith.constant 48 : index
        %c1_63 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_9[%c0_55, %c96_56, %c0_57] [%c6_58, %c96_59, %c8_60] [%c8_61, %c48_62, %c1_63]) : (memref<576x48xi8, 1 : i32>)
        %c0_64 = arith.constant 0 : index
        %c3_65 = arith.constant 3 : index
        %c0_66 = arith.constant 0 : index
        %c6_67 = arith.constant 6 : index
        %c3_68 = arith.constant 3 : index
        %c8_69 = arith.constant 8 : index
        %c8_70 = arith.constant 8 : index
        %c48_71 = arith.constant 48 : index
        %c1_72 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_10[%c0_64, %c3_65, %c0_66] [%c6_67, %c3_68, %c8_69] [%c8_70, %c48_71, %c1_72]) : (memref<18x48xf32, 1 : i32>)
        %c0_73 = arith.constant 0 : index
        %c192_74 = arith.constant 192 : index
        %c0_75 = arith.constant 0 : index
        %c6_76 = arith.constant 6 : index
        %c96_77 = arith.constant 96 : index
        %c8_78 = arith.constant 8 : index
        %c8_79 = arith.constant 8 : index
        %c48_80 = arith.constant 48 : index
        %c1_81 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_9[%c0_73, %c192_74, %c0_75] [%c6_76, %c96_77, %c8_78] [%c8_79, %c48_80, %c1_81]) : (memref<576x48xi8, 1 : i32>)
        %c0_82 = arith.constant 0 : index
        %c6_83 = arith.constant 6 : index
        %c0_84 = arith.constant 0 : index
        %c6_85 = arith.constant 6 : index
        %c3_86 = arith.constant 3 : index
        %c8_87 = arith.constant 8 : index
        %c8_88 = arith.constant 8 : index
        %c48_89 = arith.constant 48 : index
        %c1_90 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_10[%c0_82, %c6_83, %c0_84] [%c6_85, %c3_86, %c8_87] [%c8_88, %c48_89, %c1_90]) : (memref<18x48xf32, 1 : i32>)
        %c0_91 = arith.constant 0 : index
        %c288_92 = arith.constant 288 : index
        %c0_93 = arith.constant 0 : index
        %c6_94 = arith.constant 6 : index
        %c96_95 = arith.constant 96 : index
        %c8_96 = arith.constant 8 : index
        %c8_97 = arith.constant 8 : index
        %c48_98 = arith.constant 48 : index
        %c1_99 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_9[%c0_91, %c288_92, %c0_93] [%c6_94, %c96_95, %c8_96] [%c8_97, %c48_98, %c1_99]) : (memref<576x48xi8, 1 : i32>)
        %c0_100 = arith.constant 0 : index
        %c9_101 = arith.constant 9 : index
        %c0_102 = arith.constant 0 : index
        %c6_103 = arith.constant 6 : index
        %c3_104 = arith.constant 3 : index
        %c8_105 = arith.constant 8 : index
        %c8_106 = arith.constant 8 : index
        %c48_107 = arith.constant 48 : index
        %c1_108 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_10[%c0_100, %c9_101, %c0_102] [%c6_103, %c3_104, %c8_105] [%c8_106, %c48_107, %c1_108]) : (memref<18x48xf32, 1 : i32>)
        %c0_109 = arith.constant 0 : index
        %c384_110 = arith.constant 384 : index
        %c0_111 = arith.constant 0 : index
        %c6_112 = arith.constant 6 : index
        %c96_113 = arith.constant 96 : index
        %c8_114 = arith.constant 8 : index
        %c8_115 = arith.constant 8 : index
        %c48_116 = arith.constant 48 : index
        %c1_117 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_9[%c0_109, %c384_110, %c0_111] [%c6_112, %c96_113, %c8_114] [%c8_115, %c48_116, %c1_117]) : (memref<576x48xi8, 1 : i32>)
        %c0_118 = arith.constant 0 : index
        %c12_119 = arith.constant 12 : index
        %c0_120 = arith.constant 0 : index
        %c6_121 = arith.constant 6 : index
        %c3_122 = arith.constant 3 : index
        %c8_123 = arith.constant 8 : index
        %c8_124 = arith.constant 8 : index
        %c48_125 = arith.constant 48 : index
        %c1_126 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_10[%c0_118, %c12_119, %c0_120] [%c6_121, %c3_122, %c8_123] [%c8_124, %c48_125, %c1_126]) : (memref<18x48xf32, 1 : i32>)
        %c0_127 = arith.constant 0 : index
        %c480_128 = arith.constant 480 : index
        %c0_129 = arith.constant 0 : index
        %c6_130 = arith.constant 6 : index
        %c96_131 = arith.constant 96 : index
        %c8_132 = arith.constant 8 : index
        %c8_133 = arith.constant 8 : index
        %c48_134 = arith.constant 48 : index
        %c1_135 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_9[%c0_127, %c480_128, %c0_129] [%c6_130, %c96_131, %c8_132] [%c8_133, %c48_134, %c1_135]) : (memref<576x48xi8, 1 : i32>)
        %c0_136 = arith.constant 0 : index
        %c15_137 = arith.constant 15 : index
        %c0_138 = arith.constant 0 : index
        %c6_139 = arith.constant 6 : index
        %c3_140 = arith.constant 3 : index
        %c8_141 = arith.constant 8 : index
        %c8_142 = arith.constant 8 : index
        %c48_143 = arith.constant 48 : index
        %c1_144 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_10[%c0_136, %c15_137, %c0_138] [%c6_139, %c3_140, %c8_141] [%c8_142, %c48_143, %c1_144]) : (memref<18x48xf32, 1 : i32>)
        %alloc_145 = memref.alloc() : memref<48xf32, 1 : i32>
        air.channel.get  @cL1ToL2[] (%alloc_145[] [] []) : (memref<48xf32, 1 : i32>)
        %c1_146 = arith.constant 1 : index
        %c1_147 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg14, %arg15) in (%arg16=%c1_146, %arg17=%c1_147) attributes {link_with = "vm.o"} {
          %alloc_148 = memref.alloc() : memref<6x8xf32, 2 : i32>
          %cst = arith.constant 0.000000e+00 : f32
          func.call @linalg_fill_i32_view16x8xi32as2(%cst, %alloc_148) : (f32, memref<6x8xf32, 2 : i32>) -> ()
          %c0_149 = arith.constant 0 : index
          %c576_150 = arith.constant 576 : index
          %c96_151 = arith.constant 96 : index
          scf.for %arg18 = %c0_149 to %c576_150 step %c96_151 {
            %alloc_152 = memref.alloc() : memref<6x16xi8, 2 : i32>
            %alloc_153 = memref.alloc() : memref<3xf32, 2 : i32>
            air.channel.get  @aL2ToL1[] (%alloc_152[] [] []) : (memref<6x16xi8, 2 : i32>)
            air.channel.get  @aL2ToL1[] (%alloc_153[] [] []) : (memref<3xf32, 2 : i32>)
            %alloc_154 = memref.alloc() : memref<6x6x16x8xi8, 2 : i32>
            %alloc_155 = memref.alloc() : memref<6x3x8xf32, 2 : i32>
            air.channel.get  @bL2ToL1[] (%alloc_154[] [] []) : (memref<6x6x16x8xi8, 2 : i32>)
            air.channel.get  @bL2ToL1[] (%alloc_155[] [] []) : (memref<6x3x8xf32, 2 : i32>)
            func.call @vecmat_i8_f32_i32_32(%alloc_152, %alloc_153, %alloc_154, %alloc_155, %alloc_148) : (memref<6x16xi8, 2 : i32>, memref<3xf32, 2 : i32>, memref<6x6x16x8xi8, 2 : i32>, memref<6x3x8xf32, 2 : i32>, memref<6x8xf32, 2 : i32>) -> ()
            memref.dealloc %alloc_152 : memref<6x16xi8, 2 : i32>
            memref.dealloc %alloc_153 : memref<3xf32, 2 : i32>
            memref.dealloc %alloc_154 : memref<6x6x16x8xi8, 2 : i32>
            memref.dealloc %alloc_155 : memref<6x3x8xf32, 2 : i32>
          }
          air.channel.put  @cL1ToL2[] (%alloc_148[] [] []) : (memref<6x8xf32, 2 : i32>)
          memref.dealloc %alloc_148 : memref<6x8xf32, 2 : i32>
        }
        air.channel.put  @cL2ToL3[] (%alloc_145[] [] []) : (memref<48xf32, 1 : i32>)
        memref.dealloc %alloc : memref<576xi8, 1 : i32>
        memref.dealloc %alloc_8 : memref<18xf32, 1 : i32>
        memref.dealloc %alloc_9 : memref<576x48xi8, 1 : i32>
        memref.dealloc %alloc_10 : memref<18x48xf32, 1 : i32>
        memref.dealloc %alloc_145 : memref<48xf32, 1 : i32>
      }
    }
    return
  }
}

