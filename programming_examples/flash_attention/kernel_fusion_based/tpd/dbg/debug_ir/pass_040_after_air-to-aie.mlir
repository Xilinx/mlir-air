#loop_annotation = #llvm.loop_annotation<mustProgress = true>
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
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
module {
  aie.device(npu2_4col) @attn_seg_0_0 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %lock_0_2 = aie.lock(%tile_0_2, 7) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 6) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_7 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_8 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 7) {init = 1 : i32}
    %lock_1_2_9 = aie.lock(%tile_1_2, 6) {init = 0 : i32}
    %lock_1_2_10 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_11 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_12 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_13 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_14 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_15 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 7) {init = 1 : i32}
    %lock_2_2_16 = aie.lock(%tile_2_2, 6) {init = 0 : i32}
    %lock_2_2_17 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_18 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_19 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_20 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_21 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_22 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 7) {init = 1 : i32}
    %lock_3_2_23 = aie.lock(%tile_3_2, 6) {init = 0 : i32}
    %lock_3_2_24 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_25 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_26 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_27 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_28 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_29 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 7) {init = 1 : i32}
    %lock_0_3_30 = aie.lock(%tile_0_3, 6) {init = 0 : i32}
    %lock_0_3_31 = aie.lock(%tile_0_3, 5) {init = 1 : i32}
    %lock_0_3_32 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_33 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_34 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_35 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_36 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 7) {init = 1 : i32}
    %lock_1_3_37 = aie.lock(%tile_1_3, 6) {init = 0 : i32}
    %lock_1_3_38 = aie.lock(%tile_1_3, 5) {init = 1 : i32}
    %lock_1_3_39 = aie.lock(%tile_1_3, 4) {init = 0 : i32}
    %lock_1_3_40 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_41 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_42 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_43 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 7) {init = 1 : i32}
    %lock_2_3_44 = aie.lock(%tile_2_3, 6) {init = 0 : i32}
    %lock_2_3_45 = aie.lock(%tile_2_3, 5) {init = 1 : i32}
    %lock_2_3_46 = aie.lock(%tile_2_3, 4) {init = 0 : i32}
    %lock_2_3_47 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_48 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_49 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_50 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 7) {init = 1 : i32}
    %lock_3_3_51 = aie.lock(%tile_3_3, 6) {init = 0 : i32}
    %lock_3_3_52 = aie.lock(%tile_3_3, 5) {init = 1 : i32}
    %lock_3_3_53 = aie.lock(%tile_3_3, 4) {init = 0 : i32}
    %lock_3_3_54 = aie.lock(%tile_3_3, 3) {init = 1 : i32}
    %lock_3_3_55 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_56 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_57 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 7) {init = 1 : i32}
    %lock_0_4_58 = aie.lock(%tile_0_4, 6) {init = 0 : i32}
    %lock_0_4_59 = aie.lock(%tile_0_4, 5) {init = 1 : i32}
    %lock_0_4_60 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_61 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_62 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_63 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_64 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 7) {init = 1 : i32}
    %lock_1_4_65 = aie.lock(%tile_1_4, 6) {init = 0 : i32}
    %lock_1_4_66 = aie.lock(%tile_1_4, 5) {init = 1 : i32}
    %lock_1_4_67 = aie.lock(%tile_1_4, 4) {init = 0 : i32}
    %lock_1_4_68 = aie.lock(%tile_1_4, 3) {init = 1 : i32}
    %lock_1_4_69 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_70 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_71 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 7) {init = 1 : i32}
    %lock_2_4_72 = aie.lock(%tile_2_4, 6) {init = 0 : i32}
    %lock_2_4_73 = aie.lock(%tile_2_4, 5) {init = 1 : i32}
    %lock_2_4_74 = aie.lock(%tile_2_4, 4) {init = 0 : i32}
    %lock_2_4_75 = aie.lock(%tile_2_4, 3) {init = 1 : i32}
    %lock_2_4_76 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_77 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_78 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 7) {init = 1 : i32}
    %lock_3_4_79 = aie.lock(%tile_3_4, 6) {init = 0 : i32}
    %lock_3_4_80 = aie.lock(%tile_3_4, 5) {init = 1 : i32}
    %lock_3_4_81 = aie.lock(%tile_3_4, 4) {init = 0 : i32}
    %lock_3_4_82 = aie.lock(%tile_3_4, 3) {init = 1 : i32}
    %lock_3_4_83 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_84 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_85 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 7) {init = 1 : i32}
    %lock_0_5_86 = aie.lock(%tile_0_5, 6) {init = 0 : i32}
    %lock_0_5_87 = aie.lock(%tile_0_5, 5) {init = 1 : i32}
    %lock_0_5_88 = aie.lock(%tile_0_5, 4) {init = 0 : i32}
    %lock_0_5_89 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_90 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_91 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_92 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 7) {init = 1 : i32}
    %lock_1_5_93 = aie.lock(%tile_1_5, 6) {init = 0 : i32}
    %lock_1_5_94 = aie.lock(%tile_1_5, 5) {init = 1 : i32}
    %lock_1_5_95 = aie.lock(%tile_1_5, 4) {init = 0 : i32}
    %lock_1_5_96 = aie.lock(%tile_1_5, 3) {init = 1 : i32}
    %lock_1_5_97 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_98 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_99 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 7) {init = 1 : i32}
    %lock_2_5_100 = aie.lock(%tile_2_5, 6) {init = 0 : i32}
    %lock_2_5_101 = aie.lock(%tile_2_5, 5) {init = 1 : i32}
    %lock_2_5_102 = aie.lock(%tile_2_5, 4) {init = 0 : i32}
    %lock_2_5_103 = aie.lock(%tile_2_5, 3) {init = 1 : i32}
    %lock_2_5_104 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_105 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_106 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 7) {init = 1 : i32}
    %lock_3_5_107 = aie.lock(%tile_3_5, 6) {init = 0 : i32}
    %lock_3_5_108 = aie.lock(%tile_3_5, 5) {init = 1 : i32}
    %lock_3_5_109 = aie.lock(%tile_3_5, 4) {init = 0 : i32}
    %lock_3_5_110 = aie.lock(%tile_3_5, 3) {init = 1 : i32}
    %lock_3_5_111 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_112 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_113 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf159 = aie.buffer(%tile_3_5) {sym_name = "buf159"} : memref<3xi32, 2 : i32> 
    %buf158 = aie.buffer(%tile_3_5) {sym_name = "buf158"} : memref<64x1xbf16, 2 : i32> 
    %buf157 = aie.buffer(%tile_3_5) {sym_name = "buf157"} : memref<64x1xbf16, 2 : i32> 
    %buf156 = aie.buffer(%tile_3_5) {sym_name = "buf156"} : memref<64x64xbf16, 2 : i32> 
    %buf155 = aie.buffer(%tile_3_5) {sym_name = "buf155"} : memref<64x64xbf16, 2 : i32> 
    %buf154 = aie.buffer(%tile_3_5) {sym_name = "buf154"} : memref<64x64xbf16, 2 : i32> 
    %buf153 = aie.buffer(%tile_3_5) {sym_name = "buf153"} : memref<64x64xbf16, 2 : i32> 
    %buf152 = aie.buffer(%tile_3_5) {sym_name = "buf152"} : memref<64x64xbf16, 2 : i32> 
    %buf151 = aie.buffer(%tile_3_5) {sym_name = "buf151"} : memref<64x1xbf16, 2 : i32> 
    %buf150 = aie.buffer(%tile_3_5) {sym_name = "buf150"} : memref<64x1xbf16, 2 : i32> 
    %buf149 = aie.buffer(%tile_2_5) {sym_name = "buf149"} : memref<3xi32, 2 : i32> 
    %buf148 = aie.buffer(%tile_2_5) {sym_name = "buf148"} : memref<64x1xbf16, 2 : i32> 
    %buf147 = aie.buffer(%tile_2_5) {sym_name = "buf147"} : memref<64x1xbf16, 2 : i32> 
    %buf146 = aie.buffer(%tile_2_5) {sym_name = "buf146"} : memref<64x64xbf16, 2 : i32> 
    %buf145 = aie.buffer(%tile_2_5) {sym_name = "buf145"} : memref<64x64xbf16, 2 : i32> 
    %buf144 = aie.buffer(%tile_2_5) {sym_name = "buf144"} : memref<64x64xbf16, 2 : i32> 
    %buf143 = aie.buffer(%tile_2_5) {sym_name = "buf143"} : memref<64x64xbf16, 2 : i32> 
    %buf142 = aie.buffer(%tile_2_5) {sym_name = "buf142"} : memref<64x64xbf16, 2 : i32> 
    %buf141 = aie.buffer(%tile_2_5) {sym_name = "buf141"} : memref<64x1xbf16, 2 : i32> 
    %buf140 = aie.buffer(%tile_2_5) {sym_name = "buf140"} : memref<64x1xbf16, 2 : i32> 
    %buf139 = aie.buffer(%tile_1_5) {sym_name = "buf139"} : memref<3xi32, 2 : i32> 
    %buf138 = aie.buffer(%tile_1_5) {sym_name = "buf138"} : memref<64x1xbf16, 2 : i32> 
    %buf137 = aie.buffer(%tile_1_5) {sym_name = "buf137"} : memref<64x1xbf16, 2 : i32> 
    %buf136 = aie.buffer(%tile_1_5) {sym_name = "buf136"} : memref<64x64xbf16, 2 : i32> 
    %buf135 = aie.buffer(%tile_1_5) {sym_name = "buf135"} : memref<64x64xbf16, 2 : i32> 
    %buf134 = aie.buffer(%tile_1_5) {sym_name = "buf134"} : memref<64x64xbf16, 2 : i32> 
    %buf133 = aie.buffer(%tile_1_5) {sym_name = "buf133"} : memref<64x64xbf16, 2 : i32> 
    %buf132 = aie.buffer(%tile_1_5) {sym_name = "buf132"} : memref<64x64xbf16, 2 : i32> 
    %buf131 = aie.buffer(%tile_1_5) {sym_name = "buf131"} : memref<64x1xbf16, 2 : i32> 
    %buf130 = aie.buffer(%tile_1_5) {sym_name = "buf130"} : memref<64x1xbf16, 2 : i32> 
    %buf129 = aie.buffer(%tile_0_5) {sym_name = "buf129"} : memref<3xi32, 2 : i32> 
    %buf128 = aie.buffer(%tile_0_5) {sym_name = "buf128"} : memref<64x1xbf16, 2 : i32> 
    %buf127 = aie.buffer(%tile_0_5) {sym_name = "buf127"} : memref<64x1xbf16, 2 : i32> 
    %buf126 = aie.buffer(%tile_0_5) {sym_name = "buf126"} : memref<64x64xbf16, 2 : i32> 
    %buf125 = aie.buffer(%tile_0_5) {sym_name = "buf125"} : memref<64x64xbf16, 2 : i32> 
    %buf124 = aie.buffer(%tile_0_5) {sym_name = "buf124"} : memref<64x64xbf16, 2 : i32> 
    %buf123 = aie.buffer(%tile_0_5) {sym_name = "buf123"} : memref<64x64xbf16, 2 : i32> 
    %buf122 = aie.buffer(%tile_0_5) {sym_name = "buf122"} : memref<64x64xbf16, 2 : i32> 
    %buf121 = aie.buffer(%tile_0_5) {sym_name = "buf121"} : memref<64x1xbf16, 2 : i32> 
    %buf120 = aie.buffer(%tile_0_5) {sym_name = "buf120"} : memref<64x1xbf16, 2 : i32> 
    %buf119 = aie.buffer(%tile_3_4) {sym_name = "buf119"} : memref<3xi32, 2 : i32> 
    %buf118 = aie.buffer(%tile_3_4) {sym_name = "buf118"} : memref<64x1xbf16, 2 : i32> 
    %buf117 = aie.buffer(%tile_3_4) {sym_name = "buf117"} : memref<64x1xbf16, 2 : i32> 
    %buf116 = aie.buffer(%tile_3_4) {sym_name = "buf116"} : memref<64x64xbf16, 2 : i32> 
    %buf115 = aie.buffer(%tile_3_4) {sym_name = "buf115"} : memref<64x64xbf16, 2 : i32> 
    %buf114 = aie.buffer(%tile_3_4) {sym_name = "buf114"} : memref<64x64xbf16, 2 : i32> 
    %buf113 = aie.buffer(%tile_3_4) {sym_name = "buf113"} : memref<64x64xbf16, 2 : i32> 
    %buf112 = aie.buffer(%tile_3_4) {sym_name = "buf112"} : memref<64x64xbf16, 2 : i32> 
    %buf111 = aie.buffer(%tile_3_4) {sym_name = "buf111"} : memref<64x1xbf16, 2 : i32> 
    %buf110 = aie.buffer(%tile_3_4) {sym_name = "buf110"} : memref<64x1xbf16, 2 : i32> 
    %buf109 = aie.buffer(%tile_2_4) {sym_name = "buf109"} : memref<3xi32, 2 : i32> 
    %buf108 = aie.buffer(%tile_2_4) {sym_name = "buf108"} : memref<64x1xbf16, 2 : i32> 
    %buf107 = aie.buffer(%tile_2_4) {sym_name = "buf107"} : memref<64x1xbf16, 2 : i32> 
    %buf106 = aie.buffer(%tile_2_4) {sym_name = "buf106"} : memref<64x64xbf16, 2 : i32> 
    %buf105 = aie.buffer(%tile_2_4) {sym_name = "buf105"} : memref<64x64xbf16, 2 : i32> 
    %buf104 = aie.buffer(%tile_2_4) {sym_name = "buf104"} : memref<64x64xbf16, 2 : i32> 
    %buf103 = aie.buffer(%tile_2_4) {sym_name = "buf103"} : memref<64x64xbf16, 2 : i32> 
    %buf102 = aie.buffer(%tile_2_4) {sym_name = "buf102"} : memref<64x64xbf16, 2 : i32> 
    %buf101 = aie.buffer(%tile_2_4) {sym_name = "buf101"} : memref<64x1xbf16, 2 : i32> 
    %buf100 = aie.buffer(%tile_2_4) {sym_name = "buf100"} : memref<64x1xbf16, 2 : i32> 
    %buf99 = aie.buffer(%tile_1_4) {sym_name = "buf99"} : memref<3xi32, 2 : i32> 
    %buf98 = aie.buffer(%tile_1_4) {sym_name = "buf98"} : memref<64x1xbf16, 2 : i32> 
    %buf97 = aie.buffer(%tile_1_4) {sym_name = "buf97"} : memref<64x1xbf16, 2 : i32> 
    %buf96 = aie.buffer(%tile_1_4) {sym_name = "buf96"} : memref<64x64xbf16, 2 : i32> 
    %buf95 = aie.buffer(%tile_1_4) {sym_name = "buf95"} : memref<64x64xbf16, 2 : i32> 
    %buf94 = aie.buffer(%tile_1_4) {sym_name = "buf94"} : memref<64x64xbf16, 2 : i32> 
    %buf93 = aie.buffer(%tile_1_4) {sym_name = "buf93"} : memref<64x64xbf16, 2 : i32> 
    %buf92 = aie.buffer(%tile_1_4) {sym_name = "buf92"} : memref<64x64xbf16, 2 : i32> 
    %buf91 = aie.buffer(%tile_1_4) {sym_name = "buf91"} : memref<64x1xbf16, 2 : i32> 
    %buf90 = aie.buffer(%tile_1_4) {sym_name = "buf90"} : memref<64x1xbf16, 2 : i32> 
    %buf89 = aie.buffer(%tile_0_4) {sym_name = "buf89"} : memref<3xi32, 2 : i32> 
    %buf88 = aie.buffer(%tile_0_4) {sym_name = "buf88"} : memref<64x1xbf16, 2 : i32> 
    %buf87 = aie.buffer(%tile_0_4) {sym_name = "buf87"} : memref<64x1xbf16, 2 : i32> 
    %buf86 = aie.buffer(%tile_0_4) {sym_name = "buf86"} : memref<64x64xbf16, 2 : i32> 
    %buf85 = aie.buffer(%tile_0_4) {sym_name = "buf85"} : memref<64x64xbf16, 2 : i32> 
    %buf84 = aie.buffer(%tile_0_4) {sym_name = "buf84"} : memref<64x64xbf16, 2 : i32> 
    %buf83 = aie.buffer(%tile_0_4) {sym_name = "buf83"} : memref<64x64xbf16, 2 : i32> 
    %buf82 = aie.buffer(%tile_0_4) {sym_name = "buf82"} : memref<64x64xbf16, 2 : i32> 
    %buf81 = aie.buffer(%tile_0_4) {sym_name = "buf81"} : memref<64x1xbf16, 2 : i32> 
    %buf80 = aie.buffer(%tile_0_4) {sym_name = "buf80"} : memref<64x1xbf16, 2 : i32> 
    %buf79 = aie.buffer(%tile_3_3) {sym_name = "buf79"} : memref<3xi32, 2 : i32> 
    %buf78 = aie.buffer(%tile_3_3) {sym_name = "buf78"} : memref<64x1xbf16, 2 : i32> 
    %buf77 = aie.buffer(%tile_3_3) {sym_name = "buf77"} : memref<64x1xbf16, 2 : i32> 
    %buf76 = aie.buffer(%tile_3_3) {sym_name = "buf76"} : memref<64x64xbf16, 2 : i32> 
    %buf75 = aie.buffer(%tile_3_3) {sym_name = "buf75"} : memref<64x64xbf16, 2 : i32> 
    %buf74 = aie.buffer(%tile_3_3) {sym_name = "buf74"} : memref<64x64xbf16, 2 : i32> 
    %buf73 = aie.buffer(%tile_3_3) {sym_name = "buf73"} : memref<64x64xbf16, 2 : i32> 
    %buf72 = aie.buffer(%tile_3_3) {sym_name = "buf72"} : memref<64x64xbf16, 2 : i32> 
    %buf71 = aie.buffer(%tile_3_3) {sym_name = "buf71"} : memref<64x1xbf16, 2 : i32> 
    %buf70 = aie.buffer(%tile_3_3) {sym_name = "buf70"} : memref<64x1xbf16, 2 : i32> 
    %buf69 = aie.buffer(%tile_2_3) {sym_name = "buf69"} : memref<3xi32, 2 : i32> 
    %buf68 = aie.buffer(%tile_2_3) {sym_name = "buf68"} : memref<64x1xbf16, 2 : i32> 
    %buf67 = aie.buffer(%tile_2_3) {sym_name = "buf67"} : memref<64x1xbf16, 2 : i32> 
    %buf66 = aie.buffer(%tile_2_3) {sym_name = "buf66"} : memref<64x64xbf16, 2 : i32> 
    %buf65 = aie.buffer(%tile_2_3) {sym_name = "buf65"} : memref<64x64xbf16, 2 : i32> 
    %buf64 = aie.buffer(%tile_2_3) {sym_name = "buf64"} : memref<64x64xbf16, 2 : i32> 
    %buf63 = aie.buffer(%tile_2_3) {sym_name = "buf63"} : memref<64x64xbf16, 2 : i32> 
    %buf62 = aie.buffer(%tile_2_3) {sym_name = "buf62"} : memref<64x64xbf16, 2 : i32> 
    %buf61 = aie.buffer(%tile_2_3) {sym_name = "buf61"} : memref<64x1xbf16, 2 : i32> 
    %buf60 = aie.buffer(%tile_2_3) {sym_name = "buf60"} : memref<64x1xbf16, 2 : i32> 
    %buf59 = aie.buffer(%tile_1_3) {sym_name = "buf59"} : memref<3xi32, 2 : i32> 
    %buf58 = aie.buffer(%tile_1_3) {sym_name = "buf58"} : memref<64x1xbf16, 2 : i32> 
    %buf57 = aie.buffer(%tile_1_3) {sym_name = "buf57"} : memref<64x1xbf16, 2 : i32> 
    %buf56 = aie.buffer(%tile_1_3) {sym_name = "buf56"} : memref<64x64xbf16, 2 : i32> 
    %buf55 = aie.buffer(%tile_1_3) {sym_name = "buf55"} : memref<64x64xbf16, 2 : i32> 
    %buf54 = aie.buffer(%tile_1_3) {sym_name = "buf54"} : memref<64x64xbf16, 2 : i32> 
    %buf53 = aie.buffer(%tile_1_3) {sym_name = "buf53"} : memref<64x64xbf16, 2 : i32> 
    %buf52 = aie.buffer(%tile_1_3) {sym_name = "buf52"} : memref<64x64xbf16, 2 : i32> 
    %buf51 = aie.buffer(%tile_1_3) {sym_name = "buf51"} : memref<64x1xbf16, 2 : i32> 
    %buf50 = aie.buffer(%tile_1_3) {sym_name = "buf50"} : memref<64x1xbf16, 2 : i32> 
    %buf49 = aie.buffer(%tile_0_3) {sym_name = "buf49"} : memref<3xi32, 2 : i32> 
    %buf48 = aie.buffer(%tile_0_3) {sym_name = "buf48"} : memref<64x1xbf16, 2 : i32> 
    %buf47 = aie.buffer(%tile_0_3) {sym_name = "buf47"} : memref<64x1xbf16, 2 : i32> 
    %buf46 = aie.buffer(%tile_0_3) {sym_name = "buf46"} : memref<64x64xbf16, 2 : i32> 
    %buf45 = aie.buffer(%tile_0_3) {sym_name = "buf45"} : memref<64x64xbf16, 2 : i32> 
    %buf44 = aie.buffer(%tile_0_3) {sym_name = "buf44"} : memref<64x64xbf16, 2 : i32> 
    %buf43 = aie.buffer(%tile_0_3) {sym_name = "buf43"} : memref<64x64xbf16, 2 : i32> 
    %buf42 = aie.buffer(%tile_0_3) {sym_name = "buf42"} : memref<64x64xbf16, 2 : i32> 
    %buf41 = aie.buffer(%tile_0_3) {sym_name = "buf41"} : memref<64x1xbf16, 2 : i32> 
    %buf40 = aie.buffer(%tile_0_3) {sym_name = "buf40"} : memref<64x1xbf16, 2 : i32> 
    %buf39 = aie.buffer(%tile_3_2) {sym_name = "buf39"} : memref<3xi32, 2 : i32> 
    %buf38 = aie.buffer(%tile_3_2) {sym_name = "buf38"} : memref<64x1xbf16, 2 : i32> 
    %buf37 = aie.buffer(%tile_3_2) {sym_name = "buf37"} : memref<64x1xbf16, 2 : i32> 
    %buf36 = aie.buffer(%tile_3_2) {sym_name = "buf36"} : memref<64x64xbf16, 2 : i32> 
    %buf35 = aie.buffer(%tile_3_2) {sym_name = "buf35"} : memref<64x64xbf16, 2 : i32> 
    %buf34 = aie.buffer(%tile_3_2) {sym_name = "buf34"} : memref<64x64xbf16, 2 : i32> 
    %buf33 = aie.buffer(%tile_3_2) {sym_name = "buf33"} : memref<64x64xbf16, 2 : i32> 
    %buf32 = aie.buffer(%tile_3_2) {sym_name = "buf32"} : memref<64x64xbf16, 2 : i32> 
    %buf31 = aie.buffer(%tile_3_2) {sym_name = "buf31"} : memref<64x1xbf16, 2 : i32> 
    %buf30 = aie.buffer(%tile_3_2) {sym_name = "buf30"} : memref<64x1xbf16, 2 : i32> 
    %buf29 = aie.buffer(%tile_2_2) {sym_name = "buf29"} : memref<3xi32, 2 : i32> 
    %buf28 = aie.buffer(%tile_2_2) {sym_name = "buf28"} : memref<64x1xbf16, 2 : i32> 
    %buf27 = aie.buffer(%tile_2_2) {sym_name = "buf27"} : memref<64x1xbf16, 2 : i32> 
    %buf26 = aie.buffer(%tile_2_2) {sym_name = "buf26"} : memref<64x64xbf16, 2 : i32> 
    %buf25 = aie.buffer(%tile_2_2) {sym_name = "buf25"} : memref<64x64xbf16, 2 : i32> 
    %buf24 = aie.buffer(%tile_2_2) {sym_name = "buf24"} : memref<64x64xbf16, 2 : i32> 
    %buf23 = aie.buffer(%tile_2_2) {sym_name = "buf23"} : memref<64x64xbf16, 2 : i32> 
    %buf22 = aie.buffer(%tile_2_2) {sym_name = "buf22"} : memref<64x64xbf16, 2 : i32> 
    %buf21 = aie.buffer(%tile_2_2) {sym_name = "buf21"} : memref<64x1xbf16, 2 : i32> 
    %buf20 = aie.buffer(%tile_2_2) {sym_name = "buf20"} : memref<64x1xbf16, 2 : i32> 
    %buf19 = aie.buffer(%tile_1_2) {sym_name = "buf19"} : memref<3xi32, 2 : i32> 
    %buf18 = aie.buffer(%tile_1_2) {sym_name = "buf18"} : memref<64x1xbf16, 2 : i32> 
    %buf17 = aie.buffer(%tile_1_2) {sym_name = "buf17"} : memref<64x1xbf16, 2 : i32> 
    %buf16 = aie.buffer(%tile_1_2) {sym_name = "buf16"} : memref<64x64xbf16, 2 : i32> 
    %buf15 = aie.buffer(%tile_1_2) {sym_name = "buf15"} : memref<64x64xbf16, 2 : i32> 
    %buf14 = aie.buffer(%tile_1_2) {sym_name = "buf14"} : memref<64x64xbf16, 2 : i32> 
    %buf13 = aie.buffer(%tile_1_2) {sym_name = "buf13"} : memref<64x64xbf16, 2 : i32> 
    %buf12 = aie.buffer(%tile_1_2) {sym_name = "buf12"} : memref<64x64xbf16, 2 : i32> 
    %buf11 = aie.buffer(%tile_1_2) {sym_name = "buf11"} : memref<64x1xbf16, 2 : i32> 
    %buf10 = aie.buffer(%tile_1_2) {sym_name = "buf10"} : memref<64x1xbf16, 2 : i32> 
    %buf9 = aie.buffer(%tile_0_2) {sym_name = "buf9"} : memref<3xi32, 2 : i32> 
    %buf8 = aie.buffer(%tile_0_2) {sym_name = "buf8"} : memref<64x1xbf16, 2 : i32> 
    %buf7 = aie.buffer(%tile_0_2) {sym_name = "buf7"} : memref<64x1xbf16, 2 : i32> 
    %buf6 = aie.buffer(%tile_0_2) {sym_name = "buf6"} : memref<64x64xbf16, 2 : i32> 
    %buf5 = aie.buffer(%tile_0_2) {sym_name = "buf5"} : memref<64x64xbf16, 2 : i32> 
    %buf4 = aie.buffer(%tile_0_2) {sym_name = "buf4"} : memref<64x64xbf16, 2 : i32> 
    %buf3 = aie.buffer(%tile_0_2) {sym_name = "buf3"} : memref<64x64xbf16, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<64x64xbf16, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<64x1xbf16, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<512x128xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<512x128xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<512x512xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<512x512xbf16>
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    %mem_3_5 = aie.mem(%tile_3_5) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_113, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf156 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_112, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_5_110, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_111, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_5_110, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_111, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_5_110, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_111, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_5_110, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_111, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_5_108, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf153 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_5_109, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_107, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c1_117 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_5_112, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf156) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf158) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf157) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf159[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf159[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf159[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf159[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_5_110, Release, %c1_i32)
      aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_5_110, Release, %c1_i32)
      aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_5_110, Release, %c1_i32)
      aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf155, %buf154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_110, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf152 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_107, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf152 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf154, %buf155, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5, Release, %c1_i32)
        aie.use_lock(%lock_3_5_109, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf159[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c3_i32 : i32
        func.call @apply_causal_mask(%buf152, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf152 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf157, %buf151, %buf150) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf150, %buf156) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf152 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf153, %buf156) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf158, %buf150, %buf151) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf151, %buf158) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_108, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf158, %buf156) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf159[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf159[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf159[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf159[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf159[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_5_113, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_106, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf146 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_105, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_5_103, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_104, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_5_103, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_104, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_5_103, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_104, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_5_103, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_104, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_5_101, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf143 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_5_102, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_100, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_117 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_118 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_5_105, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf146) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf148) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf147) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf149[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf149[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf149[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf149[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_5_104, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_5_103, Release, %c1_i32)
      aie.use_lock(%lock_2_5_104, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_5_103, Release, %c1_i32)
      aie.use_lock(%lock_2_5_104, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf145, %buf144) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_103, Release, %c1_i32)
      aie.use_lock(%lock_2_5_104, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_5_103, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf142 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_100, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf142 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf144, %buf145, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5, Release, %c1_i32)
        aie.use_lock(%lock_2_5_102, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf149[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c2_i32 : i32
        func.call @apply_causal_mask(%buf142, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf142 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf147, %buf141, %buf140) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf140, %buf146) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf142 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf143, %buf146) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf148, %buf140, %buf141) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf141, %buf148) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_101, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf148, %buf146) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf149[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf149[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf149[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf149[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf149[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_5_106, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_99, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf136 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_98, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_5_96, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_97, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_5_96, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_97, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_5_96, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_97, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_5_96, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_97, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_5_94, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf133 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_5_95, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_93, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_5_98, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf136) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf138) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf137) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf139[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf139[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf139[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf139[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_5_97, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_5_96, Release, %c1_i32)
      aie.use_lock(%lock_1_5_97, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf135, %buf134) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_96, Release, %c1_i32)
      aie.use_lock(%lock_1_5_97, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_5_96, Release, %c1_i32)
      aie.use_lock(%lock_1_5_97, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_5_96, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf132 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_93, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf132 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf134, %buf135, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5, Release, %c1_i32)
        aie.use_lock(%lock_1_5_95, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf139[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c1_i32 : i32
        func.call @apply_causal_mask(%buf132, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf132 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf137, %buf131, %buf130) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf130, %buf136) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf132 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf133, %buf136) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf138, %buf130, %buf131) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf131, %buf138) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_94, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf138, %buf136) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf139[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf139[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf139[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf139[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf139[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_5_99, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_92, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf126 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_91, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_5_89, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_90, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_5_89, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_90, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_5_89, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_90, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_5_89, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_90, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_5_87, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf123 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_5_88, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_86, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c1_117 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_5_91, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf126) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf128) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf127) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf129[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf129[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf129[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf129[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_5_90, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf125, %buf124) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_89, Release, %c1_i32)
      aie.use_lock(%lock_0_5_90, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_5_89, Release, %c1_i32)
      aie.use_lock(%lock_0_5_90, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_5_89, Release, %c1_i32)
      aie.use_lock(%lock_0_5_90, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_5_89, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf122 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_86, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf122 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf124, %buf125, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5, Release, %c1_i32)
        aie.use_lock(%lock_0_5_88, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf129[%c0_118] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf122, %15, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf122 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf127, %buf121, %buf120) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf120, %buf126) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf122 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf123, %buf126) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf128, %buf120, %buf121) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf121, %buf128) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_87, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf128, %buf126) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf129[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf129[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf129[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf129[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf129[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_5_92, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_85, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf116 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_84, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_4_82, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_83, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_4_82, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_83, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_4_82, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_83, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_4_82, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_83, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_4_80, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf113 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_4_81, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_79, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_117 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_118 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_4_84, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf116) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf118) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf117) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf119[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf119[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf119[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf119[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_4_83, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_4_82, Release, %c1_i32)
      aie.use_lock(%lock_3_4_83, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_4_82, Release, %c1_i32)
      aie.use_lock(%lock_3_4_83, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_4_82, Release, %c1_i32)
      aie.use_lock(%lock_3_4_83, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf115, %buf114) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_82, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf112 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_79, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf112 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf114, %buf115, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4, Release, %c1_i32)
        aie.use_lock(%lock_3_4_81, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf119[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c3_i32 : i32
        func.call @apply_causal_mask(%buf112, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf112 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf117, %buf111, %buf110) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf110, %buf116) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf112 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf113, %buf116) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf118, %buf110, %buf111) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf111, %buf118) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_80, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf118, %buf116) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf119[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf119[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf119[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf119[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf119[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_4_85, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_78, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf106 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_77, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_4_75, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_76, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_4_75, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_76, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_4_75, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_76, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_4_75, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_76, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_4_73, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf103 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_4_74, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_72, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_117 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_118 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_4_77, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf106) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf108) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf107) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf109[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf109[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf109[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf109[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_4_75, Release, %c1_i32)
      aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_4_75, Release, %c1_i32)
      aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf105, %buf104) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_75, Release, %c1_i32)
      aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_4_75, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf102 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_72, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf102 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf104, %buf105, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4, Release, %c1_i32)
        aie.use_lock(%lock_2_4_74, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf109[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c2_i32 : i32
        func.call @apply_causal_mask(%buf102, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf102 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf107, %buf101, %buf100) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf100, %buf106) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf102 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf103, %buf106) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf108, %buf100, %buf101) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf101, %buf108) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_73, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf108, %buf106) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf109[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf109[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf109[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf109[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf109[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_4_78, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_71, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf96 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_70, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_4_68, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_69, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_4_68, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_69, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_4_68, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_69, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_4_68, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_69, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_4_66, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf93 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_4_67, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_65, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_4_70, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf96) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf98) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf97) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf99[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf99[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf99[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf99[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_4_69, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_4_68, Release, %c1_i32)
      aie.use_lock(%lock_1_4_69, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf95, %buf94) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_68, Release, %c1_i32)
      aie.use_lock(%lock_1_4_69, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_4_68, Release, %c1_i32)
      aie.use_lock(%lock_1_4_69, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_4_68, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf92 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_65, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf92 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf94, %buf95, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4, Release, %c1_i32)
        aie.use_lock(%lock_1_4_67, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf99[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c1_i32 : i32
        func.call @apply_causal_mask(%buf92, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf92 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf97, %buf91, %buf90) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf90, %buf96) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf92 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf93, %buf96) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf98, %buf90, %buf91) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf91, %buf98) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_66, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf98, %buf96) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf99[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf99[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf99[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf99[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf99[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_4_71, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_64, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf86 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_63, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_4_61, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_62, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_4_61, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_62, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_4_61, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_62, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_4_61, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_62, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_4_59, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf83 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_4_60, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_58, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_117 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c2 = arith.constant 2 : index
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_4_63, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf86) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf88) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf87) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf89[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf89[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf89[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf89[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_4_62, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf85, %buf84) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_61, Release, %c1_i32)
      aie.use_lock(%lock_0_4_62, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_61, Release, %c1_i32)
      aie.use_lock(%lock_0_4_62, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_61, Release, %c1_i32)
      aie.use_lock(%lock_0_4_62, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_61, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf82 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_58, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf82 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf84, %buf85, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4, Release, %c1_i32)
        aie.use_lock(%lock_0_4_60, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf89[%c0_118] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf82, %15, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf82 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf87, %buf81, %buf80) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf80, %buf86) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf82 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf83, %buf86) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf88, %buf80, %buf81) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf81, %buf88) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_59, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf88, %buf86) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf89[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf89[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf89[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf89[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf89[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_4_64, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_57, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf76 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_56, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_3_54, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_55, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_3_54, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_55, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_3_54, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_55, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_3_54, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_55, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_3_52, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf73 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_3_53, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_51, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_3_56, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf76) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf78) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf77) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf79[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf79[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf79[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf79[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_3_55, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_3_54, Release, %c1_i32)
      aie.use_lock(%lock_3_3_55, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_3_54, Release, %c1_i32)
      aie.use_lock(%lock_3_3_55, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_3_54, Release, %c1_i32)
      aie.use_lock(%lock_3_3_55, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf75, %buf74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_54, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf72 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf72 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf74, %buf75, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3, Release, %c1_i32)
        aie.use_lock(%lock_3_3_53, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf79[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c3_i32 : i32
        func.call @apply_causal_mask(%buf72, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf72 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf77, %buf71, %buf70) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf70, %buf76) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf72 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf73, %buf76) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf78, %buf70, %buf71) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf71, %buf78) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_52, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf78, %buf76) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf79[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf79[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf79[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf79[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf79[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_3_57, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_50, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf66 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_49, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_3_45, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf63 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_3_46, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_44, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_3_49, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf66) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf68) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf67) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf69[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf69[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf69[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf69[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_3_47, Release, %c1_i32)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_3_47, Release, %c1_i32)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf65, %buf64) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, %c1_i32)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_3_47, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf62 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf62 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf64, %buf65, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3, Release, %c1_i32)
        aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf69[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c2_i32 : i32
        func.call @apply_causal_mask(%buf62, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf62 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf67, %buf61, %buf60) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf60, %buf66) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf62 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf63, %buf66) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf68, %buf60, %buf61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf61, %buf68) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_45, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf68, %buf66) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf69[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf69[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf69[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf69[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf69[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_3_50, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf56 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_42, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_41, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_41, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_41, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_41, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_3_38, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf53 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_3_39, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_37, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_3_42, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf56) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf58) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf57) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf59[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf59[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf59[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf59[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_3_40, Release, %c1_i32)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf55, %buf54) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_40, Release, %c1_i32)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_3_40, Release, %c1_i32)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_3_40, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf52 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf52 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf54, %buf55, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3, Release, %c1_i32)
        aie.use_lock(%lock_1_3_39, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf59[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c1_i32 : i32
        func.call @apply_causal_mask(%buf52, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf52 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf57, %buf51, %buf50) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf50, %buf56) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf52 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf53, %buf56) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf58, %buf50, %buf51) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf51, %buf58) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_38, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf58, %buf56) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf59[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf59[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf59[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf59[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf59[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_3_43, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_36, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf46 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_35, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_3_31, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf43 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_3_32, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_30, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1_117 = arith.constant 1 : index
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_3_35, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf46) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf48) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf47) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf49[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf49[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf49[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf49[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf45, %buf44) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_33, Release, %c1_i32)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_33, Release, %c1_i32)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_33, Release, %c1_i32)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_33, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf42 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_30, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf42 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf44, %buf45, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3, Release, %c1_i32)
        aie.use_lock(%lock_0_3_32, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf49[%c0_118] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf42, %15, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf42 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf47, %buf41, %buf40) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf40, %buf46) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf42 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf43, %buf46) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf48, %buf40, %buf41) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf41, %buf48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_31, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf48, %buf46) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf49[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf49[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf49[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf49[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf49[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_3_36, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf36 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_28, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_2_24, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf33 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_2_25, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_23, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c1_117 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_2_28, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf36) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf38) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf37) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf39[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf39[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf39[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf39[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_2_26, Release, %c1_i32)
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_2_26, Release, %c1_i32)
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_2_26, Release, %c1_i32)
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf35, %buf34) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_26, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf32 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_23, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf32 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf34, %buf35, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2, Release, %c1_i32)
        aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf39[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c3_i32 : i32
        func.call @apply_causal_mask(%buf32, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf32 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf37, %buf31, %buf30) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf30, %buf36) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf32 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf33, %buf36) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf38, %buf30, %buf31) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf31, %buf38) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_24, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf38, %buf36) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf39[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf39[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf39[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf39[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf39[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_2_29, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_22, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf26 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_21, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_2_17, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf23 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_2_18, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_16, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_117 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_118 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_2_21, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf26) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf28) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf27) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf29[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf29[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf29[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf29[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_2_19, Release, %c1_i32)
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_2_19, Release, %c1_i32)
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf25, %buf24) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_19, Release, %c1_i32)
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_2_19, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf22 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_16, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf22 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf24, %buf25, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2, Release, %c1_i32)
        aie.use_lock(%lock_2_2_18, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf29[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c2_i32 : i32
        func.call @apply_causal_mask(%buf22, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf22 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf27, %buf21, %buf20) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf20, %buf26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf22 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf23, %buf26) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf28, %buf20, %buf21) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf21, %buf28) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_17, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf28, %buf26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf29[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf29[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf29[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf29[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf29[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_2_22, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_15, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf16 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_14, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_2_12, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_2_12, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_2_12, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_2_12, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_2_10, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf13 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_2_11, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_9, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_14, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf16) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf18) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf17) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf19[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf19[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf19[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf19[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_2_12, Release, %c1_i32)
      aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf15, %buf14) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_12, Release, %c1_i32)
      aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_2_12, Release, %c1_i32)
      aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_2_12, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf12 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf12 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf14, %buf15, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2, Release, %c1_i32)
        aie.use_lock(%lock_1_2_11, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf19[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c1_i32 : i32
        func.call @apply_causal_mask(%buf12, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf12 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf17, %buf11, %buf10) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf10, %buf16) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf12 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf13, %buf16) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf18, %buf10, %buf11) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf11, %buf18) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_10, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf18, %buf16) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf19[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf19[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf19[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf19[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf19[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_2_15, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_8, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf6 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_7, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf3 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c1_117 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf6) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf8) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf7) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf9[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf9[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf9[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf9[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf5, %buf4) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_5, Release, %c1_i32)
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_5, Release, %c1_i32)
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_5, Release, %c1_i32)
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_5, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf2 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf2 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf4, %buf5, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2, Release, %c1_i32)
        aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf9[%c0_118] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf2, %15, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf2 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf7, %buf1, %buf0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf0, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf2 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf3, %buf6) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf8, %buf0, %buf1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf1, %buf8) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf8, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf9[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf9[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf9[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf9[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf9[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_2_8, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    air.channel @channel_22 [1, 1]
    air.channel @Q2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @channel_20 [1, 1]
    air.channel @K2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @channel_18 [1, 1]
    air.channel @V2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @channel_2 [1, 1]
    air.channel @channel_3 [1, 1]
    air.channel @channel_4 [1, 1]
    air.channel @channel_5 [1, 1]
    air.channel @channel_6 [1, 1]
    air.channel @channel_7 [1, 1]
    air.channel @channel_8 [1, 1]
    air.channel @channel_9 [1, 1]
    air.channel @channel_10 [1, 1]
    air.channel @channel_11 [1, 1]
    air.channel @channel_12 [1, 1]
    air.channel @channel_13 [1, 1]
    air.channel @channel_14 [1, 1]
    air.channel @channel_15 [1, 1]
    air.channel @channel_16 [1, 1]
    air.channel @channel_17 [1, 1]
    air.channel @channel_0 [1, 1]
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_114 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_115 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_116 = aie.logical_tile<MemTile>(?, ?)
    %0 = aie.lock(%logical_mem_114, 1) {init = 4 : i32}
    %1 = aie.lock(%logical_mem_114, 0) {init = 0 : i32}
    %2 = aie.lock(%logical_mem_115, 1) {init = 4 : i32}
    %3 = aie.lock(%logical_mem_115, 0) {init = 0 : i32}
    %4 = aie.lock(%logical_mem, 1) {init = 2 : i32}
    %5 = aie.lock(%logical_mem, 0) {init = 0 : i32}
    %buf162 = aie.buffer(%logical_mem) {sym_name = "buf162"} : memref<64x64xbf16, 1 : i32> 
    %buf161 = aie.buffer(%logical_mem_114) {sym_name = "buf161"} : memref<256x64xbf16, 1 : i32> 
    %buf160 = aie.buffer(%logical_mem_115) {sym_name = "buf160"} : memref<64x64xbf16, 1 : i32> 
    func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @apply_causal_mask(memref<64x64xbf16, 2 : i32>, i32, i32) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    aie.flow(%logical_shim_noc, DMA : 0, %logical_mem, DMA : 0)
    aie.flow(%logical_shim_noc_1, DMA : 0, %logical_mem_115, DMA : 0)
    aie.flow(%logical_shim_noc, DMA : 1, %logical_mem, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%logical_mem, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%logical_mem, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%logical_mem, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%logical_mem, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%logical_mem, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%logical_mem, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%logical_mem, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%logical_mem, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%logical_mem, DMA : 2, %tile_1_4, DMA : 0)
    aie.flow(%logical_mem, DMA : 2, %tile_2_4, DMA : 0)
    aie.flow(%logical_mem, DMA : 2, %tile_3_4, DMA : 0)
    aie.flow(%logical_mem, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%logical_mem, DMA : 3, %tile_1_5, DMA : 0)
    aie.flow(%logical_mem, DMA : 3, %tile_2_5, DMA : 0)
    aie.flow(%logical_mem, DMA : 3, %tile_3_5, DMA : 0)
    aie.flow(%logical_mem, DMA : 4, %tile_0_2, DMA : 1)
    aie.flow(%logical_mem, DMA : 4, %tile_1_2, DMA : 1)
    aie.flow(%logical_mem, DMA : 4, %tile_2_2, DMA : 1)
    aie.flow(%logical_mem, DMA : 4, %tile_3_2, DMA : 1)
    aie.flow(%logical_mem, DMA : 5, %tile_0_3, DMA : 1)
    aie.flow(%logical_mem, DMA : 5, %tile_1_3, DMA : 1)
    aie.flow(%logical_mem, DMA : 5, %tile_2_3, DMA : 1)
    aie.flow(%logical_mem, DMA : 5, %tile_3_3, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_0_4, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_1_4, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_2_4, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_3_4, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_0_5, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_1_5, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_2_5, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_3_5, DMA : 1)
    aie.flow(%logical_mem_115, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 2, %tile_1_4, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 2, %tile_2_4, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 2, %tile_3_4, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 3, %tile_1_5, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 3, %tile_2_5, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 3, %tile_3_5, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %logical_mem_114, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %logical_mem_114, DMA : 2)
    aie.flow(%tile_3_2, DMA : 0, %logical_mem_114, DMA : 3)
    aie.flow(%logical_mem_114, DMA : 0, %logical_shim_noc_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %logical_mem_114, DMA : 4)
    aie.flow(%tile_1_3, DMA : 0, %logical_mem_114, DMA : 5)
    aie.flow(%tile_2_3, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_3_3, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_1_4, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_2_4, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_3_4, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_1_5, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_2_5, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_3_5, DMA : 0, %logical_mem_114, DMA : 0)
    %6 = aie.memtile_dma(%logical_mem) {
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb15
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %11 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %12 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %13 = aie.dma_start(MM2S, 4, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %14 = aie.dma_start(MM2S, 5, ^bb12, ^bb13)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb11
      %15 = aie.dma_start(S2MM, 0, ^bb14, ^bb15)
    ^bb14:  // 2 preds: ^bb13, ^bb14
      aie.use_lock(%4, AcquireGreaterEqual, %c2_i32)
      aie.dma_bd(%buf162 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%5, Release, %c2_i32)
      aie.next_bd ^bb14
    ^bb15:  // pred: ^bb13
      %16 = aie.dma_start(S2MM, 1, ^bb16, ^bb2)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%4, AcquireGreaterEqual, %c2_i32)
      aie.dma_bd(%buf162 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%5, Release, %c2_i32)
      aie.next_bd ^bb16
    }
    %7 = aie.memtile_dma(%logical_mem_115) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf160 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%2, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf160 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%2, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %11 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf160 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%2, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %12 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf160 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%2, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %13 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%2, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%buf160 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%3, Release, %c4_i32)
      aie.next_bd ^bb10
    }
    %8 = aie.memtile_dma(%logical_mem_114) {
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%1, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 0 len = 16384) {task_id = 0 : i32}
      aie.use_lock(%0, Release, %c4_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb23
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb15)
    ^bb4:  // 2 preds: ^bb3, ^bb14
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb15:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 1, ^bb16, ^bb17)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb16
    ^bb17:  // pred: ^bb15
      %12 = aie.dma_start(S2MM, 2, ^bb18, ^bb19)
    ^bb18:  // 2 preds: ^bb17, ^bb18
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb18
    ^bb19:  // pred: ^bb17
      %13 = aie.dma_start(S2MM, 3, ^bb20, ^bb21)
    ^bb20:  // 2 preds: ^bb19, ^bb20
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb20
    ^bb21:  // pred: ^bb19
      %14 = aie.dma_start(S2MM, 4, ^bb22, ^bb23)
    ^bb22:  // 2 preds: ^bb21, ^bb22
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb22
    ^bb23:  // pred: ^bb21
      %15 = aie.dma_start(S2MM, 5, ^bb24, ^bb2)
    ^bb24:  // 2 preds: ^bb23, ^bb24
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb24
    }
    aie.shim_dma_allocation @air_GpOut_0_0(%logical_shim_noc_0, S2MM, 0)
    aie.shim_dma_allocation @air_KIn_0_0(%logical_shim_noc, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_0_0(%logical_shim_noc_1, MM2S, 0)
    aie.shim_dma_allocation @air_QIn_0_0(%logical_shim_noc, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 0 : i64, segment_unroll_y = 0 : i64}
  aie.device(npu2_4col) @attn_seg_1_0 {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %lock_0_2 = aie.lock(%tile_0_2, 7) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 6) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_7 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_8 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 7) {init = 1 : i32}
    %lock_1_2_9 = aie.lock(%tile_1_2, 6) {init = 0 : i32}
    %lock_1_2_10 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_11 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_12 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_13 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_14 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_15 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 7) {init = 1 : i32}
    %lock_2_2_16 = aie.lock(%tile_2_2, 6) {init = 0 : i32}
    %lock_2_2_17 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_18 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_19 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_20 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_21 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_22 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 7) {init = 1 : i32}
    %lock_3_2_23 = aie.lock(%tile_3_2, 6) {init = 0 : i32}
    %lock_3_2_24 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_25 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_26 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_27 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_28 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_29 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 7) {init = 1 : i32}
    %lock_0_3_30 = aie.lock(%tile_0_3, 6) {init = 0 : i32}
    %lock_0_3_31 = aie.lock(%tile_0_3, 5) {init = 1 : i32}
    %lock_0_3_32 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_33 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_34 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_35 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_36 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 7) {init = 1 : i32}
    %lock_1_3_37 = aie.lock(%tile_1_3, 6) {init = 0 : i32}
    %lock_1_3_38 = aie.lock(%tile_1_3, 5) {init = 1 : i32}
    %lock_1_3_39 = aie.lock(%tile_1_3, 4) {init = 0 : i32}
    %lock_1_3_40 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_41 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_42 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_43 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 7) {init = 1 : i32}
    %lock_2_3_44 = aie.lock(%tile_2_3, 6) {init = 0 : i32}
    %lock_2_3_45 = aie.lock(%tile_2_3, 5) {init = 1 : i32}
    %lock_2_3_46 = aie.lock(%tile_2_3, 4) {init = 0 : i32}
    %lock_2_3_47 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_48 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_49 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_50 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 7) {init = 1 : i32}
    %lock_3_3_51 = aie.lock(%tile_3_3, 6) {init = 0 : i32}
    %lock_3_3_52 = aie.lock(%tile_3_3, 5) {init = 1 : i32}
    %lock_3_3_53 = aie.lock(%tile_3_3, 4) {init = 0 : i32}
    %lock_3_3_54 = aie.lock(%tile_3_3, 3) {init = 1 : i32}
    %lock_3_3_55 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_56 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_57 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 7) {init = 1 : i32}
    %lock_0_4_58 = aie.lock(%tile_0_4, 6) {init = 0 : i32}
    %lock_0_4_59 = aie.lock(%tile_0_4, 5) {init = 1 : i32}
    %lock_0_4_60 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_61 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_62 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_63 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_64 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 7) {init = 1 : i32}
    %lock_1_4_65 = aie.lock(%tile_1_4, 6) {init = 0 : i32}
    %lock_1_4_66 = aie.lock(%tile_1_4, 5) {init = 1 : i32}
    %lock_1_4_67 = aie.lock(%tile_1_4, 4) {init = 0 : i32}
    %lock_1_4_68 = aie.lock(%tile_1_4, 3) {init = 1 : i32}
    %lock_1_4_69 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_70 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_71 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 7) {init = 1 : i32}
    %lock_2_4_72 = aie.lock(%tile_2_4, 6) {init = 0 : i32}
    %lock_2_4_73 = aie.lock(%tile_2_4, 5) {init = 1 : i32}
    %lock_2_4_74 = aie.lock(%tile_2_4, 4) {init = 0 : i32}
    %lock_2_4_75 = aie.lock(%tile_2_4, 3) {init = 1 : i32}
    %lock_2_4_76 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_77 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_78 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 7) {init = 1 : i32}
    %lock_3_4_79 = aie.lock(%tile_3_4, 6) {init = 0 : i32}
    %lock_3_4_80 = aie.lock(%tile_3_4, 5) {init = 1 : i32}
    %lock_3_4_81 = aie.lock(%tile_3_4, 4) {init = 0 : i32}
    %lock_3_4_82 = aie.lock(%tile_3_4, 3) {init = 1 : i32}
    %lock_3_4_83 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_84 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_85 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 7) {init = 1 : i32}
    %lock_0_5_86 = aie.lock(%tile_0_5, 6) {init = 0 : i32}
    %lock_0_5_87 = aie.lock(%tile_0_5, 5) {init = 1 : i32}
    %lock_0_5_88 = aie.lock(%tile_0_5, 4) {init = 0 : i32}
    %lock_0_5_89 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_90 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_91 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_92 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 7) {init = 1 : i32}
    %lock_1_5_93 = aie.lock(%tile_1_5, 6) {init = 0 : i32}
    %lock_1_5_94 = aie.lock(%tile_1_5, 5) {init = 1 : i32}
    %lock_1_5_95 = aie.lock(%tile_1_5, 4) {init = 0 : i32}
    %lock_1_5_96 = aie.lock(%tile_1_5, 3) {init = 1 : i32}
    %lock_1_5_97 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_98 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_99 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 7) {init = 1 : i32}
    %lock_2_5_100 = aie.lock(%tile_2_5, 6) {init = 0 : i32}
    %lock_2_5_101 = aie.lock(%tile_2_5, 5) {init = 1 : i32}
    %lock_2_5_102 = aie.lock(%tile_2_5, 4) {init = 0 : i32}
    %lock_2_5_103 = aie.lock(%tile_2_5, 3) {init = 1 : i32}
    %lock_2_5_104 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_105 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_106 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 7) {init = 1 : i32}
    %lock_3_5_107 = aie.lock(%tile_3_5, 6) {init = 0 : i32}
    %lock_3_5_108 = aie.lock(%tile_3_5, 5) {init = 1 : i32}
    %lock_3_5_109 = aie.lock(%tile_3_5, 4) {init = 0 : i32}
    %lock_3_5_110 = aie.lock(%tile_3_5, 3) {init = 1 : i32}
    %lock_3_5_111 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_112 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_113 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf322 = aie.buffer(%tile_3_5) {sym_name = "buf322"} : memref<3xi32, 2 : i32> 
    %buf321 = aie.buffer(%tile_3_5) {sym_name = "buf321"} : memref<64x1xbf16, 2 : i32> 
    %buf320 = aie.buffer(%tile_3_5) {sym_name = "buf320"} : memref<64x1xbf16, 2 : i32> 
    %buf319 = aie.buffer(%tile_3_5) {sym_name = "buf319"} : memref<64x64xbf16, 2 : i32> 
    %buf318 = aie.buffer(%tile_3_5) {sym_name = "buf318"} : memref<64x64xbf16, 2 : i32> 
    %buf317 = aie.buffer(%tile_3_5) {sym_name = "buf317"} : memref<64x64xbf16, 2 : i32> 
    %buf316 = aie.buffer(%tile_3_5) {sym_name = "buf316"} : memref<64x64xbf16, 2 : i32> 
    %buf315 = aie.buffer(%tile_3_5) {sym_name = "buf315"} : memref<64x64xbf16, 2 : i32> 
    %buf314 = aie.buffer(%tile_3_5) {sym_name = "buf314"} : memref<64x1xbf16, 2 : i32> 
    %buf313 = aie.buffer(%tile_3_5) {sym_name = "buf313"} : memref<64x1xbf16, 2 : i32> 
    %buf312 = aie.buffer(%tile_2_5) {sym_name = "buf312"} : memref<3xi32, 2 : i32> 
    %buf311 = aie.buffer(%tile_2_5) {sym_name = "buf311"} : memref<64x1xbf16, 2 : i32> 
    %buf310 = aie.buffer(%tile_2_5) {sym_name = "buf310"} : memref<64x1xbf16, 2 : i32> 
    %buf309 = aie.buffer(%tile_2_5) {sym_name = "buf309"} : memref<64x64xbf16, 2 : i32> 
    %buf308 = aie.buffer(%tile_2_5) {sym_name = "buf308"} : memref<64x64xbf16, 2 : i32> 
    %buf307 = aie.buffer(%tile_2_5) {sym_name = "buf307"} : memref<64x64xbf16, 2 : i32> 
    %buf306 = aie.buffer(%tile_2_5) {sym_name = "buf306"} : memref<64x64xbf16, 2 : i32> 
    %buf305 = aie.buffer(%tile_2_5) {sym_name = "buf305"} : memref<64x64xbf16, 2 : i32> 
    %buf304 = aie.buffer(%tile_2_5) {sym_name = "buf304"} : memref<64x1xbf16, 2 : i32> 
    %buf303 = aie.buffer(%tile_2_5) {sym_name = "buf303"} : memref<64x1xbf16, 2 : i32> 
    %buf302 = aie.buffer(%tile_1_5) {sym_name = "buf302"} : memref<3xi32, 2 : i32> 
    %buf301 = aie.buffer(%tile_1_5) {sym_name = "buf301"} : memref<64x1xbf16, 2 : i32> 
    %buf300 = aie.buffer(%tile_1_5) {sym_name = "buf300"} : memref<64x1xbf16, 2 : i32> 
    %buf299 = aie.buffer(%tile_1_5) {sym_name = "buf299"} : memref<64x64xbf16, 2 : i32> 
    %buf298 = aie.buffer(%tile_1_5) {sym_name = "buf298"} : memref<64x64xbf16, 2 : i32> 
    %buf297 = aie.buffer(%tile_1_5) {sym_name = "buf297"} : memref<64x64xbf16, 2 : i32> 
    %buf296 = aie.buffer(%tile_1_5) {sym_name = "buf296"} : memref<64x64xbf16, 2 : i32> 
    %buf295 = aie.buffer(%tile_1_5) {sym_name = "buf295"} : memref<64x64xbf16, 2 : i32> 
    %buf294 = aie.buffer(%tile_1_5) {sym_name = "buf294"} : memref<64x1xbf16, 2 : i32> 
    %buf293 = aie.buffer(%tile_1_5) {sym_name = "buf293"} : memref<64x1xbf16, 2 : i32> 
    %buf292 = aie.buffer(%tile_0_5) {sym_name = "buf292"} : memref<3xi32, 2 : i32> 
    %buf291 = aie.buffer(%tile_0_5) {sym_name = "buf291"} : memref<64x1xbf16, 2 : i32> 
    %buf290 = aie.buffer(%tile_0_5) {sym_name = "buf290"} : memref<64x1xbf16, 2 : i32> 
    %buf289 = aie.buffer(%tile_0_5) {sym_name = "buf289"} : memref<64x64xbf16, 2 : i32> 
    %buf288 = aie.buffer(%tile_0_5) {sym_name = "buf288"} : memref<64x64xbf16, 2 : i32> 
    %buf287 = aie.buffer(%tile_0_5) {sym_name = "buf287"} : memref<64x64xbf16, 2 : i32> 
    %buf286 = aie.buffer(%tile_0_5) {sym_name = "buf286"} : memref<64x64xbf16, 2 : i32> 
    %buf285 = aie.buffer(%tile_0_5) {sym_name = "buf285"} : memref<64x64xbf16, 2 : i32> 
    %buf284 = aie.buffer(%tile_0_5) {sym_name = "buf284"} : memref<64x1xbf16, 2 : i32> 
    %buf283 = aie.buffer(%tile_0_5) {sym_name = "buf283"} : memref<64x1xbf16, 2 : i32> 
    %buf282 = aie.buffer(%tile_3_4) {sym_name = "buf282"} : memref<3xi32, 2 : i32> 
    %buf281 = aie.buffer(%tile_3_4) {sym_name = "buf281"} : memref<64x1xbf16, 2 : i32> 
    %buf280 = aie.buffer(%tile_3_4) {sym_name = "buf280"} : memref<64x1xbf16, 2 : i32> 
    %buf279 = aie.buffer(%tile_3_4) {sym_name = "buf279"} : memref<64x64xbf16, 2 : i32> 
    %buf278 = aie.buffer(%tile_3_4) {sym_name = "buf278"} : memref<64x64xbf16, 2 : i32> 
    %buf277 = aie.buffer(%tile_3_4) {sym_name = "buf277"} : memref<64x64xbf16, 2 : i32> 
    %buf276 = aie.buffer(%tile_3_4) {sym_name = "buf276"} : memref<64x64xbf16, 2 : i32> 
    %buf275 = aie.buffer(%tile_3_4) {sym_name = "buf275"} : memref<64x64xbf16, 2 : i32> 
    %buf274 = aie.buffer(%tile_3_4) {sym_name = "buf274"} : memref<64x1xbf16, 2 : i32> 
    %buf273 = aie.buffer(%tile_3_4) {sym_name = "buf273"} : memref<64x1xbf16, 2 : i32> 
    %buf272 = aie.buffer(%tile_2_4) {sym_name = "buf272"} : memref<3xi32, 2 : i32> 
    %buf271 = aie.buffer(%tile_2_4) {sym_name = "buf271"} : memref<64x1xbf16, 2 : i32> 
    %buf270 = aie.buffer(%tile_2_4) {sym_name = "buf270"} : memref<64x1xbf16, 2 : i32> 
    %buf269 = aie.buffer(%tile_2_4) {sym_name = "buf269"} : memref<64x64xbf16, 2 : i32> 
    %buf268 = aie.buffer(%tile_2_4) {sym_name = "buf268"} : memref<64x64xbf16, 2 : i32> 
    %buf267 = aie.buffer(%tile_2_4) {sym_name = "buf267"} : memref<64x64xbf16, 2 : i32> 
    %buf266 = aie.buffer(%tile_2_4) {sym_name = "buf266"} : memref<64x64xbf16, 2 : i32> 
    %buf265 = aie.buffer(%tile_2_4) {sym_name = "buf265"} : memref<64x64xbf16, 2 : i32> 
    %buf264 = aie.buffer(%tile_2_4) {sym_name = "buf264"} : memref<64x1xbf16, 2 : i32> 
    %buf263 = aie.buffer(%tile_2_4) {sym_name = "buf263"} : memref<64x1xbf16, 2 : i32> 
    %buf262 = aie.buffer(%tile_1_4) {sym_name = "buf262"} : memref<3xi32, 2 : i32> 
    %buf261 = aie.buffer(%tile_1_4) {sym_name = "buf261"} : memref<64x1xbf16, 2 : i32> 
    %buf260 = aie.buffer(%tile_1_4) {sym_name = "buf260"} : memref<64x1xbf16, 2 : i32> 
    %buf259 = aie.buffer(%tile_1_4) {sym_name = "buf259"} : memref<64x64xbf16, 2 : i32> 
    %buf258 = aie.buffer(%tile_1_4) {sym_name = "buf258"} : memref<64x64xbf16, 2 : i32> 
    %buf257 = aie.buffer(%tile_1_4) {sym_name = "buf257"} : memref<64x64xbf16, 2 : i32> 
    %buf256 = aie.buffer(%tile_1_4) {sym_name = "buf256"} : memref<64x64xbf16, 2 : i32> 
    %buf255 = aie.buffer(%tile_1_4) {sym_name = "buf255"} : memref<64x64xbf16, 2 : i32> 
    %buf254 = aie.buffer(%tile_1_4) {sym_name = "buf254"} : memref<64x1xbf16, 2 : i32> 
    %buf253 = aie.buffer(%tile_1_4) {sym_name = "buf253"} : memref<64x1xbf16, 2 : i32> 
    %buf252 = aie.buffer(%tile_0_4) {sym_name = "buf252"} : memref<3xi32, 2 : i32> 
    %buf251 = aie.buffer(%tile_0_4) {sym_name = "buf251"} : memref<64x1xbf16, 2 : i32> 
    %buf250 = aie.buffer(%tile_0_4) {sym_name = "buf250"} : memref<64x1xbf16, 2 : i32> 
    %buf249 = aie.buffer(%tile_0_4) {sym_name = "buf249"} : memref<64x64xbf16, 2 : i32> 
    %buf248 = aie.buffer(%tile_0_4) {sym_name = "buf248"} : memref<64x64xbf16, 2 : i32> 
    %buf247 = aie.buffer(%tile_0_4) {sym_name = "buf247"} : memref<64x64xbf16, 2 : i32> 
    %buf246 = aie.buffer(%tile_0_4) {sym_name = "buf246"} : memref<64x64xbf16, 2 : i32> 
    %buf245 = aie.buffer(%tile_0_4) {sym_name = "buf245"} : memref<64x64xbf16, 2 : i32> 
    %buf244 = aie.buffer(%tile_0_4) {sym_name = "buf244"} : memref<64x1xbf16, 2 : i32> 
    %buf243 = aie.buffer(%tile_0_4) {sym_name = "buf243"} : memref<64x1xbf16, 2 : i32> 
    %buf242 = aie.buffer(%tile_3_3) {sym_name = "buf242"} : memref<3xi32, 2 : i32> 
    %buf241 = aie.buffer(%tile_3_3) {sym_name = "buf241"} : memref<64x1xbf16, 2 : i32> 
    %buf240 = aie.buffer(%tile_3_3) {sym_name = "buf240"} : memref<64x1xbf16, 2 : i32> 
    %buf239 = aie.buffer(%tile_3_3) {sym_name = "buf239"} : memref<64x64xbf16, 2 : i32> 
    %buf238 = aie.buffer(%tile_3_3) {sym_name = "buf238"} : memref<64x64xbf16, 2 : i32> 
    %buf237 = aie.buffer(%tile_3_3) {sym_name = "buf237"} : memref<64x64xbf16, 2 : i32> 
    %buf236 = aie.buffer(%tile_3_3) {sym_name = "buf236"} : memref<64x64xbf16, 2 : i32> 
    %buf235 = aie.buffer(%tile_3_3) {sym_name = "buf235"} : memref<64x64xbf16, 2 : i32> 
    %buf234 = aie.buffer(%tile_3_3) {sym_name = "buf234"} : memref<64x1xbf16, 2 : i32> 
    %buf233 = aie.buffer(%tile_3_3) {sym_name = "buf233"} : memref<64x1xbf16, 2 : i32> 
    %buf232 = aie.buffer(%tile_2_3) {sym_name = "buf232"} : memref<3xi32, 2 : i32> 
    %buf231 = aie.buffer(%tile_2_3) {sym_name = "buf231"} : memref<64x1xbf16, 2 : i32> 
    %buf230 = aie.buffer(%tile_2_3) {sym_name = "buf230"} : memref<64x1xbf16, 2 : i32> 
    %buf229 = aie.buffer(%tile_2_3) {sym_name = "buf229"} : memref<64x64xbf16, 2 : i32> 
    %buf228 = aie.buffer(%tile_2_3) {sym_name = "buf228"} : memref<64x64xbf16, 2 : i32> 
    %buf227 = aie.buffer(%tile_2_3) {sym_name = "buf227"} : memref<64x64xbf16, 2 : i32> 
    %buf226 = aie.buffer(%tile_2_3) {sym_name = "buf226"} : memref<64x64xbf16, 2 : i32> 
    %buf225 = aie.buffer(%tile_2_3) {sym_name = "buf225"} : memref<64x64xbf16, 2 : i32> 
    %buf224 = aie.buffer(%tile_2_3) {sym_name = "buf224"} : memref<64x1xbf16, 2 : i32> 
    %buf223 = aie.buffer(%tile_2_3) {sym_name = "buf223"} : memref<64x1xbf16, 2 : i32> 
    %buf222 = aie.buffer(%tile_1_3) {sym_name = "buf222"} : memref<3xi32, 2 : i32> 
    %buf221 = aie.buffer(%tile_1_3) {sym_name = "buf221"} : memref<64x1xbf16, 2 : i32> 
    %buf220 = aie.buffer(%tile_1_3) {sym_name = "buf220"} : memref<64x1xbf16, 2 : i32> 
    %buf219 = aie.buffer(%tile_1_3) {sym_name = "buf219"} : memref<64x64xbf16, 2 : i32> 
    %buf218 = aie.buffer(%tile_1_3) {sym_name = "buf218"} : memref<64x64xbf16, 2 : i32> 
    %buf217 = aie.buffer(%tile_1_3) {sym_name = "buf217"} : memref<64x64xbf16, 2 : i32> 
    %buf216 = aie.buffer(%tile_1_3) {sym_name = "buf216"} : memref<64x64xbf16, 2 : i32> 
    %buf215 = aie.buffer(%tile_1_3) {sym_name = "buf215"} : memref<64x64xbf16, 2 : i32> 
    %buf214 = aie.buffer(%tile_1_3) {sym_name = "buf214"} : memref<64x1xbf16, 2 : i32> 
    %buf213 = aie.buffer(%tile_1_3) {sym_name = "buf213"} : memref<64x1xbf16, 2 : i32> 
    %buf212 = aie.buffer(%tile_0_3) {sym_name = "buf212"} : memref<3xi32, 2 : i32> 
    %buf211 = aie.buffer(%tile_0_3) {sym_name = "buf211"} : memref<64x1xbf16, 2 : i32> 
    %buf210 = aie.buffer(%tile_0_3) {sym_name = "buf210"} : memref<64x1xbf16, 2 : i32> 
    %buf209 = aie.buffer(%tile_0_3) {sym_name = "buf209"} : memref<64x64xbf16, 2 : i32> 
    %buf208 = aie.buffer(%tile_0_3) {sym_name = "buf208"} : memref<64x64xbf16, 2 : i32> 
    %buf207 = aie.buffer(%tile_0_3) {sym_name = "buf207"} : memref<64x64xbf16, 2 : i32> 
    %buf206 = aie.buffer(%tile_0_3) {sym_name = "buf206"} : memref<64x64xbf16, 2 : i32> 
    %buf205 = aie.buffer(%tile_0_3) {sym_name = "buf205"} : memref<64x64xbf16, 2 : i32> 
    %buf204 = aie.buffer(%tile_0_3) {sym_name = "buf204"} : memref<64x1xbf16, 2 : i32> 
    %buf203 = aie.buffer(%tile_0_3) {sym_name = "buf203"} : memref<64x1xbf16, 2 : i32> 
    %buf202 = aie.buffer(%tile_3_2) {sym_name = "buf202"} : memref<3xi32, 2 : i32> 
    %buf201 = aie.buffer(%tile_3_2) {sym_name = "buf201"} : memref<64x1xbf16, 2 : i32> 
    %buf200 = aie.buffer(%tile_3_2) {sym_name = "buf200"} : memref<64x1xbf16, 2 : i32> 
    %buf199 = aie.buffer(%tile_3_2) {sym_name = "buf199"} : memref<64x64xbf16, 2 : i32> 
    %buf198 = aie.buffer(%tile_3_2) {sym_name = "buf198"} : memref<64x64xbf16, 2 : i32> 
    %buf197 = aie.buffer(%tile_3_2) {sym_name = "buf197"} : memref<64x64xbf16, 2 : i32> 
    %buf196 = aie.buffer(%tile_3_2) {sym_name = "buf196"} : memref<64x64xbf16, 2 : i32> 
    %buf195 = aie.buffer(%tile_3_2) {sym_name = "buf195"} : memref<64x64xbf16, 2 : i32> 
    %buf194 = aie.buffer(%tile_3_2) {sym_name = "buf194"} : memref<64x1xbf16, 2 : i32> 
    %buf193 = aie.buffer(%tile_3_2) {sym_name = "buf193"} : memref<64x1xbf16, 2 : i32> 
    %buf192 = aie.buffer(%tile_2_2) {sym_name = "buf192"} : memref<3xi32, 2 : i32> 
    %buf191 = aie.buffer(%tile_2_2) {sym_name = "buf191"} : memref<64x1xbf16, 2 : i32> 
    %buf190 = aie.buffer(%tile_2_2) {sym_name = "buf190"} : memref<64x1xbf16, 2 : i32> 
    %buf189 = aie.buffer(%tile_2_2) {sym_name = "buf189"} : memref<64x64xbf16, 2 : i32> 
    %buf188 = aie.buffer(%tile_2_2) {sym_name = "buf188"} : memref<64x64xbf16, 2 : i32> 
    %buf187 = aie.buffer(%tile_2_2) {sym_name = "buf187"} : memref<64x64xbf16, 2 : i32> 
    %buf186 = aie.buffer(%tile_2_2) {sym_name = "buf186"} : memref<64x64xbf16, 2 : i32> 
    %buf185 = aie.buffer(%tile_2_2) {sym_name = "buf185"} : memref<64x64xbf16, 2 : i32> 
    %buf184 = aie.buffer(%tile_2_2) {sym_name = "buf184"} : memref<64x1xbf16, 2 : i32> 
    %buf183 = aie.buffer(%tile_2_2) {sym_name = "buf183"} : memref<64x1xbf16, 2 : i32> 
    %buf182 = aie.buffer(%tile_1_2) {sym_name = "buf182"} : memref<3xi32, 2 : i32> 
    %buf181 = aie.buffer(%tile_1_2) {sym_name = "buf181"} : memref<64x1xbf16, 2 : i32> 
    %buf180 = aie.buffer(%tile_1_2) {sym_name = "buf180"} : memref<64x1xbf16, 2 : i32> 
    %buf179 = aie.buffer(%tile_1_2) {sym_name = "buf179"} : memref<64x64xbf16, 2 : i32> 
    %buf178 = aie.buffer(%tile_1_2) {sym_name = "buf178"} : memref<64x64xbf16, 2 : i32> 
    %buf177 = aie.buffer(%tile_1_2) {sym_name = "buf177"} : memref<64x64xbf16, 2 : i32> 
    %buf176 = aie.buffer(%tile_1_2) {sym_name = "buf176"} : memref<64x64xbf16, 2 : i32> 
    %buf175 = aie.buffer(%tile_1_2) {sym_name = "buf175"} : memref<64x64xbf16, 2 : i32> 
    %buf174 = aie.buffer(%tile_1_2) {sym_name = "buf174"} : memref<64x1xbf16, 2 : i32> 
    %buf173 = aie.buffer(%tile_1_2) {sym_name = "buf173"} : memref<64x1xbf16, 2 : i32> 
    %buf172 = aie.buffer(%tile_0_2) {sym_name = "buf172"} : memref<3xi32, 2 : i32> 
    %buf171 = aie.buffer(%tile_0_2) {sym_name = "buf171"} : memref<64x1xbf16, 2 : i32> 
    %buf170 = aie.buffer(%tile_0_2) {sym_name = "buf170"} : memref<64x1xbf16, 2 : i32> 
    %buf169 = aie.buffer(%tile_0_2) {sym_name = "buf169"} : memref<64x64xbf16, 2 : i32> 
    %buf168 = aie.buffer(%tile_0_2) {sym_name = "buf168"} : memref<64x64xbf16, 2 : i32> 
    %buf167 = aie.buffer(%tile_0_2) {sym_name = "buf167"} : memref<64x64xbf16, 2 : i32> 
    %buf166 = aie.buffer(%tile_0_2) {sym_name = "buf166"} : memref<64x64xbf16, 2 : i32> 
    %buf165 = aie.buffer(%tile_0_2) {sym_name = "buf165"} : memref<64x64xbf16, 2 : i32> 
    %buf164 = aie.buffer(%tile_0_2) {sym_name = "buf164"} : memref<64x1xbf16, 2 : i32> 
    %buf163 = aie.buffer(%tile_0_2) {sym_name = "buf163"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<512x128xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<512x128xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<512x512xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<512x512xbf16>
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    %mem_3_5 = aie.mem(%tile_3_5) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_113, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf319 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_112, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_5_110, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_111, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_5_110, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_111, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_5_110, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_111, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_5_110, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_111, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_5_108, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf316 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_5_109, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_107, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_5_112, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf319) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf321) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf320) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf322[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf322[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf322[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf322[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_5_110, Release, %c1_i32)
      aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_5_110, Release, %c1_i32)
      aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_5_110, Release, %c1_i32)
      aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf318, %buf317) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_110, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf315 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_107, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf315 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf317, %buf318, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5, Release, %c1_i32)
        aie.use_lock(%lock_3_5_109, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf322[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c3_i32 : i32
        func.call @apply_causal_mask(%buf315, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf315 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf320, %buf314, %buf313) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf313, %buf319) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf315 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf316, %buf319) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf321, %buf313, %buf314) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf314, %buf321) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_108, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf321, %buf319) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf322[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf322[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf322[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf322[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf322[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_5_113, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_106, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf309 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_105, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_5_103, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_104, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_5_103, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_104, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_5_103, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_104, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_5_103, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_104, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_5_101, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf306 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_5_102, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_100, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_5_105, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf309) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf311) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf310) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf312[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf312[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf312[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf312[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_5_104, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_5_103, Release, %c1_i32)
      aie.use_lock(%lock_2_5_104, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_5_103, Release, %c1_i32)
      aie.use_lock(%lock_2_5_104, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf308, %buf307) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_103, Release, %c1_i32)
      aie.use_lock(%lock_2_5_104, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_5_103, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf305 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_100, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf305 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf307, %buf308, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5, Release, %c1_i32)
        aie.use_lock(%lock_2_5_102, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf312[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c2_i32 : i32
        func.call @apply_causal_mask(%buf305, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf305 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf310, %buf304, %buf303) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf303, %buf309) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf305 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf306, %buf309) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf311, %buf303, %buf304) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf304, %buf311) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_101, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf311, %buf309) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf312[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf312[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf312[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf312[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf312[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_5_106, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_99, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf299 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_98, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_5_96, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_97, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_5_96, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_97, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_5_96, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_97, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_5_96, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_97, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_5_94, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf296 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_5_95, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_93, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_5_98, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf299) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf301) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf300) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf302[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf302[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf302[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf302[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_5_97, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_5_96, Release, %c1_i32)
      aie.use_lock(%lock_1_5_97, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf298, %buf297) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_96, Release, %c1_i32)
      aie.use_lock(%lock_1_5_97, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_5_96, Release, %c1_i32)
      aie.use_lock(%lock_1_5_97, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_5_96, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf295 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_93, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf295 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf297, %buf298, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5, Release, %c1_i32)
        aie.use_lock(%lock_1_5_95, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf302[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c1_i32 : i32
        func.call @apply_causal_mask(%buf295, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf295 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf300, %buf294, %buf293) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf293, %buf299) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf295 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf296, %buf299) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf301, %buf293, %buf294) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf294, %buf301) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_94, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf301, %buf299) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf302[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf302[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf302[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf302[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf302[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_5_99, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_92, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf289 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_91, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_5_89, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_90, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_5_89, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_90, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_5_89, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_90, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_5_89, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_90, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_5_87, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf286 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_5_88, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_86, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1_117 = arith.constant 1 : index
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_5_91, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf289) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf291) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf290) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf292[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf292[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf292[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf292[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_5_90, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf288, %buf287) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_89, Release, %c1_i32)
      aie.use_lock(%lock_0_5_90, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_5_89, Release, %c1_i32)
      aie.use_lock(%lock_0_5_90, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_5_89, Release, %c1_i32)
      aie.use_lock(%lock_0_5_90, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_5_89, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf285 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_86, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf285 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf287, %buf288, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5, Release, %c1_i32)
        aie.use_lock(%lock_0_5_88, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf292[%c0_118] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf285, %15, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf285 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf290, %buf284, %buf283) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf283, %buf289) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf285 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf286, %buf289) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf291, %buf283, %buf284) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf284, %buf291) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_87, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf291, %buf289) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf292[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf292[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf292[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf292[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf292[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_5_92, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_85, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf279 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_84, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_4_82, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_83, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_4_82, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_83, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_4_82, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_83, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_4_82, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_83, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_4_80, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf276 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_4_81, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_79, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_4_84, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf279) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf281) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf280) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf282[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf282[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf282[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf282[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_4_83, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_4_82, Release, %c1_i32)
      aie.use_lock(%lock_3_4_83, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_4_82, Release, %c1_i32)
      aie.use_lock(%lock_3_4_83, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_4_82, Release, %c1_i32)
      aie.use_lock(%lock_3_4_83, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf278, %buf277) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_82, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf275 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_79, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf275 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf277, %buf278, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4, Release, %c1_i32)
        aie.use_lock(%lock_3_4_81, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf282[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c3_i32 : i32
        func.call @apply_causal_mask(%buf275, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf275 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf280, %buf274, %buf273) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf273, %buf279) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf275 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf276, %buf279) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf281, %buf273, %buf274) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf274, %buf281) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_80, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf281, %buf279) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf282[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf282[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf282[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf282[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf282[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_4_85, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_78, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf269 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_77, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_4_75, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_76, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_4_75, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_76, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_4_75, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_76, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_4_75, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_76, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_4_73, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf266 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_4_74, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_72, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_4_77, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf269) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf271) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf270) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf272[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf272[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf272[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf272[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_4_75, Release, %c1_i32)
      aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_4_75, Release, %c1_i32)
      aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf268, %buf267) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_75, Release, %c1_i32)
      aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_4_75, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf265 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_72, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf267, %buf268, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4, Release, %c1_i32)
        aie.use_lock(%lock_2_4_74, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf272[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c2_i32 : i32
        func.call @apply_causal_mask(%buf265, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf270, %buf264, %buf263) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf263, %buf269) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf266, %buf269) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf271, %buf263, %buf264) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf264, %buf271) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_73, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf271, %buf269) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf272[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf272[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf272[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf272[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf272[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_4_78, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_71, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf259 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_70, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_4_68, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_69, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_4_68, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_69, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_4_68, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_69, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_4_68, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_69, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_4_66, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf256 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_4_67, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_65, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_4_70, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf259) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf261) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf260) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf262[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf262[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf262[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf262[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_4_69, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_4_68, Release, %c1_i32)
      aie.use_lock(%lock_1_4_69, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf258, %buf257) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_68, Release, %c1_i32)
      aie.use_lock(%lock_1_4_69, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_4_68, Release, %c1_i32)
      aie.use_lock(%lock_1_4_69, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_4_68, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf255 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_65, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf255 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf257, %buf258, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4, Release, %c1_i32)
        aie.use_lock(%lock_1_4_67, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf262[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c1_i32 : i32
        func.call @apply_causal_mask(%buf255, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf255 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf260, %buf254, %buf253) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf253, %buf259) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf255 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf256, %buf259) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf261, %buf253, %buf254) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf254, %buf261) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_66, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf261, %buf259) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf262[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf262[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf262[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf262[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf262[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_4_71, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_64, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf249 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_63, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_4_61, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_62, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_4_61, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_62, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_4_61, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_62, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_4_61, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_62, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_4_59, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf246 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_4_60, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_58, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c1_117 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_4_63, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf249) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf251) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf250) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf252[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf252[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf252[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf252[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_4_62, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf248, %buf247) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_61, Release, %c1_i32)
      aie.use_lock(%lock_0_4_62, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_61, Release, %c1_i32)
      aie.use_lock(%lock_0_4_62, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_61, Release, %c1_i32)
      aie.use_lock(%lock_0_4_62, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_61, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf245 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_58, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf245 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf247, %buf248, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4, Release, %c1_i32)
        aie.use_lock(%lock_0_4_60, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf252[%c0_118] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf245, %15, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf245 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf250, %buf244, %buf243) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf243, %buf249) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf245 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf246, %buf249) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf251, %buf243, %buf244) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf244, %buf251) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_59, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf251, %buf249) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf252[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf252[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf252[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf252[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf252[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_4_64, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_57, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf239 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_56, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_3_54, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_55, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_3_54, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_55, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_3_54, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_55, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_3_54, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_55, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_3_52, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf236 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_3_53, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_51, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_3_56, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf239) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf241) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf240) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf242[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf242[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf242[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf242[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_3_55, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_3_54, Release, %c1_i32)
      aie.use_lock(%lock_3_3_55, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_3_54, Release, %c1_i32)
      aie.use_lock(%lock_3_3_55, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_3_54, Release, %c1_i32)
      aie.use_lock(%lock_3_3_55, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf238, %buf237) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_54, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf235 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf235 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf237, %buf238, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3, Release, %c1_i32)
        aie.use_lock(%lock_3_3_53, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf242[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c3_i32 : i32
        func.call @apply_causal_mask(%buf235, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf235 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf240, %buf234, %buf233) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf233, %buf239) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf235 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf236, %buf239) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf241, %buf233, %buf234) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf234, %buf241) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_52, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf241, %buf239) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf242[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf242[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf242[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf242[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf242[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_3_57, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_50, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf229 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_49, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_3_45, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf226 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_3_46, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_44, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_3_49, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf229) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf231) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf230) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf232[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf232[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf232[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf232[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_3_47, Release, %c1_i32)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_3_47, Release, %c1_i32)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf228, %buf227) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, %c1_i32)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_3_47, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf225 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf225 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf227, %buf228, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3, Release, %c1_i32)
        aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf232[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c2_i32 : i32
        func.call @apply_causal_mask(%buf225, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf225 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf230, %buf224, %buf223) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf223, %buf229) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf225 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf226, %buf229) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf231, %buf223, %buf224) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf224, %buf231) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_45, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf231, %buf229) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf232[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf232[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf232[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf232[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf232[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_3_50, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf219 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_42, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_41, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_41, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_41, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_41, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_3_38, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf216 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_3_39, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_37, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_3_42, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf219) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf221) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf220) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf222[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf222[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf222[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf222[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_3_40, Release, %c1_i32)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf218, %buf217) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_40, Release, %c1_i32)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_3_40, Release, %c1_i32)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_3_40, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf215 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf215 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf217, %buf218, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3, Release, %c1_i32)
        aie.use_lock(%lock_1_3_39, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf222[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c1_i32 : i32
        func.call @apply_causal_mask(%buf215, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf215 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf220, %buf214, %buf213) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf213, %buf219) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf215 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf216, %buf219) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf221, %buf213, %buf214) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf214, %buf221) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_38, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf221, %buf219) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf222[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf222[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf222[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf222[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf222[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_3_43, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_36, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf209 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_35, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_3_31, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf206 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_3_32, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_30, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1_117 = arith.constant 1 : index
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_3_35, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf209) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf211) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf210) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf212[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf212[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf212[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf212[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf208, %buf207) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_33, Release, %c1_i32)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_33, Release, %c1_i32)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_33, Release, %c1_i32)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_33, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf205 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_30, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf205 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf207, %buf208, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3, Release, %c1_i32)
        aie.use_lock(%lock_0_3_32, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf212[%c0_118] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf205, %15, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf205 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf210, %buf204, %buf203) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf203, %buf209) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf205 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf206, %buf209) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf211, %buf203, %buf204) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf204, %buf211) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_31, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf211, %buf209) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf212[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf212[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf212[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf212[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf212[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_3_36, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf199 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_28, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_2_24, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf196 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_2_25, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_23, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1_117 = arith.constant 1 : index
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_2_28, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf199) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf201) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf200) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf202[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf202[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf202[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf202[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_2_26, Release, %c1_i32)
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_2_26, Release, %c1_i32)
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_2_26, Release, %c1_i32)
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf198, %buf197) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_26, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf195 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_23, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf195 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf197, %buf198, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2, Release, %c1_i32)
        aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf202[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c3_i32 : i32
        func.call @apply_causal_mask(%buf195, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf195 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf200, %buf194, %buf193) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf193, %buf199) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf195 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf196, %buf199) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf201, %buf193, %buf194) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf194, %buf201) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_24, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf201, %buf199) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf202[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf202[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf202[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf202[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf202[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_2_29, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_22, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf189 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_21, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_2_17, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf186 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_2_18, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_16, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c1_117 = arith.constant 1 : index
      %c0_118 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_2_21, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf189) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf191) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf190) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf192[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf192[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf192[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf192[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_2_19, Release, %c1_i32)
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_2_19, Release, %c1_i32)
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf188, %buf187) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_19, Release, %c1_i32)
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_2_19, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf185 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_16, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf187, %buf188, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2, Release, %c1_i32)
        aie.use_lock(%lock_2_2_18, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf192[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c2_i32 : i32
        func.call @apply_causal_mask(%buf185, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf190, %buf184, %buf183) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf183, %buf189) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf186, %buf189) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf191, %buf183, %buf184) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf184, %buf191) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_17, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf191, %buf189) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf192[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf192[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf192[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf192[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf192[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_2_22, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_15, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf179 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_14, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_2_12, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_2_12, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_2_12, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_2_12, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_2_10, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf176 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_2_11, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_9, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_14, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf179) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf181) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf180) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf182[%c1_118] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf182[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf182[%c1_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf182[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_2_12, Release, %c1_i32)
      aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf178, %buf177) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_12, Release, %c1_i32)
      aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_2_12, Release, %c1_i32)
      aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_2_12, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf175 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf175 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf177, %buf178, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2, Release, %c1_i32)
        aie.use_lock(%lock_1_2_11, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf182[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c1_i32 : i32
        func.call @apply_causal_mask(%buf175, %16, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf175 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf180, %buf174, %buf173) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf173, %buf179) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf175 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf176, %buf179) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf181, %buf173, %buf174) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf174, %buf181) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_10, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf181, %buf179) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf182[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf182[%c0_117] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf182[%c0_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf182[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf182[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_2_15, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_8, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf169 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_7, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf166 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %12 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1_117 = arith.constant 1 : index
      %c0_118 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, %c1_i32)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf169) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf171) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf170) : (memref<64x1xbf16, 2 : i32>) -> ()
      %9 = memref.load %buf172[%c1_117] : memref<3xi32, 2 : i32>
      %10 = arith.cmpi eq, %9, %c0_i32 : i32
      scf.if %10 {
        memref.store %c0_i32, %buf172[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf172[%c1_117] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf172[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf168, %buf167) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_5, Release, %c1_i32)
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_5, Release, %c1_i32)
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_5, Release, %c1_i32)
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_5, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf165 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, %c1_i32)
        %collapse_shape_119 = memref.collapse_shape %buf165 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf167, %buf168, %collapse_shape_119) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2, Release, %c1_i32)
        aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
        %15 = memref.load %buf172[%c0_118] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf165, %15, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        %collapse_shape_120 = memref.collapse_shape %buf165 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_120, %buf170, %buf164, %buf163) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf163, %buf169) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_121 = memref.collapse_shape %buf165 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_121, %buf166, %buf169) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf171, %buf163, %buf164) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf164, %buf171) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf171, %buf169) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %11 = memref.load %buf172[%c2] : memref<3xi32, 2 : i32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c1_i32 : i32
      scf.if %13 {
        %15 = memref.load %buf172[%c0_118] : memref<3xi32, 2 : i32>
        %16 = arith.addi %15, %c4_i32 : i32
        memref.store %16, %buf172[%c0_118] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf172[%c2] : memref<3xi32, 2 : i32>
      }
      %14 = arith.cmpi slt, %12, %c1_i32 : i32
      scf.if %14 {
        memref.store %12, %buf172[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_2_8, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    air.channel @channel_23 [1, 1]
    air.channel @Q2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @Q2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @channel_21 [1, 1]
    air.channel @K2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @K2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @channel_19 [1, 1]
    air.channel @V2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @V2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @channel_2 [1, 1]
    air.channel @channel_3 [1, 1]
    air.channel @channel_4 [1, 1]
    air.channel @channel_5 [1, 1]
    air.channel @channel_6 [1, 1]
    air.channel @channel_7 [1, 1]
    air.channel @channel_8 [1, 1]
    air.channel @channel_9 [1, 1]
    air.channel @channel_10 [1, 1]
    air.channel @channel_11 [1, 1]
    air.channel @channel_12 [1, 1]
    air.channel @channel_13 [1, 1]
    air.channel @channel_14 [1, 1]
    air.channel @channel_15 [1, 1]
    air.channel @channel_16 [1, 1]
    air.channel @channel_17 [1, 1]
    air.channel @channel_1 [1, 1]
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_114 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_115 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_116 = aie.logical_tile<MemTile>(?, ?)
    %0 = aie.lock(%logical_mem_114, 1) {init = 4 : i32}
    %1 = aie.lock(%logical_mem_114, 0) {init = 0 : i32}
    %2 = aie.lock(%logical_mem_115, 1) {init = 4 : i32}
    %3 = aie.lock(%logical_mem_115, 0) {init = 0 : i32}
    %4 = aie.lock(%logical_mem, 1) {init = 2 : i32}
    %5 = aie.lock(%logical_mem, 0) {init = 0 : i32}
    %buf325 = aie.buffer(%logical_mem) {sym_name = "buf325"} : memref<64x64xbf16, 1 : i32> 
    %buf324 = aie.buffer(%logical_mem_114) {sym_name = "buf324"} : memref<256x64xbf16, 1 : i32> 
    %buf323 = aie.buffer(%logical_mem_115) {sym_name = "buf323"} : memref<64x64xbf16, 1 : i32> 
    func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @apply_causal_mask(memref<64x64xbf16, 2 : i32>, i32, i32) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
    aie.flow(%logical_shim_noc, DMA : 0, %logical_mem, DMA : 0)
    aie.flow(%logical_shim_noc_1, DMA : 0, %logical_mem_115, DMA : 0)
    aie.flow(%logical_shim_noc, DMA : 1, %logical_mem, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%logical_mem, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%logical_mem, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%logical_mem, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%logical_mem, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%logical_mem, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%logical_mem, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%logical_mem, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%logical_mem, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%logical_mem, DMA : 2, %tile_1_4, DMA : 0)
    aie.flow(%logical_mem, DMA : 2, %tile_2_4, DMA : 0)
    aie.flow(%logical_mem, DMA : 2, %tile_3_4, DMA : 0)
    aie.flow(%logical_mem, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%logical_mem, DMA : 3, %tile_1_5, DMA : 0)
    aie.flow(%logical_mem, DMA : 3, %tile_2_5, DMA : 0)
    aie.flow(%logical_mem, DMA : 3, %tile_3_5, DMA : 0)
    aie.flow(%logical_mem, DMA : 4, %tile_0_2, DMA : 1)
    aie.flow(%logical_mem, DMA : 4, %tile_1_2, DMA : 1)
    aie.flow(%logical_mem, DMA : 4, %tile_2_2, DMA : 1)
    aie.flow(%logical_mem, DMA : 4, %tile_3_2, DMA : 1)
    aie.flow(%logical_mem, DMA : 5, %tile_0_3, DMA : 1)
    aie.flow(%logical_mem, DMA : 5, %tile_1_3, DMA : 1)
    aie.flow(%logical_mem, DMA : 5, %tile_2_3, DMA : 1)
    aie.flow(%logical_mem, DMA : 5, %tile_3_3, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_0_4, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_1_4, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_2_4, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_3_4, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_0_5, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_1_5, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_2_5, DMA : 1)
    aie.flow(%logical_mem, DMA : 0, %tile_3_5, DMA : 1)
    aie.flow(%logical_mem_115, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 2, %tile_1_4, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 2, %tile_2_4, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 2, %tile_3_4, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 3, %tile_1_5, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 3, %tile_2_5, DMA : 0)
    aie.flow(%logical_mem_115, DMA : 3, %tile_3_5, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %logical_mem_114, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %logical_mem_114, DMA : 2)
    aie.flow(%tile_3_2, DMA : 0, %logical_mem_114, DMA : 3)
    aie.flow(%logical_mem_114, DMA : 0, %logical_shim_noc_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %logical_mem_114, DMA : 4)
    aie.flow(%tile_1_3, DMA : 0, %logical_mem_114, DMA : 5)
    aie.flow(%tile_2_3, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_3_3, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_1_4, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_2_4, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_3_4, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_1_5, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_2_5, DMA : 0, %logical_mem_114, DMA : 0)
    aie.flow(%tile_3_5, DMA : 0, %logical_mem_114, DMA : 0)
    %6 = aie.memtile_dma(%logical_mem) {
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb15
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %11 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %12 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %13 = aie.dma_start(MM2S, 4, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %14 = aie.dma_start(MM2S, 5, ^bb12, ^bb13)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%4, Release, %c1_i32)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb11
      %15 = aie.dma_start(S2MM, 0, ^bb14, ^bb15)
    ^bb14:  // 2 preds: ^bb13, ^bb14
      aie.use_lock(%4, AcquireGreaterEqual, %c2_i32)
      aie.dma_bd(%buf325 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%5, Release, %c2_i32)
      aie.next_bd ^bb14
    ^bb15:  // pred: ^bb13
      %16 = aie.dma_start(S2MM, 1, ^bb16, ^bb2)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%4, AcquireGreaterEqual, %c2_i32)
      aie.dma_bd(%buf325 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%5, Release, %c2_i32)
      aie.next_bd ^bb16
    }
    %7 = aie.memtile_dma(%logical_mem_115) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf323 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%2, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf323 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%2, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %11 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf323 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%2, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %12 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf323 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%2, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %13 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%2, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%buf323 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%3, Release, %c4_i32)
      aie.next_bd ^bb10
    }
    %8 = aie.memtile_dma(%logical_mem_114) {
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %9 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%1, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 0 len = 16384) {task_id = 0 : i32}
      aie.use_lock(%0, Release, %c4_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb23
      aie.end
    ^bb3:  // pred: ^bb0
      %10 = aie.dma_start(S2MM, 0, ^bb4, ^bb15)
    ^bb4:  // 2 preds: ^bb3, ^bb14
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb15:  // pred: ^bb3
      %11 = aie.dma_start(S2MM, 1, ^bb16, ^bb17)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb16
    ^bb17:  // pred: ^bb15
      %12 = aie.dma_start(S2MM, 2, ^bb18, ^bb19)
    ^bb18:  // 2 preds: ^bb17, ^bb18
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb18
    ^bb19:  // pred: ^bb17
      %13 = aie.dma_start(S2MM, 3, ^bb20, ^bb21)
    ^bb20:  // 2 preds: ^bb19, ^bb20
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb20
    ^bb21:  // pred: ^bb19
      %14 = aie.dma_start(S2MM, 4, ^bb22, ^bb23)
    ^bb22:  // 2 preds: ^bb21, ^bb22
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb22
    ^bb23:  // pred: ^bb21
      %15 = aie.dma_start(S2MM, 5, ^bb24, ^bb2)
    ^bb24:  // 2 preds: ^bb23, ^bb24
      aie.use_lock(%0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%1, Release, %c1_i32)
      aie.next_bd ^bb24
    }
    aie.shim_dma_allocation @air_GpOut_1_0_0(%logical_shim_noc_0, S2MM, 0)
    aie.shim_dma_allocation @air_KIn_1_0_0(%logical_shim_noc, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_1_0_0(%logical_shim_noc_1, MM2S, 0)
    aie.shim_dma_allocation @air_QIn_1_0_0(%logical_shim_noc, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 1 : i64, segment_unroll_y = 0 : i64}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = -1 : i64, id = 25 : i64, location = -1 : i64, row = -3 : i64}, {channel = 2 : i64, col = -1 : i64, id = 34 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 13 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 16 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 19 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 22 : i64, location = -1 : i64, row = -3 : i64}], sym_name = "attn_seg_0_0"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = -1 : i64, id = 25 : i64, location = -1 : i64, row = -3 : i64}, {channel = 2 : i64, col = -1 : i64, id = 34 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 13 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 16 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 19 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 22 : i64, location = -1 : i64, row = -3 : i64}], sym_name = "attn_seg_1_0"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
  }
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
  air.channel @Q2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QIn [2]
  air.channel @K2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @KIn [2]
  air.channel @V2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @VIn [2]
  air.channel @Gp2L2 [4, 4]
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<512x512xbf16>, %arg1: memref<512x128xbf16>, %arg2: memref<512x128xbf16>, %arg3: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<512x512xbf16>, memref<512x128xbf16>, memref<512x128xbf16>, memref<512x512xbf16> attributes {id = 1 : i32} {
      %c256 = arith.constant 256 : index
      %c2_0 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c32768 = arith.constant 32768 : index
      %c4 = arith.constant 4 : index
      %c128 = arith.constant 128 : index
      %c8192 = arith.constant 8192 : index
      %c64 = arith.constant 64 : index
      %c1_1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = affine.apply #map1()[%arg5]
      %3 = air.channel.put async  @KIn[%c0] (%arg9[%c0, %2] [%c8, %c1_1, %c64, %c64] [%c8192, %c64, %c128, %c1_1]) {id = 1 : i32, metadataArray = [{base = "air_KIn_0_0", index = 0 : i32}, {base = "air_KIn_1_0_0", index = 1 : i32}]} : (memref<512x128xbf16>)
      %4 = air.channel.put async  @VIn[%c0] (%arg10[%c0, %2] [%c8, %c64, %c64] [%c8192, %c128, %c1_1]) {id = 2 : i32, metadataArray = [{base = "air_VIn_0_0", index = 0 : i32}, {base = "air_VIn_1_0_0", index = 1 : i32}]} : (memref<512x128xbf16>)
      %5 = affine.apply #map2()[%arg5]
      %6 = air.channel.put async  @QIn[%c0] (%arg8[%1, %5] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 3 : i32, metadataArray = [{base = "air_QIn_0_0", index = 0 : i32}, {base = "air_QIn_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %7 = affine.apply #map3()[%arg5]
      %8 = air.channel.put async [%6]  @QIn[%c0] (%arg8[%1, %7] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 4 : i32, metadataArray = [{base = "air_QIn_0_0", index = 0 : i32}, {base = "air_QIn_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %9 = affine.apply #map4()[%arg5]
      %10 = air.channel.put async [%8]  @QIn[%c0] (%arg8[%1, %9] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 5 : i32, metadataArray = [{base = "air_QIn_0_0", index = 0 : i32}, {base = "air_QIn_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %11 = affine.apply #map5()[%arg5]
      %12 = air.channel.put async [%10]  @QIn[%c0] (%arg8[%1, %11] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 6 : i32, metadataArray = [{base = "air_QIn_0_0", index = 0 : i32}, {base = "air_QIn_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %13 = affine.apply #map6()[%arg5]
      %14 = air.channel.put async  @KIn[%c1_1] (%arg9[%c0, %13] [%c8, %c1_1, %c64, %c64] [%c8192, %c64, %c128, %c1_1]) {id = 7 : i32, metadataArray = [{base = "air_KIn_0_0", index = 0 : i32}, {base = "air_KIn_1_0_0", index = 1 : i32}]} : (memref<512x128xbf16>)
      %15 = air.channel.put async  @VIn[%c1_1] (%arg10[%c0, %13] [%c8, %c64, %c64] [%c8192, %c128, %c1_1]) {id = 8 : i32, metadataArray = [{base = "air_VIn_0_0", index = 0 : i32}, {base = "air_VIn_1_0_0", index = 1 : i32}]} : (memref<512x128xbf16>)
      %16 = affine.apply #map7()[%arg5]
      %17 = air.channel.put async  @QIn[%c1_1] (%arg8[%1, %16] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 9 : i32, metadataArray = [{base = "air_QIn_0_0", index = 0 : i32}, {base = "air_QIn_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %18 = affine.apply #map8()[%arg5]
      %19 = air.channel.put async [%17]  @QIn[%c1_1] (%arg8[%1, %18] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 10 : i32, metadataArray = [{base = "air_QIn_0_0", index = 0 : i32}, {base = "air_QIn_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %20 = affine.apply #map9()[%arg5]
      %21 = air.channel.put async [%19]  @QIn[%c1_1] (%arg8[%1, %20] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 11 : i32, metadataArray = [{base = "air_QIn_0_0", index = 0 : i32}, {base = "air_QIn_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %22 = affine.apply #map10()[%arg5]
      %23 = air.channel.put async [%21]  @QIn[%c1_1] (%arg8[%1, %22] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 12 : i32, metadataArray = [{base = "air_QIn_0_0", index = 0 : i32}, {base = "air_QIn_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %24 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2_0, %arg15=%c1_1) attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c12288 = arith.constant 12288 : index
        %c8192_2 = arith.constant 8192 : index
        %c4096 = arith.constant 4096 : index
        %c3 = arith.constant 3 : index
        %c2_3 = arith.constant 2 : index
        %c64_4 = arith.constant 64 : index
        %c1_5 = arith.constant 1 : index
        %c8_6 = arith.constant 8 : index
        %c0_7 = arith.constant 0 : index
        %c4_8 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %33 = air.wait_all async 
        %async_token_9, %results_10 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        }
        %34 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @Q2L1_0_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 14 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @Q2L1_1_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 15 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          scf.yield %63 : !air.async.token
        }
        %35 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %34) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 16 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @Q2L1_0_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @Q2L1_1_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          scf.yield %63 : !air.async.token
        }
        %36 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %35) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @Q2L1_0_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @Q2L1_1_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          scf.yield %63 : !air.async.token
        }
        %37 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %36) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @Q2L1_0_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @Q2L1_1_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          scf.yield %63 : !air.async.token
        }
        %38 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %37) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @KIn[%arg12] (%results[] [] []) {id = 25 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63:4 = scf.if %62 -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %65 = air.channel.put async [%61]  @K2L1_0_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 26 : i32} : (memref<64x64xbf16, 1 : i32>)
            %66 = air.channel.put async [%61]  @K2L1_0_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
            %67 = air.channel.put async [%61]  @K2L1_0_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
            %68 = air.channel.put async [%61]  @K2L1_0_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %65, %66, %67, %68 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          } else {
            %65 = air.channel.put async [%61]  @K2L1_1_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
            %66 = air.channel.put async [%61]  @K2L1_1_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
            %67 = air.channel.put async [%61]  @K2L1_1_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
            %68 = air.channel.put async [%61]  @K2L1_1_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %65, %66, %67, %68 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %64 = air.wait_all async [%63#0, %63#1, %63#2, %63#3] 
          scf.yield %64 : !air.async.token
        }
        %39 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %33) -> (!air.async.token) {
          %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %61 = air.channel.get async [%async_token_13, %arg17]  @VIn[%arg12] (%results_14[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63:4 = scf.if %62 -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %65 = air.channel.put async [%61]  @V2L1_0_0[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            %66 = air.channel.put async [%61]  @V2L1_0_1[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
            %67 = air.channel.put async [%61]  @V2L1_0_2[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
            %68 = air.channel.put async [%61]  @V2L1_0_3[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %65, %66, %67, %68 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          } else {
            %65 = air.channel.put async [%61]  @V2L1_1_0[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
            %66 = air.channel.put async [%61]  @V2L1_1_1[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            %67 = air.channel.put async [%61]  @V2L1_1_2[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            %68 = air.channel.put async [%61]  @V2L1_1_3[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %65, %66, %67, %68 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %64 = air.wait_all async [%63#0, %63#1, %63#2, %63#3] 
          %async_token_15 = air.execute [%63#0, %61] {
            memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %64 : !air.async.token
        }
        %40 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn_npu2.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c64_13 = arith.constant 64 : index
          %c8_i32 = arith.constant 8 : i32
          %c0_14 = arith.constant 0 : index
          %c1_15 = arith.constant 1 : index
          %c2_16 = arith.constant 2 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c8_17 = arith.constant 8 : index
          %c512_18 = arith.constant 512 : index
          %c4_i32 = arith.constant 4 : i32
          %async_token_19, %results_20 = air.execute -> (memref<3xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<3xi32, 2 : i32>
            air.execute_terminator %alloc : memref<3xi32, 2 : i32>
          }
          %async_token_21, %results_22 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_23, %results_24 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_25, %results_26 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_27, %results_28 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_31 = air.execute [%async_token_25] {
            func.call @zero_fill_gp_bf16(%results_26) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_32 = air.execute [%async_token_21] {
            func.call @zero_fill_sp_bf16(%results_22) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_33 = air.execute [%async_token_23] {
            func.call @neg_inf_fill_up_bf16(%results_24) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_34, %results_35 = air.execute [%async_token_19] -> (i32) {
            %74 = memref.load %results_20[%c1_15] : memref<3xi32, 2 : i32>
            air.execute_terminator %74 : i32
          }
          %61 = arith.cmpi eq, %results_35, %c0_i32 : i32
          scf.if %61 {
            %async_token_45 = air.execute [%async_token_34] {
              memref.store %c0_i32, %results_20[%c0_14] : memref<3xi32, 2 : i32>
            }
            %async_token_46 = air.execute [%async_token_45] {
              memref.store %c1_i32, %results_20[%c1_15] : memref<3xi32, 2 : i32>
            }
            %async_token_47 = air.execute [%async_token_46] {
              memref.store %c0_i32, %results_20[%c2_16] : memref<3xi32, 2 : i32>
            }
          }
          %62 = arith.cmpi eq, %arg20, %c0_14 : index
          scf.if %62 {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          } else {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          }
          %63 = arith.index_cast %arg16 : index to i32
          %64 = arith.cmpi eq, %63, %c0_i32 : i32
          scf.if %64 {
            %async_token_45 = air.execute [%async_token_27, %async_token_29] {
              func.call @copy_tile(%results_28, %results_30) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %62 {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          } else {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          }
          %65 = arith.cmpi eq, %63, %c1_i32 : i32
          scf.if %65 {
            %async_token_45 = air.execute [%async_token_27, %async_token_29] {
              func.call @copy_tile(%results_28, %results_30) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %62 {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          } else {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          }
          %66 = arith.cmpi eq, %63, %c2_i32 : i32
          scf.if %66 {
            %async_token_45 = air.execute [%async_token_27, %async_token_29] {
              func.call @copy_tile(%results_28, %results_30) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %62 {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          } else {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          }
          %67 = arith.cmpi eq, %63, %c3_i32 : i32
          scf.if %67 {
            %async_token_45 = air.execute [%async_token_27, %async_token_29] {
              func.call @copy_tile(%results_28, %results_30) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %68 = air.wait_all async [%async_token_19, %async_token_27, %async_token_29, %async_token_31, %async_token_32, %async_token_33] 
          %69 = scf.for %arg21 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg22 = %68) -> (!air.async.token)  : i32 {
            %async_token_45, %results_46 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_47, %results_48 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_49 = air.execute [%async_token_47, %arg22] {
              %collapse_shape = memref.collapse_shape %results_48 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %62 {
              %76 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%arg22]  @K2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%arg22]  @K2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %79 = air.channel.get async [%arg22]  @K2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  } else {
                    %79 = air.channel.get async [%arg22]  @K2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  }
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
            } else {
              %76 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%arg22]  @K2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%arg22]  @K2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %79 = air.channel.get async [%arg22]  @K2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  } else {
                    %79 = air.channel.get async [%arg22]  @K2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  }
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
            }
            %async_token_50 = air.execute [%async_token_49] {
              %collapse_shape = memref.collapse_shape %results_48 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_30, %results_28, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %62 {
              %76 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%async_token_45, %arg22]  @V2L1_0_0[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%async_token_45, %arg22]  @V2L1_0_1[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %79 = air.channel.get async [%async_token_45, %arg22]  @V2L1_0_2[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  } else {
                    %79 = air.channel.get async [%async_token_45, %arg22]  @V2L1_0_3[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  }
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
            } else {
              %76 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%async_token_45, %arg22]  @V2L1_1_0[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%async_token_45, %arg22]  @V2L1_1_1[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %79 = air.channel.get async [%async_token_45, %arg22]  @V2L1_1_2[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  } else {
                    %79 = air.channel.get async [%async_token_45, %arg22]  @V2L1_1_3[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  }
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
            }
            %async_token_51, %results_52 = air.execute [%arg22] -> (i32) {
              %76 = memref.load %results_20[%c0_14] : memref<3xi32, 2 : i32>
              air.execute_terminator %76 : i32
            }
            %74 = arith.addi %results_52, %63 : i32
            %async_token_53 = air.execute [%async_token_50] {
              func.call @apply_causal_mask(%results_48, %74, %arg21) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
            }
            %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_58 = air.execute [%async_token_53, %async_token_54, %async_token_56, %arg22] {
              %collapse_shape = memref.collapse_shape %results_48 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_24, %results_55, %results_57) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_59 = air.execute [%async_token_58] {
              func.call @mul_r_gp(%results_57, %results_26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_60 = air.execute [%async_token_59, %async_token_45, %async_token_47] {
              %collapse_shape = memref.collapse_shape %results_48 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_46, %results_26) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_61 = air.execute [%async_token_59] {
              func.call @accum_sp_r_s(%results_22, %results_57, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_62 = air.execute [%async_token_61] {
              func.call @vector_copy_32elems(%c0_i32, %results_55, %results_22) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_63 = air.execute [%async_token_62] {
              memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_64 = air.execute [%async_token_61] {
              memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
            }
            %75 = air.wait_all async [%async_token_51, %async_token_60, %async_token_62] 
            %async_token_65 = air.execute [%async_token_58, %async_token_60] {
              memref.dealloc %results_48 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_66 = air.execute [%async_token_60] {
              memref.dealloc %results_46 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %75 : !air.async.token
          }
          %async_token_36 = air.execute [%async_token_21, %async_token_25, %69] {
            func.call @div_gp_sp(%results_22, %results_26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %70 = air.channel.put async [%async_token_36]  @Gp2L2[%arg17, %arg16] (%results_26[%c0_14, %c0_14, %c0_14] [%c64_13, %c8_17, %c8_17] [%c8_17, %c512_18, %c1_15]) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
          %async_token_37, %results_38 = air.execute [%async_token_19, %69] -> (i32) {
            %74 = memref.load %results_20[%c2_16] : memref<3xi32, 2 : i32>
            air.execute_terminator %74 : i32
          }
          %71 = arith.addi %results_38, %c1_i32 : i32
          %72 = arith.cmpi sge, %71, %c1_i32 : i32
          scf.if %72 {
            %async_token_45, %results_46 = air.execute [%async_token_37] -> (i32) {
              %75 = memref.load %results_20[%c0_14] : memref<3xi32, 2 : i32>
              air.execute_terminator %75 : i32
            }
            %74 = arith.addi %results_46, %c4_i32 : i32
            %async_token_47 = air.execute [%async_token_45] {
              memref.store %74, %results_20[%c0_14] : memref<3xi32, 2 : i32>
            }
            %async_token_48 = air.execute [%async_token_47] {
              memref.store %c0_i32, %results_20[%c2_16] : memref<3xi32, 2 : i32>
            }
          }
          %73 = arith.cmpi slt, %71, %c1_i32 : i32
          scf.if %73 {
            %async_token_45 = air.execute [%async_token_19] {
              memref.store %71, %results_20[%c2_16] : memref<3xi32, 2 : i32>
            }
          }
          %async_token_39 = air.execute [%69] {
            memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_40 = air.execute [%69] {
            memref.dealloc %results_28 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_41 = air.execute [%async_token_31, %70] {
            memref.dealloc %results_26 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_42 = air.execute [%69, %async_token_33] {
            memref.dealloc %results_24 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_43 = air.execute [%async_token_32, %async_token_36] {
            memref.dealloc %results_22 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_44 = air.execute [%async_token_34, %async_token_37] {
            memref.dealloc %results_20 : memref<3xi32, 2 : i32>
          }
        }
        %41 = air.channel.get async [%async_token_9]  @Gp2L2[%c0_7, %c0_7] (%results_10[%c0_7] [%c4096] [%c1_5]) {id = 92 : i32} : (memref<256x64xbf16, 1 : i32>)
        %42 = air.channel.get async [%async_token_9]  @Gp2L2[%c0_7, %c1_5] (%results_10[%c4096] [%c4096] [%c1_5]) {id = 93 : i32} : (memref<256x64xbf16, 1 : i32>)
        %43 = air.channel.get async [%async_token_9]  @Gp2L2[%c0_7, %c2_3] (%results_10[%c8192_2] [%c4096] [%c1_5]) {id = 94 : i32} : (memref<256x64xbf16, 1 : i32>)
        %44 = air.channel.get async [%async_token_9]  @Gp2L2[%c0_7, %c3] (%results_10[%c12288] [%c4096] [%c1_5]) {id = 95 : i32} : (memref<256x64xbf16, 1 : i32>)
        %45 = air.channel.put async [%41, %42, %43, %44]  @GpOut[%arg12] (%results_10[] [] []) {id = 96 : i32} : (memref<256x64xbf16, 1 : i32>)
        %46 = air.channel.get async [%45]  @Gp2L2[%c1_5, %c0_7] (%results_10[%c0_7] [%c4096] [%c1_5]) {id = 97 : i32} : (memref<256x64xbf16, 1 : i32>)
        %47 = air.channel.get async [%45]  @Gp2L2[%c1_5, %c1_5] (%results_10[%c4096] [%c4096] [%c1_5]) {id = 98 : i32} : (memref<256x64xbf16, 1 : i32>)
        %48 = air.channel.get async [%45]  @Gp2L2[%c1_5, %c2_3] (%results_10[%c8192_2] [%c4096] [%c1_5]) {id = 99 : i32} : (memref<256x64xbf16, 1 : i32>)
        %49 = air.channel.get async [%45]  @Gp2L2[%c1_5, %c3] (%results_10[%c12288] [%c4096] [%c1_5]) {id = 100 : i32} : (memref<256x64xbf16, 1 : i32>)
        %50 = air.channel.put async [%45, %46, %47, %48, %49]  @GpOut[%arg12] (%results_10[] [] []) {id = 101 : i32} : (memref<256x64xbf16, 1 : i32>)
        %51 = air.channel.get async [%50]  @Gp2L2[%c2_3, %c0_7] (%results_10[%c0_7] [%c4096] [%c1_5]) {id = 102 : i32} : (memref<256x64xbf16, 1 : i32>)
        %52 = air.channel.get async [%50]  @Gp2L2[%c2_3, %c1_5] (%results_10[%c4096] [%c4096] [%c1_5]) {id = 103 : i32} : (memref<256x64xbf16, 1 : i32>)
        %53 = air.channel.get async [%50]  @Gp2L2[%c2_3, %c2_3] (%results_10[%c8192_2] [%c4096] [%c1_5]) {id = 104 : i32} : (memref<256x64xbf16, 1 : i32>)
        %54 = air.channel.get async [%50]  @Gp2L2[%c2_3, %c3] (%results_10[%c12288] [%c4096] [%c1_5]) {id = 105 : i32} : (memref<256x64xbf16, 1 : i32>)
        %55 = air.channel.put async [%50, %51, %52, %53, %54]  @GpOut[%arg12] (%results_10[] [] []) {id = 106 : i32} : (memref<256x64xbf16, 1 : i32>)
        %56 = air.channel.get async [%55]  @Gp2L2[%c3, %c0_7] (%results_10[%c0_7] [%c4096] [%c1_5]) {id = 107 : i32} : (memref<256x64xbf16, 1 : i32>)
        %57 = air.channel.get async [%55]  @Gp2L2[%c3, %c1_5] (%results_10[%c4096] [%c4096] [%c1_5]) {id = 108 : i32} : (memref<256x64xbf16, 1 : i32>)
        %58 = air.channel.get async [%55]  @Gp2L2[%c3, %c2_3] (%results_10[%c8192_2] [%c4096] [%c1_5]) {id = 109 : i32} : (memref<256x64xbf16, 1 : i32>)
        %59 = air.channel.get async [%55]  @Gp2L2[%c3, %c3] (%results_10[%c12288] [%c4096] [%c1_5]) {id = 110 : i32} : (memref<256x64xbf16, 1 : i32>)
        %60 = air.channel.put async [%55, %56, %57, %58, %59]  @GpOut[%arg12] (%results_10[] [] []) {id = 111 : i32} : (memref<256x64xbf16, 1 : i32>)
        %async_token_11 = air.execute [%38] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_12 = air.execute [%60] {
          memref.dealloc %results_10 : memref<256x64xbf16, 1 : i32>
        }
        air.wait_all [%39, %40, %async_token_11, %async_token_12]  {air.segment_end}
      }
      %25 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %5] [%c256, %c64] [%c512, %c1_1]) {id = 112 : i32, metadataArray = [{base = "air_GpOut_0_0", index = 0 : i32}, {base = "air_GpOut_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %26 = air.channel.get async [%24, %25]  @GpOut[%c0] (%arg11[%1, %7] [%c256, %c64] [%c512, %c1_1]) {id = 113 : i32, metadataArray = [{base = "air_GpOut_0_0", index = 0 : i32}, {base = "air_GpOut_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %27 = air.channel.get async [%24, %26]  @GpOut[%c0] (%arg11[%1, %9] [%c256, %c64] [%c512, %c1_1]) {id = 114 : i32, metadataArray = [{base = "air_GpOut_0_0", index = 0 : i32}, {base = "air_GpOut_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %28 = air.channel.get async [%24, %27]  @GpOut[%c0] (%arg11[%1, %11] [%c256, %c64] [%c512, %c1_1]) {id = 115 : i32, metadataArray = [{base = "air_GpOut_0_0", index = 0 : i32}, {base = "air_GpOut_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %29 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %16] [%c256, %c64] [%c512, %c1_1]) {id = 116 : i32, metadataArray = [{base = "air_GpOut_0_0", index = 0 : i32}, {base = "air_GpOut_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %30 = air.channel.get async [%24, %29]  @GpOut[%c1_1] (%arg11[%1, %18] [%c256, %c64] [%c512, %c1_1]) {id = 117 : i32, metadataArray = [{base = "air_GpOut_0_0", index = 0 : i32}, {base = "air_GpOut_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %31 = air.channel.get async [%24, %30]  @GpOut[%c1_1] (%arg11[%1, %20] [%c256, %c64] [%c512, %c1_1]) {id = 118 : i32, metadataArray = [{base = "air_GpOut_0_0", index = 0 : i32}, {base = "air_GpOut_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
      %32 = air.channel.get async [%24, %31]  @GpOut[%c1_1] (%arg11[%1, %22] [%c256, %c64] [%c512, %c1_1]) {id = 119 : i32, metadataArray = [{base = "air_GpOut_0_0", index = 0 : i32}, {base = "air_GpOut_1_0_0", index = 1 : i32}]} : (memref<512x512xbf16>)
    }
    return
  }
}
