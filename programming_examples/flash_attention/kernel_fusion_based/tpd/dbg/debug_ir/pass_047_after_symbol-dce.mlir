#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @attn_seg {
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
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
    %lock_0_2_0 = aie.lock(%tile_0_2, 6) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 7) {init = 1 : i32}
    %lock_1_2_7 = aie.lock(%tile_1_2, 6) {init = 0 : i32}
    %lock_1_2_8 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_9 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_10 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_11 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_12 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_13 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 7) {init = 1 : i32}
    %lock_2_2_14 = aie.lock(%tile_2_2, 6) {init = 0 : i32}
    %lock_2_2_15 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_16 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_17 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_18 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_19 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_20 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 7) {init = 1 : i32}
    %lock_3_2_21 = aie.lock(%tile_3_2, 6) {init = 0 : i32}
    %lock_3_2_22 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_23 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_24 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_25 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_26 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_27 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 7) {init = 1 : i32}
    %lock_0_3_28 = aie.lock(%tile_0_3, 6) {init = 0 : i32}
    %lock_0_3_29 = aie.lock(%tile_0_3, 5) {init = 1 : i32}
    %lock_0_3_30 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_31 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_32 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_33 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_34 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 7) {init = 1 : i32}
    %lock_1_3_35 = aie.lock(%tile_1_3, 6) {init = 0 : i32}
    %lock_1_3_36 = aie.lock(%tile_1_3, 5) {init = 1 : i32}
    %lock_1_3_37 = aie.lock(%tile_1_3, 4) {init = 0 : i32}
    %lock_1_3_38 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_39 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_40 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_41 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 7) {init = 1 : i32}
    %lock_2_3_42 = aie.lock(%tile_2_3, 6) {init = 0 : i32}
    %lock_2_3_43 = aie.lock(%tile_2_3, 5) {init = 1 : i32}
    %lock_2_3_44 = aie.lock(%tile_2_3, 4) {init = 0 : i32}
    %lock_2_3_45 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_46 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_47 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_48 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 7) {init = 1 : i32}
    %lock_3_3_49 = aie.lock(%tile_3_3, 6) {init = 0 : i32}
    %lock_3_3_50 = aie.lock(%tile_3_3, 5) {init = 1 : i32}
    %lock_3_3_51 = aie.lock(%tile_3_3, 4) {init = 0 : i32}
    %lock_3_3_52 = aie.lock(%tile_3_3, 3) {init = 1 : i32}
    %lock_3_3_53 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_54 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_55 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 7) {init = 1 : i32}
    %lock_0_4_56 = aie.lock(%tile_0_4, 6) {init = 0 : i32}
    %lock_0_4_57 = aie.lock(%tile_0_4, 5) {init = 1 : i32}
    %lock_0_4_58 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_59 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_60 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_61 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_62 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 7) {init = 1 : i32}
    %lock_1_4_63 = aie.lock(%tile_1_4, 6) {init = 0 : i32}
    %lock_1_4_64 = aie.lock(%tile_1_4, 5) {init = 1 : i32}
    %lock_1_4_65 = aie.lock(%tile_1_4, 4) {init = 0 : i32}
    %lock_1_4_66 = aie.lock(%tile_1_4, 3) {init = 1 : i32}
    %lock_1_4_67 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_68 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_69 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 7) {init = 1 : i32}
    %lock_2_4_70 = aie.lock(%tile_2_4, 6) {init = 0 : i32}
    %lock_2_4_71 = aie.lock(%tile_2_4, 5) {init = 1 : i32}
    %lock_2_4_72 = aie.lock(%tile_2_4, 4) {init = 0 : i32}
    %lock_2_4_73 = aie.lock(%tile_2_4, 3) {init = 1 : i32}
    %lock_2_4_74 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_75 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_76 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 7) {init = 1 : i32}
    %lock_3_4_77 = aie.lock(%tile_3_4, 6) {init = 0 : i32}
    %lock_3_4_78 = aie.lock(%tile_3_4, 5) {init = 1 : i32}
    %lock_3_4_79 = aie.lock(%tile_3_4, 4) {init = 0 : i32}
    %lock_3_4_80 = aie.lock(%tile_3_4, 3) {init = 1 : i32}
    %lock_3_4_81 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_82 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_83 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 7) {init = 1 : i32}
    %lock_0_5_84 = aie.lock(%tile_0_5, 6) {init = 0 : i32}
    %lock_0_5_85 = aie.lock(%tile_0_5, 5) {init = 1 : i32}
    %lock_0_5_86 = aie.lock(%tile_0_5, 4) {init = 0 : i32}
    %lock_0_5_87 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_88 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_89 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_90 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 7) {init = 1 : i32}
    %lock_1_5_91 = aie.lock(%tile_1_5, 6) {init = 0 : i32}
    %lock_1_5_92 = aie.lock(%tile_1_5, 5) {init = 1 : i32}
    %lock_1_5_93 = aie.lock(%tile_1_5, 4) {init = 0 : i32}
    %lock_1_5_94 = aie.lock(%tile_1_5, 3) {init = 1 : i32}
    %lock_1_5_95 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_96 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_97 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 7) {init = 1 : i32}
    %lock_2_5_98 = aie.lock(%tile_2_5, 6) {init = 0 : i32}
    %lock_2_5_99 = aie.lock(%tile_2_5, 5) {init = 1 : i32}
    %lock_2_5_100 = aie.lock(%tile_2_5, 4) {init = 0 : i32}
    %lock_2_5_101 = aie.lock(%tile_2_5, 3) {init = 1 : i32}
    %lock_2_5_102 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_103 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_104 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 7) {init = 1 : i32}
    %lock_3_5_105 = aie.lock(%tile_3_5, 6) {init = 0 : i32}
    %lock_3_5_106 = aie.lock(%tile_3_5, 5) {init = 1 : i32}
    %lock_3_5_107 = aie.lock(%tile_3_5, 4) {init = 0 : i32}
    %lock_3_5_108 = aie.lock(%tile_3_5, 3) {init = 1 : i32}
    %lock_3_5_109 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_110 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_111 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf159_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf159_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf158_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf158_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf157_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf157_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf156_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf156_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf155_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf155_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf154_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf154_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf153_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf153_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf152_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf152_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf151_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf151_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf150_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf150_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf149_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf149_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf148_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf148_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf147_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf147_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf146_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf146_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf145_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf145_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf144_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf144_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf143_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf143_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf142_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf142_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf141_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf141_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf140_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf140_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf139_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf139_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf138_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf138_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf137_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf137_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf136_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf136_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf135_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf135_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf134_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf134_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf133_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf133_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf132_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf132_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf131_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf131_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf130_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf130_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf129_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf129_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf128_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf128_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf127_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf127_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf126_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf126_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf125_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf125_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf124_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf124_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf123_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf123_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf122_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf122_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf121_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf121_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf120_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf120_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf119_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf119_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf118_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf118_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf117_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf117_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf116_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf116_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf115_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf115_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf114_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf114_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf113_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf113_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf112_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf112_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf111_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf111_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf110_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf110_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf109_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf109_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf108_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf108_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf107_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf107_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf106_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf106_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf105_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf105_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf104_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf104_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf103_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf103_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf102_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf102_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf101_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf101_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf100_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf100_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf99_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf99_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf98_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf98_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf97_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf97_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf96_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf96_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf95_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf95_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf94_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf94_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf93_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf93_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf92_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf92_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf91_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf91_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf90_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf90_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf89_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf89_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf88_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf88_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf87_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf87_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf86_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf86_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf85_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf85_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf84_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf84_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf83_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf83_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf82_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf82_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf81_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf81_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf80_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf80_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf79_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf79_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf78_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf78_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf77_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf77_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf76_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf76_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf75_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf75_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf74_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf74_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf73_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf73_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf72_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf72_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf71_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf71_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf70_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf70_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf69_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf69_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf68_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf68_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf67_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf67_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf66_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf66_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf65_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf65_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf64_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf64_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf63_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf63_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf62_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf62_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf61_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf61_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf60_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf60_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf59_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf59_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf58_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf58_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf57_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf57_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf56_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf56_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf55_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf55_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf54_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf54_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf53_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf53_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf52_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf52_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf51_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf51_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf50_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf50_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf49_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf49_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf48_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf48_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf47_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf47_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf46_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf46_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf45_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf45_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf44_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf44_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf43_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf43_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf42_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf42_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf41_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf41_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf40_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf40_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf39_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf39_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf38_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf38_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf37_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf37_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf36_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf36_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf35_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf35_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf34_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf34_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf33_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf33_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf32_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf32_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf31_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf31_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf30_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf30_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf29_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf29_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf28_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf28_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf27_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf27_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf26_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf26_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf25_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf25_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf24_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf24_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf23_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf23_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf22_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf22_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf21_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf21_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf20_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf20_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf19_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf19_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf18_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf18_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf17_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf17_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf16_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf16_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf15_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf15_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf14_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf14_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf13_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf13_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf12_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf12_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf11_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf11_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf10_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf10_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf9_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf9_unroll_0"} : memref<3xi32, 2 : i32> 
    %buf8_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf8_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf7_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf7_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf6_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf6_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf5_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf5_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf4_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf4_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf3_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf3_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf2_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf2_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf1_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf1_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf0_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf0_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer_unroll_0 = aie.external_buffer {sym_name = "__air_external_buffer_unroll_0"} : memref<512x128xbf16>
    %__air_external_buffer_1_unroll_0 = aie.external_buffer {sym_name = "__air_external_buffer_1_unroll_0"} : memref<512x128xbf16>
    %__air_external_buffer_2_unroll_0 = aie.external_buffer {sym_name = "__air_external_buffer_2_unroll_0"} : memref<512x512xbf16>
    %__air_external_buffer_3_unroll_0 = aie.external_buffer {sym_name = "__air_external_buffer_3_unroll_0"} : memref<512x512xbf16>
    %mem_3_5 = aie.mem(%tile_3_5) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf156_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_110, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_5_108, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_109, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_5_108, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_109, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_5_108, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_109, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_5_108, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_109, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_5_106, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf153_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_5_107, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf155_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_105, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_110, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf156_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf158_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf157_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf159_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf159_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf159_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf159_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_5_109, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_5_108, Release, %c1_i32)
      aie.use_lock(%lock_3_5_109, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_5_108, Release, %c1_i32)
      aie.use_lock(%lock_3_5_109, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_5_108, Release, %c1_i32)
      aie.use_lock(%lock_3_5_109, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf155_unroll_0, %buf154_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_108, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf152_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_105, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf154_unroll_0, %buf155_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5, Release, %c1_i32)
        aie.use_lock(%lock_3_5_107, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf159_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c3_i32 : i32
        func.call @apply_causal_mask(%buf152_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf157_unroll_0, %buf151_unroll_0, %buf150_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf150_unroll_0, %buf156_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf153_unroll_0, %buf156_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf158_unroll_0, %buf150_unroll_0, %buf151_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf151_unroll_0, %buf158_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_106, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf158_unroll_0, %buf156_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf159_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf159_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf159_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf159_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf159_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_5_111, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_104, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf146_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_103, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_5_101, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_102, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_5_101, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_102, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_5_101, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_102, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_5_101, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_102, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_5_99, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf143_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_5_100, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf145_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_98, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_103, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf146_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf148_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf147_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf149_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf149_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf149_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf149_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_5_102, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_5_101, Release, %c1_i32)
      aie.use_lock(%lock_2_5_102, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_5_101, Release, %c1_i32)
      aie.use_lock(%lock_2_5_102, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf145_unroll_0, %buf144_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_101, Release, %c1_i32)
      aie.use_lock(%lock_2_5_102, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_5_101, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf142_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_98, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf144_unroll_0, %buf145_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5, Release, %c1_i32)
        aie.use_lock(%lock_2_5_100, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf149_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c2_i32 : i32
        func.call @apply_causal_mask(%buf142_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf147_unroll_0, %buf141_unroll_0, %buf140_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf140_unroll_0, %buf146_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf143_unroll_0, %buf146_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf148_unroll_0, %buf140_unroll_0, %buf141_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf141_unroll_0, %buf148_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_99, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf148_unroll_0, %buf146_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf149_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf149_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf149_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf149_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf149_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_5_104, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_97, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf136_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_96, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_5_94, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_95, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_5_94, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_95, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_5_94, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_95, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_5_94, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_95, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_5_92, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf133_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_5_93, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf135_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_91, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_96, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf136_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf138_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf137_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf139_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf139_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf139_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf139_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_5_95, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_5_94, Release, %c1_i32)
      aie.use_lock(%lock_1_5_95, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf135_unroll_0, %buf134_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_94, Release, %c1_i32)
      aie.use_lock(%lock_1_5_95, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_5_94, Release, %c1_i32)
      aie.use_lock(%lock_1_5_95, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_5_94, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf132_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_91, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf134_unroll_0, %buf135_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5, Release, %c1_i32)
        aie.use_lock(%lock_1_5_93, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf139_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c1_i32 : i32
        func.call @apply_causal_mask(%buf132_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf137_unroll_0, %buf131_unroll_0, %buf130_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf130_unroll_0, %buf136_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf133_unroll_0, %buf136_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf138_unroll_0, %buf130_unroll_0, %buf131_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf131_unroll_0, %buf138_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_92, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf138_unroll_0, %buf136_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf139_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf139_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf139_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf139_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf139_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_5_97, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_90, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf126_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_89, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_5_87, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_88, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_5_87, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_88, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_5_87, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_88, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_5_87, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_88, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_5_85, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf123_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_5_86, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf125_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_84, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_89, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf126_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf128_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf127_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf129_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf129_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf129_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf129_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_5_88, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf125_unroll_0, %buf124_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_87, Release, %c1_i32)
      aie.use_lock(%lock_0_5_88, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_5_87, Release, %c1_i32)
      aie.use_lock(%lock_0_5_88, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_5_87, Release, %c1_i32)
      aie.use_lock(%lock_0_5_88, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_5_87, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf122_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_84, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf124_unroll_0, %buf125_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5, Release, %c1_i32)
        aie.use_lock(%lock_0_5_86, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf129_unroll_0[%c0] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf122_unroll_0, %6, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf127_unroll_0, %buf121_unroll_0, %buf120_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf120_unroll_0, %buf126_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf123_unroll_0, %buf126_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf128_unroll_0, %buf120_unroll_0, %buf121_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf121_unroll_0, %buf128_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_85, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf128_unroll_0, %buf126_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf129_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf129_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf129_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf129_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf129_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_5_90, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_83, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf116_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_82, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_4_80, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_81, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_4_80, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_81, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_4_80, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_81, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_4_80, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_81, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_4_78, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf113_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_4_79, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf115_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_77, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_82, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf116_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf118_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf117_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf119_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf119_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf119_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf119_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_4_81, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_4_80, Release, %c1_i32)
      aie.use_lock(%lock_3_4_81, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_4_80, Release, %c1_i32)
      aie.use_lock(%lock_3_4_81, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_4_80, Release, %c1_i32)
      aie.use_lock(%lock_3_4_81, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf115_unroll_0, %buf114_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_80, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf112_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_77, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf114_unroll_0, %buf115_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4, Release, %c1_i32)
        aie.use_lock(%lock_3_4_79, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf119_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c3_i32 : i32
        func.call @apply_causal_mask(%buf112_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf117_unroll_0, %buf111_unroll_0, %buf110_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf110_unroll_0, %buf116_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf113_unroll_0, %buf116_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf118_unroll_0, %buf110_unroll_0, %buf111_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf111_unroll_0, %buf118_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_78, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf118_unroll_0, %buf116_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf119_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf119_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf119_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf119_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf119_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_4_83, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf106_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_75, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_4_73, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_74, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_4_73, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_74, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_4_73, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_74, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_4_73, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_74, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_4_71, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf103_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_4_72, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf105_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_70, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_75, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf106_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf108_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf107_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf109_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf109_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf109_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf109_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_4_74, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_4_73, Release, %c1_i32)
      aie.use_lock(%lock_2_4_74, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_4_73, Release, %c1_i32)
      aie.use_lock(%lock_2_4_74, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf105_unroll_0, %buf104_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_73, Release, %c1_i32)
      aie.use_lock(%lock_2_4_74, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_4_73, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf102_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_70, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf104_unroll_0, %buf105_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4, Release, %c1_i32)
        aie.use_lock(%lock_2_4_72, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf109_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c2_i32 : i32
        func.call @apply_causal_mask(%buf102_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf107_unroll_0, %buf101_unroll_0, %buf100_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf100_unroll_0, %buf106_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf103_unroll_0, %buf106_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf108_unroll_0, %buf100_unroll_0, %buf101_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf101_unroll_0, %buf108_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_71, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf108_unroll_0, %buf106_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf109_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf109_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf109_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf109_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf109_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_4_76, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_69, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf96_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_68, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_4_66, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_67, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_4_66, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_67, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_4_66, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_67, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_4_66, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_67, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_4_64, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf93_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_4_65, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf95_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_63, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_68, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf96_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf98_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf97_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf99_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf99_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf99_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf99_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_4_67, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_4_66, Release, %c1_i32)
      aie.use_lock(%lock_1_4_67, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf95_unroll_0, %buf94_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_66, Release, %c1_i32)
      aie.use_lock(%lock_1_4_67, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_4_66, Release, %c1_i32)
      aie.use_lock(%lock_1_4_67, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_4_66, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf92_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_63, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf94_unroll_0, %buf95_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4, Release, %c1_i32)
        aie.use_lock(%lock_1_4_65, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf99_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c1_i32 : i32
        func.call @apply_causal_mask(%buf92_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf97_unroll_0, %buf91_unroll_0, %buf90_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf90_unroll_0, %buf96_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf93_unroll_0, %buf96_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf98_unroll_0, %buf90_unroll_0, %buf91_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf91_unroll_0, %buf98_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_64, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf98_unroll_0, %buf96_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf99_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf99_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf99_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf99_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf99_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_4_69, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_62, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf86_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_61, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_4_59, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_60, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_4_59, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_60, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_4_59, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_60, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_4_59, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_60, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_4_57, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf83_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_4_58, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf85_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_56, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_61, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf86_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf88_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf87_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf89_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf89_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf89_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf89_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_4_60, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf85_unroll_0, %buf84_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_59, Release, %c1_i32)
      aie.use_lock(%lock_0_4_60, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_59, Release, %c1_i32)
      aie.use_lock(%lock_0_4_60, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_59, Release, %c1_i32)
      aie.use_lock(%lock_0_4_60, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_59, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf82_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_56, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf84_unroll_0, %buf85_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4, Release, %c1_i32)
        aie.use_lock(%lock_0_4_58, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf89_unroll_0[%c0] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf82_unroll_0, %6, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf87_unroll_0, %buf81_unroll_0, %buf80_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf80_unroll_0, %buf86_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf83_unroll_0, %buf86_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf88_unroll_0, %buf80_unroll_0, %buf81_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf81_unroll_0, %buf88_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_57, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf88_unroll_0, %buf86_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf89_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf89_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf89_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf89_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf89_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_4_62, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_55, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf76_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_54, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_3_52, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_53, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_3_52, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_53, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_3_52, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_53, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_3_52, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_53, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_3_50, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf73_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_3_51, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf75_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_49, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_54, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf76_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf78_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf77_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf79_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf79_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf79_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf79_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_3_53, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_3_52, Release, %c1_i32)
      aie.use_lock(%lock_3_3_53, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_3_52, Release, %c1_i32)
      aie.use_lock(%lock_3_3_53, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_3_52, Release, %c1_i32)
      aie.use_lock(%lock_3_3_53, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf75_unroll_0, %buf74_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_52, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf72_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf74_unroll_0, %buf75_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3, Release, %c1_i32)
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf79_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c3_i32 : i32
        func.call @apply_causal_mask(%buf72_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf77_unroll_0, %buf71_unroll_0, %buf70_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf70_unroll_0, %buf76_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf73_unroll_0, %buf76_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf78_unroll_0, %buf70_unroll_0, %buf71_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf71_unroll_0, %buf78_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_50, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf78_unroll_0, %buf76_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf79_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf79_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf79_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf79_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf79_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_3_55, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf66_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_47, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_3_45, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_46, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_3_45, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_46, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_3_45, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_46, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_3_45, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_46, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_3_43, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf63_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_3_44, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf65_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_42, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf66_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf68_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf67_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf69_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf69_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf69_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf69_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_3_45, Release, %c1_i32)
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_3_45, Release, %c1_i32)
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf65_unroll_0, %buf64_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_45, Release, %c1_i32)
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_3_45, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf62_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_42, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf64_unroll_0, %buf65_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3, Release, %c1_i32)
        aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf69_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c2_i32 : i32
        func.call @apply_causal_mask(%buf62_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf67_unroll_0, %buf61_unroll_0, %buf60_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf60_unroll_0, %buf66_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf63_unroll_0, %buf66_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf68_unroll_0, %buf60_unroll_0, %buf61_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf61_unroll_0, %buf68_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_43, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf68_unroll_0, %buf66_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf69_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf69_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf69_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf69_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf69_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_3_48, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf56_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_40, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_3_38, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_39, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_3_38, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_39, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_3_38, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_39, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_3_38, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_39, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_3_36, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf53_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_3_37, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf55_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_35, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf56_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf58_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf57_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf59_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf59_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf59_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf59_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_3_39, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_3_38, Release, %c1_i32)
      aie.use_lock(%lock_1_3_39, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf55_unroll_0, %buf54_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_38, Release, %c1_i32)
      aie.use_lock(%lock_1_3_39, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_3_38, Release, %c1_i32)
      aie.use_lock(%lock_1_3_39, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_3_38, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf52_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_35, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf54_unroll_0, %buf55_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3, Release, %c1_i32)
        aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf59_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c1_i32 : i32
        func.call @apply_causal_mask(%buf52_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf57_unroll_0, %buf51_unroll_0, %buf50_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf50_unroll_0, %buf56_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf53_unroll_0, %buf56_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf58_unroll_0, %buf50_unroll_0, %buf51_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf51_unroll_0, %buf58_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_36, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf58_unroll_0, %buf56_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf59_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf59_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf59_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf59_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf59_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_3_41, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf46_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_33, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_3_31, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_32, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_3_31, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_32, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_3_31, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_32, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_3_31, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_32, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_3_29, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf43_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_3_30, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf45_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_28, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf46_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf48_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf47_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf49_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf49_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf49_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf49_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_3_32, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf45_unroll_0, %buf44_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_31, Release, %c1_i32)
      aie.use_lock(%lock_0_3_32, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_31, Release, %c1_i32)
      aie.use_lock(%lock_0_3_32, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_31, Release, %c1_i32)
      aie.use_lock(%lock_0_3_32, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_31, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf42_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_28, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf44_unroll_0, %buf45_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3, Release, %c1_i32)
        aie.use_lock(%lock_0_3_30, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf49_unroll_0[%c0] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf42_unroll_0, %6, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf47_unroll_0, %buf41_unroll_0, %buf40_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf40_unroll_0, %buf46_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf43_unroll_0, %buf46_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf48_unroll_0, %buf40_unroll_0, %buf41_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf41_unroll_0, %buf48_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_29, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf48_unroll_0, %buf46_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf49_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf49_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf49_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf49_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf49_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_3_34, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf36_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_26, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_3_2_24, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_25, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_2_24, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_25, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_2_24, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_25, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_2_24, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_25, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_3_2_22, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf33_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_3_2_23, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf35_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_21, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf36_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf38_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf37_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf39_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf39_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf39_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf39_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_2_24, Release, %c1_i32)
      aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_2_24, Release, %c1_i32)
      aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_3_2_24, Release, %c1_i32)
      aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf35_unroll_0, %buf34_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_24, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf32_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_21, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf34_unroll_0, %buf35_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2, Release, %c1_i32)
        aie.use_lock(%lock_3_2_23, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf39_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c3_i32 : i32
        func.call @apply_causal_mask(%buf32_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf37_unroll_0, %buf31_unroll_0, %buf30_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf30_unroll_0, %buf36_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf33_unroll_0, %buf36_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf38_unroll_0, %buf30_unroll_0, %buf31_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf31_unroll_0, %buf38_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_22, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf38_unroll_0, %buf36_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf39_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf39_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf39_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf39_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf39_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_3_2_27, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf26_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_19, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_2_17, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_18, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_2_17, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_18, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_2_17, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_18, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_2_17, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_18, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_2_2_15, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf23_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_2_2_16, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_14, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf26_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf28_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf27_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf29_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf29_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf29_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf29_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_2_18, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_2_17, Release, %c1_i32)
      aie.use_lock(%lock_2_2_18, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_2_17, Release, %c1_i32)
      aie.use_lock(%lock_2_2_18, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf25_unroll_0, %buf24_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_17, Release, %c1_i32)
      aie.use_lock(%lock_2_2_18, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_2_2_17, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf22_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_14, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf24_unroll_0, %buf25_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2, Release, %c1_i32)
        aie.use_lock(%lock_2_2_16, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf29_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c2_i32 : i32
        func.call @apply_causal_mask(%buf22_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf27_unroll_0, %buf21_unroll_0, %buf20_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf20_unroll_0, %buf26_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf23_unroll_0, %buf26_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf28_unroll_0, %buf20_unroll_0, %buf21_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf21_unroll_0, %buf28_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_15, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf28_unroll_0, %buf26_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf29_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf29_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf29_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf29_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf29_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_2_2_20, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf16_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_12, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_2_10, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_11, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_2_10, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_11, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_2_10, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_11, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_2_10, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_11, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_1_2_8, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf13_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_1_2_9, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_7, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_12, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf16_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf18_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf17_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf19_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf19_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf19_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf19_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_2_11, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_2_10, Release, %c1_i32)
      aie.use_lock(%lock_1_2_11, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf15_unroll_0, %buf14_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_10, Release, %c1_i32)
      aie.use_lock(%lock_1_2_11, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_2_10, Release, %c1_i32)
      aie.use_lock(%lock_1_2_11, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_1_2_10, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf12_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_7, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf14_unroll_0, %buf15_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2, Release, %c1_i32)
        aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf19_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c1_i32 : i32
        func.call @apply_causal_mask(%buf12_unroll_0, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf17_unroll_0, %buf11_unroll_0, %buf10_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf10_unroll_0, %buf16_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf13_unroll_0, %buf16_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf18_unroll_0, %buf10_unroll_0, %buf11_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf11_unroll_0, %buf18_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_8, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf18_unroll_0, %buf16_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf19_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf19_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf19_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf19_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf19_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_1_2_13, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf6_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_5, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf3_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5_unroll_0 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf6_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf8_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf7_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf9_unroll_0[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf9_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf9_unroll_0[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf9_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf5_unroll_0, %buf4_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf2_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf4_unroll_0, %buf5_unroll_0, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2, Release, %c1_i32)
        aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf9_unroll_0[%c0] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf2_unroll_0, %6, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf7_unroll_0, %buf1_unroll_0, %buf0_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf0_unroll_0, %buf6_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf3_unroll_0, %buf6_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf8_unroll_0, %buf0_unroll_0, %buf1_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf1_unroll_0, %buf8_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_1, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf8_unroll_0, %buf6_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf9_unroll_0[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf9_unroll_0[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf9_unroll_0[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf9_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf9_unroll_0[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_2_6, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 1) {init = 4 : i32}
    %lock_0_1_112 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 1) {init = 4 : i32}
    %lock_2_1_113 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 1) {init = 2 : i32}
    %lock_1_1_114 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %buf162_unroll_0 = aie.buffer(%mem_tile_1_1) {sym_name = "buf162_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf161_unroll_0 = aie.buffer(%mem_tile_0_1) {sym_name = "buf161_unroll_0"} : memref<256x64xbf16, 1 : i32> 
    %buf160_unroll_0 = aie.buffer(%mem_tile_2_1) {sym_name = "buf160_unroll_0"} : memref<64x64xbf16, 1 : i32> 
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
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_1_1, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 3, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 3, %tile_2_5, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 3, %tile_3_5, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 4, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 4, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 4, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 4, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 5, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 5, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 5, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 5, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_3_5, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 3, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 3, %tile_2_5, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 3, %tile_3_5, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_0_1, DMA : 3)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 4)
    aie.flow(%tile_1_3, DMA : 0, %mem_tile_0_1, DMA : 5)
    aie.flow(%tile_2_3, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_3_3, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_1_4, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_2_4, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_3_4, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_1_5, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_2_5, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_3_5, DMA : 0, %mem_tile_0_1, DMA : 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_114, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb15
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_114, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1_114, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_114, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(MM2S, 4, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1_114, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, %c1_i32)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(MM2S, 5, ^bb12, ^bb13)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_1_1_114, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf162_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, %c1_i32)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb11
      %6 = aie.dma_start(S2MM, 0, ^bb14, ^bb15)
    ^bb14:  // 2 preds: ^bb13, ^bb14
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, %c2_i32)
      aie.dma_bd(%buf162_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_114, Release, %c2_i32)
      aie.next_bd ^bb14
    ^bb15:  // pred: ^bb13
      %7 = aie.dma_start(S2MM, 1, ^bb16, ^bb2)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, %c2_i32)
      aie.dma_bd(%buf162_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_114, Release, %c2_i32)
      aie.next_bd ^bb16
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_113, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf160_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_113, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf160_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1_113, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf160_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_113, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf160_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%buf160_unroll_0 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_113, Release, %c4_i32)
      aie.next_bd ^bb10
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_112, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 0 len = 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, %c4_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb23
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb15)
    ^bb4:  // 2 preds: ^bb3, ^bb14
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb15:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb16, ^bb17)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb16
    ^bb17:  // pred: ^bb15
      %3 = aie.dma_start(S2MM, 2, ^bb18, ^bb19)
    ^bb18:  // 2 preds: ^bb17, ^bb18
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb18
    ^bb19:  // pred: ^bb17
      %4 = aie.dma_start(S2MM, 3, ^bb20, ^bb21)
    ^bb20:  // 2 preds: ^bb19, ^bb20
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb20
    ^bb21:  // pred: ^bb19
      %5 = aie.dma_start(S2MM, 4, ^bb22, ^bb23)
    ^bb22:  // 2 preds: ^bb21, ^bb22
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb22
    ^bb23:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 5, ^bb24, ^bb2)
    ^bb24:  // 2 preds: ^bb23, ^bb24
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf161_unroll_0 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_112, Release, %c1_i32)
      aie.next_bd ^bb24
    }
    aie.shim_dma_allocation @air_GpOut_0_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_KIn_0_0(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_0_0(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_QIn_0_0(%shim_noc_tile_1_0, MM2S, 1)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %tile_4_3 = aie.tile(4, 3)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_3 = aie.tile(7, 3)
    %tile_4_4 = aie.tile(4, 4)
    %tile_5_4 = aie.tile(5, 4)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_4 = aie.tile(7, 4)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_5 = aie.tile(7, 5)
    %lock_4_2 = aie.lock(%tile_4_2, 7) {init = 1 : i32}
    %lock_4_2_115 = aie.lock(%tile_4_2, 6) {init = 0 : i32}
    %lock_4_2_116 = aie.lock(%tile_4_2, 5) {init = 1 : i32}
    %lock_4_2_117 = aie.lock(%tile_4_2, 4) {init = 0 : i32}
    %lock_4_2_118 = aie.lock(%tile_4_2, 3) {init = 1 : i32}
    %lock_4_2_119 = aie.lock(%tile_4_2, 2) {init = 0 : i32}
    %lock_4_2_120 = aie.lock(%tile_4_2, 1) {init = 1 : i32}
    %lock_4_2_121 = aie.lock(%tile_4_2, 0) {init = 0 : i32}
    %lock_5_2 = aie.lock(%tile_5_2, 7) {init = 1 : i32}
    %lock_5_2_122 = aie.lock(%tile_5_2, 6) {init = 0 : i32}
    %lock_5_2_123 = aie.lock(%tile_5_2, 5) {init = 1 : i32}
    %lock_5_2_124 = aie.lock(%tile_5_2, 4) {init = 0 : i32}
    %lock_5_2_125 = aie.lock(%tile_5_2, 3) {init = 1 : i32}
    %lock_5_2_126 = aie.lock(%tile_5_2, 2) {init = 0 : i32}
    %lock_5_2_127 = aie.lock(%tile_5_2, 1) {init = 1 : i32}
    %lock_5_2_128 = aie.lock(%tile_5_2, 0) {init = 0 : i32}
    %lock_6_2 = aie.lock(%tile_6_2, 7) {init = 1 : i32}
    %lock_6_2_129 = aie.lock(%tile_6_2, 6) {init = 0 : i32}
    %lock_6_2_130 = aie.lock(%tile_6_2, 5) {init = 1 : i32}
    %lock_6_2_131 = aie.lock(%tile_6_2, 4) {init = 0 : i32}
    %lock_6_2_132 = aie.lock(%tile_6_2, 3) {init = 1 : i32}
    %lock_6_2_133 = aie.lock(%tile_6_2, 2) {init = 0 : i32}
    %lock_6_2_134 = aie.lock(%tile_6_2, 1) {init = 1 : i32}
    %lock_6_2_135 = aie.lock(%tile_6_2, 0) {init = 0 : i32}
    %lock_7_2 = aie.lock(%tile_7_2, 7) {init = 1 : i32}
    %lock_7_2_136 = aie.lock(%tile_7_2, 6) {init = 0 : i32}
    %lock_7_2_137 = aie.lock(%tile_7_2, 5) {init = 1 : i32}
    %lock_7_2_138 = aie.lock(%tile_7_2, 4) {init = 0 : i32}
    %lock_7_2_139 = aie.lock(%tile_7_2, 3) {init = 1 : i32}
    %lock_7_2_140 = aie.lock(%tile_7_2, 2) {init = 0 : i32}
    %lock_7_2_141 = aie.lock(%tile_7_2, 1) {init = 1 : i32}
    %lock_7_2_142 = aie.lock(%tile_7_2, 0) {init = 0 : i32}
    %lock_4_3 = aie.lock(%tile_4_3, 7) {init = 1 : i32}
    %lock_4_3_143 = aie.lock(%tile_4_3, 6) {init = 0 : i32}
    %lock_4_3_144 = aie.lock(%tile_4_3, 5) {init = 1 : i32}
    %lock_4_3_145 = aie.lock(%tile_4_3, 4) {init = 0 : i32}
    %lock_4_3_146 = aie.lock(%tile_4_3, 3) {init = 1 : i32}
    %lock_4_3_147 = aie.lock(%tile_4_3, 2) {init = 0 : i32}
    %lock_4_3_148 = aie.lock(%tile_4_3, 1) {init = 1 : i32}
    %lock_4_3_149 = aie.lock(%tile_4_3, 0) {init = 0 : i32}
    %lock_5_3 = aie.lock(%tile_5_3, 7) {init = 1 : i32}
    %lock_5_3_150 = aie.lock(%tile_5_3, 6) {init = 0 : i32}
    %lock_5_3_151 = aie.lock(%tile_5_3, 5) {init = 1 : i32}
    %lock_5_3_152 = aie.lock(%tile_5_3, 4) {init = 0 : i32}
    %lock_5_3_153 = aie.lock(%tile_5_3, 3) {init = 1 : i32}
    %lock_5_3_154 = aie.lock(%tile_5_3, 2) {init = 0 : i32}
    %lock_5_3_155 = aie.lock(%tile_5_3, 1) {init = 1 : i32}
    %lock_5_3_156 = aie.lock(%tile_5_3, 0) {init = 0 : i32}
    %lock_6_3 = aie.lock(%tile_6_3, 7) {init = 1 : i32}
    %lock_6_3_157 = aie.lock(%tile_6_3, 6) {init = 0 : i32}
    %lock_6_3_158 = aie.lock(%tile_6_3, 5) {init = 1 : i32}
    %lock_6_3_159 = aie.lock(%tile_6_3, 4) {init = 0 : i32}
    %lock_6_3_160 = aie.lock(%tile_6_3, 3) {init = 1 : i32}
    %lock_6_3_161 = aie.lock(%tile_6_3, 2) {init = 0 : i32}
    %lock_6_3_162 = aie.lock(%tile_6_3, 1) {init = 1 : i32}
    %lock_6_3_163 = aie.lock(%tile_6_3, 0) {init = 0 : i32}
    %lock_7_3 = aie.lock(%tile_7_3, 7) {init = 1 : i32}
    %lock_7_3_164 = aie.lock(%tile_7_3, 6) {init = 0 : i32}
    %lock_7_3_165 = aie.lock(%tile_7_3, 5) {init = 1 : i32}
    %lock_7_3_166 = aie.lock(%tile_7_3, 4) {init = 0 : i32}
    %lock_7_3_167 = aie.lock(%tile_7_3, 3) {init = 1 : i32}
    %lock_7_3_168 = aie.lock(%tile_7_3, 2) {init = 0 : i32}
    %lock_7_3_169 = aie.lock(%tile_7_3, 1) {init = 1 : i32}
    %lock_7_3_170 = aie.lock(%tile_7_3, 0) {init = 0 : i32}
    %lock_4_4 = aie.lock(%tile_4_4, 7) {init = 1 : i32}
    %lock_4_4_171 = aie.lock(%tile_4_4, 6) {init = 0 : i32}
    %lock_4_4_172 = aie.lock(%tile_4_4, 5) {init = 1 : i32}
    %lock_4_4_173 = aie.lock(%tile_4_4, 4) {init = 0 : i32}
    %lock_4_4_174 = aie.lock(%tile_4_4, 3) {init = 1 : i32}
    %lock_4_4_175 = aie.lock(%tile_4_4, 2) {init = 0 : i32}
    %lock_4_4_176 = aie.lock(%tile_4_4, 1) {init = 1 : i32}
    %lock_4_4_177 = aie.lock(%tile_4_4, 0) {init = 0 : i32}
    %lock_5_4 = aie.lock(%tile_5_4, 7) {init = 1 : i32}
    %lock_5_4_178 = aie.lock(%tile_5_4, 6) {init = 0 : i32}
    %lock_5_4_179 = aie.lock(%tile_5_4, 5) {init = 1 : i32}
    %lock_5_4_180 = aie.lock(%tile_5_4, 4) {init = 0 : i32}
    %lock_5_4_181 = aie.lock(%tile_5_4, 3) {init = 1 : i32}
    %lock_5_4_182 = aie.lock(%tile_5_4, 2) {init = 0 : i32}
    %lock_5_4_183 = aie.lock(%tile_5_4, 1) {init = 1 : i32}
    %lock_5_4_184 = aie.lock(%tile_5_4, 0) {init = 0 : i32}
    %lock_6_4 = aie.lock(%tile_6_4, 7) {init = 1 : i32}
    %lock_6_4_185 = aie.lock(%tile_6_4, 6) {init = 0 : i32}
    %lock_6_4_186 = aie.lock(%tile_6_4, 5) {init = 1 : i32}
    %lock_6_4_187 = aie.lock(%tile_6_4, 4) {init = 0 : i32}
    %lock_6_4_188 = aie.lock(%tile_6_4, 3) {init = 1 : i32}
    %lock_6_4_189 = aie.lock(%tile_6_4, 2) {init = 0 : i32}
    %lock_6_4_190 = aie.lock(%tile_6_4, 1) {init = 1 : i32}
    %lock_6_4_191 = aie.lock(%tile_6_4, 0) {init = 0 : i32}
    %lock_7_4 = aie.lock(%tile_7_4, 7) {init = 1 : i32}
    %lock_7_4_192 = aie.lock(%tile_7_4, 6) {init = 0 : i32}
    %lock_7_4_193 = aie.lock(%tile_7_4, 5) {init = 1 : i32}
    %lock_7_4_194 = aie.lock(%tile_7_4, 4) {init = 0 : i32}
    %lock_7_4_195 = aie.lock(%tile_7_4, 3) {init = 1 : i32}
    %lock_7_4_196 = aie.lock(%tile_7_4, 2) {init = 0 : i32}
    %lock_7_4_197 = aie.lock(%tile_7_4, 1) {init = 1 : i32}
    %lock_7_4_198 = aie.lock(%tile_7_4, 0) {init = 0 : i32}
    %lock_4_5 = aie.lock(%tile_4_5, 7) {init = 1 : i32}
    %lock_4_5_199 = aie.lock(%tile_4_5, 6) {init = 0 : i32}
    %lock_4_5_200 = aie.lock(%tile_4_5, 5) {init = 1 : i32}
    %lock_4_5_201 = aie.lock(%tile_4_5, 4) {init = 0 : i32}
    %lock_4_5_202 = aie.lock(%tile_4_5, 3) {init = 1 : i32}
    %lock_4_5_203 = aie.lock(%tile_4_5, 2) {init = 0 : i32}
    %lock_4_5_204 = aie.lock(%tile_4_5, 1) {init = 1 : i32}
    %lock_4_5_205 = aie.lock(%tile_4_5, 0) {init = 0 : i32}
    %lock_5_5 = aie.lock(%tile_5_5, 7) {init = 1 : i32}
    %lock_5_5_206 = aie.lock(%tile_5_5, 6) {init = 0 : i32}
    %lock_5_5_207 = aie.lock(%tile_5_5, 5) {init = 1 : i32}
    %lock_5_5_208 = aie.lock(%tile_5_5, 4) {init = 0 : i32}
    %lock_5_5_209 = aie.lock(%tile_5_5, 3) {init = 1 : i32}
    %lock_5_5_210 = aie.lock(%tile_5_5, 2) {init = 0 : i32}
    %lock_5_5_211 = aie.lock(%tile_5_5, 1) {init = 1 : i32}
    %lock_5_5_212 = aie.lock(%tile_5_5, 0) {init = 0 : i32}
    %lock_6_5 = aie.lock(%tile_6_5, 7) {init = 1 : i32}
    %lock_6_5_213 = aie.lock(%tile_6_5, 6) {init = 0 : i32}
    %lock_6_5_214 = aie.lock(%tile_6_5, 5) {init = 1 : i32}
    %lock_6_5_215 = aie.lock(%tile_6_5, 4) {init = 0 : i32}
    %lock_6_5_216 = aie.lock(%tile_6_5, 3) {init = 1 : i32}
    %lock_6_5_217 = aie.lock(%tile_6_5, 2) {init = 0 : i32}
    %lock_6_5_218 = aie.lock(%tile_6_5, 1) {init = 1 : i32}
    %lock_6_5_219 = aie.lock(%tile_6_5, 0) {init = 0 : i32}
    %lock_7_5 = aie.lock(%tile_7_5, 7) {init = 1 : i32}
    %lock_7_5_220 = aie.lock(%tile_7_5, 6) {init = 0 : i32}
    %lock_7_5_221 = aie.lock(%tile_7_5, 5) {init = 1 : i32}
    %lock_7_5_222 = aie.lock(%tile_7_5, 4) {init = 0 : i32}
    %lock_7_5_223 = aie.lock(%tile_7_5, 3) {init = 1 : i32}
    %lock_7_5_224 = aie.lock(%tile_7_5, 2) {init = 0 : i32}
    %lock_7_5_225 = aie.lock(%tile_7_5, 1) {init = 1 : i32}
    %lock_7_5_226 = aie.lock(%tile_7_5, 0) {init = 0 : i32}
    %buf322_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf322_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf321_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf321_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf320_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf320_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf319_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf319_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf318_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf318_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf317_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf317_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf316_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf316_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf315_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf315_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf314_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf314_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf313_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf313_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf312_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf312_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf311_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf311_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf310_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf310_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf309_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf309_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf308_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf308_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf307_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf307_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf306_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf306_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf305_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf305_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf304_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf304_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf303_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf303_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf302_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf302_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf301_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf301_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf300_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf300_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf299_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf299_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf298_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf298_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf297_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf297_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf296_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf296_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf295_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf295_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf294_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf294_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf293_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf293_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf292_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf292_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf291_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf291_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf290_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf290_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf289_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf289_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf288_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf288_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf287_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf287_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf286_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf286_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf285_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf285_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf284_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf284_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf283_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf283_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf282_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf282_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf281_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf281_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf280_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf280_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf279_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf279_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf278_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf278_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf277_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf277_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf276_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf276_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf275_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf275_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf274_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf274_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf273_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf273_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf272_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf272_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf271_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf271_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf270_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf270_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf269_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf269_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf268_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf268_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf267_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf267_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf266_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf266_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf265_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf265_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf264_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf264_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf263_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf263_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf262_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf262_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf261_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf261_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf260_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf260_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf259_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf259_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf258_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf258_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf257_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf257_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf256_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf256_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf255_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf255_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf254_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf254_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf253_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf253_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf252_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf252_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf251_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf251_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf250_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf250_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf249_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf249_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf248_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf248_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf247_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf247_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf246_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf246_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf245_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf245_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf244_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf244_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf243_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf243_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf242_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf242_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf241_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf241_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf240_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf240_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf239_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf239_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf238_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf238_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf237_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf237_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf236_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf236_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf235_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf235_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf234_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf234_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf233_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf233_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf232_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf232_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf231_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf231_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf230_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf230_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf229_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf229_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf228_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf228_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf227_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf227_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf226_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf226_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf225_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf225_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf224_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf224_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf223_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf223_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf222_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf222_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf221_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf221_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf220_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf220_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf219_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf219_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf218_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf218_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf217_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf217_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf216_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf216_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf215_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf215_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf214_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf214_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf213_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf213_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf212_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf212_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf211_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf211_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf210_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf210_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf209_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf209_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf208_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf208_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf207_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf207_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf206_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf206_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf205_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf205_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf204_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf204_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf203_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf203_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf202_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf202_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf201_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf201_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf200_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf200_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf199_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf199_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf198_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf198_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf197_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf197_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf196_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf196_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf195_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf195_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf194_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf194_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf193_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf193_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf192_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf192_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf191_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf191_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf190_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf190_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf189_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf189_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf188_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf188_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf187_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf187_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf186_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf186_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf185_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf185_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf184_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf184_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf183_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf183_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf182_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf182_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf181_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf181_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf180_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf180_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf179_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf179_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf178_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf178_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf177_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf177_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf176_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf176_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf175_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf175_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf174_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf174_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf173_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf173_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf172_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf172_unroll_1"} : memref<3xi32, 2 : i32> 
    %buf171_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf171_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf170_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf170_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf169_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf169_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf168_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf168_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf167_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf167_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf166_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf166_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf165_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf165_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf164_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf164_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf163_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf163_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer_unroll_1 = aie.external_buffer {sym_name = "__air_external_buffer_unroll_1"} : memref<512x128xbf16>
    %__air_external_buffer_1_unroll_1 = aie.external_buffer {sym_name = "__air_external_buffer_1_unroll_1"} : memref<512x128xbf16>
    %__air_external_buffer_2_unroll_1 = aie.external_buffer {sym_name = "__air_external_buffer_2_unroll_1"} : memref<512x512xbf16>
    %__air_external_buffer_3_unroll_1 = aie.external_buffer {sym_name = "__air_external_buffer_3_unroll_1"} : memref<512x512xbf16>
    %mem_7_5 = aie.mem(%tile_7_5) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_5_226, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf319_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_5_225, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_7_5_223, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_5_224, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_7_5_223, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_5_224, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_7_5_223, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_5_224, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_7_5_223, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_5_224, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_7_5_221, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf316_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_7_5_222, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_7_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf318_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_5_220, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_7_5 = aie.core(%tile_7_5) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_5_225, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf319_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf321_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf320_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf322_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf322_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf322_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf322_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_7_5_224, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_5_223, Release, %c1_i32)
      aie.use_lock(%lock_7_5_224, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_5_223, Release, %c1_i32)
      aie.use_lock(%lock_7_5_224, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_5_223, Release, %c1_i32)
      aie.use_lock(%lock_7_5_224, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf318_unroll_1, %buf317_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_5_223, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf315_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_5_220, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf317_unroll_1, %buf318_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_5, Release, %c1_i32)
        aie.use_lock(%lock_7_5_222, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf322_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c3_i32 : i32
        func.call @apply_causal_mask(%buf315_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf320_unroll_1, %buf314_unroll_1, %buf313_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf313_unroll_1, %buf319_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf316_unroll_1, %buf319_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf321_unroll_1, %buf313_unroll_1, %buf314_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf314_unroll_1, %buf321_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_5_221, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf321_unroll_1, %buf319_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf322_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf322_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf322_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf322_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf322_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_7_5_226, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_6_5 = aie.mem(%tile_6_5) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_5_219, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf309_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_5_218, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_6_5_216, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_5_217, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_6_5_216, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_5_217, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_6_5_216, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_5_217, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_6_5_216, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_5_217, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_6_5_214, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf306_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_6_5_215, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_6_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf308_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_5_213, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_6_5 = aie.core(%tile_6_5) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_5_218, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf309_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf311_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf310_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf312_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf312_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf312_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf312_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_6_5_217, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_5_216, Release, %c1_i32)
      aie.use_lock(%lock_6_5_217, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_5_216, Release, %c1_i32)
      aie.use_lock(%lock_6_5_217, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf308_unroll_1, %buf307_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_5_216, Release, %c1_i32)
      aie.use_lock(%lock_6_5_217, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_5_216, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf305_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_5_213, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf307_unroll_1, %buf308_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_5, Release, %c1_i32)
        aie.use_lock(%lock_6_5_215, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf312_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c2_i32 : i32
        func.call @apply_causal_mask(%buf305_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf310_unroll_1, %buf304_unroll_1, %buf303_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf303_unroll_1, %buf309_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf306_unroll_1, %buf309_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf311_unroll_1, %buf303_unroll_1, %buf304_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf304_unroll_1, %buf311_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_5_214, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf311_unroll_1, %buf309_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf312_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf312_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf312_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf312_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf312_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_6_5_219, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_5_5 = aie.mem(%tile_5_5) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_5_212, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf299_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_5_211, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_5_5_209, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_5_210, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_5_5_209, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_5_210, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_5_5_209, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_5_210, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_5_5_209, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_5_210, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_5_5_207, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf296_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_5_5_208, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_5_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf298_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_5_206, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_5_5 = aie.core(%tile_5_5) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_5_211, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf299_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf301_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf300_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf302_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf302_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf302_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf302_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_5_5_210, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_5_209, Release, %c1_i32)
      aie.use_lock(%lock_5_5_210, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf298_unroll_1, %buf297_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_5_209, Release, %c1_i32)
      aie.use_lock(%lock_5_5_210, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_5_209, Release, %c1_i32)
      aie.use_lock(%lock_5_5_210, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_5_209, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf295_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_5_206, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf297_unroll_1, %buf298_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_5, Release, %c1_i32)
        aie.use_lock(%lock_5_5_208, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf302_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c1_i32 : i32
        func.call @apply_causal_mask(%buf295_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf300_unroll_1, %buf294_unroll_1, %buf293_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf293_unroll_1, %buf299_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf296_unroll_1, %buf299_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf301_unroll_1, %buf293_unroll_1, %buf294_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf294_unroll_1, %buf301_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_5_207, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf301_unroll_1, %buf299_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf302_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf302_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf302_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf302_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf302_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_5_5_212, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_4_5 = aie.mem(%tile_4_5) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_5_205, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf289_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_5_204, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_4_5_202, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_5_203, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_4_5_202, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_5_203, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_4_5_202, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_5_203, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_4_5_202, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_5_203, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_4_5_200, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf286_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_4_5_201, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_4_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf288_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_5_199, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_4_5 = aie.core(%tile_4_5) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_5_204, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf289_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf291_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf290_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf292_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf292_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf292_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf292_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_4_5_203, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf288_unroll_1, %buf287_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_5_202, Release, %c1_i32)
      aie.use_lock(%lock_4_5_203, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_5_202, Release, %c1_i32)
      aie.use_lock(%lock_4_5_203, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_5_202, Release, %c1_i32)
      aie.use_lock(%lock_4_5_203, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_5_202, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf285_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_5_199, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf287_unroll_1, %buf288_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_5, Release, %c1_i32)
        aie.use_lock(%lock_4_5_201, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf292_unroll_1[%c0] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf285_unroll_1, %6, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf290_unroll_1, %buf284_unroll_1, %buf283_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf283_unroll_1, %buf289_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf286_unroll_1, %buf289_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf291_unroll_1, %buf283_unroll_1, %buf284_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf284_unroll_1, %buf291_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_5_200, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf291_unroll_1, %buf289_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf292_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf292_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf292_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf292_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf292_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_4_5_205, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 3>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_7_4 = aie.mem(%tile_7_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_4_198, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf279_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_4_197, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_7_4_195, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_4_196, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_7_4_195, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_4_196, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_7_4_195, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_4_196, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_7_4_195, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_4_196, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_7_4_193, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf276_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_7_4_194, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_7_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf278_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_4_192, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_7_4 = aie.core(%tile_7_4) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_4_197, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf279_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf281_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf280_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf282_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf282_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf282_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf282_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_7_4_196, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_4_195, Release, %c1_i32)
      aie.use_lock(%lock_7_4_196, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_4_195, Release, %c1_i32)
      aie.use_lock(%lock_7_4_196, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_4_195, Release, %c1_i32)
      aie.use_lock(%lock_7_4_196, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf278_unroll_1, %buf277_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_4_195, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf275_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_4_192, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf277_unroll_1, %buf278_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_4, Release, %c1_i32)
        aie.use_lock(%lock_7_4_194, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf282_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c3_i32 : i32
        func.call @apply_causal_mask(%buf275_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf280_unroll_1, %buf274_unroll_1, %buf273_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf273_unroll_1, %buf279_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf276_unroll_1, %buf279_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf281_unroll_1, %buf273_unroll_1, %buf274_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf274_unroll_1, %buf281_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_4_193, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf281_unroll_1, %buf279_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf282_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf282_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf282_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf282_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf282_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_7_4_198, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_6_4 = aie.mem(%tile_6_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_4_191, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf269_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_4_190, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_6_4_188, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_4_189, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_6_4_188, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_4_189, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_6_4_188, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_4_189, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_6_4_188, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_4_189, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_6_4_186, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf266_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_6_4_187, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_6_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf268_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_4_185, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_6_4 = aie.core(%tile_6_4) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_4_190, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf269_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf271_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf270_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf272_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf272_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf272_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf272_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_6_4_189, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_4_188, Release, %c1_i32)
      aie.use_lock(%lock_6_4_189, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_4_188, Release, %c1_i32)
      aie.use_lock(%lock_6_4_189, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf268_unroll_1, %buf267_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_4_188, Release, %c1_i32)
      aie.use_lock(%lock_6_4_189, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_4_188, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf265_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_4_185, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf267_unroll_1, %buf268_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_4, Release, %c1_i32)
        aie.use_lock(%lock_6_4_187, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf272_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c2_i32 : i32
        func.call @apply_causal_mask(%buf265_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf270_unroll_1, %buf264_unroll_1, %buf263_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf263_unroll_1, %buf269_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf266_unroll_1, %buf269_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf271_unroll_1, %buf263_unroll_1, %buf264_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf264_unroll_1, %buf271_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_4_186, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf271_unroll_1, %buf269_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf272_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf272_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf272_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf272_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf272_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_6_4_191, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_5_4 = aie.mem(%tile_5_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_4_184, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf259_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_4_183, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_5_4_181, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_4_182, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_5_4_181, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_4_182, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_5_4_181, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_4_182, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_5_4_181, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_4_182, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_5_4_179, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf256_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_5_4_180, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_5_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf258_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_4_178, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_5_4 = aie.core(%tile_5_4) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_4_183, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf259_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf261_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf260_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf262_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf262_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf262_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf262_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_5_4_182, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_4_181, Release, %c1_i32)
      aie.use_lock(%lock_5_4_182, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf258_unroll_1, %buf257_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_4_181, Release, %c1_i32)
      aie.use_lock(%lock_5_4_182, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_4_181, Release, %c1_i32)
      aie.use_lock(%lock_5_4_182, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_4_181, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf255_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_4_178, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf257_unroll_1, %buf258_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_4, Release, %c1_i32)
        aie.use_lock(%lock_5_4_180, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf262_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c1_i32 : i32
        func.call @apply_causal_mask(%buf255_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf260_unroll_1, %buf254_unroll_1, %buf253_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf253_unroll_1, %buf259_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf256_unroll_1, %buf259_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf261_unroll_1, %buf253_unroll_1, %buf254_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf254_unroll_1, %buf261_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_4_179, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf261_unroll_1, %buf259_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf262_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf262_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf262_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf262_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf262_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_5_4_184, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_4_4 = aie.mem(%tile_4_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_4_177, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf249_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_4_176, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_4_4_174, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_4_175, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_4_4_174, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_4_175, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_4_4_174, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_4_175, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_4_4_174, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_4_175, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_4_4_172, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf246_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_4_4_173, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_4_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf248_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_4_171, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_4_4 = aie.core(%tile_4_4) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_4_176, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf249_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf251_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf250_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf252_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf252_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf252_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf252_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_4_4_175, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf248_unroll_1, %buf247_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_4_174, Release, %c1_i32)
      aie.use_lock(%lock_4_4_175, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_4_174, Release, %c1_i32)
      aie.use_lock(%lock_4_4_175, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_4_174, Release, %c1_i32)
      aie.use_lock(%lock_4_4_175, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_4_174, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf245_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_4_171, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf247_unroll_1, %buf248_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_4, Release, %c1_i32)
        aie.use_lock(%lock_4_4_173, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf252_unroll_1[%c0] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf245_unroll_1, %6, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf250_unroll_1, %buf244_unroll_1, %buf243_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf243_unroll_1, %buf249_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf246_unroll_1, %buf249_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf251_unroll_1, %buf243_unroll_1, %buf244_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf244_unroll_1, %buf251_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_4_172, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf251_unroll_1, %buf249_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf252_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf252_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf252_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf252_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf252_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_4_4_177, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 2>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_7_3 = aie.mem(%tile_7_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_3_170, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf239_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_3_169, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_7_3_167, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_3_168, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_7_3_167, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_3_168, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_7_3_167, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_3_168, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_7_3_167, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_3_168, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_7_3_165, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf236_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_7_3_166, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_7_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf238_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_3_164, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_7_3 = aie.core(%tile_7_3) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_3_169, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf239_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf241_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf240_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf242_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf242_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf242_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf242_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_7_3_168, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_3_167, Release, %c1_i32)
      aie.use_lock(%lock_7_3_168, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_3_167, Release, %c1_i32)
      aie.use_lock(%lock_7_3_168, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_3_167, Release, %c1_i32)
      aie.use_lock(%lock_7_3_168, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf238_unroll_1, %buf237_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_3_167, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf235_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_3_164, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf237_unroll_1, %buf238_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_3, Release, %c1_i32)
        aie.use_lock(%lock_7_3_166, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf242_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c3_i32 : i32
        func.call @apply_causal_mask(%buf235_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf240_unroll_1, %buf234_unroll_1, %buf233_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf233_unroll_1, %buf239_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf236_unroll_1, %buf239_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf241_unroll_1, %buf233_unroll_1, %buf234_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf234_unroll_1, %buf241_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_3_165, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf241_unroll_1, %buf239_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf242_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf242_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf242_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf242_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf242_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_7_3_170, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_6_3 = aie.mem(%tile_6_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_3_163, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf229_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_3_162, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_6_3_160, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_3_161, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_6_3_160, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_3_161, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_6_3_160, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_3_161, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_6_3_160, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_3_161, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_6_3_158, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf226_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_6_3_159, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_6_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf228_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_3_157, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_6_3 = aie.core(%tile_6_3) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_3_162, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf229_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf231_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf230_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf232_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf232_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf232_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf232_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_6_3_161, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_3_160, Release, %c1_i32)
      aie.use_lock(%lock_6_3_161, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_3_160, Release, %c1_i32)
      aie.use_lock(%lock_6_3_161, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf228_unroll_1, %buf227_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_3_160, Release, %c1_i32)
      aie.use_lock(%lock_6_3_161, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_3_160, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf225_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_3_157, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf227_unroll_1, %buf228_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_3, Release, %c1_i32)
        aie.use_lock(%lock_6_3_159, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf232_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c2_i32 : i32
        func.call @apply_causal_mask(%buf225_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf230_unroll_1, %buf224_unroll_1, %buf223_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf223_unroll_1, %buf229_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf226_unroll_1, %buf229_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf231_unroll_1, %buf223_unroll_1, %buf224_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf224_unroll_1, %buf231_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_3_158, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf231_unroll_1, %buf229_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf232_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf232_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf232_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf232_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf232_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_6_3_163, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_5_3 = aie.mem(%tile_5_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_3_156, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf219_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_3_155, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_5_3_153, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_3_154, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_5_3_153, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_3_154, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_5_3_153, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_3_154, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_5_3_153, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_3_154, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_5_3_151, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf216_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_5_3_152, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_5_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf218_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_3_150, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_5_3 = aie.core(%tile_5_3) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_3_155, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf219_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf221_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf220_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf222_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf222_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf222_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf222_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_5_3_154, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_3_153, Release, %c1_i32)
      aie.use_lock(%lock_5_3_154, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf218_unroll_1, %buf217_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_3_153, Release, %c1_i32)
      aie.use_lock(%lock_5_3_154, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_3_153, Release, %c1_i32)
      aie.use_lock(%lock_5_3_154, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_3_153, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf215_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_3_150, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf217_unroll_1, %buf218_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_3, Release, %c1_i32)
        aie.use_lock(%lock_5_3_152, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf222_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c1_i32 : i32
        func.call @apply_causal_mask(%buf215_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf220_unroll_1, %buf214_unroll_1, %buf213_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf213_unroll_1, %buf219_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf216_unroll_1, %buf219_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf221_unroll_1, %buf213_unroll_1, %buf214_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf214_unroll_1, %buf221_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_3_151, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf221_unroll_1, %buf219_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf222_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf222_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf222_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf222_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf222_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_5_3_156, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_4_3 = aie.mem(%tile_4_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_3_149, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf209_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_3_148, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_4_3_146, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_3_147, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_4_3_146, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_3_147, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_4_3_146, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_3_147, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_4_3_146, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_3_147, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_4_3_144, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf206_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_4_3_145, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_4_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf208_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_3_143, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_4_3 = aie.core(%tile_4_3) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_3_148, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf209_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf211_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf210_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf212_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf212_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf212_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf212_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_4_3_147, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf208_unroll_1, %buf207_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_3_146, Release, %c1_i32)
      aie.use_lock(%lock_4_3_147, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_3_146, Release, %c1_i32)
      aie.use_lock(%lock_4_3_147, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_3_146, Release, %c1_i32)
      aie.use_lock(%lock_4_3_147, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_3_146, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf205_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_3_143, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf207_unroll_1, %buf208_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_3, Release, %c1_i32)
        aie.use_lock(%lock_4_3_145, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf212_unroll_1[%c0] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf205_unroll_1, %6, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf210_unroll_1, %buf204_unroll_1, %buf203_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf203_unroll_1, %buf209_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf206_unroll_1, %buf209_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf211_unroll_1, %buf203_unroll_1, %buf204_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf204_unroll_1, %buf211_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_3_144, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf211_unroll_1, %buf209_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf212_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf212_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf212_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf212_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf212_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_4_3_149, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 1>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_7_2 = aie.mem(%tile_7_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_142, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf199_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_141, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_7_2_139, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_140, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_7_2_139, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_140, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_7_2_139, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_140, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_7_2_139, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_140, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_7_2_137, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf196_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_7_2_138, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf198_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_136, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_141, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf199_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf201_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf200_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf202_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf202_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf202_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf202_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_7_2_140, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_2_139, Release, %c1_i32)
      aie.use_lock(%lock_7_2_140, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_2_139, Release, %c1_i32)
      aie.use_lock(%lock_7_2_140, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_7_2_139, Release, %c1_i32)
      aie.use_lock(%lock_7_2_140, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf198_unroll_1, %buf197_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_2_139, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf195_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_136, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf197_unroll_1, %buf198_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2, Release, %c1_i32)
        aie.use_lock(%lock_7_2_138, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf202_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c3_i32 : i32
        func.call @apply_causal_mask(%buf195_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf200_unroll_1, %buf194_unroll_1, %buf193_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf193_unroll_1, %buf199_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf196_unroll_1, %buf199_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf201_unroll_1, %buf193_unroll_1, %buf194_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf194_unroll_1, %buf201_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_137, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf201_unroll_1, %buf199_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf202_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf202_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf202_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf202_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf202_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_7_2_142, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_135, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf189_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_134, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_6_2_132, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_133, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_6_2_132, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_133, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_6_2_132, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_133, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_6_2_132, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_133, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_6_2_130, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf186_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_6_2_131, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf188_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_129, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_134, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf189_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf191_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf190_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf192_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf192_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf192_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf192_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_6_2_133, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_2_132, Release, %c1_i32)
      aie.use_lock(%lock_6_2_133, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_2_132, Release, %c1_i32)
      aie.use_lock(%lock_6_2_133, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf188_unroll_1, %buf187_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_2_132, Release, %c1_i32)
      aie.use_lock(%lock_6_2_133, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_6_2_132, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf185_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_129, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf187_unroll_1, %buf188_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2, Release, %c1_i32)
        aie.use_lock(%lock_6_2_131, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf192_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c2_i32 : i32
        func.call @apply_causal_mask(%buf185_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf190_unroll_1, %buf184_unroll_1, %buf183_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf183_unroll_1, %buf189_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf186_unroll_1, %buf189_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf191_unroll_1, %buf183_unroll_1, %buf184_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf184_unroll_1, %buf191_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_130, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf191_unroll_1, %buf189_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf192_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf192_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf192_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf192_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf192_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_6_2_135, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_128, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf179_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_127, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_5_2_125, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_126, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_5_2_125, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_126, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_5_2_125, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_126, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_5_2_125, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_126, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_5_2_123, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf176_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_5_2_124, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf178_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_122, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_127, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf179_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf181_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf180_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf182_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf182_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf182_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf182_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_5_2_126, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_2_125, Release, %c1_i32)
      aie.use_lock(%lock_5_2_126, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf178_unroll_1, %buf177_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_2_125, Release, %c1_i32)
      aie.use_lock(%lock_5_2_126, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_2_125, Release, %c1_i32)
      aie.use_lock(%lock_5_2_126, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_5_2_125, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf175_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_122, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf177_unroll_1, %buf178_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2, Release, %c1_i32)
        aie.use_lock(%lock_5_2_124, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf182_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c1_i32 : i32
        func.call @apply_causal_mask(%buf175_unroll_1, %7, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf180_unroll_1, %buf174_unroll_1, %buf173_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf173_unroll_1, %buf179_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf176_unroll_1, %buf179_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf181_unroll_1, %buf173_unroll_1, %buf174_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf174_unroll_1, %buf181_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_123, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf181_unroll_1, %buf179_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf182_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf182_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf182_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf182_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf182_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_5_2_128, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_121, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf169_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096 sizes = [64, 8, 8] strides = [8, 512, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_120, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // 3 preds: ^bb7, ^bb9, ^bb10
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_4_2_118, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_119, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_4_2_118, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_119, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_4_2_118, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_119, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_4_2_118, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_119, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb9, ^bb10, repeat_count = 7)
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_4_2_116, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf166_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 1 : i32}
      aie.use_lock(%lock_4_2_117, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb11, ^bb2)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf168_unroll_1 : memref<64x64xbf16, 2 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_115, Release, %c1_i32)
      aie.next_bd ^bb11
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c8_i32 = arith.constant 8 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_120, AcquireGreaterEqual, %c1_i32)
      func.call @zero_fill_gp_bf16(%buf169_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf171_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf170_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      %0 = memref.load %buf172_unroll_1[%c1] : memref<3xi32, 2 : i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      scf.if %1 {
        memref.store %c0_i32, %buf172_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c1_i32, %buf172_unroll_1[%c1] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf172_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_4_2_119, AcquireGreaterEqual, %c1_i32)
      func.call @copy_tile(%buf168_unroll_1, %buf167_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_2_118, Release, %c1_i32)
      aie.use_lock(%lock_4_2_119, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_2_118, Release, %c1_i32)
      aie.use_lock(%lock_4_2_119, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_2_118, Release, %c1_i32)
      aie.use_lock(%lock_4_2_119, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_4_2_118, Release, %c1_i32)
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %collapse_shape = memref.collapse_shape %buf165_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_115, AcquireGreaterEqual, %c1_i32)
        func.call @matmul_a_b_bf16(%buf167_unroll_1, %buf168_unroll_1, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2, Release, %c1_i32)
        aie.use_lock(%lock_4_2_117, AcquireGreaterEqual, %c1_i32)
        %6 = memref.load %buf172_unroll_1[%c0] : memref<3xi32, 2 : i32>
        func.call @apply_causal_mask(%buf165_unroll_1, %6, %arg0) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
        func.call @fused_softmax(%collapse_shape, %buf170_unroll_1, %buf164_unroll_1, %buf163_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf163_unroll_1, %buf169_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape, %buf166_unroll_1, %buf169_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf171_unroll_1, %buf163_unroll_1, %buf164_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf164_unroll_1, %buf171_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_116, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf171_unroll_1, %buf169_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %2 = memref.load %buf172_unroll_1[%c2] : memref<3xi32, 2 : i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi sge, %3, %c1_i32 : i32
      scf.if %4 {
        %6 = memref.load %buf172_unroll_1[%c0] : memref<3xi32, 2 : i32>
        %7 = arith.addi %6, %c4_i32 : i32
        memref.store %7, %buf172_unroll_1[%c0] : memref<3xi32, 2 : i32>
        memref.store %c0_i32, %buf172_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      %5 = arith.cmpi slt, %3, %c1_i32 : i32
      scf.if %5 {
        memref.store %3, %buf172_unroll_1[%c2] : memref<3xi32, 2 : i32>
      }
      aie.use_lock(%lock_4_2_121, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 4, 4>, link_with = "attn_npu2.o"}
    %lock_4_1 = aie.lock(%mem_tile_4_1, 1) {init = 4 : i32}
    %lock_4_1_227 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_6_1 = aie.lock(%mem_tile_6_1, 1) {init = 4 : i32}
    %lock_6_1_228 = aie.lock(%mem_tile_6_1, 0) {init = 0 : i32}
    %lock_5_1 = aie.lock(%mem_tile_5_1, 1) {init = 2 : i32}
    %lock_5_1_229 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %buf325_unroll_1 = aie.buffer(%mem_tile_5_1) {sym_name = "buf325_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf324_unroll_1 = aie.buffer(%mem_tile_4_1) {sym_name = "buf324_unroll_1"} : memref<256x64xbf16, 1 : i32> 
    %buf323_unroll_1 = aie.buffer(%mem_tile_6_1) {sym_name = "buf323_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %mem_tile_5_1, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_4_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_5_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_6_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_7_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_4_4, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_5_4, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_6_4, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_7_4, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 3, %tile_4_5, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 3, %tile_5_5, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 3, %tile_6_5, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 3, %tile_7_5, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 4, %tile_4_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 4, %tile_5_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 4, %tile_6_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 4, %tile_7_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 5, %tile_4_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 5, %tile_5_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 5, %tile_6_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 5, %tile_7_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_4_4, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_4, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_6_4, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_7_4, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_4_5, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_5, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_6_5, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_7_5, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_4_3, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_5_3, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_6_3, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_7_3, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_4_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_5_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_6_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_7_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 3, %tile_4_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 3, %tile_5_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 3, %tile_6_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 3, %tile_7_5, DMA : 0)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_4_1, DMA : 1)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_4_1, DMA : 2)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_4_1, DMA : 3)
    aie.flow(%mem_tile_4_1, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_4_3, DMA : 0, %mem_tile_4_1, DMA : 4)
    aie.flow(%tile_5_3, DMA : 0, %mem_tile_4_1, DMA : 5)
    aie.flow(%tile_6_3, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_7_3, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_4_4, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_5_4, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_6_4, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_7_4, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_4_5, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_5_5, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_6_5, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_7_5, DMA : 0, %mem_tile_4_1, DMA : 0)
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_229, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb15
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1_229, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_1_229, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_5_1_229, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(MM2S, 4, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_5_1_229, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, %c1_i32)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(MM2S, 5, ^bb12, ^bb13)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_5_1_229, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf325_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, %c1_i32)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb11
      %6 = aie.dma_start(S2MM, 0, ^bb14, ^bb15)
    ^bb14:  // 2 preds: ^bb13, ^bb14
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, %c2_i32)
      aie.dma_bd(%buf325_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_229, Release, %c2_i32)
      aie.next_bd ^bb14
    ^bb15:  // pred: ^bb13
      %7 = aie.dma_start(S2MM, 1, ^bb16, ^bb2)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, %c2_i32)
      aie.dma_bd(%buf325_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_229, Release, %c2_i32)
      aie.next_bd ^bb16
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_228, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf323_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1_228, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf323_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_1_228, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf323_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_6_1_228, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf323_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096 sizes = [8, 64, 8] strides = [8, 64, 1]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%buf323_unroll_1 : memref<64x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_228, Release, %c4_i32)
      aie.next_bd ^bb10
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_227, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 0 len = 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, %c4_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb23
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb15)
    ^bb4:  // 2 preds: ^bb3, ^bb14
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb15:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb16, ^bb17)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb16
    ^bb17:  // pred: ^bb15
      %3 = aie.dma_start(S2MM, 2, ^bb18, ^bb19)
    ^bb18:  // 2 preds: ^bb17, ^bb18
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 8192 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb18
    ^bb19:  // pred: ^bb17
      %4 = aie.dma_start(S2MM, 3, ^bb20, ^bb21)
    ^bb20:  // 2 preds: ^bb19, ^bb20
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 12288 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb20
    ^bb21:  // pred: ^bb19
      %5 = aie.dma_start(S2MM, 4, ^bb22, ^bb23)
    ^bb22:  // 2 preds: ^bb21, ^bb22
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 0 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb22
    ^bb23:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 5, ^bb24, ^bb2)
    ^bb24:  // 2 preds: ^bb23, ^bb24
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf324_unroll_1 : memref<256x64xbf16, 1 : i32> offset = 4096 len = 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_227, Release, %c1_i32)
      aie.next_bd ^bb24
    }
    aie.shim_dma_allocation @air_GpOut_1_0_0(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_KIn_1_0_0(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_1_0_0(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_QIn_1_0_0(%shim_noc_tile_5_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = -1 : i64, id = 25 : i64, location = -1 : i64, row = -3 : i64}, {channel = 2 : i64, col = -1 : i64, id = 34 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 13 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 16 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 19 : i64, location = -1 : i64, row = -3 : i64}, {channel = 3 : i64, col = -1 : i64, id = 22 : i64, location = -1 : i64, row = -3 : i64}], sym_name = "attn_seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
  }
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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = airrt.wait_all : !airrt.event
    affine.for %arg4 = 0 to 1 {
      %p = airrt.segment_load "attn_seg" : i64
      %p_0 = airrt.segment_load "attn_seg" : i64
      %c448 = arith.constant 448 : index
      %c384 = arith.constant 384 : index
      %c320 = arith.constant 320 : index
      %c192 = arith.constant 192 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0_1 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i64 = arith.constant 0 : i64
      %c0_3 = arith.constant 0 : index
      %c1_4 = arith.constant 1 : index
      %c25_i32 = arith.constant 25 : i32
      %1 = arith.index_cast %arg4 : index to i64
      %2 = arith.index_cast %c0_3 : index to i64
      %3 = arith.index_cast %c0_3 : index to i64
      %4 = arith.index_cast %c0_1 : index to i64
      %5 = arith.index_cast %c0_1 : index to i64
      %6 = arith.index_cast %c0_3 : index to i64
      %7 = arith.index_cast %c0_3 : index to i64
      %8 = arith.index_cast %c128 : index to i64
      %9 = arith.index_cast %c1_2 : index to i64
      %10 = arith.index_cast %c1_4 : index to i64
      %11 = arith.index_cast %c1_4 : index to i64
      %12 = arith.index_cast %c512 : index to i64
      %13 = arith.index_cast %c64 : index to i64
      %c96_i32 = arith.constant 96 : i32
      %14 = arith.index_cast %arg4 : index to i64
      %c0_i64_5 = arith.constant 0 : i64
      %c0_6 = arith.constant 0 : index
      %15 = arith.index_cast %c0_6 : index to i64
      %16 = arith.index_cast %c0_6 : index to i64
      %17 = arith.index_cast %c0_1 : index to i64
      %18 = arith.index_cast %c0_1 : index to i64
      %c1_7 = arith.constant 1 : index
      %19 = arith.index_cast %c1_7 : index to i64
      %20 = arith.index_cast %c1_7 : index to i64
      %21 = arith.index_cast %c256 : index to i64
      %22 = arith.index_cast %c64 : index to i64
      %23 = arith.index_cast %c0_6 : index to i64
      %24 = arith.index_cast %c0_6 : index to i64
      %25 = arith.index_cast %c512 : index to i64
      %26 = arith.index_cast %c1_2 : index to i64
      %27 = airrt.dma_memcpy_nd(%c96_i32, %14, %c0_i64_5, %arg3[%15, %16, %17, %18], [%19, %20, %21, %22], [%23, %24, %25, %26]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_8 = arith.constant 96 : i32
      %28 = arith.index_cast %arg4 : index to i64
      %c0_i64_9 = arith.constant 0 : i64
      %c0_10 = arith.constant 0 : index
      %29 = arith.index_cast %c0_10 : index to i64
      %30 = arith.index_cast %c0_10 : index to i64
      %31 = arith.index_cast %c0_1 : index to i64
      %32 = arith.index_cast %c64 : index to i64
      %c1_11 = arith.constant 1 : index
      %33 = arith.index_cast %c1_11 : index to i64
      %34 = arith.index_cast %c1_11 : index to i64
      %35 = arith.index_cast %c256 : index to i64
      %36 = arith.index_cast %c64 : index to i64
      %37 = arith.index_cast %c0_10 : index to i64
      %38 = arith.index_cast %c0_10 : index to i64
      %39 = arith.index_cast %c512 : index to i64
      %40 = arith.index_cast %c1_2 : index to i64
      %41 = airrt.dma_memcpy_nd(%c96_i32_8, %28, %c0_i64_9, %arg3[%29, %30, %31, %32], [%33, %34, %35, %36], [%37, %38, %39, %40]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_12 = arith.constant 96 : i32
      %42 = arith.index_cast %arg4 : index to i64
      %c0_i64_13 = arith.constant 0 : i64
      %c0_14 = arith.constant 0 : index
      %43 = arith.index_cast %c0_14 : index to i64
      %44 = arith.index_cast %c0_14 : index to i64
      %45 = arith.index_cast %c0_1 : index to i64
      %46 = arith.index_cast %c128 : index to i64
      %c1_15 = arith.constant 1 : index
      %47 = arith.index_cast %c1_15 : index to i64
      %48 = arith.index_cast %c1_15 : index to i64
      %49 = arith.index_cast %c256 : index to i64
      %50 = arith.index_cast %c64 : index to i64
      %51 = arith.index_cast %c0_14 : index to i64
      %52 = arith.index_cast %c0_14 : index to i64
      %53 = arith.index_cast %c512 : index to i64
      %54 = arith.index_cast %c1_2 : index to i64
      %55 = airrt.dma_memcpy_nd(%c96_i32_12, %42, %c0_i64_13, %arg3[%43, %44, %45, %46], [%47, %48, %49, %50], [%51, %52, %53, %54]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_16 = arith.constant 96 : i32
      %56 = arith.index_cast %arg4 : index to i64
      %c0_i64_17 = arith.constant 0 : i64
      %c0_18 = arith.constant 0 : index
      %57 = arith.index_cast %c0_18 : index to i64
      %58 = arith.index_cast %c0_18 : index to i64
      %59 = arith.index_cast %c0_1 : index to i64
      %60 = arith.index_cast %c192 : index to i64
      %c1_19 = arith.constant 1 : index
      %61 = arith.index_cast %c1_19 : index to i64
      %62 = arith.index_cast %c1_19 : index to i64
      %63 = arith.index_cast %c256 : index to i64
      %64 = arith.index_cast %c64 : index to i64
      %65 = arith.index_cast %c0_18 : index to i64
      %66 = arith.index_cast %c0_18 : index to i64
      %67 = arith.index_cast %c512 : index to i64
      %68 = arith.index_cast %c1_2 : index to i64
      %69 = airrt.dma_memcpy_nd(%c96_i32_16, %56, %c0_i64_17, %arg3[%57, %58, %59, %60], [%61, %62, %63, %64], [%65, %66, %67, %68]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_20 = arith.constant 96 : i32
      %70 = arith.index_cast %arg4 : index to i64
      %c0_i64_21 = arith.constant 0 : i64
      %c0_22 = arith.constant 0 : index
      %71 = arith.index_cast %c0_22 : index to i64
      %72 = arith.index_cast %c0_22 : index to i64
      %73 = arith.index_cast %c0_1 : index to i64
      %74 = arith.index_cast %c256 : index to i64
      %c1_23 = arith.constant 1 : index
      %75 = arith.index_cast %c1_23 : index to i64
      %76 = arith.index_cast %c1_23 : index to i64
      %77 = arith.index_cast %c256 : index to i64
      %78 = arith.index_cast %c64 : index to i64
      %79 = arith.index_cast %c0_22 : index to i64
      %80 = arith.index_cast %c0_22 : index to i64
      %81 = arith.index_cast %c512 : index to i64
      %82 = arith.index_cast %c1_2 : index to i64
      %83 = airrt.dma_memcpy_nd(%c96_i32_20, %70, %c0_i64_21, %arg3[%71, %72, %73, %74], [%75, %76, %77, %78], [%79, %80, %81, %82]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_24 = arith.constant 96 : i32
      %84 = arith.index_cast %arg4 : index to i64
      %c0_i64_25 = arith.constant 0 : i64
      %c0_26 = arith.constant 0 : index
      %85 = arith.index_cast %c0_26 : index to i64
      %86 = arith.index_cast %c0_26 : index to i64
      %87 = arith.index_cast %c0_1 : index to i64
      %88 = arith.index_cast %c320 : index to i64
      %c1_27 = arith.constant 1 : index
      %89 = arith.index_cast %c1_27 : index to i64
      %90 = arith.index_cast %c1_27 : index to i64
      %91 = arith.index_cast %c256 : index to i64
      %92 = arith.index_cast %c64 : index to i64
      %93 = arith.index_cast %c0_26 : index to i64
      %94 = arith.index_cast %c0_26 : index to i64
      %95 = arith.index_cast %c512 : index to i64
      %96 = arith.index_cast %c1_2 : index to i64
      %97 = airrt.dma_memcpy_nd(%c96_i32_24, %84, %c0_i64_25, %arg3[%85, %86, %87, %88], [%89, %90, %91, %92], [%93, %94, %95, %96]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_28 = arith.constant 96 : i32
      %98 = arith.index_cast %arg4 : index to i64
      %c0_i64_29 = arith.constant 0 : i64
      %c0_30 = arith.constant 0 : index
      %99 = arith.index_cast %c0_30 : index to i64
      %100 = arith.index_cast %c0_30 : index to i64
      %101 = arith.index_cast %c0_1 : index to i64
      %102 = arith.index_cast %c384 : index to i64
      %c1_31 = arith.constant 1 : index
      %103 = arith.index_cast %c1_31 : index to i64
      %104 = arith.index_cast %c1_31 : index to i64
      %105 = arith.index_cast %c256 : index to i64
      %106 = arith.index_cast %c64 : index to i64
      %107 = arith.index_cast %c0_30 : index to i64
      %108 = arith.index_cast %c0_30 : index to i64
      %109 = arith.index_cast %c512 : index to i64
      %110 = arith.index_cast %c1_2 : index to i64
      %111 = airrt.dma_memcpy_nd(%c96_i32_28, %98, %c0_i64_29, %arg3[%99, %100, %101, %102], [%103, %104, %105, %106], [%107, %108, %109, %110]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_32 = arith.constant 96 : i32
      %112 = arith.index_cast %arg4 : index to i64
      %c0_i64_33 = arith.constant 0 : i64
      %c0_34 = arith.constant 0 : index
      %113 = arith.index_cast %c0_34 : index to i64
      %114 = arith.index_cast %c0_34 : index to i64
      %115 = arith.index_cast %c0_1 : index to i64
      %116 = arith.index_cast %c448 : index to i64
      %c1_35 = arith.constant 1 : index
      %117 = arith.index_cast %c1_35 : index to i64
      %118 = arith.index_cast %c1_35 : index to i64
      %119 = arith.index_cast %c256 : index to i64
      %120 = arith.index_cast %c64 : index to i64
      %121 = arith.index_cast %c0_34 : index to i64
      %122 = arith.index_cast %c0_34 : index to i64
      %123 = arith.index_cast %c512 : index to i64
      %124 = arith.index_cast %c1_2 : index to i64
      %125 = airrt.dma_memcpy_nd(%c96_i32_32, %112, %c0_i64_33, %arg3[%113, %114, %115, %116], [%117, %118, %119, %120], [%121, %122, %123, %124]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %126 = airrt.dma_memcpy_nd(%c25_i32, %1, %c0_i64, %arg1[%2, %3, %4, %5], [%10, %11, %12, %13], [%6, %7, %8, %9]) {chan_name = @KIn, metadata = @air_KIn_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %127 = airrt.wait_all %126 : !airrt.event
      %c0_i64_36 = arith.constant 0 : i64
      %c0_37 = arith.constant 0 : index
      %c1_38 = arith.constant 1 : index
      %c34_i32 = arith.constant 34 : i32
      %128 = arith.index_cast %arg4 : index to i64
      %129 = arith.index_cast %c0_37 : index to i64
      %130 = arith.index_cast %c0_37 : index to i64
      %131 = arith.index_cast %c0_1 : index to i64
      %132 = arith.index_cast %c0_1 : index to i64
      %133 = arith.index_cast %c0_37 : index to i64
      %134 = arith.index_cast %c0_37 : index to i64
      %135 = arith.index_cast %c128 : index to i64
      %136 = arith.index_cast %c1_2 : index to i64
      %137 = arith.index_cast %c1_38 : index to i64
      %138 = arith.index_cast %c1_38 : index to i64
      %139 = arith.index_cast %c512 : index to i64
      %140 = arith.index_cast %c64 : index to i64
      %141 = airrt.dma_memcpy_nd(%c34_i32, %128, %c0_i64_36, %arg2[%129, %130, %131, %132], [%137, %138, %139, %140], [%133, %134, %135, %136]) {chan_name = @VIn, metadata = @air_VIn_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %142 = airrt.wait_all %141 : !airrt.event
      %c0_i64_39 = arith.constant 0 : i64
      %c0_40 = arith.constant 0 : index
      %c1_41 = arith.constant 1 : index
      %c13_i32 = arith.constant 13 : i32
      %143 = arith.index_cast %arg4 : index to i64
      %144 = arith.index_cast %c0_40 : index to i64
      %145 = arith.index_cast %c0_40 : index to i64
      %146 = arith.index_cast %c0_1 : index to i64
      %147 = arith.index_cast %c0_1 : index to i64
      %148 = arith.index_cast %c0_40 : index to i64
      %149 = arith.index_cast %c0_40 : index to i64
      %150 = arith.index_cast %c512 : index to i64
      %151 = arith.index_cast %c1_2 : index to i64
      %152 = arith.index_cast %c1_41 : index to i64
      %153 = arith.index_cast %c1_41 : index to i64
      %154 = arith.index_cast %c256 : index to i64
      %155 = arith.index_cast %c64 : index to i64
      %156 = airrt.dma_memcpy_nd(%c13_i32, %143, %c0_i64_39, %arg0[%144, %145, %146, %147], [%152, %153, %154, %155], [%148, %149, %150, %151]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %157 = airrt.wait_all %156 : !airrt.event
      %c0_i64_42 = arith.constant 0 : i64
      %c0_43 = arith.constant 0 : index
      %c1_44 = arith.constant 1 : index
      %c13_i32_45 = arith.constant 13 : i32
      %158 = arith.index_cast %arg4 : index to i64
      %159 = arith.index_cast %c0_43 : index to i64
      %160 = arith.index_cast %c0_43 : index to i64
      %161 = arith.index_cast %c0_1 : index to i64
      %162 = arith.index_cast %c64 : index to i64
      %163 = arith.index_cast %c0_43 : index to i64
      %164 = arith.index_cast %c0_43 : index to i64
      %165 = arith.index_cast %c512 : index to i64
      %166 = arith.index_cast %c1_2 : index to i64
      %167 = arith.index_cast %c1_44 : index to i64
      %168 = arith.index_cast %c1_44 : index to i64
      %169 = arith.index_cast %c256 : index to i64
      %170 = arith.index_cast %c64 : index to i64
      %171 = airrt.dma_memcpy_nd(%c13_i32_45, %158, %c0_i64_42, %arg0[%159, %160, %161, %162], [%167, %168, %169, %170], [%163, %164, %165, %166]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %172 = airrt.wait_all %171, %157 : !airrt.event
      %c0_i64_46 = arith.constant 0 : i64
      %c0_47 = arith.constant 0 : index
      %c1_48 = arith.constant 1 : index
      %c13_i32_49 = arith.constant 13 : i32
      %173 = arith.index_cast %arg4 : index to i64
      %174 = arith.index_cast %c0_47 : index to i64
      %175 = arith.index_cast %c0_47 : index to i64
      %176 = arith.index_cast %c0_1 : index to i64
      %177 = arith.index_cast %c128 : index to i64
      %178 = arith.index_cast %c0_47 : index to i64
      %179 = arith.index_cast %c0_47 : index to i64
      %180 = arith.index_cast %c512 : index to i64
      %181 = arith.index_cast %c1_2 : index to i64
      %182 = arith.index_cast %c1_48 : index to i64
      %183 = arith.index_cast %c1_48 : index to i64
      %184 = arith.index_cast %c256 : index to i64
      %185 = arith.index_cast %c64 : index to i64
      %186 = airrt.dma_memcpy_nd(%c13_i32_49, %173, %c0_i64_46, %arg0[%174, %175, %176, %177], [%182, %183, %184, %185], [%178, %179, %180, %181]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %187 = airrt.wait_all %186, %172 : !airrt.event
      %c0_i64_50 = arith.constant 0 : i64
      %c0_51 = arith.constant 0 : index
      %c1_52 = arith.constant 1 : index
      %c13_i32_53 = arith.constant 13 : i32
      %188 = arith.index_cast %arg4 : index to i64
      %189 = arith.index_cast %c0_51 : index to i64
      %190 = arith.index_cast %c0_51 : index to i64
      %191 = arith.index_cast %c0_1 : index to i64
      %192 = arith.index_cast %c192 : index to i64
      %193 = arith.index_cast %c0_51 : index to i64
      %194 = arith.index_cast %c0_51 : index to i64
      %195 = arith.index_cast %c512 : index to i64
      %196 = arith.index_cast %c1_2 : index to i64
      %197 = arith.index_cast %c1_52 : index to i64
      %198 = arith.index_cast %c1_52 : index to i64
      %199 = arith.index_cast %c256 : index to i64
      %200 = arith.index_cast %c64 : index to i64
      %201 = airrt.dma_memcpy_nd(%c13_i32_53, %188, %c0_i64_50, %arg0[%189, %190, %191, %192], [%197, %198, %199, %200], [%193, %194, %195, %196]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %202 = airrt.wait_all %201, %187 : !airrt.event
      %c0_i64_54 = arith.constant 0 : i64
      %c0_55 = arith.constant 0 : index
      %c1_56 = arith.constant 1 : index
      %c25_i32_57 = arith.constant 25 : i32
      %203 = arith.index_cast %arg4 : index to i64
      %204 = arith.index_cast %c0_55 : index to i64
      %205 = arith.index_cast %c0_55 : index to i64
      %206 = arith.index_cast %c0_1 : index to i64
      %207 = arith.index_cast %c64 : index to i64
      %208 = arith.index_cast %c0_55 : index to i64
      %209 = arith.index_cast %c0_55 : index to i64
      %210 = arith.index_cast %c128 : index to i64
      %211 = arith.index_cast %c1_2 : index to i64
      %212 = arith.index_cast %c1_56 : index to i64
      %213 = arith.index_cast %c1_56 : index to i64
      %214 = arith.index_cast %c512 : index to i64
      %215 = arith.index_cast %c64 : index to i64
      %216 = airrt.dma_memcpy_nd(%c25_i32_57, %203, %c0_i64_54, %arg1[%204, %205, %206, %207], [%212, %213, %214, %215], [%208, %209, %210, %211]) {chan_name = @KIn, metadata = @air_KIn_1_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %217 = airrt.wait_all %216 : !airrt.event
      %c0_i64_58 = arith.constant 0 : i64
      %c0_59 = arith.constant 0 : index
      %c1_60 = arith.constant 1 : index
      %c34_i32_61 = arith.constant 34 : i32
      %218 = arith.index_cast %arg4 : index to i64
      %219 = arith.index_cast %c0_59 : index to i64
      %220 = arith.index_cast %c0_59 : index to i64
      %221 = arith.index_cast %c0_1 : index to i64
      %222 = arith.index_cast %c64 : index to i64
      %223 = arith.index_cast %c0_59 : index to i64
      %224 = arith.index_cast %c0_59 : index to i64
      %225 = arith.index_cast %c128 : index to i64
      %226 = arith.index_cast %c1_2 : index to i64
      %227 = arith.index_cast %c1_60 : index to i64
      %228 = arith.index_cast %c1_60 : index to i64
      %229 = arith.index_cast %c512 : index to i64
      %230 = arith.index_cast %c64 : index to i64
      %231 = airrt.dma_memcpy_nd(%c34_i32_61, %218, %c0_i64_58, %arg2[%219, %220, %221, %222], [%227, %228, %229, %230], [%223, %224, %225, %226]) {chan_name = @VIn, metadata = @air_VIn_1_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %232 = airrt.wait_all %231 : !airrt.event
      %c0_i64_62 = arith.constant 0 : i64
      %c0_63 = arith.constant 0 : index
      %c1_64 = arith.constant 1 : index
      %c13_i32_65 = arith.constant 13 : i32
      %233 = arith.index_cast %arg4 : index to i64
      %234 = arith.index_cast %c0_63 : index to i64
      %235 = arith.index_cast %c0_63 : index to i64
      %236 = arith.index_cast %c0_1 : index to i64
      %237 = arith.index_cast %c256 : index to i64
      %238 = arith.index_cast %c0_63 : index to i64
      %239 = arith.index_cast %c0_63 : index to i64
      %240 = arith.index_cast %c512 : index to i64
      %241 = arith.index_cast %c1_2 : index to i64
      %242 = arith.index_cast %c1_64 : index to i64
      %243 = arith.index_cast %c1_64 : index to i64
      %244 = arith.index_cast %c256 : index to i64
      %245 = arith.index_cast %c64 : index to i64
      %246 = airrt.dma_memcpy_nd(%c13_i32_65, %233, %c0_i64_62, %arg0[%234, %235, %236, %237], [%242, %243, %244, %245], [%238, %239, %240, %241]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %247 = airrt.wait_all %246 : !airrt.event
      %c0_i64_66 = arith.constant 0 : i64
      %c0_67 = arith.constant 0 : index
      %c1_68 = arith.constant 1 : index
      %c13_i32_69 = arith.constant 13 : i32
      %248 = arith.index_cast %arg4 : index to i64
      %249 = arith.index_cast %c0_67 : index to i64
      %250 = arith.index_cast %c0_67 : index to i64
      %251 = arith.index_cast %c0_1 : index to i64
      %252 = arith.index_cast %c320 : index to i64
      %253 = arith.index_cast %c0_67 : index to i64
      %254 = arith.index_cast %c0_67 : index to i64
      %255 = arith.index_cast %c512 : index to i64
      %256 = arith.index_cast %c1_2 : index to i64
      %257 = arith.index_cast %c1_68 : index to i64
      %258 = arith.index_cast %c1_68 : index to i64
      %259 = arith.index_cast %c256 : index to i64
      %260 = arith.index_cast %c64 : index to i64
      %261 = airrt.dma_memcpy_nd(%c13_i32_69, %248, %c0_i64_66, %arg0[%249, %250, %251, %252], [%257, %258, %259, %260], [%253, %254, %255, %256]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %262 = airrt.wait_all %261, %247 : !airrt.event
      %c0_i64_70 = arith.constant 0 : i64
      %c0_71 = arith.constant 0 : index
      %c1_72 = arith.constant 1 : index
      %c13_i32_73 = arith.constant 13 : i32
      %263 = arith.index_cast %arg4 : index to i64
      %264 = arith.index_cast %c0_71 : index to i64
      %265 = arith.index_cast %c0_71 : index to i64
      %266 = arith.index_cast %c0_1 : index to i64
      %267 = arith.index_cast %c384 : index to i64
      %268 = arith.index_cast %c0_71 : index to i64
      %269 = arith.index_cast %c0_71 : index to i64
      %270 = arith.index_cast %c512 : index to i64
      %271 = arith.index_cast %c1_2 : index to i64
      %272 = arith.index_cast %c1_72 : index to i64
      %273 = arith.index_cast %c1_72 : index to i64
      %274 = arith.index_cast %c256 : index to i64
      %275 = arith.index_cast %c64 : index to i64
      %276 = airrt.dma_memcpy_nd(%c13_i32_73, %263, %c0_i64_70, %arg0[%264, %265, %266, %267], [%272, %273, %274, %275], [%268, %269, %270, %271]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %277 = airrt.wait_all %276, %262 : !airrt.event
      %c0_i64_74 = arith.constant 0 : i64
      %c0_75 = arith.constant 0 : index
      %c1_76 = arith.constant 1 : index
      %c13_i32_77 = arith.constant 13 : i32
      %278 = arith.index_cast %arg4 : index to i64
      %279 = arith.index_cast %c0_75 : index to i64
      %280 = arith.index_cast %c0_75 : index to i64
      %281 = arith.index_cast %c0_1 : index to i64
      %282 = arith.index_cast %c448 : index to i64
      %283 = arith.index_cast %c0_75 : index to i64
      %284 = arith.index_cast %c0_75 : index to i64
      %285 = arith.index_cast %c512 : index to i64
      %286 = arith.index_cast %c1_2 : index to i64
      %287 = arith.index_cast %c1_76 : index to i64
      %288 = arith.index_cast %c1_76 : index to i64
      %289 = arith.index_cast %c256 : index to i64
      %290 = arith.index_cast %c64 : index to i64
      %291 = airrt.dma_memcpy_nd(%c13_i32_77, %278, %c0_i64_74, %arg0[%279, %280, %281, %282], [%287, %288, %289, %290], [%283, %284, %285, %286]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %292 = airrt.wait_all %291, %277 : !airrt.event
      %c0_78 = arith.constant 0 : index
      %c1_79 = arith.constant 1 : index
      %293 = airrt.wait_all : !airrt.event
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 1 {
          %c12288 = arith.constant 12288 : index
          %c8192 = arith.constant 8192 : index
          %c4096 = arith.constant 4096 : index
          %c3 = arith.constant 3 : index
          %c2_162 = arith.constant 2 : index
          %c64_163 = arith.constant 64 : index
          %c1_164 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c0_165 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %603 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %604 = airrt.wait_all : !airrt.event
          %605 = airrt.wait_all : !airrt.event
          %606 = airrt.alloc : memref<256x64xbf16, 1 : i32>
          %607 = airrt.wait_all : !airrt.event
          %608 = scf.for %arg7 = %c0_165 to %c8 step %c1_164 iter_args(%arg8 = %605) -> (!airrt.event) {
            %632 = airrt.alloc : memref<64x64xbf16, 1 : i32>
            %633 = airrt.wait_all : !airrt.event
            %634 = airrt.wait_all %633, %arg8 : !airrt.event
            %635 = arith.cmpi eq, %arg5, %c0_165 : index
            %636 = airrt.wait_all %634 : !airrt.event
            %637 = airrt.wait_all %634 : !airrt.event
            %638 = airrt.wait_all %634 : !airrt.event
            %639 = airrt.wait_all %634 : !airrt.event
            %640 = airrt.wait_all %634 : !airrt.event
            %641 = airrt.wait_all %634 : !airrt.event
            %642 = airrt.wait_all %634 : !airrt.event
            %643 = airrt.wait_all %634 : !airrt.event
            %644 = airrt.wait_all %636, %638, %640, %642 : !airrt.event
            airrt.dealloc %632 : memref<64x64xbf16, 1 : i32>
            %645 = airrt.wait_all : !airrt.event
            scf.yield %644 : !airrt.event
          }
          %h = airrt.herd_load "herd_0" (%arg5) {segment_name = "attn_seg"} : (index) -> i64
          %609 = airrt.wait_all : !airrt.event
          %610 = airrt.wait_all %607 : !airrt.event
          %611 = airrt.wait_all %607 : !airrt.event
          %612 = airrt.wait_all %607 : !airrt.event
          %613 = airrt.wait_all %607 : !airrt.event
          %614 = airrt.wait_all %610, %611, %612, %613 : !airrt.event
          %615 = airrt.wait_all %614 : !airrt.event
          %616 = airrt.wait_all %614 : !airrt.event
          %617 = airrt.wait_all %614 : !airrt.event
          %618 = airrt.wait_all %614 : !airrt.event
          %619 = airrt.wait_all %618, %617, %616, %615 : !airrt.event
          %620 = airrt.wait_all %619 : !airrt.event
          %621 = airrt.wait_all %619 : !airrt.event
          %622 = airrt.wait_all %619 : !airrt.event
          %623 = airrt.wait_all %619 : !airrt.event
          %624 = airrt.wait_all %623, %622, %621, %620 : !airrt.event
          %625 = airrt.wait_all %624 : !airrt.event
          %626 = airrt.wait_all %624 : !airrt.event
          %627 = airrt.wait_all %624 : !airrt.event
          %628 = airrt.wait_all %624 : !airrt.event
          %629 = airrt.wait_all %628, %627, %626, %625 : !airrt.event
          airrt.dealloc %603 : memref<64x64xbf16, 1 : i32>
          %630 = airrt.wait_all : !airrt.event
          airrt.dealloc %606 : memref<256x64xbf16, 1 : i32>
          %631 = airrt.wait_all : !airrt.event
          airrt.wait_all %608, %609, %630, %631 {air.segment_end}
        }
      }
      %294 = airrt.wait_all %293 : !airrt.event
      %295 = airrt.wait_all %294 : !airrt.event
      %296 = airrt.wait_all %293, %295 : !airrt.event
      %297 = airrt.wait_all %296 : !airrt.event
      %298 = airrt.wait_all %293 : !airrt.event
      %299 = airrt.wait_all %298 : !airrt.event
      %300 = airrt.wait_all %293, %299 : !airrt.event
      %301 = airrt.wait_all %300 : !airrt.event
      airrt.wait_all %127, %202, %232, %297, %301, %292, %217, %142, %27, %41, %55, %69, %83, %97, %111, %125 {air.launch_end}
      %c0_i64_80 = arith.constant 0 : i64
      %c0_81 = arith.constant 0 : index
      %c1_82 = arith.constant 1 : index
      %c25_i32_83 = arith.constant 25 : i32
      %302 = arith.index_cast %arg4 : index to i64
      %303 = arith.index_cast %c0_81 : index to i64
      %304 = arith.index_cast %c0_81 : index to i64
      %305 = arith.index_cast %c0_1 : index to i64
      %306 = arith.index_cast %c0_1 : index to i64
      %307 = arith.index_cast %c0_81 : index to i64
      %308 = arith.index_cast %c0_81 : index to i64
      %309 = arith.index_cast %c128 : index to i64
      %310 = arith.index_cast %c1_2 : index to i64
      %311 = arith.index_cast %c1_82 : index to i64
      %312 = arith.index_cast %c1_82 : index to i64
      %313 = arith.index_cast %c512 : index to i64
      %314 = arith.index_cast %c64 : index to i64
      %c96_i32_84 = arith.constant 96 : i32
      %315 = arith.index_cast %arg4 : index to i64
      %c0_i64_85 = arith.constant 0 : i64
      %c0_86 = arith.constant 0 : index
      %316 = arith.index_cast %c0_86 : index to i64
      %317 = arith.index_cast %c0_86 : index to i64
      %318 = arith.index_cast %c256 : index to i64
      %319 = arith.index_cast %c0_1 : index to i64
      %c1_87 = arith.constant 1 : index
      %320 = arith.index_cast %c1_87 : index to i64
      %321 = arith.index_cast %c1_87 : index to i64
      %322 = arith.index_cast %c256 : index to i64
      %323 = arith.index_cast %c64 : index to i64
      %324 = arith.index_cast %c0_86 : index to i64
      %325 = arith.index_cast %c0_86 : index to i64
      %326 = arith.index_cast %c512 : index to i64
      %327 = arith.index_cast %c1_2 : index to i64
      %328 = airrt.dma_memcpy_nd(%c96_i32_84, %315, %c0_i64_85, %arg3[%316, %317, %318, %319], [%320, %321, %322, %323], [%324, %325, %326, %327]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_88 = arith.constant 96 : i32
      %329 = arith.index_cast %arg4 : index to i64
      %c0_i64_89 = arith.constant 0 : i64
      %c0_90 = arith.constant 0 : index
      %330 = arith.index_cast %c0_90 : index to i64
      %331 = arith.index_cast %c0_90 : index to i64
      %332 = arith.index_cast %c256 : index to i64
      %333 = arith.index_cast %c64 : index to i64
      %c1_91 = arith.constant 1 : index
      %334 = arith.index_cast %c1_91 : index to i64
      %335 = arith.index_cast %c1_91 : index to i64
      %336 = arith.index_cast %c256 : index to i64
      %337 = arith.index_cast %c64 : index to i64
      %338 = arith.index_cast %c0_90 : index to i64
      %339 = arith.index_cast %c0_90 : index to i64
      %340 = arith.index_cast %c512 : index to i64
      %341 = arith.index_cast %c1_2 : index to i64
      %342 = airrt.dma_memcpy_nd(%c96_i32_88, %329, %c0_i64_89, %arg3[%330, %331, %332, %333], [%334, %335, %336, %337], [%338, %339, %340, %341]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_92 = arith.constant 96 : i32
      %343 = arith.index_cast %arg4 : index to i64
      %c0_i64_93 = arith.constant 0 : i64
      %c0_94 = arith.constant 0 : index
      %344 = arith.index_cast %c0_94 : index to i64
      %345 = arith.index_cast %c0_94 : index to i64
      %346 = arith.index_cast %c256 : index to i64
      %347 = arith.index_cast %c128 : index to i64
      %c1_95 = arith.constant 1 : index
      %348 = arith.index_cast %c1_95 : index to i64
      %349 = arith.index_cast %c1_95 : index to i64
      %350 = arith.index_cast %c256 : index to i64
      %351 = arith.index_cast %c64 : index to i64
      %352 = arith.index_cast %c0_94 : index to i64
      %353 = arith.index_cast %c0_94 : index to i64
      %354 = arith.index_cast %c512 : index to i64
      %355 = arith.index_cast %c1_2 : index to i64
      %356 = airrt.dma_memcpy_nd(%c96_i32_92, %343, %c0_i64_93, %arg3[%344, %345, %346, %347], [%348, %349, %350, %351], [%352, %353, %354, %355]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_96 = arith.constant 96 : i32
      %357 = arith.index_cast %arg4 : index to i64
      %c0_i64_97 = arith.constant 0 : i64
      %c0_98 = arith.constant 0 : index
      %358 = arith.index_cast %c0_98 : index to i64
      %359 = arith.index_cast %c0_98 : index to i64
      %360 = arith.index_cast %c256 : index to i64
      %361 = arith.index_cast %c192 : index to i64
      %c1_99 = arith.constant 1 : index
      %362 = arith.index_cast %c1_99 : index to i64
      %363 = arith.index_cast %c1_99 : index to i64
      %364 = arith.index_cast %c256 : index to i64
      %365 = arith.index_cast %c64 : index to i64
      %366 = arith.index_cast %c0_98 : index to i64
      %367 = arith.index_cast %c0_98 : index to i64
      %368 = arith.index_cast %c512 : index to i64
      %369 = arith.index_cast %c1_2 : index to i64
      %370 = airrt.dma_memcpy_nd(%c96_i32_96, %357, %c0_i64_97, %arg3[%358, %359, %360, %361], [%362, %363, %364, %365], [%366, %367, %368, %369]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_100 = arith.constant 96 : i32
      %371 = arith.index_cast %arg4 : index to i64
      %c0_i64_101 = arith.constant 0 : i64
      %c0_102 = arith.constant 0 : index
      %372 = arith.index_cast %c0_102 : index to i64
      %373 = arith.index_cast %c0_102 : index to i64
      %374 = arith.index_cast %c256 : index to i64
      %375 = arith.index_cast %c256 : index to i64
      %c1_103 = arith.constant 1 : index
      %376 = arith.index_cast %c1_103 : index to i64
      %377 = arith.index_cast %c1_103 : index to i64
      %378 = arith.index_cast %c256 : index to i64
      %379 = arith.index_cast %c64 : index to i64
      %380 = arith.index_cast %c0_102 : index to i64
      %381 = arith.index_cast %c0_102 : index to i64
      %382 = arith.index_cast %c512 : index to i64
      %383 = arith.index_cast %c1_2 : index to i64
      %384 = airrt.dma_memcpy_nd(%c96_i32_100, %371, %c0_i64_101, %arg3[%372, %373, %374, %375], [%376, %377, %378, %379], [%380, %381, %382, %383]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_104 = arith.constant 96 : i32
      %385 = arith.index_cast %arg4 : index to i64
      %c0_i64_105 = arith.constant 0 : i64
      %c0_106 = arith.constant 0 : index
      %386 = arith.index_cast %c0_106 : index to i64
      %387 = arith.index_cast %c0_106 : index to i64
      %388 = arith.index_cast %c256 : index to i64
      %389 = arith.index_cast %c320 : index to i64
      %c1_107 = arith.constant 1 : index
      %390 = arith.index_cast %c1_107 : index to i64
      %391 = arith.index_cast %c1_107 : index to i64
      %392 = arith.index_cast %c256 : index to i64
      %393 = arith.index_cast %c64 : index to i64
      %394 = arith.index_cast %c0_106 : index to i64
      %395 = arith.index_cast %c0_106 : index to i64
      %396 = arith.index_cast %c512 : index to i64
      %397 = arith.index_cast %c1_2 : index to i64
      %398 = airrt.dma_memcpy_nd(%c96_i32_104, %385, %c0_i64_105, %arg3[%386, %387, %388, %389], [%390, %391, %392, %393], [%394, %395, %396, %397]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_108 = arith.constant 96 : i32
      %399 = arith.index_cast %arg4 : index to i64
      %c0_i64_109 = arith.constant 0 : i64
      %c0_110 = arith.constant 0 : index
      %400 = arith.index_cast %c0_110 : index to i64
      %401 = arith.index_cast %c0_110 : index to i64
      %402 = arith.index_cast %c256 : index to i64
      %403 = arith.index_cast %c384 : index to i64
      %c1_111 = arith.constant 1 : index
      %404 = arith.index_cast %c1_111 : index to i64
      %405 = arith.index_cast %c1_111 : index to i64
      %406 = arith.index_cast %c256 : index to i64
      %407 = arith.index_cast %c64 : index to i64
      %408 = arith.index_cast %c0_110 : index to i64
      %409 = arith.index_cast %c0_110 : index to i64
      %410 = arith.index_cast %c512 : index to i64
      %411 = arith.index_cast %c1_2 : index to i64
      %412 = airrt.dma_memcpy_nd(%c96_i32_108, %399, %c0_i64_109, %arg3[%400, %401, %402, %403], [%404, %405, %406, %407], [%408, %409, %410, %411]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %c96_i32_112 = arith.constant 96 : i32
      %413 = arith.index_cast %arg4 : index to i64
      %c0_i64_113 = arith.constant 0 : i64
      %c0_114 = arith.constant 0 : index
      %414 = arith.index_cast %c0_114 : index to i64
      %415 = arith.index_cast %c0_114 : index to i64
      %416 = arith.index_cast %c256 : index to i64
      %417 = arith.index_cast %c448 : index to i64
      %c1_115 = arith.constant 1 : index
      %418 = arith.index_cast %c1_115 : index to i64
      %419 = arith.index_cast %c1_115 : index to i64
      %420 = arith.index_cast %c256 : index to i64
      %421 = arith.index_cast %c64 : index to i64
      %422 = arith.index_cast %c0_114 : index to i64
      %423 = arith.index_cast %c0_114 : index to i64
      %424 = arith.index_cast %c512 : index to i64
      %425 = arith.index_cast %c1_2 : index to i64
      %426 = airrt.dma_memcpy_nd(%c96_i32_112, %413, %c0_i64_113, %arg3[%414, %415, %416, %417], [%418, %419, %420, %421], [%422, %423, %424, %425]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %427 = airrt.dma_memcpy_nd(%c25_i32_83, %302, %c0_i64_80, %arg1[%303, %304, %305, %306], [%311, %312, %313, %314], [%307, %308, %309, %310]) {chan_name = @KIn, metadata = @air_KIn_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %428 = airrt.wait_all %427 : !airrt.event
      %c0_i64_116 = arith.constant 0 : i64
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      %c34_i32_119 = arith.constant 34 : i32
      %429 = arith.index_cast %arg4 : index to i64
      %430 = arith.index_cast %c0_117 : index to i64
      %431 = arith.index_cast %c0_117 : index to i64
      %432 = arith.index_cast %c0_1 : index to i64
      %433 = arith.index_cast %c0_1 : index to i64
      %434 = arith.index_cast %c0_117 : index to i64
      %435 = arith.index_cast %c0_117 : index to i64
      %436 = arith.index_cast %c128 : index to i64
      %437 = arith.index_cast %c1_2 : index to i64
      %438 = arith.index_cast %c1_118 : index to i64
      %439 = arith.index_cast %c1_118 : index to i64
      %440 = arith.index_cast %c512 : index to i64
      %441 = arith.index_cast %c64 : index to i64
      %442 = airrt.dma_memcpy_nd(%c34_i32_119, %429, %c0_i64_116, %arg2[%430, %431, %432, %433], [%438, %439, %440, %441], [%434, %435, %436, %437]) {chan_name = @VIn, metadata = @air_VIn_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %443 = airrt.wait_all %442 : !airrt.event
      %c0_i64_120 = arith.constant 0 : i64
      %c0_121 = arith.constant 0 : index
      %c1_122 = arith.constant 1 : index
      %c13_i32_123 = arith.constant 13 : i32
      %444 = arith.index_cast %arg4 : index to i64
      %445 = arith.index_cast %c0_121 : index to i64
      %446 = arith.index_cast %c0_121 : index to i64
      %447 = arith.index_cast %c256 : index to i64
      %448 = arith.index_cast %c0_1 : index to i64
      %449 = arith.index_cast %c0_121 : index to i64
      %450 = arith.index_cast %c0_121 : index to i64
      %451 = arith.index_cast %c512 : index to i64
      %452 = arith.index_cast %c1_2 : index to i64
      %453 = arith.index_cast %c1_122 : index to i64
      %454 = arith.index_cast %c1_122 : index to i64
      %455 = arith.index_cast %c256 : index to i64
      %456 = arith.index_cast %c64 : index to i64
      %457 = airrt.dma_memcpy_nd(%c13_i32_123, %444, %c0_i64_120, %arg0[%445, %446, %447, %448], [%453, %454, %455, %456], [%449, %450, %451, %452]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %458 = airrt.wait_all %457 : !airrt.event
      %c0_i64_124 = arith.constant 0 : i64
      %c0_125 = arith.constant 0 : index
      %c1_126 = arith.constant 1 : index
      %c13_i32_127 = arith.constant 13 : i32
      %459 = arith.index_cast %arg4 : index to i64
      %460 = arith.index_cast %c0_125 : index to i64
      %461 = arith.index_cast %c0_125 : index to i64
      %462 = arith.index_cast %c256 : index to i64
      %463 = arith.index_cast %c64 : index to i64
      %464 = arith.index_cast %c0_125 : index to i64
      %465 = arith.index_cast %c0_125 : index to i64
      %466 = arith.index_cast %c512 : index to i64
      %467 = arith.index_cast %c1_2 : index to i64
      %468 = arith.index_cast %c1_126 : index to i64
      %469 = arith.index_cast %c1_126 : index to i64
      %470 = arith.index_cast %c256 : index to i64
      %471 = arith.index_cast %c64 : index to i64
      %472 = airrt.dma_memcpy_nd(%c13_i32_127, %459, %c0_i64_124, %arg0[%460, %461, %462, %463], [%468, %469, %470, %471], [%464, %465, %466, %467]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %473 = airrt.wait_all %472, %458 : !airrt.event
      %c0_i64_128 = arith.constant 0 : i64
      %c0_129 = arith.constant 0 : index
      %c1_130 = arith.constant 1 : index
      %c13_i32_131 = arith.constant 13 : i32
      %474 = arith.index_cast %arg4 : index to i64
      %475 = arith.index_cast %c0_129 : index to i64
      %476 = arith.index_cast %c0_129 : index to i64
      %477 = arith.index_cast %c256 : index to i64
      %478 = arith.index_cast %c128 : index to i64
      %479 = arith.index_cast %c0_129 : index to i64
      %480 = arith.index_cast %c0_129 : index to i64
      %481 = arith.index_cast %c512 : index to i64
      %482 = arith.index_cast %c1_2 : index to i64
      %483 = arith.index_cast %c1_130 : index to i64
      %484 = arith.index_cast %c1_130 : index to i64
      %485 = arith.index_cast %c256 : index to i64
      %486 = arith.index_cast %c64 : index to i64
      %487 = airrt.dma_memcpy_nd(%c13_i32_131, %474, %c0_i64_128, %arg0[%475, %476, %477, %478], [%483, %484, %485, %486], [%479, %480, %481, %482]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %488 = airrt.wait_all %487, %473 : !airrt.event
      %c0_i64_132 = arith.constant 0 : i64
      %c0_133 = arith.constant 0 : index
      %c1_134 = arith.constant 1 : index
      %c13_i32_135 = arith.constant 13 : i32
      %489 = arith.index_cast %arg4 : index to i64
      %490 = arith.index_cast %c0_133 : index to i64
      %491 = arith.index_cast %c0_133 : index to i64
      %492 = arith.index_cast %c256 : index to i64
      %493 = arith.index_cast %c192 : index to i64
      %494 = arith.index_cast %c0_133 : index to i64
      %495 = arith.index_cast %c0_133 : index to i64
      %496 = arith.index_cast %c512 : index to i64
      %497 = arith.index_cast %c1_2 : index to i64
      %498 = arith.index_cast %c1_134 : index to i64
      %499 = arith.index_cast %c1_134 : index to i64
      %500 = arith.index_cast %c256 : index to i64
      %501 = arith.index_cast %c64 : index to i64
      %502 = airrt.dma_memcpy_nd(%c13_i32_135, %489, %c0_i64_132, %arg0[%490, %491, %492, %493], [%498, %499, %500, %501], [%494, %495, %496, %497]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %503 = airrt.wait_all %502, %488 : !airrt.event
      %c0_i64_136 = arith.constant 0 : i64
      %c0_137 = arith.constant 0 : index
      %c1_138 = arith.constant 1 : index
      %c25_i32_139 = arith.constant 25 : i32
      %504 = arith.index_cast %arg4 : index to i64
      %505 = arith.index_cast %c0_137 : index to i64
      %506 = arith.index_cast %c0_137 : index to i64
      %507 = arith.index_cast %c0_1 : index to i64
      %508 = arith.index_cast %c64 : index to i64
      %509 = arith.index_cast %c0_137 : index to i64
      %510 = arith.index_cast %c0_137 : index to i64
      %511 = arith.index_cast %c128 : index to i64
      %512 = arith.index_cast %c1_2 : index to i64
      %513 = arith.index_cast %c1_138 : index to i64
      %514 = arith.index_cast %c1_138 : index to i64
      %515 = arith.index_cast %c512 : index to i64
      %516 = arith.index_cast %c64 : index to i64
      %517 = airrt.dma_memcpy_nd(%c25_i32_139, %504, %c0_i64_136, %arg1[%505, %506, %507, %508], [%513, %514, %515, %516], [%509, %510, %511, %512]) {chan_name = @KIn, metadata = @air_KIn_1_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %518 = airrt.wait_all %517 : !airrt.event
      %c0_i64_140 = arith.constant 0 : i64
      %c0_141 = arith.constant 0 : index
      %c1_142 = arith.constant 1 : index
      %c34_i32_143 = arith.constant 34 : i32
      %519 = arith.index_cast %arg4 : index to i64
      %520 = arith.index_cast %c0_141 : index to i64
      %521 = arith.index_cast %c0_141 : index to i64
      %522 = arith.index_cast %c0_1 : index to i64
      %523 = arith.index_cast %c64 : index to i64
      %524 = arith.index_cast %c0_141 : index to i64
      %525 = arith.index_cast %c0_141 : index to i64
      %526 = arith.index_cast %c128 : index to i64
      %527 = arith.index_cast %c1_2 : index to i64
      %528 = arith.index_cast %c1_142 : index to i64
      %529 = arith.index_cast %c1_142 : index to i64
      %530 = arith.index_cast %c512 : index to i64
      %531 = arith.index_cast %c64 : index to i64
      %532 = airrt.dma_memcpy_nd(%c34_i32_143, %519, %c0_i64_140, %arg2[%520, %521, %522, %523], [%528, %529, %530, %531], [%524, %525, %526, %527]) {chan_name = @VIn, metadata = @air_VIn_1_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %533 = airrt.wait_all %532 : !airrt.event
      %c0_i64_144 = arith.constant 0 : i64
      %c0_145 = arith.constant 0 : index
      %c1_146 = arith.constant 1 : index
      %c13_i32_147 = arith.constant 13 : i32
      %534 = arith.index_cast %arg4 : index to i64
      %535 = arith.index_cast %c0_145 : index to i64
      %536 = arith.index_cast %c0_145 : index to i64
      %537 = arith.index_cast %c256 : index to i64
      %538 = arith.index_cast %c256 : index to i64
      %539 = arith.index_cast %c0_145 : index to i64
      %540 = arith.index_cast %c0_145 : index to i64
      %541 = arith.index_cast %c512 : index to i64
      %542 = arith.index_cast %c1_2 : index to i64
      %543 = arith.index_cast %c1_146 : index to i64
      %544 = arith.index_cast %c1_146 : index to i64
      %545 = arith.index_cast %c256 : index to i64
      %546 = arith.index_cast %c64 : index to i64
      %547 = airrt.dma_memcpy_nd(%c13_i32_147, %534, %c0_i64_144, %arg0[%535, %536, %537, %538], [%543, %544, %545, %546], [%539, %540, %541, %542]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %548 = airrt.wait_all %547 : !airrt.event
      %c0_i64_148 = arith.constant 0 : i64
      %c0_149 = arith.constant 0 : index
      %c1_150 = arith.constant 1 : index
      %c13_i32_151 = arith.constant 13 : i32
      %549 = arith.index_cast %arg4 : index to i64
      %550 = arith.index_cast %c0_149 : index to i64
      %551 = arith.index_cast %c0_149 : index to i64
      %552 = arith.index_cast %c256 : index to i64
      %553 = arith.index_cast %c320 : index to i64
      %554 = arith.index_cast %c0_149 : index to i64
      %555 = arith.index_cast %c0_149 : index to i64
      %556 = arith.index_cast %c512 : index to i64
      %557 = arith.index_cast %c1_2 : index to i64
      %558 = arith.index_cast %c1_150 : index to i64
      %559 = arith.index_cast %c1_150 : index to i64
      %560 = arith.index_cast %c256 : index to i64
      %561 = arith.index_cast %c64 : index to i64
      %562 = airrt.dma_memcpy_nd(%c13_i32_151, %549, %c0_i64_148, %arg0[%550, %551, %552, %553], [%558, %559, %560, %561], [%554, %555, %556, %557]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %563 = airrt.wait_all %562, %548 : !airrt.event
      %c0_i64_152 = arith.constant 0 : i64
      %c0_153 = arith.constant 0 : index
      %c1_154 = arith.constant 1 : index
      %c13_i32_155 = arith.constant 13 : i32
      %564 = arith.index_cast %arg4 : index to i64
      %565 = arith.index_cast %c0_153 : index to i64
      %566 = arith.index_cast %c0_153 : index to i64
      %567 = arith.index_cast %c256 : index to i64
      %568 = arith.index_cast %c384 : index to i64
      %569 = arith.index_cast %c0_153 : index to i64
      %570 = arith.index_cast %c0_153 : index to i64
      %571 = arith.index_cast %c512 : index to i64
      %572 = arith.index_cast %c1_2 : index to i64
      %573 = arith.index_cast %c1_154 : index to i64
      %574 = arith.index_cast %c1_154 : index to i64
      %575 = arith.index_cast %c256 : index to i64
      %576 = arith.index_cast %c64 : index to i64
      %577 = airrt.dma_memcpy_nd(%c13_i32_155, %564, %c0_i64_152, %arg0[%565, %566, %567, %568], [%573, %574, %575, %576], [%569, %570, %571, %572]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %578 = airrt.wait_all %577, %563 : !airrt.event
      %c0_i64_156 = arith.constant 0 : i64
      %c0_157 = arith.constant 0 : index
      %c1_158 = arith.constant 1 : index
      %c13_i32_159 = arith.constant 13 : i32
      %579 = arith.index_cast %arg4 : index to i64
      %580 = arith.index_cast %c0_157 : index to i64
      %581 = arith.index_cast %c0_157 : index to i64
      %582 = arith.index_cast %c256 : index to i64
      %583 = arith.index_cast %c448 : index to i64
      %584 = arith.index_cast %c0_157 : index to i64
      %585 = arith.index_cast %c0_157 : index to i64
      %586 = arith.index_cast %c512 : index to i64
      %587 = arith.index_cast %c1_2 : index to i64
      %588 = arith.index_cast %c1_158 : index to i64
      %589 = arith.index_cast %c1_158 : index to i64
      %590 = arith.index_cast %c256 : index to i64
      %591 = arith.index_cast %c64 : index to i64
      %592 = airrt.dma_memcpy_nd(%c13_i32_159, %579, %c0_i64_156, %arg0[%580, %581, %582, %583], [%588, %589, %590, %591], [%584, %585, %586, %587]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %593 = airrt.wait_all %592, %578 : !airrt.event
      %c0_160 = arith.constant 0 : index
      %c1_161 = arith.constant 1 : index
      %594 = airrt.wait_all : !airrt.event
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 1 {
          %c12288 = arith.constant 12288 : index
          %c8192 = arith.constant 8192 : index
          %c4096 = arith.constant 4096 : index
          %c3 = arith.constant 3 : index
          %c2_162 = arith.constant 2 : index
          %c64_163 = arith.constant 64 : index
          %c1_164 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c0_165 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %603 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %604 = airrt.wait_all : !airrt.event
          %605 = airrt.wait_all : !airrt.event
          %606 = airrt.alloc : memref<256x64xbf16, 1 : i32>
          %607 = airrt.wait_all : !airrt.event
          %608 = scf.for %arg7 = %c0_165 to %c8 step %c1_164 iter_args(%arg8 = %605) -> (!airrt.event) {
            %632 = airrt.alloc : memref<64x64xbf16, 1 : i32>
            %633 = airrt.wait_all : !airrt.event
            %634 = airrt.wait_all %633, %arg8 : !airrt.event
            %635 = arith.cmpi eq, %arg5, %c0_165 : index
            %636 = airrt.wait_all %634 : !airrt.event
            %637 = airrt.wait_all %634 : !airrt.event
            %638 = airrt.wait_all %634 : !airrt.event
            %639 = airrt.wait_all %634 : !airrt.event
            %640 = airrt.wait_all %634 : !airrt.event
            %641 = airrt.wait_all %634 : !airrt.event
            %642 = airrt.wait_all %634 : !airrt.event
            %643 = airrt.wait_all %634 : !airrt.event
            %644 = airrt.wait_all %636, %638, %640, %642 : !airrt.event
            airrt.dealloc %632 : memref<64x64xbf16, 1 : i32>
            %645 = airrt.wait_all : !airrt.event
            scf.yield %644 : !airrt.event
          }
          %h = airrt.herd_load "herd_0" (%arg5) {segment_name = "attn_seg"} : (index) -> i64
          %609 = airrt.wait_all : !airrt.event
          %610 = airrt.wait_all %607 : !airrt.event
          %611 = airrt.wait_all %607 : !airrt.event
          %612 = airrt.wait_all %607 : !airrt.event
          %613 = airrt.wait_all %607 : !airrt.event
          %614 = airrt.wait_all %610, %611, %612, %613 : !airrt.event
          %615 = airrt.wait_all %614 : !airrt.event
          %616 = airrt.wait_all %614 : !airrt.event
          %617 = airrt.wait_all %614 : !airrt.event
          %618 = airrt.wait_all %614 : !airrt.event
          %619 = airrt.wait_all %618, %617, %616, %615 : !airrt.event
          %620 = airrt.wait_all %619 : !airrt.event
          %621 = airrt.wait_all %619 : !airrt.event
          %622 = airrt.wait_all %619 : !airrt.event
          %623 = airrt.wait_all %619 : !airrt.event
          %624 = airrt.wait_all %623, %622, %621, %620 : !airrt.event
          %625 = airrt.wait_all %624 : !airrt.event
          %626 = airrt.wait_all %624 : !airrt.event
          %627 = airrt.wait_all %624 : !airrt.event
          %628 = airrt.wait_all %624 : !airrt.event
          %629 = airrt.wait_all %628, %627, %626, %625 : !airrt.event
          airrt.dealloc %603 : memref<64x64xbf16, 1 : i32>
          %630 = airrt.wait_all : !airrt.event
          airrt.dealloc %606 : memref<256x64xbf16, 1 : i32>
          %631 = airrt.wait_all : !airrt.event
          airrt.wait_all %608, %609, %630, %631 {air.segment_end}
        }
      }
      %595 = airrt.wait_all %594 : !airrt.event
      %596 = airrt.wait_all %595 : !airrt.event
      %597 = airrt.wait_all %594, %596 : !airrt.event
      %598 = airrt.wait_all %597 : !airrt.event
      %599 = airrt.wait_all %594 : !airrt.event
      %600 = airrt.wait_all %599 : !airrt.event
      %601 = airrt.wait_all %594, %600 : !airrt.event
      %602 = airrt.wait_all %601 : !airrt.event
      airrt.wait_all %443, %518, %593, %602, %598, %533, %503, %428, %328, %342, %356, %370, %384, %398, %412, %426 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
