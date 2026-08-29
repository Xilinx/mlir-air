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
  func.func @attention_bf16(%arg0: memref<512x512xbf16>, %arg1: memref<512x128xbf16>, %arg2: memref<512x128xbf16>, %arg3: memref<512x512xbf16>) {
    %c448_i64 = arith.constant 448 : i64
    %c384_i64 = arith.constant 384 : i64
    %c320_i64 = arith.constant 320 : i64
    %c192_i64 = arith.constant 192 : i64
    %c256_i64 = arith.constant 256 : i64
    %c64_i64 = arith.constant 64 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c8 = arith.constant 8 : index
    %c13_i32 = arith.constant 13 : i32
    %c34_i32 = arith.constant 34 : i32
    %c96_i32 = arith.constant 96 : i32
    %c25_i32 = arith.constant 25 : i32
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.for %arg4 = 0 to 1 {
      %p = airrt.segment_load "attn_seg" : i64
      %0 = arith.index_cast %arg4 : index to i64
      %1 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %2 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %3 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c128_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %4 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c192_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %5 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c256_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %6 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c320_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %7 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c384_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %8 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c448_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %9 = airrt.dma_memcpy_nd(%c25_i32, %0, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c512_i64, %c64_i64], [%c0_i64, %c0_i64, %c128_i64, %c1_i64]) {chan_name = @KIn, metadata = @air_KIn_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %10 = airrt.dma_memcpy_nd(%c34_i32, %0, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c512_i64, %c64_i64], [%c0_i64, %c0_i64, %c128_i64, %c1_i64]) {chan_name = @VIn, metadata = @air_VIn_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %11 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %12 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %13 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c128_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %14 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c192_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %15 = airrt.dma_memcpy_nd(%c25_i32, %0, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c512_i64, %c64_i64], [%c0_i64, %c0_i64, %c128_i64, %c1_i64]) {chan_name = @KIn, metadata = @air_KIn_1_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %16 = airrt.dma_memcpy_nd(%c34_i32, %0, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c512_i64, %c64_i64], [%c0_i64, %c0_i64, %c128_i64, %c1_i64]) {chan_name = @VIn, metadata = @air_VIn_1_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %17 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c256_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %18 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c320_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %19 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c384_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %20 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c448_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 1 {
          %41 = airrt.wait_all : !airrt.event
          %42 = scf.for %arg7 = %c0 to %c8 step %c1 iter_args(%arg8 = %41) -> (!airrt.event) {
            %43 = airrt.wait_all %arg8 : !airrt.event
            scf.yield %43 : !airrt.event
          }
          %h = airrt.herd_load "herd_0" (%arg5) {segment_name = "attn_seg"} : (index) -> i64
          airrt.wait_all %42 {air.segment_end}
        }
      }
      airrt.wait_all %1, %2, %3, %4, %5, %6, %7, %8, %9, %14, %13, %12, %11, %16, %20, %19, %18, %17, %15, %10 {air.launch_end}
      %21 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %22 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c256_i64, %c64_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %23 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c256_i64, %c128_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %24 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c256_i64, %c192_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %25 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c256_i64, %c256_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %26 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c256_i64, %c320_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %27 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c256_i64, %c384_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %28 = airrt.dma_memcpy_nd(%c96_i32, %0, %c0_i64, %arg3[%c0_i64, %c0_i64, %c256_i64, %c448_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @GpOut, metadata = @air_GpOut_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %29 = airrt.dma_memcpy_nd(%c25_i32, %0, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c512_i64, %c64_i64], [%c0_i64, %c0_i64, %c128_i64, %c1_i64]) {chan_name = @KIn, metadata = @air_KIn_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %30 = airrt.dma_memcpy_nd(%c34_i32, %0, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c512_i64, %c64_i64], [%c0_i64, %c0_i64, %c128_i64, %c1_i64]) {chan_name = @VIn, metadata = @air_VIn_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %31 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %32 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c64_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %33 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c128_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %34 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c192_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %35 = airrt.dma_memcpy_nd(%c25_i32, %0, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c512_i64, %c64_i64], [%c0_i64, %c0_i64, %c128_i64, %c1_i64]) {chan_name = @KIn, metadata = @air_KIn_1_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %36 = airrt.dma_memcpy_nd(%c34_i32, %0, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c512_i64, %c64_i64], [%c0_i64, %c0_i64, %c128_i64, %c1_i64]) {chan_name = @VIn, metadata = @air_VIn_1_0_0} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %37 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c256_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %38 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c320_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %39 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c384_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %40 = airrt.dma_memcpy_nd(%c13_i32, %0, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c448_i64], [%c1_i64, %c1_i64, %c256_i64, %c64_i64], [%c0_i64, %c0_i64, %c512_i64, %c1_i64]) {chan_name = @QIn, metadata = @air_QIn_1_0_0} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 1 {
          %41 = airrt.wait_all : !airrt.event
          %42 = scf.for %arg7 = %c0 to %c8 step %c1 iter_args(%arg8 = %41) -> (!airrt.event) {
            %43 = airrt.wait_all %arg8 : !airrt.event
            scf.yield %43 : !airrt.event
          }
          %h = airrt.herd_load "herd_0" (%arg5) {segment_name = "attn_seg"} : (index) -> i64
          airrt.wait_all %42 {air.segment_end}
        }
      }
      airrt.wait_all %21, %22, %23, %24, %25, %26, %27, %28, %30, %35, %40, %39, %38, %37, %36, %34, %33, %32, %31, %29 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
