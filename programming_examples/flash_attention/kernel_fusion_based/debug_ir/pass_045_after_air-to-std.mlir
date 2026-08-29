#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @attn_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
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
    %lock_3_1 = aie.lock(%mem_tile_3_1, 5) {init = 1 : i32}
    %lock_3_1_0 = aie.lock(%mem_tile_3_1, 4) {init = 0 : i32}
    %lock_3_1_1 = aie.lock(%mem_tile_3_1, 3) {init = 1 : i32}
    %lock_3_1_2 = aie.lock(%mem_tile_3_1, 2) {init = 0 : i32}
    %lock_3_1_3 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_4 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 5) {init = 1 : i32}
    %lock_2_1_5 = aie.lock(%mem_tile_2_1, 4) {init = 0 : i32}
    %lock_2_1_6 = aie.lock(%mem_tile_2_1, 3) {init = 1 : i32}
    %lock_2_1_7 = aie.lock(%mem_tile_2_1, 2) {init = 0 : i32}
    %lock_2_1_8 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_9 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 5) {init = 1 : i32}
    %lock_1_1_10 = aie.lock(%mem_tile_1_1, 4) {init = 0 : i32}
    %lock_1_1_11 = aie.lock(%mem_tile_1_1, 3) {init = 1 : i32}
    %lock_1_1_12 = aie.lock(%mem_tile_1_1, 2) {init = 0 : i32}
    %lock_1_1_13 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_14 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 5) {init = 1 : i32}
    %lock_0_1_15 = aie.lock(%mem_tile_0_1, 4) {init = 0 : i32}
    %lock_0_1_16 = aie.lock(%mem_tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_17 = aie.lock(%mem_tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_18 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_19 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_20 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_21 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_22 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_23 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_24 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_25 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_26 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_27 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_28 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_29 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_30 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_31 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_32 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_33 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_34 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_35 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_36 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_37 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_38 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_39 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_40 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_41 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_42 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_43 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_44 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_45 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_46 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_47 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_48 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 3) {init = 1 : i32}
    %lock_3_3_49 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_50 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_51 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_52 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_53 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_54 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 3) {init = 1 : i32}
    %lock_1_4_55 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_56 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_57 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 3) {init = 1 : i32}
    %lock_2_4_58 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_59 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_60 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 3) {init = 1 : i32}
    %lock_3_4_61 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_62 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_63 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_64 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_65 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_66 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 3) {init = 1 : i32}
    %lock_1_5_67 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_68 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_69 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 3) {init = 1 : i32}
    %lock_2_5_70 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_71 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_72 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 3) {init = 1 : i32}
    %lock_3_5_73 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_74 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_75 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf255_unroll_0 = aie.buffer(%mem_tile_0_1) {sym_name = "buf255_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf254_unroll_0 = aie.buffer(%mem_tile_1_1) {sym_name = "buf254_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf253_unroll_0 = aie.buffer(%mem_tile_2_1) {sym_name = "buf253_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf252_unroll_0 = aie.buffer(%mem_tile_3_1) {sym_name = "buf252_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf251_unroll_0 = aie.buffer(%mem_tile_0_1) {sym_name = "buf251_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf250_unroll_0 = aie.buffer(%mem_tile_1_1) {sym_name = "buf250_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf249_unroll_0 = aie.buffer(%mem_tile_2_1) {sym_name = "buf249_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf248_unroll_0 = aie.buffer(%mem_tile_3_1) {sym_name = "buf248_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf247_unroll_0 = aie.buffer(%mem_tile_0_1) {sym_name = "buf247_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf246_unroll_0 = aie.buffer(%mem_tile_1_1) {sym_name = "buf246_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf245_unroll_0 = aie.buffer(%mem_tile_2_1) {sym_name = "buf245_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf244_unroll_0 = aie.buffer(%mem_tile_3_1) {sym_name = "buf244_unroll_0"} : memref<64x64xbf16, 1 : i32> 
    %buf243_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf243_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf242_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf242_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf241_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf241_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf240_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf240_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf239_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf239_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf238_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf238_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf237_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf237_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf236_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf236_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf235_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf235_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf234_unroll_0 = aie.buffer(%tile_3_5) {sym_name = "buf234_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf233_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf233_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf232_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf232_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf231_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf231_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf230_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf230_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf229_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf229_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf228_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf228_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf227_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf227_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf226_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf226_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf225_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf225_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf224_unroll_0 = aie.buffer(%tile_2_5) {sym_name = "buf224_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf223_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf223_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf222_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf222_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf221_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf221_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf220_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf220_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf219_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf219_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf218_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf218_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf217_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf217_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf216_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf216_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf215_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf215_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf214_unroll_0 = aie.buffer(%tile_1_5) {sym_name = "buf214_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf213_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf213_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf212_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf212_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf211_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf211_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf210_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf210_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf209_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf209_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf208_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf208_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf207_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf207_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf206_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf206_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf205_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf205_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf204_unroll_0 = aie.buffer(%tile_0_5) {sym_name = "buf204_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf203_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf203_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf202_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf202_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf201_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf201_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf200_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf200_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf199_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf199_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf198_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf198_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf197_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf197_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf196_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf196_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf195_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf195_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf194_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf194_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf193_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf193_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf192_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf192_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf191_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf191_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf190_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf190_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf189_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf189_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf188_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf188_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf187_unroll_0 = aie.buffer(%tile_3_4) {sym_name = "buf187_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf186_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf186_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf185_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf185_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf184_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf184_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf183_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf183_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf182_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf182_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf181_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf181_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf180_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf180_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf179_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf179_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf178_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf178_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf177_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf177_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf176_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf176_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf175_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf175_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf174_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf174_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf173_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf173_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf172_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf172_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf171_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf171_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf170_unroll_0 = aie.buffer(%tile_2_4) {sym_name = "buf170_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf169_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf169_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf168_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf168_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf167_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf167_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf166_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf166_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf165_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf165_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf164_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf164_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf163_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf163_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf162_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf162_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf161_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf161_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf160_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf160_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf159_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf159_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf158_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf158_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf157_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf157_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf156_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf156_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf155_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf155_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf154_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf154_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf153_unroll_0 = aie.buffer(%tile_1_4) {sym_name = "buf153_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf152_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf152_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf151_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf151_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf150_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf150_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf149_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf149_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf148_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf148_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf147_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf147_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf146_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf146_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf145_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf145_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf144_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf144_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf143_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf143_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf142_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf142_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf141_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf141_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf140_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf140_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf139_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf139_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf138_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf138_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf137_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf137_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf136_unroll_0 = aie.buffer(%tile_0_4) {sym_name = "buf136_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf135_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf135_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf134_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf134_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf133_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf133_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf132_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf132_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf131_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf131_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf130_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf130_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf129_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf129_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf128_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf128_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf127_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf127_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf126_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf126_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf125_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf125_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf124_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf124_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf123_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf123_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf122_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf122_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf121_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf121_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf120_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf120_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf119_unroll_0 = aie.buffer(%tile_3_3) {sym_name = "buf119_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf118_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf118_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf117_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf117_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf116_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf116_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf115_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf115_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf114_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf114_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf113_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf113_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf112_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf112_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf111_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf111_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf110_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf110_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf109_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf109_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf108_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf108_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf107_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf107_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf106_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf106_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf105_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf105_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf104_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf104_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf103_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf103_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf102_unroll_0 = aie.buffer(%tile_2_3) {sym_name = "buf102_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf101_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf101_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf100_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf100_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf99_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf99_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf98_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf98_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf97_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf97_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf96_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf96_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf95_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf95_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf94_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf94_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf93_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf93_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf92_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf92_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf91_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf91_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf90_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf90_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf89_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf89_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf88_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf88_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf87_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf87_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf86_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf86_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf85_unroll_0 = aie.buffer(%tile_1_3) {sym_name = "buf85_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf84_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf84_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf83_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf83_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf82_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf82_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf81_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf81_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf80_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf80_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf79_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf79_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf78_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf78_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf77_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf77_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf76_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf76_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf75_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf75_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf74_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf74_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf73_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf73_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf72_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf72_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf71_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf71_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf70_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf70_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf69_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf69_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf68_unroll_0 = aie.buffer(%tile_0_3) {sym_name = "buf68_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf67_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf67_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf66_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf66_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf65_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf65_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf64_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf64_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf63_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf63_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf62_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf62_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf61_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf61_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf60_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf60_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf59_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf59_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf58_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf58_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf57_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf57_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf56_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf56_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf55_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf55_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf54_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf54_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf53_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf53_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf52_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf52_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf51_unroll_0 = aie.buffer(%tile_3_2) {sym_name = "buf51_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf50_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf50_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf49_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf49_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf48_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf48_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf47_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf47_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf46_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf46_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf45_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf45_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf44_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf44_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf43_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf43_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf42_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf42_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf41_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf41_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf40_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf40_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf39_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf39_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf38_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf38_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf37_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf37_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf36_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf36_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf35_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf35_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf34_unroll_0 = aie.buffer(%tile_2_2) {sym_name = "buf34_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf33_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf33_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf32_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf32_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf31_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf31_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf30_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf30_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf29_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf29_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf28_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf28_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf27_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf27_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf26_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf26_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf25_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf25_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf24_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf24_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf23_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf23_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf22_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf22_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf21_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf21_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf20_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf20_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf19_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf19_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf18_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf18_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf17_unroll_0 = aie.buffer(%tile_1_2) {sym_name = "buf17_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf16_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf16_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf15_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf15_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf14_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf14_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf13_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf13_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf12_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf12_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf11_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf11_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf10_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf10_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf9_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf9_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf8_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf8_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf7_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf7_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf6_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf6_unroll_0"} : memref<64x64xbf16, 2 : i32> 
    %buf5_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf5_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf4_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf4_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf3_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf3_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf2_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf2_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf1_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf1_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %buf0_unroll_0 = aie.buffer(%tile_0_2) {sym_name = "buf0_unroll_0"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer_unroll_0 = aie.external_buffer {sym_name = "__air_external_buffer_unroll_0"} : memref<2x256x128xbf16>
    %__air_external_buffer_1_unroll_0 = aie.external_buffer {sym_name = "__air_external_buffer_1_unroll_0"} : memref<2x512x128xbf16>
    %__air_external_buffer_2_unroll_0 = aie.external_buffer {sym_name = "__air_external_buffer_2_unroll_0"} : memref<2x512x64xbf16>
    %__air_external_buffer_3_unroll_0 = aie.external_buffer {sym_name = "__air_external_buffer_3_unroll_0"} : memref<2x256x64xbf16>
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_74, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf240_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_75, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf237_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_73, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf241_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf243_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf242_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf240_unroll_0, %buf238_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf240_unroll_0, %buf239_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf236_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf238_unroll_0, %buf240_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_74, Release, 1)
        aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf239_unroll_0, %buf240_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_74, Release, 1)
        aie.use_lock(%lock_3_5_73, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf242_unroll_0, %buf235_unroll_0, %buf234_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf234_unroll_0, %buf241_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf237_unroll_0, %buf241_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf243_unroll_0, %buf234_unroll_0, %buf235_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf235_unroll_0, %buf243_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf241_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_152 = memref.collapse_shape %buf242_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_153 = memref.collapse_shape %buf243_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_71, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf227_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_70, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf231_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf233_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf232_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf230_unroll_0, %buf228_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf230_unroll_0, %buf229_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf226_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf228_unroll_0, %buf230_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_71, Release, 1)
        aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf229_unroll_0, %buf230_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_71, Release, 1)
        aie.use_lock(%lock_2_5_70, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf232_unroll_0, %buf225_unroll_0, %buf224_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf224_unroll_0, %buf231_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf227_unroll_0, %buf231_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf233_unroll_0, %buf224_unroll_0, %buf225_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf225_unroll_0, %buf233_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf231_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_152 = memref.collapse_shape %buf232_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_153 = memref.collapse_shape %buf233_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_68, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf220_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_69, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf217_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_67, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf221_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf223_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf222_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf220_unroll_0, %buf218_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf220_unroll_0, %buf219_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf216_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf218_unroll_0, %buf220_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_68, Release, 1)
        aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf219_unroll_0, %buf220_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_68, Release, 1)
        aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf222_unroll_0, %buf215_unroll_0, %buf214_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf214_unroll_0, %buf221_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf217_unroll_0, %buf221_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf223_unroll_0, %buf214_unroll_0, %buf215_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf215_unroll_0, %buf223_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf221_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_152 = memref.collapse_shape %buf222_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_153 = memref.collapse_shape %buf223_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_65, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf210_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_66, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf207_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_64, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf211_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf213_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf212_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf210_unroll_0, %buf208_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf210_unroll_0, %buf209_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf206_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf208_unroll_0, %buf210_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_65, Release, 1)
        aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf209_unroll_0, %buf210_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_65, Release, 1)
        aie.use_lock(%lock_0_5_64, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf212_unroll_0, %buf205_unroll_0, %buf204_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf204_unroll_0, %buf211_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf207_unroll_0, %buf211_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf213_unroll_0, %buf204_unroll_0, %buf205_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf205_unroll_0, %buf213_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf211_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_152 = memref.collapse_shape %buf212_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_153 = memref.collapse_shape %buf213_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf200_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_63, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf197_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_61, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf201_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf203_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf202_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf200_unroll_0, %buf198_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf200_unroll_0, %buf199_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf196_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf198_unroll_0, %buf200_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_62, Release, 1)
        aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf199_unroll_0, %buf200_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_62, Release, 1)
        aie.use_lock(%lock_3_4_61, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf202_unroll_0, %buf195_unroll_0, %buf194_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf194_unroll_0, %buf201_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf197_unroll_0, %buf201_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf203_unroll_0, %buf194_unroll_0, %buf195_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf195_unroll_0, %buf203_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf193_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf192_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf191_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf202_unroll_0, %buf190_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf192_unroll_0, %buf202_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf192_unroll_0, %buf202_unroll_0, %buf189_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf190_unroll_0, %buf202_unroll_0, %buf188_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf189_unroll_0, %buf193_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf188_unroll_0, %buf201_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf201_unroll_0, %buf193_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf187_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf191_unroll_0, %buf189_unroll_0, %buf187_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf203_unroll_0, %buf188_unroll_0, %buf187_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf187_unroll_0, %buf191_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf202_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_59, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf183_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf180_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_58, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf184_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf186_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf185_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf183_unroll_0, %buf181_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf183_unroll_0, %buf182_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf179_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf181_unroll_0, %buf183_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_59, Release, 1)
        aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf182_unroll_0, %buf183_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_59, Release, 1)
        aie.use_lock(%lock_2_4_58, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf185_unroll_0, %buf178_unroll_0, %buf177_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf177_unroll_0, %buf184_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf180_unroll_0, %buf184_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf186_unroll_0, %buf177_unroll_0, %buf178_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf178_unroll_0, %buf186_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf176_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf175_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf174_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf185_unroll_0, %buf173_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf175_unroll_0, %buf185_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf175_unroll_0, %buf185_unroll_0, %buf172_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf173_unroll_0, %buf185_unroll_0, %buf171_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf172_unroll_0, %buf176_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf171_unroll_0, %buf184_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf184_unroll_0, %buf176_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf170_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf174_unroll_0, %buf172_unroll_0, %buf170_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf186_unroll_0, %buf171_unroll_0, %buf170_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf170_unroll_0, %buf174_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf185_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_56, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf166_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf163_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_55, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf167_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf169_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf168_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf166_unroll_0, %buf164_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf166_unroll_0, %buf165_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf162_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf164_unroll_0, %buf166_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_56, Release, 1)
        aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf165_unroll_0, %buf166_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_56, Release, 1)
        aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf168_unroll_0, %buf161_unroll_0, %buf160_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf160_unroll_0, %buf167_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf163_unroll_0, %buf167_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf169_unroll_0, %buf160_unroll_0, %buf161_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf161_unroll_0, %buf169_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf159_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf158_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf157_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf168_unroll_0, %buf156_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf158_unroll_0, %buf168_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf158_unroll_0, %buf168_unroll_0, %buf155_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf156_unroll_0, %buf168_unroll_0, %buf154_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf155_unroll_0, %buf159_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf154_unroll_0, %buf167_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf167_unroll_0, %buf159_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf153_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf157_unroll_0, %buf155_unroll_0, %buf153_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf169_unroll_0, %buf154_unroll_0, %buf153_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf153_unroll_0, %buf157_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf168_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf149_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf146_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_52, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf150_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf152_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf151_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf149_unroll_0, %buf147_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf149_unroll_0, %buf148_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf145_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf147_unroll_0, %buf149_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_53, Release, 1)
        aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf148_unroll_0, %buf149_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_53, Release, 1)
        aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf151_unroll_0, %buf144_unroll_0, %buf143_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf143_unroll_0, %buf150_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf146_unroll_0, %buf150_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf152_unroll_0, %buf143_unroll_0, %buf144_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf144_unroll_0, %buf152_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf142_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf141_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf140_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf151_unroll_0, %buf139_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf141_unroll_0, %buf151_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf141_unroll_0, %buf151_unroll_0, %buf138_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf139_unroll_0, %buf151_unroll_0, %buf137_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf138_unroll_0, %buf142_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf137_unroll_0, %buf150_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf150_unroll_0, %buf142_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf136_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf140_unroll_0, %buf138_unroll_0, %buf136_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf152_unroll_0, %buf137_unroll_0, %buf136_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf136_unroll_0, %buf140_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf151_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf132_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_51, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf129_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_49, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf133_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf135_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf134_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf132_unroll_0, %buf130_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf132_unroll_0, %buf131_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf128_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf130_unroll_0, %buf132_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_50, Release, 1)
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf131_unroll_0, %buf132_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_50, Release, 1)
        aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf134_unroll_0, %buf127_unroll_0, %buf126_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf126_unroll_0, %buf133_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf129_unroll_0, %buf133_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf135_unroll_0, %buf126_unroll_0, %buf127_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf127_unroll_0, %buf135_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf125_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf124_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf123_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf134_unroll_0, %buf122_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf124_unroll_0, %buf134_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf124_unroll_0, %buf134_unroll_0, %buf121_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf122_unroll_0, %buf134_unroll_0, %buf120_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf121_unroll_0, %buf125_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf120_unroll_0, %buf133_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf133_unroll_0, %buf125_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf119_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf123_unroll_0, %buf121_unroll_0, %buf119_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf135_unroll_0, %buf120_unroll_0, %buf119_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf119_unroll_0, %buf123_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf134_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf115_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf112_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_46, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf116_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf118_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf117_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf115_unroll_0, %buf113_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf115_unroll_0, %buf114_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf111_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf113_unroll_0, %buf115_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_47, Release, 1)
        aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf114_unroll_0, %buf115_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_47, Release, 1)
        aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf117_unroll_0, %buf110_unroll_0, %buf109_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf109_unroll_0, %buf116_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf112_unroll_0, %buf116_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf118_unroll_0, %buf109_unroll_0, %buf110_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf110_unroll_0, %buf118_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf108_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf107_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf106_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf117_unroll_0, %buf105_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf107_unroll_0, %buf117_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf107_unroll_0, %buf117_unroll_0, %buf104_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf105_unroll_0, %buf117_unroll_0, %buf103_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf104_unroll_0, %buf108_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf103_unroll_0, %buf116_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf116_unroll_0, %buf108_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf102_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf106_unroll_0, %buf104_unroll_0, %buf102_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf118_unroll_0, %buf103_unroll_0, %buf102_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf102_unroll_0, %buf106_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf117_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_44, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf98_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf95_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_43, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf99_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf101_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf100_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf98_unroll_0, %buf96_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf98_unroll_0, %buf97_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf94_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf96_unroll_0, %buf98_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_44, Release, 1)
        aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf97_unroll_0, %buf98_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_44, Release, 1)
        aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf100_unroll_0, %buf93_unroll_0, %buf92_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf92_unroll_0, %buf99_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf95_unroll_0, %buf99_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf101_unroll_0, %buf92_unroll_0, %buf93_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf93_unroll_0, %buf101_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf91_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf90_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf89_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf100_unroll_0, %buf88_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf90_unroll_0, %buf100_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf90_unroll_0, %buf100_unroll_0, %buf87_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf88_unroll_0, %buf100_unroll_0, %buf86_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf87_unroll_0, %buf91_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf86_unroll_0, %buf99_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf99_unroll_0, %buf91_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf85_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf89_unroll_0, %buf87_unroll_0, %buf85_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf101_unroll_0, %buf86_unroll_0, %buf85_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf85_unroll_0, %buf89_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf100_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf81_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf78_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_40, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf82_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf84_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf83_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf81_unroll_0, %buf79_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf81_unroll_0, %buf80_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf77_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf79_unroll_0, %buf81_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_41, Release, 1)
        aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf80_unroll_0, %buf81_unroll_0, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_41, Release, 1)
        aie.use_lock(%lock_0_3_40, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf83_unroll_0, %buf76_unroll_0, %buf75_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf75_unroll_0, %buf82_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf78_unroll_0, %buf82_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf84_unroll_0, %buf75_unroll_0, %buf76_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf76_unroll_0, %buf84_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf74_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf73_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf72_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf83_unroll_0, %buf71_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf73_unroll_0, %buf83_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf73_unroll_0, %buf83_unroll_0, %buf70_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf71_unroll_0, %buf83_unroll_0, %buf69_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf70_unroll_0, %buf74_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf69_unroll_0, %buf82_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf82_unroll_0, %buf74_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf68_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf72_unroll_0, %buf70_unroll_0, %buf68_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf84_unroll_0, %buf69_unroll_0, %buf68_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf68_unroll_0, %buf72_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf83_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf57_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf64_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf61_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_35, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_38, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf65_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf67_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf66_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf64_unroll_0, %buf62_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf64_unroll_0, %buf63_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf60_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf62_unroll_0, %buf64_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_36, Release, 1)
        aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf63_unroll_0, %buf64_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_36, Release, 1)
        aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf66_unroll_0, %buf59_unroll_0, %buf58_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf58_unroll_0, %buf65_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf61_unroll_0, %buf65_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf67_unroll_0, %buf58_unroll_0, %buf59_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf59_unroll_0, %buf67_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf57_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf56_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf55_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf66_unroll_0, %buf54_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf56_unroll_0, %buf66_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf56_unroll_0, %buf66_unroll_0, %buf53_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf54_unroll_0, %buf66_unroll_0, %buf52_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf53_unroll_0, %buf57_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf52_unroll_0, %buf65_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf65_unroll_0, %buf57_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf51_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf55_unroll_0, %buf53_unroll_0, %buf51_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf67_unroll_0, %buf52_unroll_0, %buf51_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf51_unroll_0, %buf55_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf55_unroll_0, %buf57_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_39, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf40_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf47_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf44_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_30, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_33, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf48_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf50_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf49_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf47_unroll_0, %buf45_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf47_unroll_0, %buf46_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf43_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf45_unroll_0, %buf47_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_31, Release, 1)
        aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf46_unroll_0, %buf47_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_31, Release, 1)
        aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf49_unroll_0, %buf42_unroll_0, %buf41_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf41_unroll_0, %buf48_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf44_unroll_0, %buf48_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf50_unroll_0, %buf41_unroll_0, %buf42_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf42_unroll_0, %buf50_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf40_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf39_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf38_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf49_unroll_0, %buf37_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf39_unroll_0, %buf49_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf39_unroll_0, %buf49_unroll_0, %buf36_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf37_unroll_0, %buf49_unroll_0, %buf35_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf36_unroll_0, %buf40_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf35_unroll_0, %buf48_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf48_unroll_0, %buf40_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf34_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf38_unroll_0, %buf36_unroll_0, %buf34_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf50_unroll_0, %buf35_unroll_0, %buf34_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf34_unroll_0, %buf38_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf38_unroll_0, %buf40_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_34, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf23_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_27, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf27_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_25, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_28, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf31_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf33_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf32_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf30_unroll_0, %buf28_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf30_unroll_0, %buf29_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf26_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf28_unroll_0, %buf30_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_26, Release, 1)
        aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf29_unroll_0, %buf30_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_26, Release, 1)
        aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf32_unroll_0, %buf25_unroll_0, %buf24_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf24_unroll_0, %buf31_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf27_unroll_0, %buf31_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf33_unroll_0, %buf24_unroll_0, %buf25_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf25_unroll_0, %buf33_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf23_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf22_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf21_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf32_unroll_0, %buf20_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf22_unroll_0, %buf32_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf22_unroll_0, %buf32_unroll_0, %buf19_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf20_unroll_0, %buf32_unroll_0, %buf18_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf19_unroll_0, %buf23_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf18_unroll_0, %buf31_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf31_unroll_0, %buf23_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf17_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf21_unroll_0, %buf19_unroll_0, %buf17_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf33_unroll_0, %buf18_unroll_0, %buf17_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf17_unroll_0, %buf21_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf21_unroll_0, %buf23_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_29, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_23, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10_unroll_0 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_20, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_23, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf14_unroll_0) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf16_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf15_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf13_unroll_0, %buf11_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf13_unroll_0, %buf12_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf9_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf11_unroll_0, %buf13_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_21, Release, 1)
        aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf12_unroll_0, %buf13_unroll_0, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_21, Release, 1)
        aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf15_unroll_0, %buf8_unroll_0, %buf7_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf7_unroll_0, %buf14_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf10_unroll_0, %buf14_unroll_0) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf16_unroll_0, %buf7_unroll_0, %buf8_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf8_unroll_0, %buf16_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf6_unroll_0 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf5_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf4_unroll_0 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf15_unroll_0, %buf3_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf5_unroll_0, %buf15_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf5_unroll_0, %buf15_unroll_0, %buf2_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf3_unroll_0, %buf15_unroll_0, %buf1_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf2_unroll_0, %buf6_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf1_unroll_0, %buf14_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf14_unroll_0, %buf6_unroll_0) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf0_unroll_0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf4_unroll_0, %buf2_unroll_0, %buf0_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf16_unroll_0, %buf1_unroll_0, %buf0_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf0_unroll_0, %buf4_unroll_0) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf4_unroll_0, %buf6_unroll_0) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_24, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>)
    func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>)
    func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>)
    func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>)
    func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>)
    func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>)
    func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>)
    func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>)
    func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>)
    func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>)
    func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>)
    func.func private @maximum_up_u_bf16(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>)
    func.func private @exp_up_minus_u(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>)
    func.func private @add_gp_g(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>)
    func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_1_1, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_2_1, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_3_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_2_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_3_2, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_2_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_3_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 2)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 2)
    aie.cascade_flow(%tile_3_5, %tile_3_4)
    aie.cascade_flow(%tile_2_5, %tile_2_4)
    aie.cascade_flow(%tile_1_5, %tile_1_4)
    aie.cascade_flow(%tile_0_5, %tile_0_4)
    aie.cascade_flow(%tile_3_4, %tile_3_3)
    aie.cascade_flow(%tile_2_4, %tile_2_3)
    aie.cascade_flow(%tile_1_4, %tile_1_3)
    aie.cascade_flow(%tile_0_4, %tile_0_3)
    aie.cascade_flow(%tile_3_3, %tile_3_2)
    aie.cascade_flow(%tile_2_3, %tile_2_2)
    aie.cascade_flow(%tile_1_3, %tile_1_2)
    aie.cascade_flow(%tile_0_3, %tile_0_2)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf251_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf255_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_16, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf247_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf255_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_17, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf247_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_15, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_0_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf251_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_19, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf250_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf254_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_11, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf246_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf254_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_12, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf246_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_10, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_1_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf250_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_14, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf249_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf253_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_6, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf245_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf253_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_7, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf245_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_2_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf249_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf248_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf252_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf244_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf252_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf244_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_3_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf248_unroll_0 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_4, Release, 1)
      aie.next_bd ^bb12
    }
    aie.shim_dma_allocation @air_channel_0_0_0_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_0_0_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_0_0_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_0_0_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_QKIn_0_0_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_1_0_0(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_2_0_0(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_3_0_0(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_0_0_0(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_1_0_0(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_2_0_0(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_3_0_0(%shim_noc_tile_3_0, MM2S, 1)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
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
    %lock_7_1 = aie.lock(%mem_tile_7_1, 5) {init = 1 : i32}
    %lock_7_1_76 = aie.lock(%mem_tile_7_1, 4) {init = 0 : i32}
    %lock_7_1_77 = aie.lock(%mem_tile_7_1, 3) {init = 1 : i32}
    %lock_7_1_78 = aie.lock(%mem_tile_7_1, 2) {init = 0 : i32}
    %lock_7_1_79 = aie.lock(%mem_tile_7_1, 1) {init = 1 : i32}
    %lock_7_1_80 = aie.lock(%mem_tile_7_1, 0) {init = 0 : i32}
    %lock_6_1 = aie.lock(%mem_tile_6_1, 5) {init = 1 : i32}
    %lock_6_1_81 = aie.lock(%mem_tile_6_1, 4) {init = 0 : i32}
    %lock_6_1_82 = aie.lock(%mem_tile_6_1, 3) {init = 1 : i32}
    %lock_6_1_83 = aie.lock(%mem_tile_6_1, 2) {init = 0 : i32}
    %lock_6_1_84 = aie.lock(%mem_tile_6_1, 1) {init = 1 : i32}
    %lock_6_1_85 = aie.lock(%mem_tile_6_1, 0) {init = 0 : i32}
    %lock_5_1 = aie.lock(%mem_tile_5_1, 5) {init = 1 : i32}
    %lock_5_1_86 = aie.lock(%mem_tile_5_1, 4) {init = 0 : i32}
    %lock_5_1_87 = aie.lock(%mem_tile_5_1, 3) {init = 1 : i32}
    %lock_5_1_88 = aie.lock(%mem_tile_5_1, 2) {init = 0 : i32}
    %lock_5_1_89 = aie.lock(%mem_tile_5_1, 1) {init = 1 : i32}
    %lock_5_1_90 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %lock_4_1 = aie.lock(%mem_tile_4_1, 5) {init = 1 : i32}
    %lock_4_1_91 = aie.lock(%mem_tile_4_1, 4) {init = 0 : i32}
    %lock_4_1_92 = aie.lock(%mem_tile_4_1, 3) {init = 1 : i32}
    %lock_4_1_93 = aie.lock(%mem_tile_4_1, 2) {init = 0 : i32}
    %lock_4_1_94 = aie.lock(%mem_tile_4_1, 1) {init = 1 : i32}
    %lock_4_1_95 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_4_2 = aie.lock(%tile_4_2, 5) {init = 1 : i32}
    %lock_4_2_96 = aie.lock(%tile_4_2, 4) {init = 0 : i32}
    %lock_4_2_97 = aie.lock(%tile_4_2, 3) {init = 1 : i32}
    %lock_4_2_98 = aie.lock(%tile_4_2, 2) {init = 0 : i32}
    %lock_4_2_99 = aie.lock(%tile_4_2, 1) {init = 1 : i32}
    %lock_4_2_100 = aie.lock(%tile_4_2, 0) {init = 0 : i32}
    %lock_5_2 = aie.lock(%tile_5_2, 5) {init = 1 : i32}
    %lock_5_2_101 = aie.lock(%tile_5_2, 4) {init = 0 : i32}
    %lock_5_2_102 = aie.lock(%tile_5_2, 3) {init = 1 : i32}
    %lock_5_2_103 = aie.lock(%tile_5_2, 2) {init = 0 : i32}
    %lock_5_2_104 = aie.lock(%tile_5_2, 1) {init = 1 : i32}
    %lock_5_2_105 = aie.lock(%tile_5_2, 0) {init = 0 : i32}
    %lock_6_2 = aie.lock(%tile_6_2, 5) {init = 1 : i32}
    %lock_6_2_106 = aie.lock(%tile_6_2, 4) {init = 0 : i32}
    %lock_6_2_107 = aie.lock(%tile_6_2, 3) {init = 1 : i32}
    %lock_6_2_108 = aie.lock(%tile_6_2, 2) {init = 0 : i32}
    %lock_6_2_109 = aie.lock(%tile_6_2, 1) {init = 1 : i32}
    %lock_6_2_110 = aie.lock(%tile_6_2, 0) {init = 0 : i32}
    %lock_7_2 = aie.lock(%tile_7_2, 5) {init = 1 : i32}
    %lock_7_2_111 = aie.lock(%tile_7_2, 4) {init = 0 : i32}
    %lock_7_2_112 = aie.lock(%tile_7_2, 3) {init = 1 : i32}
    %lock_7_2_113 = aie.lock(%tile_7_2, 2) {init = 0 : i32}
    %lock_7_2_114 = aie.lock(%tile_7_2, 1) {init = 1 : i32}
    %lock_7_2_115 = aie.lock(%tile_7_2, 0) {init = 0 : i32}
    %lock_4_3 = aie.lock(%tile_4_3, 3) {init = 1 : i32}
    %lock_4_3_116 = aie.lock(%tile_4_3, 2) {init = 0 : i32}
    %lock_4_3_117 = aie.lock(%tile_4_3, 1) {init = 1 : i32}
    %lock_4_3_118 = aie.lock(%tile_4_3, 0) {init = 0 : i32}
    %lock_5_3 = aie.lock(%tile_5_3, 3) {init = 1 : i32}
    %lock_5_3_119 = aie.lock(%tile_5_3, 2) {init = 0 : i32}
    %lock_5_3_120 = aie.lock(%tile_5_3, 1) {init = 1 : i32}
    %lock_5_3_121 = aie.lock(%tile_5_3, 0) {init = 0 : i32}
    %lock_6_3 = aie.lock(%tile_6_3, 3) {init = 1 : i32}
    %lock_6_3_122 = aie.lock(%tile_6_3, 2) {init = 0 : i32}
    %lock_6_3_123 = aie.lock(%tile_6_3, 1) {init = 1 : i32}
    %lock_6_3_124 = aie.lock(%tile_6_3, 0) {init = 0 : i32}
    %lock_7_3 = aie.lock(%tile_7_3, 3) {init = 1 : i32}
    %lock_7_3_125 = aie.lock(%tile_7_3, 2) {init = 0 : i32}
    %lock_7_3_126 = aie.lock(%tile_7_3, 1) {init = 1 : i32}
    %lock_7_3_127 = aie.lock(%tile_7_3, 0) {init = 0 : i32}
    %lock_4_4 = aie.lock(%tile_4_4, 3) {init = 1 : i32}
    %lock_4_4_128 = aie.lock(%tile_4_4, 2) {init = 0 : i32}
    %lock_4_4_129 = aie.lock(%tile_4_4, 1) {init = 1 : i32}
    %lock_4_4_130 = aie.lock(%tile_4_4, 0) {init = 0 : i32}
    %lock_5_4 = aie.lock(%tile_5_4, 3) {init = 1 : i32}
    %lock_5_4_131 = aie.lock(%tile_5_4, 2) {init = 0 : i32}
    %lock_5_4_132 = aie.lock(%tile_5_4, 1) {init = 1 : i32}
    %lock_5_4_133 = aie.lock(%tile_5_4, 0) {init = 0 : i32}
    %lock_6_4 = aie.lock(%tile_6_4, 3) {init = 1 : i32}
    %lock_6_4_134 = aie.lock(%tile_6_4, 2) {init = 0 : i32}
    %lock_6_4_135 = aie.lock(%tile_6_4, 1) {init = 1 : i32}
    %lock_6_4_136 = aie.lock(%tile_6_4, 0) {init = 0 : i32}
    %lock_7_4 = aie.lock(%tile_7_4, 3) {init = 1 : i32}
    %lock_7_4_137 = aie.lock(%tile_7_4, 2) {init = 0 : i32}
    %lock_7_4_138 = aie.lock(%tile_7_4, 1) {init = 1 : i32}
    %lock_7_4_139 = aie.lock(%tile_7_4, 0) {init = 0 : i32}
    %lock_4_5 = aie.lock(%tile_4_5, 3) {init = 1 : i32}
    %lock_4_5_140 = aie.lock(%tile_4_5, 2) {init = 0 : i32}
    %lock_4_5_141 = aie.lock(%tile_4_5, 1) {init = 1 : i32}
    %lock_4_5_142 = aie.lock(%tile_4_5, 0) {init = 0 : i32}
    %lock_5_5 = aie.lock(%tile_5_5, 3) {init = 1 : i32}
    %lock_5_5_143 = aie.lock(%tile_5_5, 2) {init = 0 : i32}
    %lock_5_5_144 = aie.lock(%tile_5_5, 1) {init = 1 : i32}
    %lock_5_5_145 = aie.lock(%tile_5_5, 0) {init = 0 : i32}
    %lock_6_5 = aie.lock(%tile_6_5, 3) {init = 1 : i32}
    %lock_6_5_146 = aie.lock(%tile_6_5, 2) {init = 0 : i32}
    %lock_6_5_147 = aie.lock(%tile_6_5, 1) {init = 1 : i32}
    %lock_6_5_148 = aie.lock(%tile_6_5, 0) {init = 0 : i32}
    %lock_7_5 = aie.lock(%tile_7_5, 3) {init = 1 : i32}
    %lock_7_5_149 = aie.lock(%tile_7_5, 2) {init = 0 : i32}
    %lock_7_5_150 = aie.lock(%tile_7_5, 1) {init = 1 : i32}
    %lock_7_5_151 = aie.lock(%tile_7_5, 0) {init = 0 : i32}
    %buf511_unroll_1 = aie.buffer(%mem_tile_4_1) {sym_name = "buf511_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf510_unroll_1 = aie.buffer(%mem_tile_5_1) {sym_name = "buf510_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf509_unroll_1 = aie.buffer(%mem_tile_6_1) {sym_name = "buf509_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf508_unroll_1 = aie.buffer(%mem_tile_7_1) {sym_name = "buf508_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf507_unroll_1 = aie.buffer(%mem_tile_4_1) {sym_name = "buf507_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf506_unroll_1 = aie.buffer(%mem_tile_5_1) {sym_name = "buf506_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf505_unroll_1 = aie.buffer(%mem_tile_6_1) {sym_name = "buf505_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf504_unroll_1 = aie.buffer(%mem_tile_7_1) {sym_name = "buf504_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf503_unroll_1 = aie.buffer(%mem_tile_4_1) {sym_name = "buf503_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf502_unroll_1 = aie.buffer(%mem_tile_5_1) {sym_name = "buf502_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf501_unroll_1 = aie.buffer(%mem_tile_6_1) {sym_name = "buf501_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf500_unroll_1 = aie.buffer(%mem_tile_7_1) {sym_name = "buf500_unroll_1"} : memref<64x64xbf16, 1 : i32> 
    %buf499_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf499_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf498_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf498_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf497_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf497_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf496_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf496_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf495_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf495_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf494_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf494_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf493_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf493_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf492_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf492_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf491_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf491_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf490_unroll_1 = aie.buffer(%tile_7_5) {sym_name = "buf490_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf489_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf489_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf488_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf488_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf487_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf487_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf486_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf486_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf485_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf485_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf484_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf484_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf483_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf483_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf482_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf482_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf481_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf481_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf480_unroll_1 = aie.buffer(%tile_6_5) {sym_name = "buf480_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf479_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf479_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf478_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf478_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf477_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf477_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf476_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf476_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf475_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf475_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf474_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf474_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf473_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf473_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf472_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf472_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf471_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf471_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf470_unroll_1 = aie.buffer(%tile_5_5) {sym_name = "buf470_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf469_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf469_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf468_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf468_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf467_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf467_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf466_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf466_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf465_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf465_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf464_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf464_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf463_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf463_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf462_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf462_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf461_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf461_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf460_unroll_1 = aie.buffer(%tile_4_5) {sym_name = "buf460_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf459_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf459_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf458_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf458_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf457_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf457_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf456_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf456_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf455_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf455_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf454_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf454_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf453_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf453_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf452_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf452_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf451_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf451_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf450_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf450_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf449_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf449_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf448_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf448_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf447_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf447_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf446_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf446_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf445_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf445_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf444_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf444_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf443_unroll_1 = aie.buffer(%tile_7_4) {sym_name = "buf443_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf442_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf442_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf441_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf441_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf440_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf440_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf439_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf439_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf438_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf438_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf437_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf437_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf436_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf436_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf435_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf435_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf434_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf434_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf433_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf433_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf432_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf432_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf431_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf431_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf430_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf430_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf429_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf429_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf428_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf428_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf427_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf427_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf426_unroll_1 = aie.buffer(%tile_6_4) {sym_name = "buf426_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf425_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf425_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf424_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf424_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf423_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf423_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf422_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf422_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf421_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf421_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf420_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf420_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf419_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf419_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf418_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf418_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf417_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf417_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf416_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf416_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf415_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf415_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf414_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf414_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf413_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf413_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf412_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf412_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf411_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf411_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf410_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf410_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf409_unroll_1 = aie.buffer(%tile_5_4) {sym_name = "buf409_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf408_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf408_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf407_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf407_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf406_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf406_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf405_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf405_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf404_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf404_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf403_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf403_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf402_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf402_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf401_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf401_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf400_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf400_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf399_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf399_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf398_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf398_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf397_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf397_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf396_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf396_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf395_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf395_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf394_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf394_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf393_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf393_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf392_unroll_1 = aie.buffer(%tile_4_4) {sym_name = "buf392_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf391_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf391_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf390_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf390_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf389_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf389_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf388_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf388_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf387_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf387_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf386_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf386_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf385_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf385_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf384_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf384_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf383_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf383_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf382_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf382_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf381_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf381_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf380_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf380_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf379_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf379_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf378_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf378_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf377_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf377_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf376_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf376_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf375_unroll_1 = aie.buffer(%tile_7_3) {sym_name = "buf375_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf374_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf374_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf373_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf373_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf372_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf372_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf371_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf371_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf370_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf370_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf369_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf369_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf368_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf368_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf367_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf367_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf366_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf366_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf365_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf365_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf364_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf364_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf363_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf363_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf362_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf362_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf361_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf361_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf360_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf360_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf359_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf359_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf358_unroll_1 = aie.buffer(%tile_6_3) {sym_name = "buf358_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf357_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf357_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf356_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf356_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf355_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf355_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf354_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf354_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf353_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf353_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf352_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf352_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf351_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf351_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf350_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf350_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf349_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf349_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf348_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf348_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf347_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf347_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf346_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf346_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf345_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf345_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf344_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf344_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf343_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf343_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf342_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf342_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf341_unroll_1 = aie.buffer(%tile_5_3) {sym_name = "buf341_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf340_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf340_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf339_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf339_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf338_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf338_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf337_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf337_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf336_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf336_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf335_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf335_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf334_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf334_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf333_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf333_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf332_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf332_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf331_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf331_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf330_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf330_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf329_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf329_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf328_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf328_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf327_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf327_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf326_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf326_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf325_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf325_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf324_unroll_1 = aie.buffer(%tile_4_3) {sym_name = "buf324_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf323_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf323_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf322_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf322_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf321_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf321_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf320_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf320_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf319_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf319_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf318_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf318_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf317_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf317_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf316_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf316_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf315_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf315_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf314_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf314_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf313_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf313_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf312_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf312_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf311_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf311_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf310_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf310_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf309_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf309_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf308_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf308_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf307_unroll_1 = aie.buffer(%tile_7_2) {sym_name = "buf307_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf306_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf306_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf305_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf305_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf304_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf304_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf303_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf303_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf302_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf302_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf301_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf301_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf300_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf300_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf299_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf299_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf298_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf298_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf297_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf297_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf296_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf296_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf295_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf295_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf294_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf294_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf293_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf293_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf292_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf292_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf291_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf291_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf290_unroll_1 = aie.buffer(%tile_6_2) {sym_name = "buf290_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf289_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf289_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf288_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf288_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf287_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf287_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf286_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf286_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf285_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf285_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf284_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf284_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf283_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf283_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf282_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf282_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf281_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf281_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf280_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf280_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf279_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf279_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf278_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf278_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf277_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf277_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf276_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf276_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf275_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf275_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf274_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf274_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf273_unroll_1 = aie.buffer(%tile_5_2) {sym_name = "buf273_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf272_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf272_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf271_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf271_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf270_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf270_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf269_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf269_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf268_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf268_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf267_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf267_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf266_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf266_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf265_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf265_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf264_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf264_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf263_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf263_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf262_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf262_unroll_1"} : memref<64x64xbf16, 2 : i32> 
    %buf261_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf261_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf260_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf260_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf259_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf259_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf258_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf258_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf257_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf257_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %buf256_unroll_1 = aie.buffer(%tile_4_2) {sym_name = "buf256_unroll_1"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer_unroll_1 = aie.external_buffer {sym_name = "__air_external_buffer_unroll_1"} : memref<2x256x128xbf16>
    %__air_external_buffer_1_unroll_1 = aie.external_buffer {sym_name = "__air_external_buffer_1_unroll_1"} : memref<2x512x128xbf16>
    %__air_external_buffer_2_unroll_1 = aie.external_buffer {sym_name = "__air_external_buffer_2_unroll_1"} : memref<2x512x64xbf16>
    %__air_external_buffer_3_unroll_1 = aie.external_buffer {sym_name = "__air_external_buffer_3_unroll_1"} : memref<2x256x64xbf16>
    %mem_7_5 = aie.mem(%tile_7_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_5_150, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf496_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_5_151, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf493_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_5_149, Release, 1)
      aie.next_bd ^bb4
    }
    %core_7_5 = aie.core(%tile_7_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf497_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf499_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf498_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_5_150, Release, 1)
      aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_5_150, Release, 1)
      aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_5_150, Release, 1)
      aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf496_unroll_1, %buf494_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_5_150, Release, 1)
      aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_5_150, Release, 1)
      aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_5_150, Release, 1)
      aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_5_150, Release, 1)
      aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf496_unroll_1, %buf495_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_5_150, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf492_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf494_unroll_1, %buf496_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_5_150, Release, 1)
        aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf495_unroll_1, %buf496_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_5_150, Release, 1)
        aie.use_lock(%lock_7_5_149, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf498_unroll_1, %buf491_unroll_1, %buf490_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf490_unroll_1, %buf497_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf493_unroll_1, %buf497_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf499_unroll_1, %buf490_unroll_1, %buf491_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf491_unroll_1, %buf499_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf497_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_152 = memref.collapse_shape %buf498_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_153 = memref.collapse_shape %buf499_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_6_5 = aie.mem(%tile_6_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_5_147, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf486_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_5_148, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf483_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_5_146, Release, 1)
      aie.next_bd ^bb4
    }
    %core_6_5 = aie.core(%tile_6_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf487_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf489_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf488_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_5_147, Release, 1)
      aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_5_147, Release, 1)
      aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf486_unroll_1, %buf484_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_5_147, Release, 1)
      aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_5_147, Release, 1)
      aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_5_147, Release, 1)
      aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_5_147, Release, 1)
      aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf486_unroll_1, %buf485_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_5_147, Release, 1)
      aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_5_147, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf482_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf484_unroll_1, %buf486_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_5_147, Release, 1)
        aie.use_lock(%lock_6_5_148, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf485_unroll_1, %buf486_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_5_147, Release, 1)
        aie.use_lock(%lock_6_5_146, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf488_unroll_1, %buf481_unroll_1, %buf480_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf480_unroll_1, %buf487_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf483_unroll_1, %buf487_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf489_unroll_1, %buf480_unroll_1, %buf481_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf481_unroll_1, %buf489_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf487_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_152 = memref.collapse_shape %buf488_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_153 = memref.collapse_shape %buf489_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_5_5 = aie.mem(%tile_5_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_5_144, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf476_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_5_145, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf473_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_5_143, Release, 1)
      aie.next_bd ^bb4
    }
    %core_5_5 = aie.core(%tile_5_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf477_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf479_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf478_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_5_144, Release, 1)
      aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf476_unroll_1, %buf474_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_5_144, Release, 1)
      aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_5_144, Release, 1)
      aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_5_144, Release, 1)
      aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_5_144, Release, 1)
      aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf476_unroll_1, %buf475_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_5_144, Release, 1)
      aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_5_144, Release, 1)
      aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_5_144, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf472_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf474_unroll_1, %buf476_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_5_144, Release, 1)
        aie.use_lock(%lock_5_5_145, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf475_unroll_1, %buf476_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_5_144, Release, 1)
        aie.use_lock(%lock_5_5_143, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf478_unroll_1, %buf471_unroll_1, %buf470_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf470_unroll_1, %buf477_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf473_unroll_1, %buf477_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf479_unroll_1, %buf470_unroll_1, %buf471_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf471_unroll_1, %buf479_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf477_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_152 = memref.collapse_shape %buf478_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_153 = memref.collapse_shape %buf479_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_4_5 = aie.mem(%tile_4_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_5_141, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf466_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_5_142, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf463_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_5_140, Release, 1)
      aie.next_bd ^bb4
    }
    %core_4_5 = aie.core(%tile_4_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf467_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf469_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf468_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf466_unroll_1, %buf464_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_5_141, Release, 1)
      aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_5_141, Release, 1)
      aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_5_141, Release, 1)
      aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_5_141, Release, 1)
      aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf466_unroll_1, %buf465_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_5_141, Release, 1)
      aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_5_141, Release, 1)
      aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_5_141, Release, 1)
      aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_5_141, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf462_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf464_unroll_1, %buf466_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_5_141, Release, 1)
        aie.use_lock(%lock_4_5_142, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf465_unroll_1, %buf466_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_5_141, Release, 1)
        aie.use_lock(%lock_4_5_140, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf468_unroll_1, %buf461_unroll_1, %buf460_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf460_unroll_1, %buf467_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf463_unroll_1, %buf467_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf469_unroll_1, %buf460_unroll_1, %buf461_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf461_unroll_1, %buf469_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf467_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_152 = memref.collapse_shape %buf468_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_153 = memref.collapse_shape %buf469_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_7_4 = aie.mem(%tile_7_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_4_138, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf456_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_4_139, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf453_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_4_137, Release, 1)
      aie.next_bd ^bb4
    }
    %core_7_4 = aie.core(%tile_7_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf457_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf459_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf458_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_4_138, Release, 1)
      aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_4_138, Release, 1)
      aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_4_138, Release, 1)
      aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf456_unroll_1, %buf454_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_4_138, Release, 1)
      aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_4_138, Release, 1)
      aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_4_138, Release, 1)
      aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_4_138, Release, 1)
      aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf456_unroll_1, %buf455_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_4_138, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf452_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf454_unroll_1, %buf456_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_4_138, Release, 1)
        aie.use_lock(%lock_7_4_139, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf455_unroll_1, %buf456_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_4_138, Release, 1)
        aie.use_lock(%lock_7_4_137, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf458_unroll_1, %buf451_unroll_1, %buf450_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf450_unroll_1, %buf457_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf453_unroll_1, %buf457_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf459_unroll_1, %buf450_unroll_1, %buf451_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf451_unroll_1, %buf459_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf449_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf448_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf447_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf458_unroll_1, %buf446_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf448_unroll_1, %buf458_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf448_unroll_1, %buf458_unroll_1, %buf445_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf446_unroll_1, %buf458_unroll_1, %buf444_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf445_unroll_1, %buf449_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf444_unroll_1, %buf457_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf457_unroll_1, %buf449_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf443_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf447_unroll_1, %buf445_unroll_1, %buf443_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf459_unroll_1, %buf444_unroll_1, %buf443_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf443_unroll_1, %buf447_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf458_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_6_4 = aie.mem(%tile_6_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_4_135, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf439_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_4_136, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf436_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_4_134, Release, 1)
      aie.next_bd ^bb4
    }
    %core_6_4 = aie.core(%tile_6_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf440_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf442_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf441_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_4_135, Release, 1)
      aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_4_135, Release, 1)
      aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf439_unroll_1, %buf437_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_4_135, Release, 1)
      aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_4_135, Release, 1)
      aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_4_135, Release, 1)
      aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_4_135, Release, 1)
      aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf439_unroll_1, %buf438_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_4_135, Release, 1)
      aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_4_135, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf435_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf437_unroll_1, %buf439_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_4_135, Release, 1)
        aie.use_lock(%lock_6_4_136, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf438_unroll_1, %buf439_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_4_135, Release, 1)
        aie.use_lock(%lock_6_4_134, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf441_unroll_1, %buf434_unroll_1, %buf433_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf433_unroll_1, %buf440_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf436_unroll_1, %buf440_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf442_unroll_1, %buf433_unroll_1, %buf434_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf434_unroll_1, %buf442_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf432_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf431_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf430_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf441_unroll_1, %buf429_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf431_unroll_1, %buf441_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf431_unroll_1, %buf441_unroll_1, %buf428_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf429_unroll_1, %buf441_unroll_1, %buf427_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf428_unroll_1, %buf432_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf427_unroll_1, %buf440_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf440_unroll_1, %buf432_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf426_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf430_unroll_1, %buf428_unroll_1, %buf426_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf442_unroll_1, %buf427_unroll_1, %buf426_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf426_unroll_1, %buf430_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf441_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_5_4 = aie.mem(%tile_5_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_4_132, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf422_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_4_133, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf419_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_4_131, Release, 1)
      aie.next_bd ^bb4
    }
    %core_5_4 = aie.core(%tile_5_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf423_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf425_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf424_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_4_132, Release, 1)
      aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf422_unroll_1, %buf420_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_4_132, Release, 1)
      aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_4_132, Release, 1)
      aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_4_132, Release, 1)
      aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_4_132, Release, 1)
      aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf422_unroll_1, %buf421_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_4_132, Release, 1)
      aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_4_132, Release, 1)
      aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_4_132, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf418_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf420_unroll_1, %buf422_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_4_132, Release, 1)
        aie.use_lock(%lock_5_4_133, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf421_unroll_1, %buf422_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_4_132, Release, 1)
        aie.use_lock(%lock_5_4_131, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf424_unroll_1, %buf417_unroll_1, %buf416_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf416_unroll_1, %buf423_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf419_unroll_1, %buf423_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf425_unroll_1, %buf416_unroll_1, %buf417_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf417_unroll_1, %buf425_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf415_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf414_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf413_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf424_unroll_1, %buf412_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf414_unroll_1, %buf424_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf414_unroll_1, %buf424_unroll_1, %buf411_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf412_unroll_1, %buf424_unroll_1, %buf410_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf411_unroll_1, %buf415_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf410_unroll_1, %buf423_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf423_unroll_1, %buf415_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf409_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf413_unroll_1, %buf411_unroll_1, %buf409_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf425_unroll_1, %buf410_unroll_1, %buf409_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf409_unroll_1, %buf413_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf424_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_4_4 = aie.mem(%tile_4_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_4_129, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf405_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_4_130, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf402_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_4_128, Release, 1)
      aie.next_bd ^bb4
    }
    %core_4_4 = aie.core(%tile_4_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf406_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf408_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf407_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf405_unroll_1, %buf403_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_4_129, Release, 1)
      aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_4_129, Release, 1)
      aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_4_129, Release, 1)
      aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_4_129, Release, 1)
      aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf405_unroll_1, %buf404_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_4_129, Release, 1)
      aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_4_129, Release, 1)
      aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_4_129, Release, 1)
      aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_4_129, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf401_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf403_unroll_1, %buf405_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_4_129, Release, 1)
        aie.use_lock(%lock_4_4_130, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf404_unroll_1, %buf405_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_4_129, Release, 1)
        aie.use_lock(%lock_4_4_128, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf407_unroll_1, %buf400_unroll_1, %buf399_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf399_unroll_1, %buf406_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf402_unroll_1, %buf406_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf408_unroll_1, %buf399_unroll_1, %buf400_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf400_unroll_1, %buf408_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf398_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf397_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf396_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf407_unroll_1, %buf395_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf397_unroll_1, %buf407_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf397_unroll_1, %buf407_unroll_1, %buf394_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf395_unroll_1, %buf407_unroll_1, %buf393_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf394_unroll_1, %buf398_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf393_unroll_1, %buf406_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf406_unroll_1, %buf398_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf392_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf396_unroll_1, %buf394_unroll_1, %buf392_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf408_unroll_1, %buf393_unroll_1, %buf392_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf392_unroll_1, %buf396_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf407_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_7_3 = aie.mem(%tile_7_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_3_126, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf388_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_3_127, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf385_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_3_125, Release, 1)
      aie.next_bd ^bb4
    }
    %core_7_3 = aie.core(%tile_7_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf389_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf391_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf390_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_3_126, Release, 1)
      aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_3_126, Release, 1)
      aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_3_126, Release, 1)
      aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf388_unroll_1, %buf386_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_3_126, Release, 1)
      aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_3_126, Release, 1)
      aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_3_126, Release, 1)
      aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_3_126, Release, 1)
      aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf388_unroll_1, %buf387_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_3_126, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf384_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf386_unroll_1, %buf388_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_3_126, Release, 1)
        aie.use_lock(%lock_7_3_127, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf387_unroll_1, %buf388_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_3_126, Release, 1)
        aie.use_lock(%lock_7_3_125, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf390_unroll_1, %buf383_unroll_1, %buf382_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf382_unroll_1, %buf389_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf385_unroll_1, %buf389_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf391_unroll_1, %buf382_unroll_1, %buf383_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf383_unroll_1, %buf391_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf381_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf380_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf379_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf390_unroll_1, %buf378_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf380_unroll_1, %buf390_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf380_unroll_1, %buf390_unroll_1, %buf377_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf378_unroll_1, %buf390_unroll_1, %buf376_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf377_unroll_1, %buf381_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf376_unroll_1, %buf389_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf389_unroll_1, %buf381_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf375_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf379_unroll_1, %buf377_unroll_1, %buf375_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf391_unroll_1, %buf376_unroll_1, %buf375_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf375_unroll_1, %buf379_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf390_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_6_3 = aie.mem(%tile_6_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_3_123, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf371_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_3_124, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf368_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_3_122, Release, 1)
      aie.next_bd ^bb4
    }
    %core_6_3 = aie.core(%tile_6_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf372_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf374_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf373_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_3_123, Release, 1)
      aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_3_123, Release, 1)
      aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf371_unroll_1, %buf369_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_3_123, Release, 1)
      aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_3_123, Release, 1)
      aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_3_123, Release, 1)
      aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_3_123, Release, 1)
      aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf371_unroll_1, %buf370_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_3_123, Release, 1)
      aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_3_123, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf367_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf369_unroll_1, %buf371_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_3_123, Release, 1)
        aie.use_lock(%lock_6_3_124, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf370_unroll_1, %buf371_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_3_123, Release, 1)
        aie.use_lock(%lock_6_3_122, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf373_unroll_1, %buf366_unroll_1, %buf365_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf365_unroll_1, %buf372_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf368_unroll_1, %buf372_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf374_unroll_1, %buf365_unroll_1, %buf366_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf366_unroll_1, %buf374_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf364_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf363_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf362_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf373_unroll_1, %buf361_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf363_unroll_1, %buf373_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf363_unroll_1, %buf373_unroll_1, %buf360_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf361_unroll_1, %buf373_unroll_1, %buf359_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf360_unroll_1, %buf364_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf359_unroll_1, %buf372_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf372_unroll_1, %buf364_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf358_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf362_unroll_1, %buf360_unroll_1, %buf358_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf374_unroll_1, %buf359_unroll_1, %buf358_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf358_unroll_1, %buf362_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf373_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_5_3 = aie.mem(%tile_5_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_3_120, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf354_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_3_121, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf351_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_3_119, Release, 1)
      aie.next_bd ^bb4
    }
    %core_5_3 = aie.core(%tile_5_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf355_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf357_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf356_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_3_120, Release, 1)
      aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf354_unroll_1, %buf352_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_3_120, Release, 1)
      aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_3_120, Release, 1)
      aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_3_120, Release, 1)
      aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_3_120, Release, 1)
      aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf354_unroll_1, %buf353_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_3_120, Release, 1)
      aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_3_120, Release, 1)
      aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_3_120, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf350_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf352_unroll_1, %buf354_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_3_120, Release, 1)
        aie.use_lock(%lock_5_3_121, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf353_unroll_1, %buf354_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_3_120, Release, 1)
        aie.use_lock(%lock_5_3_119, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf356_unroll_1, %buf349_unroll_1, %buf348_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf348_unroll_1, %buf355_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf351_unroll_1, %buf355_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf357_unroll_1, %buf348_unroll_1, %buf349_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf349_unroll_1, %buf357_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf347_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf346_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf345_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf356_unroll_1, %buf344_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf346_unroll_1, %buf356_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf346_unroll_1, %buf356_unroll_1, %buf343_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf344_unroll_1, %buf356_unroll_1, %buf342_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf343_unroll_1, %buf347_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf342_unroll_1, %buf355_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf355_unroll_1, %buf347_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf341_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf345_unroll_1, %buf343_unroll_1, %buf341_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf357_unroll_1, %buf342_unroll_1, %buf341_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf341_unroll_1, %buf345_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf356_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_4_3 = aie.mem(%tile_4_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_3_117, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf337_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_3_118, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf334_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_3_116, Release, 1)
      aie.next_bd ^bb4
    }
    %core_4_3 = aie.core(%tile_4_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf338_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf340_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf339_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf337_unroll_1, %buf335_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_3_117, Release, 1)
      aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_3_117, Release, 1)
      aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_3_117, Release, 1)
      aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_3_117, Release, 1)
      aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf337_unroll_1, %buf336_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_3_117, Release, 1)
      aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_3_117, Release, 1)
      aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_3_117, Release, 1)
      aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_3_117, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_155 = memref.collapse_shape %buf333_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_155) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf335_unroll_1, %buf337_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_3_117, Release, 1)
        aie.use_lock(%lock_4_3_118, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf336_unroll_1, %buf337_unroll_1, %collapse_shape_155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_3_117, Release, 1)
        aie.use_lock(%lock_4_3_116, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_155, %buf339_unroll_1, %buf332_unroll_1, %buf331_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf331_unroll_1, %buf338_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_155, %buf334_unroll_1, %buf338_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf340_unroll_1, %buf331_unroll_1, %buf332_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf332_unroll_1, %buf340_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf330_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf329_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf328_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf339_unroll_1, %buf327_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf329_unroll_1, %buf339_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf329_unroll_1, %buf339_unroll_1, %buf326_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf327_unroll_1, %buf339_unroll_1, %buf325_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf326_unroll_1, %buf330_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf325_unroll_1, %buf338_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf338_unroll_1, %buf330_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf324_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf328_unroll_1, %buf326_unroll_1, %buf324_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf340_unroll_1, %buf325_unroll_1, %buf324_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf324_unroll_1, %buf328_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_154 = memref.collapse_shape %buf339_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_154[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_115, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf313_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_114, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_112, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf320_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_113, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf317_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_111, Release, 1)
      aie.next_bd ^bb6
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_114, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf321_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf323_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf322_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_112, Release, 1)
      aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_112, Release, 1)
      aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_112, Release, 1)
      aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf320_unroll_1, %buf318_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_2_112, Release, 1)
      aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_112, Release, 1)
      aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_112, Release, 1)
      aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_112, Release, 1)
      aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf320_unroll_1, %buf319_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_2_112, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf316_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf318_unroll_1, %buf320_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_112, Release, 1)
        aie.use_lock(%lock_7_2_113, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf319_unroll_1, %buf320_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_112, Release, 1)
        aie.use_lock(%lock_7_2_111, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf322_unroll_1, %buf315_unroll_1, %buf314_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf314_unroll_1, %buf321_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf317_unroll_1, %buf321_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf323_unroll_1, %buf314_unroll_1, %buf315_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf315_unroll_1, %buf323_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf313_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf312_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf311_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf322_unroll_1, %buf310_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf312_unroll_1, %buf322_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf312_unroll_1, %buf322_unroll_1, %buf309_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf310_unroll_1, %buf322_unroll_1, %buf308_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf309_unroll_1, %buf313_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf308_unroll_1, %buf321_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf321_unroll_1, %buf313_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf307_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf311_unroll_1, %buf309_unroll_1, %buf307_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf323_unroll_1, %buf308_unroll_1, %buf307_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf307_unroll_1, %buf311_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf311_unroll_1, %buf313_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_7_2_115, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_110, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf296_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_109, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_107, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf303_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_108, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf300_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_106, Release, 1)
      aie.next_bd ^bb6
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_109, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf304_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf306_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf305_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_107, Release, 1)
      aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_107, Release, 1)
      aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf303_unroll_1, %buf301_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_2_107, Release, 1)
      aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_107, Release, 1)
      aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_107, Release, 1)
      aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_107, Release, 1)
      aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf303_unroll_1, %buf302_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_2_107, Release, 1)
      aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_107, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf299_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf301_unroll_1, %buf303_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_107, Release, 1)
        aie.use_lock(%lock_6_2_108, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf302_unroll_1, %buf303_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_107, Release, 1)
        aie.use_lock(%lock_6_2_106, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf305_unroll_1, %buf298_unroll_1, %buf297_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf297_unroll_1, %buf304_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf300_unroll_1, %buf304_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf306_unroll_1, %buf297_unroll_1, %buf298_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf298_unroll_1, %buf306_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf296_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf295_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf294_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf305_unroll_1, %buf293_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf295_unroll_1, %buf305_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf295_unroll_1, %buf305_unroll_1, %buf292_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf293_unroll_1, %buf305_unroll_1, %buf291_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf292_unroll_1, %buf296_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf291_unroll_1, %buf304_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf304_unroll_1, %buf296_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf290_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf294_unroll_1, %buf292_unroll_1, %buf290_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf306_unroll_1, %buf291_unroll_1, %buf290_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf290_unroll_1, %buf294_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf294_unroll_1, %buf296_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_6_2_110, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_105, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf279_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_104, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_102, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf286_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_103, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf283_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_101, Release, 1)
      aie.next_bd ^bb6
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_104, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf287_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf289_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf288_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_102, Release, 1)
      aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf286_unroll_1, %buf284_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_2_102, Release, 1)
      aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_102, Release, 1)
      aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_102, Release, 1)
      aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_102, Release, 1)
      aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf286_unroll_1, %buf285_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_2_102, Release, 1)
      aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_102, Release, 1)
      aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_102, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf282_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf284_unroll_1, %buf286_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_102, Release, 1)
        aie.use_lock(%lock_5_2_103, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf285_unroll_1, %buf286_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_102, Release, 1)
        aie.use_lock(%lock_5_2_101, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf288_unroll_1, %buf281_unroll_1, %buf280_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf280_unroll_1, %buf287_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf283_unroll_1, %buf287_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf289_unroll_1, %buf280_unroll_1, %buf281_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf281_unroll_1, %buf289_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf279_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf278_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf277_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf288_unroll_1, %buf276_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf278_unroll_1, %buf288_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf278_unroll_1, %buf288_unroll_1, %buf275_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf276_unroll_1, %buf288_unroll_1, %buf274_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf275_unroll_1, %buf279_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf274_unroll_1, %buf287_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf287_unroll_1, %buf279_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf273_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf277_unroll_1, %buf275_unroll_1, %buf273_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf289_unroll_1, %buf274_unroll_1, %buf273_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf273_unroll_1, %buf277_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf277_unroll_1, %buf279_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_5_2_105, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_100, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf262_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_99, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_97, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf269_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_98, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf266_unroll_1 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_96, Release, 1)
      aie.next_bd ^bb6
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_99, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf270_unroll_1) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf272_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf271_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf269_unroll_1, %buf267_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_2_97, Release, 1)
      aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_97, Release, 1)
      aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_97, Release, 1)
      aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_97, Release, 1)
      aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf269_unroll_1, %buf268_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_2_97, Release, 1)
      aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_97, Release, 1)
      aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_97, Release, 1)
      aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_97, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_154 = memref.collapse_shape %buf265_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_154) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf267_unroll_1, %buf269_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_97, Release, 1)
        aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf268_unroll_1, %buf269_unroll_1, %collapse_shape_154) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_97, Release, 1)
        aie.use_lock(%lock_4_2_96, AcquireGreaterEqual, 1)
        func.call @fused_softmax(%collapse_shape_154, %buf271_unroll_1, %buf264_unroll_1, %buf263_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf263_unroll_1, %buf270_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_154, %buf266_unroll_1, %buf270_unroll_1) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf272_unroll_1, %buf263_unroll_1, %buf264_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf264_unroll_1, %buf272_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf262_unroll_1 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_152 = memref.collapse_shape %buf261_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_152[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_153 = memref.collapse_shape %buf260_unroll_1 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_153[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf271_unroll_1, %buf259_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf261_unroll_1, %buf271_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf261_unroll_1, %buf271_unroll_1, %buf258_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf259_unroll_1, %buf271_unroll_1, %buf257_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf258_unroll_1, %buf262_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf257_unroll_1, %buf270_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf270_unroll_1, %buf262_unroll_1) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf256_unroll_1) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf260_unroll_1, %buf258_unroll_1, %buf256_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf272_unroll_1, %buf257_unroll_1, %buf256_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf256_unroll_1, %buf260_unroll_1) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf260_unroll_1, %buf262_unroll_1) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_4_2_100, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %mem_tile_4_1, DMA : 1)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %mem_tile_5_1, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %mem_tile_6_1, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 1, %mem_tile_7_1, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %shim_noc_tile_7_0, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_4_2, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_5_2, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_6_2, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_4_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_5_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_6_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_7_3, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_4_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_5_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_6_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_7_4, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_4_5, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_5_5, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_6_5, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_7_5, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_4_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_5_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_6_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_7_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_4_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_5_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_6_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_7_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_4_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_5_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_6_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_7_4, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 2, %tile_4_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 2, %tile_5_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 2, %tile_6_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 2, %tile_7_5, DMA : 1)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 2)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 2)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_6_1, DMA : 2)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_7_1, DMA : 2)
    aie.cascade_flow(%tile_7_5, %tile_7_4)
    aie.cascade_flow(%tile_6_5, %tile_6_4)
    aie.cascade_flow(%tile_5_5, %tile_5_4)
    aie.cascade_flow(%tile_4_5, %tile_4_4)
    aie.cascade_flow(%tile_7_4, %tile_7_3)
    aie.cascade_flow(%tile_6_4, %tile_6_3)
    aie.cascade_flow(%tile_5_4, %tile_5_3)
    aie.cascade_flow(%tile_4_4, %tile_4_3)
    aie.cascade_flow(%tile_7_3, %tile_7_2)
    aie.cascade_flow(%tile_6_3, %tile_6_2)
    aie.cascade_flow(%tile_5_3, %tile_5_2)
    aie.cascade_flow(%tile_4_3, %tile_4_2)
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_95, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf507_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_94, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1_93, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf511_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_92, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_1_91, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf503_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_4_1_92, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf511_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_93, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf503_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_91, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_4_1_94, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf507_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_95, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_90, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf506_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_89, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1_88, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf510_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_87, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_1_86, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf502_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_5_1_87, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf510_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_88, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf502_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_86, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_5_1_89, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf506_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_90, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_85, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf505_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_84, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1_83, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf509_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_82, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_1_81, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf501_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_6_1_82, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf509_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_83, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf501_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_81, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_6_1_84, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf505_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_85, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1_80, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf504_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_79, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_1_78, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf508_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_77, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_1_76, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf500_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_7_1_77, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf508_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_78, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf500_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_76, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_7_1_79, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf504_unroll_1 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_80, Release, 1)
      aie.next_bd ^bb12
    }
    aie.shim_dma_allocation @air_channel_0_1_0_0(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_1(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_2(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_3(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_QKIn_0_1_0_0(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_1_1_0_0(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_2_1_0_0(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_3_1_0_0(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_0_1_0_0(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_1_1_0_0(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_2_1_0_0(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_3_1_0_0(%shim_noc_tile_7_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 41 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 44 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 47 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 50 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 53 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 56 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 59 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 62 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 65 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 68 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 71 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 74 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 77 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 80 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 83 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 86 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 89 : i64, location = 0 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 92 : i64, location = 1 : i64, row = -1 : i64}, {channel = 3 : i64, col = 2 : i64, id = 95 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 98 : i64, location = 3 : i64, row = -1 : i64}], sym_name = "attn_seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
  }
  air.channel @channel_0 [4, 2]
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
  air.channel @QK2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QKIn_0 [2]
  air.channel @QK2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QKIn_1 [2]
  air.channel @QK2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QKIn_2 [2]
  air.channel @QK2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QK2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @QKIn_3 [2]
  air.channel @V2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @VIn_0 [2]
  air.channel @V2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @VIn_1 [2]
  air.channel @V2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @VIn_2 [2]
  air.channel @V2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @V2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
  air.channel @VIn_3 [2]
  air.channel @cascade_gp [4, 3] {channel_type = "cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  func.func @attention_bf16(%arg0: memref<2x256x128xbf16>, %arg1: memref<2x512x128xbf16>, %arg2: memref<2x512x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = airrt.wait_all : !airrt.event
    affine.for %arg4 = 0 to 1 {
      %p = airrt.segment_load "attn_seg" : i64
      %c57344 = arith.constant 57344 : index
      %c40960 = arith.constant 40960 : index
      %c114688 = arith.constant 114688 : index
      %c98304 = arith.constant 98304 : index
      %c81920 = arith.constant 81920 : index
      %c65536 = arith.constant 65536 : index
      %c32832 = arith.constant 32832 : index
      %c24576 = arith.constant 24576 : index
      %c49152 = arith.constant 49152 : index
      %c32768 = arith.constant 32768 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c2 = arith.constant 2 : index
      %c8192 = arith.constant 8192 : index
      %c16384 = arith.constant 16384 : index
      %c3 = arith.constant 3 : index
      %c0_0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c0_i64 = arith.constant 0 : i64
      %c0_2 = arith.constant 0 : index
      %c1_3 = arith.constant 1 : index
      %c41_i32 = arith.constant 41 : i32
      %1 = arith.index_cast %arg4 : index to i64
      %2 = arith.index_cast %c0_2 : index to i64
      %3 = arith.index_cast %c0_2 : index to i64
      %4 = arith.index_cast %c0_0 : index to i64
      %5 = arith.index_cast %c0_0 : index to i64
      %6 = arith.index_cast %c0_2 : index to i64
      %7 = arith.index_cast %c0_2 : index to i64
      %8 = arith.index_cast %c128 : index to i64
      %9 = arith.index_cast %c1_1 : index to i64
      %10 = arith.index_cast %c1_3 : index to i64
      %11 = arith.index_cast %c1_3 : index to i64
      %12 = arith.index_cast %c256 : index to i64
      %13 = arith.index_cast %c64 : index to i64
      %14 = airrt.dma_memcpy_nd(%c41_i32, %1, %c0_i64, %arg0[%2, %3, %4, %5], [%10, %11, %12, %13], [%6, %7, %8, %9]) {chan_name = @QKIn_0, metadata = @air_QKIn_0_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %15 = airrt.wait_all %14 : !airrt.event
      %c0_i64_4 = arith.constant 0 : i64
      %c0_5 = arith.constant 0 : index
      %c1_6 = arith.constant 1 : index
      %c41_i32_7 = arith.constant 41 : i32
      %16 = arith.index_cast %arg4 : index to i64
      %17 = arith.index_cast %c0_5 : index to i64
      %18 = arith.index_cast %c0_5 : index to i64
      %19 = arith.index_cast %c0_0 : index to i64
      %20 = arith.index_cast %c64 : index to i64
      %21 = arith.index_cast %c0_5 : index to i64
      %22 = arith.index_cast %c0_5 : index to i64
      %23 = arith.index_cast %c128 : index to i64
      %24 = arith.index_cast %c1_1 : index to i64
      %25 = arith.index_cast %c1_6 : index to i64
      %26 = arith.index_cast %c1_6 : index to i64
      %27 = arith.index_cast %c256 : index to i64
      %28 = arith.index_cast %c64 : index to i64
      %29 = airrt.dma_memcpy_nd(%c41_i32_7, %16, %c0_i64_4, %arg0[%17, %18, %19, %20], [%25, %26, %27, %28], [%21, %22, %23, %24]) {chan_name = @QKIn_0, metadata = @air_QKIn_0_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %30 = airrt.wait_all %29 : !airrt.event
      %c0_i64_8 = arith.constant 0 : i64
      %c0_9 = arith.constant 0 : index
      %c1_10 = arith.constant 1 : index
      %c41_i32_11 = arith.constant 41 : i32
      %31 = arith.index_cast %arg4 : index to i64
      %32 = arith.index_cast %c0_0 : index to i64
      %33 = arith.index_cast %c0_0 : index to i64
      %34 = arith.index_cast %c0_0 : index to i64
      %35 = arith.index_cast %c0_0 : index to i64
      %36 = arith.index_cast %c8192 : index to i64
      %37 = arith.index_cast %c64 : index to i64
      %38 = arith.index_cast %c128 : index to i64
      %39 = arith.index_cast %c1_1 : index to i64
      %40 = arith.index_cast %c2 : index to i64
      %41 = arith.index_cast %c2 : index to i64
      %42 = arith.index_cast %c64 : index to i64
      %43 = arith.index_cast %c64 : index to i64
      %44 = airrt.dma_memcpy_nd(%c41_i32_11, %31, %c0_i64_8, %arg1[%32, %33, %34, %35], [%40, %41, %42, %43], [%36, %37, %38, %39]) {chan_name = @QKIn_0, metadata = @air_QKIn_0_0_0} : (i32, i64, i64, memref<2x512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %45 = airrt.wait_all %44 : !airrt.event
      %c0_i64_12 = arith.constant 0 : i64
      %c0_13 = arith.constant 0 : index
      %c1_14 = arith.constant 1 : index
      %c53_i32 = arith.constant 53 : i32
      %46 = arith.index_cast %arg4 : index to i64
      %47 = arith.index_cast %c0_13 : index to i64
      %48 = arith.index_cast %c0_13 : index to i64
      %49 = arith.index_cast %c0_0 : index to i64
      %50 = arith.index_cast %c0_0 : index to i64
      %51 = arith.index_cast %c0_13 : index to i64
      %52 = arith.index_cast %c0_13 : index to i64
      %53 = arith.index_cast %c128 : index to i64
      %54 = arith.index_cast %c1_1 : index to i64
      %55 = arith.index_cast %c1_14 : index to i64
      %56 = arith.index_cast %c1_14 : index to i64
      %57 = arith.index_cast %c256 : index to i64
      %58 = arith.index_cast %c64 : index to i64
      %59 = airrt.dma_memcpy_nd(%c53_i32, %46, %c0_i64_12, %arg0[%47, %48, %49, %50], [%55, %56, %57, %58], [%51, %52, %53, %54]) {chan_name = @QKIn_1, metadata = @air_QKIn_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %60 = airrt.wait_all %59 : !airrt.event
      %c0_i64_15 = arith.constant 0 : i64
      %c0_16 = arith.constant 0 : index
      %c1_17 = arith.constant 1 : index
      %c53_i32_18 = arith.constant 53 : i32
      %61 = arith.index_cast %arg4 : index to i64
      %62 = arith.index_cast %c0_16 : index to i64
      %63 = arith.index_cast %c0_16 : index to i64
      %64 = arith.index_cast %c0_0 : index to i64
      %65 = arith.index_cast %c64 : index to i64
      %66 = arith.index_cast %c0_16 : index to i64
      %67 = arith.index_cast %c0_16 : index to i64
      %68 = arith.index_cast %c128 : index to i64
      %69 = arith.index_cast %c1_1 : index to i64
      %70 = arith.index_cast %c1_17 : index to i64
      %71 = arith.index_cast %c1_17 : index to i64
      %72 = arith.index_cast %c256 : index to i64
      %73 = arith.index_cast %c64 : index to i64
      %74 = airrt.dma_memcpy_nd(%c53_i32_18, %61, %c0_i64_15, %arg0[%62, %63, %64, %65], [%70, %71, %72, %73], [%66, %67, %68, %69]) {chan_name = @QKIn_1, metadata = @air_QKIn_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %75 = airrt.wait_all %74 : !airrt.event
      %c0_i64_19 = arith.constant 0 : i64
      %c0_20 = arith.constant 0 : index
      %c1_21 = arith.constant 1 : index
      %c53_i32_22 = arith.constant 53 : i32
      %76 = arith.index_cast %arg4 : index to i64
      %77 = arith.index_cast %c0_0 : index to i64
      %78 = arith.index_cast %c0_0 : index to i64
      %79 = arith.index_cast %c0_0 : index to i64
      %80 = arith.index_cast %c16384 : index to i64
      %81 = arith.index_cast %c8192 : index to i64
      %82 = arith.index_cast %c64 : index to i64
      %83 = arith.index_cast %c128 : index to i64
      %84 = arith.index_cast %c1_1 : index to i64
      %85 = arith.index_cast %c2 : index to i64
      %86 = arith.index_cast %c2 : index to i64
      %87 = arith.index_cast %c64 : index to i64
      %88 = arith.index_cast %c64 : index to i64
      %89 = airrt.dma_memcpy_nd(%c53_i32_22, %76, %c0_i64_19, %arg1[%77, %78, %79, %80], [%85, %86, %87, %88], [%81, %82, %83, %84]) {chan_name = @QKIn_1, metadata = @air_QKIn_1_0_0} : (i32, i64, i64, memref<2x512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %90 = airrt.wait_all %89 : !airrt.event
      %c0_i64_23 = arith.constant 0 : i64
      %c0_24 = arith.constant 0 : index
      %c1_25 = arith.constant 1 : index
      %c65_i32 = arith.constant 65 : i32
      %91 = arith.index_cast %arg4 : index to i64
      %92 = arith.index_cast %c0_24 : index to i64
      %93 = arith.index_cast %c0_24 : index to i64
      %94 = arith.index_cast %c0_0 : index to i64
      %95 = arith.index_cast %c0_0 : index to i64
      %96 = arith.index_cast %c0_24 : index to i64
      %97 = arith.index_cast %c0_24 : index to i64
      %98 = arith.index_cast %c128 : index to i64
      %99 = arith.index_cast %c1_1 : index to i64
      %100 = arith.index_cast %c1_25 : index to i64
      %101 = arith.index_cast %c1_25 : index to i64
      %102 = arith.index_cast %c256 : index to i64
      %103 = arith.index_cast %c64 : index to i64
      %104 = airrt.dma_memcpy_nd(%c65_i32, %91, %c0_i64_23, %arg0[%92, %93, %94, %95], [%100, %101, %102, %103], [%96, %97, %98, %99]) {chan_name = @QKIn_2, metadata = @air_QKIn_2_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %105 = airrt.wait_all %104 : !airrt.event
      %c0_i64_26 = arith.constant 0 : i64
      %c0_27 = arith.constant 0 : index
      %c1_28 = arith.constant 1 : index
      %c65_i32_29 = arith.constant 65 : i32
      %106 = arith.index_cast %arg4 : index to i64
      %107 = arith.index_cast %c0_27 : index to i64
      %108 = arith.index_cast %c0_27 : index to i64
      %109 = arith.index_cast %c0_0 : index to i64
      %110 = arith.index_cast %c64 : index to i64
      %111 = arith.index_cast %c0_27 : index to i64
      %112 = arith.index_cast %c0_27 : index to i64
      %113 = arith.index_cast %c128 : index to i64
      %114 = arith.index_cast %c1_1 : index to i64
      %115 = arith.index_cast %c1_28 : index to i64
      %116 = arith.index_cast %c1_28 : index to i64
      %117 = arith.index_cast %c256 : index to i64
      %118 = arith.index_cast %c64 : index to i64
      %119 = airrt.dma_memcpy_nd(%c65_i32_29, %106, %c0_i64_26, %arg0[%107, %108, %109, %110], [%115, %116, %117, %118], [%111, %112, %113, %114]) {chan_name = @QKIn_2, metadata = @air_QKIn_2_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %120 = airrt.wait_all %119 : !airrt.event
      %c0_i64_30 = arith.constant 0 : i64
      %c0_31 = arith.constant 0 : index
      %c1_32 = arith.constant 1 : index
      %c65_i32_33 = arith.constant 65 : i32
      %121 = arith.index_cast %arg4 : index to i64
      %122 = arith.index_cast %c0_0 : index to i64
      %123 = arith.index_cast %c0_0 : index to i64
      %124 = arith.index_cast %c0_0 : index to i64
      %125 = arith.index_cast %c32768 : index to i64
      %126 = arith.index_cast %c8192 : index to i64
      %127 = arith.index_cast %c64 : index to i64
      %128 = arith.index_cast %c128 : index to i64
      %129 = arith.index_cast %c1_1 : index to i64
      %130 = arith.index_cast %c2 : index to i64
      %131 = arith.index_cast %c2 : index to i64
      %132 = arith.index_cast %c64 : index to i64
      %133 = arith.index_cast %c64 : index to i64
      %134 = airrt.dma_memcpy_nd(%c65_i32_33, %121, %c0_i64_30, %arg1[%122, %123, %124, %125], [%130, %131, %132, %133], [%126, %127, %128, %129]) {chan_name = @QKIn_2, metadata = @air_QKIn_2_0_0} : (i32, i64, i64, memref<2x512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %135 = airrt.wait_all %134 : !airrt.event
      %c0_i64_34 = arith.constant 0 : i64
      %c0_35 = arith.constant 0 : index
      %c1_36 = arith.constant 1 : index
      %c77_i32 = arith.constant 77 : i32
      %136 = arith.index_cast %arg4 : index to i64
      %137 = arith.index_cast %c0_35 : index to i64
      %138 = arith.index_cast %c0_35 : index to i64
      %139 = arith.index_cast %c0_0 : index to i64
      %140 = arith.index_cast %c0_0 : index to i64
      %141 = arith.index_cast %c0_35 : index to i64
      %142 = arith.index_cast %c0_35 : index to i64
      %143 = arith.index_cast %c128 : index to i64
      %144 = arith.index_cast %c1_1 : index to i64
      %145 = arith.index_cast %c1_36 : index to i64
      %146 = arith.index_cast %c1_36 : index to i64
      %147 = arith.index_cast %c256 : index to i64
      %148 = arith.index_cast %c64 : index to i64
      %149 = airrt.dma_memcpy_nd(%c77_i32, %136, %c0_i64_34, %arg0[%137, %138, %139, %140], [%145, %146, %147, %148], [%141, %142, %143, %144]) {chan_name = @QKIn_3, metadata = @air_QKIn_3_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %150 = airrt.wait_all %149 : !airrt.event
      %c0_i64_37 = arith.constant 0 : i64
      %c0_38 = arith.constant 0 : index
      %c1_39 = arith.constant 1 : index
      %c77_i32_40 = arith.constant 77 : i32
      %151 = arith.index_cast %arg4 : index to i64
      %152 = arith.index_cast %c0_38 : index to i64
      %153 = arith.index_cast %c0_38 : index to i64
      %154 = arith.index_cast %c0_0 : index to i64
      %155 = arith.index_cast %c64 : index to i64
      %156 = arith.index_cast %c0_38 : index to i64
      %157 = arith.index_cast %c0_38 : index to i64
      %158 = arith.index_cast %c128 : index to i64
      %159 = arith.index_cast %c1_1 : index to i64
      %160 = arith.index_cast %c1_39 : index to i64
      %161 = arith.index_cast %c1_39 : index to i64
      %162 = arith.index_cast %c256 : index to i64
      %163 = arith.index_cast %c64 : index to i64
      %164 = airrt.dma_memcpy_nd(%c77_i32_40, %151, %c0_i64_37, %arg0[%152, %153, %154, %155], [%160, %161, %162, %163], [%156, %157, %158, %159]) {chan_name = @QKIn_3, metadata = @air_QKIn_3_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %165 = airrt.wait_all %164 : !airrt.event
      %c0_i64_41 = arith.constant 0 : i64
      %c0_42 = arith.constant 0 : index
      %c1_43 = arith.constant 1 : index
      %c77_i32_44 = arith.constant 77 : i32
      %166 = arith.index_cast %arg4 : index to i64
      %167 = arith.index_cast %c0_0 : index to i64
      %168 = arith.index_cast %c0_0 : index to i64
      %169 = arith.index_cast %c0_0 : index to i64
      %170 = arith.index_cast %c49152 : index to i64
      %171 = arith.index_cast %c8192 : index to i64
      %172 = arith.index_cast %c64 : index to i64
      %173 = arith.index_cast %c128 : index to i64
      %174 = arith.index_cast %c1_1 : index to i64
      %175 = arith.index_cast %c2 : index to i64
      %176 = arith.index_cast %c2 : index to i64
      %177 = arith.index_cast %c64 : index to i64
      %178 = arith.index_cast %c64 : index to i64
      %179 = airrt.dma_memcpy_nd(%c77_i32_44, %166, %c0_i64_41, %arg1[%167, %168, %169, %170], [%175, %176, %177, %178], [%171, %172, %173, %174]) {chan_name = @QKIn_3, metadata = @air_QKIn_3_0_0} : (i32, i64, i64, memref<2x512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %180 = airrt.wait_all %179 : !airrt.event
      %c0_i64_45 = arith.constant 0 : i64
      %c0_46 = arith.constant 0 : index
      %c1_47 = arith.constant 1 : index
      %c89_i32 = arith.constant 89 : i32
      %181 = arith.index_cast %arg4 : index to i64
      %182 = arith.index_cast %c0_46 : index to i64
      %183 = arith.index_cast %c0_46 : index to i64
      %184 = arith.index_cast %c0_46 : index to i64
      %185 = arith.index_cast %c0_0 : index to i64
      %186 = arith.index_cast %c0_46 : index to i64
      %187 = arith.index_cast %c0_46 : index to i64
      %188 = arith.index_cast %c0_46 : index to i64
      %189 = arith.index_cast %c1_1 : index to i64
      %190 = arith.index_cast %c1_47 : index to i64
      %191 = arith.index_cast %c1_47 : index to i64
      %192 = arith.index_cast %c1_47 : index to i64
      %193 = arith.index_cast %c8192 : index to i64
      %194 = airrt.dma_memcpy_nd(%c89_i32, %181, %c0_i64_45, %arg2[%182, %183, %184, %185], [%190, %191, %192, %193], [%186, %187, %188, %189]) {chan_name = @VIn_0, metadata = @air_VIn_0_0_0} : (i32, i64, i64, memref<2x512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %195 = airrt.wait_all %194 : !airrt.event
      %c0_i64_48 = arith.constant 0 : i64
      %c0_49 = arith.constant 0 : index
      %c1_50 = arith.constant 1 : index
      %c92_i32 = arith.constant 92 : i32
      %196 = arith.index_cast %arg4 : index to i64
      %197 = arith.index_cast %c0_49 : index to i64
      %198 = arith.index_cast %c0_49 : index to i64
      %199 = arith.index_cast %c0_49 : index to i64
      %200 = arith.index_cast %c8192 : index to i64
      %201 = arith.index_cast %c0_49 : index to i64
      %202 = arith.index_cast %c0_49 : index to i64
      %203 = arith.index_cast %c0_49 : index to i64
      %204 = arith.index_cast %c1_1 : index to i64
      %205 = arith.index_cast %c1_50 : index to i64
      %206 = arith.index_cast %c1_50 : index to i64
      %207 = arith.index_cast %c1_50 : index to i64
      %208 = arith.index_cast %c8192 : index to i64
      %209 = airrt.dma_memcpy_nd(%c92_i32, %196, %c0_i64_48, %arg2[%197, %198, %199, %200], [%205, %206, %207, %208], [%201, %202, %203, %204]) {chan_name = @VIn_1, metadata = @air_VIn_1_0_0} : (i32, i64, i64, memref<2x512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %210 = airrt.wait_all %209 : !airrt.event
      %c0_i64_51 = arith.constant 0 : i64
      %c0_52 = arith.constant 0 : index
      %c1_53 = arith.constant 1 : index
      %c95_i32 = arith.constant 95 : i32
      %211 = arith.index_cast %arg4 : index to i64
      %212 = arith.index_cast %c0_52 : index to i64
      %213 = arith.index_cast %c0_52 : index to i64
      %214 = arith.index_cast %c0_52 : index to i64
      %215 = arith.index_cast %c16384 : index to i64
      %216 = arith.index_cast %c0_52 : index to i64
      %217 = arith.index_cast %c0_52 : index to i64
      %218 = arith.index_cast %c0_52 : index to i64
      %219 = arith.index_cast %c1_1 : index to i64
      %220 = arith.index_cast %c1_53 : index to i64
      %221 = arith.index_cast %c1_53 : index to i64
      %222 = arith.index_cast %c1_53 : index to i64
      %223 = arith.index_cast %c8192 : index to i64
      %224 = airrt.dma_memcpy_nd(%c95_i32, %211, %c0_i64_51, %arg2[%212, %213, %214, %215], [%220, %221, %222, %223], [%216, %217, %218, %219]) {chan_name = @VIn_2, metadata = @air_VIn_2_0_0} : (i32, i64, i64, memref<2x512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %225 = airrt.wait_all %224 : !airrt.event
      %c0_i64_54 = arith.constant 0 : i64
      %c0_55 = arith.constant 0 : index
      %c1_56 = arith.constant 1 : index
      %c98_i32 = arith.constant 98 : i32
      %226 = arith.index_cast %arg4 : index to i64
      %227 = arith.index_cast %c0_55 : index to i64
      %228 = arith.index_cast %c0_55 : index to i64
      %229 = arith.index_cast %c0_55 : index to i64
      %230 = arith.index_cast %c24576 : index to i64
      %231 = arith.index_cast %c0_55 : index to i64
      %232 = arith.index_cast %c0_55 : index to i64
      %233 = arith.index_cast %c0_55 : index to i64
      %234 = arith.index_cast %c1_1 : index to i64
      %235 = arith.index_cast %c1_56 : index to i64
      %236 = arith.index_cast %c1_56 : index to i64
      %237 = arith.index_cast %c1_56 : index to i64
      %238 = arith.index_cast %c8192 : index to i64
      %239 = airrt.dma_memcpy_nd(%c98_i32, %226, %c0_i64_54, %arg2[%227, %228, %229, %230], [%235, %236, %237, %238], [%231, %232, %233, %234]) {chan_name = @VIn_3, metadata = @air_VIn_3_0_0} : (i32, i64, i64, memref<2x512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %240 = airrt.wait_all %239 : !airrt.event
      %c0_i64_57 = arith.constant 0 : i64
      %c0_58 = arith.constant 0 : index
      %c1_59 = arith.constant 1 : index
      %c105_i32 = arith.constant 105 : i32
      %241 = arith.index_cast %arg4 : index to i64
      %242 = arith.index_cast %c0_58 : index to i64
      %243 = arith.index_cast %c0_58 : index to i64
      %244 = arith.index_cast %c0_58 : index to i64
      %245 = arith.index_cast %c0_0 : index to i64
      %246 = arith.index_cast %c0_58 : index to i64
      %247 = arith.index_cast %c0_58 : index to i64
      %248 = arith.index_cast %c0_58 : index to i64
      %249 = arith.index_cast %c1_1 : index to i64
      %250 = arith.index_cast %c1_59 : index to i64
      %251 = arith.index_cast %c1_59 : index to i64
      %252 = arith.index_cast %c1_59 : index to i64
      %253 = arith.index_cast %c16384 : index to i64
      %254 = airrt.dma_memcpy_nd(%c105_i32, %241, %c0_i64_57, %arg3[%242, %243, %244, %245], [%250, %251, %252, %253], [%246, %247, %248, %249]) {chan_name = @channel_0, metadata = @air_channel_0_0_0_0} : (i32, i64, i64, memref<2x256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %255 = airrt.wait_all %254 : !airrt.event
      %c0_i64_60 = arith.constant 0 : i64
      %c0_61 = arith.constant 0 : index
      %c1_62 = arith.constant 1 : index
      %c105_i32_63 = arith.constant 105 : i32
      %256 = arith.index_cast %arg4 : index to i64
      %257 = arith.index_cast %c0_61 : index to i64
      %258 = arith.index_cast %c0_61 : index to i64
      %259 = arith.index_cast %c0_61 : index to i64
      %260 = arith.index_cast %c16384 : index to i64
      %261 = arith.index_cast %c0_61 : index to i64
      %262 = arith.index_cast %c0_61 : index to i64
      %263 = arith.index_cast %c0_61 : index to i64
      %264 = arith.index_cast %c1_1 : index to i64
      %265 = arith.index_cast %c1_62 : index to i64
      %266 = arith.index_cast %c1_62 : index to i64
      %267 = arith.index_cast %c1_62 : index to i64
      %268 = arith.index_cast %c16384 : index to i64
      %269 = airrt.dma_memcpy_nd(%c105_i32_63, %256, %c0_i64_60, %arg3[%257, %258, %259, %260], [%265, %266, %267, %268], [%261, %262, %263, %264]) {chan_name = @channel_0, metadata = @air_channel_0_0_0_1} : (i32, i64, i64, memref<2x256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %270 = airrt.wait_all %269 : !airrt.event
      %c0_i64_64 = arith.constant 0 : i64
      %c0_65 = arith.constant 0 : index
      %c1_66 = arith.constant 1 : index
      %c105_i32_67 = arith.constant 105 : i32
      %271 = arith.index_cast %arg4 : index to i64
      %272 = arith.index_cast %c0_65 : index to i64
      %273 = arith.index_cast %c0_65 : index to i64
      %274 = arith.index_cast %c0_65 : index to i64
      %275 = arith.index_cast %c32768 : index to i64
      %276 = arith.index_cast %c0_65 : index to i64
      %277 = arith.index_cast %c0_65 : index to i64
      %278 = arith.index_cast %c0_65 : index to i64
      %279 = arith.index_cast %c1_1 : index to i64
      %280 = arith.index_cast %c1_66 : index to i64
      %281 = arith.index_cast %c1_66 : index to i64
      %282 = arith.index_cast %c1_66 : index to i64
      %283 = arith.index_cast %c16384 : index to i64
      %284 = airrt.dma_memcpy_nd(%c105_i32_67, %271, %c0_i64_64, %arg3[%272, %273, %274, %275], [%280, %281, %282, %283], [%276, %277, %278, %279]) {chan_name = @channel_0, metadata = @air_channel_0_0_0_2} : (i32, i64, i64, memref<2x256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %285 = airrt.wait_all %284 : !airrt.event
      %c0_i64_68 = arith.constant 0 : i64
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      %c105_i32_71 = arith.constant 105 : i32
      %286 = arith.index_cast %arg4 : index to i64
      %287 = arith.index_cast %c0_69 : index to i64
      %288 = arith.index_cast %c0_69 : index to i64
      %289 = arith.index_cast %c0_69 : index to i64
      %290 = arith.index_cast %c49152 : index to i64
      %291 = arith.index_cast %c0_69 : index to i64
      %292 = arith.index_cast %c0_69 : index to i64
      %293 = arith.index_cast %c0_69 : index to i64
      %294 = arith.index_cast %c1_1 : index to i64
      %295 = arith.index_cast %c1_70 : index to i64
      %296 = arith.index_cast %c1_70 : index to i64
      %297 = arith.index_cast %c1_70 : index to i64
      %298 = arith.index_cast %c16384 : index to i64
      %299 = airrt.dma_memcpy_nd(%c105_i32_71, %286, %c0_i64_68, %arg3[%287, %288, %289, %290], [%295, %296, %297, %298], [%291, %292, %293, %294]) {chan_name = @channel_0, metadata = @air_channel_0_0_0_3} : (i32, i64, i64, memref<2x256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %300 = airrt.wait_all %299 : !airrt.event
      %c0_i64_72 = arith.constant 0 : i64
      %c0_73 = arith.constant 0 : index
      %c1_74 = arith.constant 1 : index
      %c41_i32_75 = arith.constant 41 : i32
      %301 = arith.index_cast %arg4 : index to i64
      %302 = arith.index_cast %c0_73 : index to i64
      %303 = arith.index_cast %c0_73 : index to i64
      %304 = arith.index_cast %c0_0 : index to i64
      %305 = arith.index_cast %c32768 : index to i64
      %306 = arith.index_cast %c0_73 : index to i64
      %307 = arith.index_cast %c0_73 : index to i64
      %308 = arith.index_cast %c128 : index to i64
      %309 = arith.index_cast %c1_1 : index to i64
      %310 = arith.index_cast %c1_74 : index to i64
      %311 = arith.index_cast %c1_74 : index to i64
      %312 = arith.index_cast %c256 : index to i64
      %313 = arith.index_cast %c64 : index to i64
      %314 = airrt.dma_memcpy_nd(%c41_i32_75, %301, %c0_i64_72, %arg0[%302, %303, %304, %305], [%310, %311, %312, %313], [%306, %307, %308, %309]) {chan_name = @QKIn_0, metadata = @air_QKIn_0_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %315 = airrt.wait_all %314 : !airrt.event
      %c0_i64_76 = arith.constant 0 : i64
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      %c41_i32_79 = arith.constant 41 : i32
      %316 = arith.index_cast %arg4 : index to i64
      %317 = arith.index_cast %c0_77 : index to i64
      %318 = arith.index_cast %c0_77 : index to i64
      %319 = arith.index_cast %c0_0 : index to i64
      %320 = arith.index_cast %c32832 : index to i64
      %321 = arith.index_cast %c0_77 : index to i64
      %322 = arith.index_cast %c0_77 : index to i64
      %323 = arith.index_cast %c128 : index to i64
      %324 = arith.index_cast %c1_1 : index to i64
      %325 = arith.index_cast %c1_78 : index to i64
      %326 = arith.index_cast %c1_78 : index to i64
      %327 = arith.index_cast %c256 : index to i64
      %328 = arith.index_cast %c64 : index to i64
      %329 = airrt.dma_memcpy_nd(%c41_i32_79, %316, %c0_i64_76, %arg0[%317, %318, %319, %320], [%325, %326, %327, %328], [%321, %322, %323, %324]) {chan_name = @QKIn_0, metadata = @air_QKIn_0_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %330 = airrt.wait_all %329 : !airrt.event
      %c0_i64_80 = arith.constant 0 : i64
      %c0_81 = arith.constant 0 : index
      %c1_82 = arith.constant 1 : index
      %c41_i32_83 = arith.constant 41 : i32
      %331 = arith.index_cast %arg4 : index to i64
      %332 = arith.index_cast %c0_0 : index to i64
      %333 = arith.index_cast %c0_0 : index to i64
      %334 = arith.index_cast %c0_0 : index to i64
      %335 = arith.index_cast %c65536 : index to i64
      %336 = arith.index_cast %c8192 : index to i64
      %337 = arith.index_cast %c64 : index to i64
      %338 = arith.index_cast %c128 : index to i64
      %339 = arith.index_cast %c1_1 : index to i64
      %340 = arith.index_cast %c2 : index to i64
      %341 = arith.index_cast %c2 : index to i64
      %342 = arith.index_cast %c64 : index to i64
      %343 = arith.index_cast %c64 : index to i64
      %344 = airrt.dma_memcpy_nd(%c41_i32_83, %331, %c0_i64_80, %arg1[%332, %333, %334, %335], [%340, %341, %342, %343], [%336, %337, %338, %339]) {chan_name = @QKIn_0, metadata = @air_QKIn_0_1_0_0} : (i32, i64, i64, memref<2x512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %345 = airrt.wait_all %344 : !airrt.event
      %c0_i64_84 = arith.constant 0 : i64
      %c0_85 = arith.constant 0 : index
      %c1_86 = arith.constant 1 : index
      %c53_i32_87 = arith.constant 53 : i32
      %346 = arith.index_cast %arg4 : index to i64
      %347 = arith.index_cast %c0_85 : index to i64
      %348 = arith.index_cast %c0_85 : index to i64
      %349 = arith.index_cast %c0_0 : index to i64
      %350 = arith.index_cast %c32768 : index to i64
      %351 = arith.index_cast %c0_85 : index to i64
      %352 = arith.index_cast %c0_85 : index to i64
      %353 = arith.index_cast %c128 : index to i64
      %354 = arith.index_cast %c1_1 : index to i64
      %355 = arith.index_cast %c1_86 : index to i64
      %356 = arith.index_cast %c1_86 : index to i64
      %357 = arith.index_cast %c256 : index to i64
      %358 = arith.index_cast %c64 : index to i64
      %359 = airrt.dma_memcpy_nd(%c53_i32_87, %346, %c0_i64_84, %arg0[%347, %348, %349, %350], [%355, %356, %357, %358], [%351, %352, %353, %354]) {chan_name = @QKIn_1, metadata = @air_QKIn_1_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %360 = airrt.wait_all %359 : !airrt.event
      %c0_i64_88 = arith.constant 0 : i64
      %c0_89 = arith.constant 0 : index
      %c1_90 = arith.constant 1 : index
      %c53_i32_91 = arith.constant 53 : i32
      %361 = arith.index_cast %arg4 : index to i64
      %362 = arith.index_cast %c0_89 : index to i64
      %363 = arith.index_cast %c0_89 : index to i64
      %364 = arith.index_cast %c0_0 : index to i64
      %365 = arith.index_cast %c32832 : index to i64
      %366 = arith.index_cast %c0_89 : index to i64
      %367 = arith.index_cast %c0_89 : index to i64
      %368 = arith.index_cast %c128 : index to i64
      %369 = arith.index_cast %c1_1 : index to i64
      %370 = arith.index_cast %c1_90 : index to i64
      %371 = arith.index_cast %c1_90 : index to i64
      %372 = arith.index_cast %c256 : index to i64
      %373 = arith.index_cast %c64 : index to i64
      %374 = airrt.dma_memcpy_nd(%c53_i32_91, %361, %c0_i64_88, %arg0[%362, %363, %364, %365], [%370, %371, %372, %373], [%366, %367, %368, %369]) {chan_name = @QKIn_1, metadata = @air_QKIn_1_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %375 = airrt.wait_all %374 : !airrt.event
      %c0_i64_92 = arith.constant 0 : i64
      %c0_93 = arith.constant 0 : index
      %c1_94 = arith.constant 1 : index
      %c53_i32_95 = arith.constant 53 : i32
      %376 = arith.index_cast %arg4 : index to i64
      %377 = arith.index_cast %c0_0 : index to i64
      %378 = arith.index_cast %c0_0 : index to i64
      %379 = arith.index_cast %c0_0 : index to i64
      %380 = arith.index_cast %c81920 : index to i64
      %381 = arith.index_cast %c8192 : index to i64
      %382 = arith.index_cast %c64 : index to i64
      %383 = arith.index_cast %c128 : index to i64
      %384 = arith.index_cast %c1_1 : index to i64
      %385 = arith.index_cast %c2 : index to i64
      %386 = arith.index_cast %c2 : index to i64
      %387 = arith.index_cast %c64 : index to i64
      %388 = arith.index_cast %c64 : index to i64
      %389 = airrt.dma_memcpy_nd(%c53_i32_95, %376, %c0_i64_92, %arg1[%377, %378, %379, %380], [%385, %386, %387, %388], [%381, %382, %383, %384]) {chan_name = @QKIn_1, metadata = @air_QKIn_1_1_0_0} : (i32, i64, i64, memref<2x512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %390 = airrt.wait_all %389 : !airrt.event
      %c0_i64_96 = arith.constant 0 : i64
      %c0_97 = arith.constant 0 : index
      %c1_98 = arith.constant 1 : index
      %c65_i32_99 = arith.constant 65 : i32
      %391 = arith.index_cast %arg4 : index to i64
      %392 = arith.index_cast %c0_97 : index to i64
      %393 = arith.index_cast %c0_97 : index to i64
      %394 = arith.index_cast %c0_0 : index to i64
      %395 = arith.index_cast %c32768 : index to i64
      %396 = arith.index_cast %c0_97 : index to i64
      %397 = arith.index_cast %c0_97 : index to i64
      %398 = arith.index_cast %c128 : index to i64
      %399 = arith.index_cast %c1_1 : index to i64
      %400 = arith.index_cast %c1_98 : index to i64
      %401 = arith.index_cast %c1_98 : index to i64
      %402 = arith.index_cast %c256 : index to i64
      %403 = arith.index_cast %c64 : index to i64
      %404 = airrt.dma_memcpy_nd(%c65_i32_99, %391, %c0_i64_96, %arg0[%392, %393, %394, %395], [%400, %401, %402, %403], [%396, %397, %398, %399]) {chan_name = @QKIn_2, metadata = @air_QKIn_2_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %405 = airrt.wait_all %404 : !airrt.event
      %c0_i64_100 = arith.constant 0 : i64
      %c0_101 = arith.constant 0 : index
      %c1_102 = arith.constant 1 : index
      %c65_i32_103 = arith.constant 65 : i32
      %406 = arith.index_cast %arg4 : index to i64
      %407 = arith.index_cast %c0_101 : index to i64
      %408 = arith.index_cast %c0_101 : index to i64
      %409 = arith.index_cast %c0_0 : index to i64
      %410 = arith.index_cast %c32832 : index to i64
      %411 = arith.index_cast %c0_101 : index to i64
      %412 = arith.index_cast %c0_101 : index to i64
      %413 = arith.index_cast %c128 : index to i64
      %414 = arith.index_cast %c1_1 : index to i64
      %415 = arith.index_cast %c1_102 : index to i64
      %416 = arith.index_cast %c1_102 : index to i64
      %417 = arith.index_cast %c256 : index to i64
      %418 = arith.index_cast %c64 : index to i64
      %419 = airrt.dma_memcpy_nd(%c65_i32_103, %406, %c0_i64_100, %arg0[%407, %408, %409, %410], [%415, %416, %417, %418], [%411, %412, %413, %414]) {chan_name = @QKIn_2, metadata = @air_QKIn_2_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %420 = airrt.wait_all %419 : !airrt.event
      %c0_i64_104 = arith.constant 0 : i64
      %c0_105 = arith.constant 0 : index
      %c1_106 = arith.constant 1 : index
      %c65_i32_107 = arith.constant 65 : i32
      %421 = arith.index_cast %arg4 : index to i64
      %422 = arith.index_cast %c0_0 : index to i64
      %423 = arith.index_cast %c0_0 : index to i64
      %424 = arith.index_cast %c0_0 : index to i64
      %425 = arith.index_cast %c98304 : index to i64
      %426 = arith.index_cast %c8192 : index to i64
      %427 = arith.index_cast %c64 : index to i64
      %428 = arith.index_cast %c128 : index to i64
      %429 = arith.index_cast %c1_1 : index to i64
      %430 = arith.index_cast %c2 : index to i64
      %431 = arith.index_cast %c2 : index to i64
      %432 = arith.index_cast %c64 : index to i64
      %433 = arith.index_cast %c64 : index to i64
      %434 = airrt.dma_memcpy_nd(%c65_i32_107, %421, %c0_i64_104, %arg1[%422, %423, %424, %425], [%430, %431, %432, %433], [%426, %427, %428, %429]) {chan_name = @QKIn_2, metadata = @air_QKIn_2_1_0_0} : (i32, i64, i64, memref<2x512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %435 = airrt.wait_all %434 : !airrt.event
      %c0_i64_108 = arith.constant 0 : i64
      %c0_109 = arith.constant 0 : index
      %c1_110 = arith.constant 1 : index
      %c77_i32_111 = arith.constant 77 : i32
      %436 = arith.index_cast %arg4 : index to i64
      %437 = arith.index_cast %c0_109 : index to i64
      %438 = arith.index_cast %c0_109 : index to i64
      %439 = arith.index_cast %c0_0 : index to i64
      %440 = arith.index_cast %c32768 : index to i64
      %441 = arith.index_cast %c0_109 : index to i64
      %442 = arith.index_cast %c0_109 : index to i64
      %443 = arith.index_cast %c128 : index to i64
      %444 = arith.index_cast %c1_1 : index to i64
      %445 = arith.index_cast %c1_110 : index to i64
      %446 = arith.index_cast %c1_110 : index to i64
      %447 = arith.index_cast %c256 : index to i64
      %448 = arith.index_cast %c64 : index to i64
      %449 = airrt.dma_memcpy_nd(%c77_i32_111, %436, %c0_i64_108, %arg0[%437, %438, %439, %440], [%445, %446, %447, %448], [%441, %442, %443, %444]) {chan_name = @QKIn_3, metadata = @air_QKIn_3_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %450 = airrt.wait_all %449 : !airrt.event
      %c0_i64_112 = arith.constant 0 : i64
      %c0_113 = arith.constant 0 : index
      %c1_114 = arith.constant 1 : index
      %c77_i32_115 = arith.constant 77 : i32
      %451 = arith.index_cast %arg4 : index to i64
      %452 = arith.index_cast %c0_113 : index to i64
      %453 = arith.index_cast %c0_113 : index to i64
      %454 = arith.index_cast %c0_0 : index to i64
      %455 = arith.index_cast %c32832 : index to i64
      %456 = arith.index_cast %c0_113 : index to i64
      %457 = arith.index_cast %c0_113 : index to i64
      %458 = arith.index_cast %c128 : index to i64
      %459 = arith.index_cast %c1_1 : index to i64
      %460 = arith.index_cast %c1_114 : index to i64
      %461 = arith.index_cast %c1_114 : index to i64
      %462 = arith.index_cast %c256 : index to i64
      %463 = arith.index_cast %c64 : index to i64
      %464 = airrt.dma_memcpy_nd(%c77_i32_115, %451, %c0_i64_112, %arg0[%452, %453, %454, %455], [%460, %461, %462, %463], [%456, %457, %458, %459]) {chan_name = @QKIn_3, metadata = @air_QKIn_3_1_0_0} : (i32, i64, i64, memref<2x256x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %465 = airrt.wait_all %464 : !airrt.event
      %c0_i64_116 = arith.constant 0 : i64
      %c0_117 = arith.constant 0 : index
      %c1_118 = arith.constant 1 : index
      %c77_i32_119 = arith.constant 77 : i32
      %466 = arith.index_cast %arg4 : index to i64
      %467 = arith.index_cast %c0_0 : index to i64
      %468 = arith.index_cast %c0_0 : index to i64
      %469 = arith.index_cast %c0_0 : index to i64
      %470 = arith.index_cast %c114688 : index to i64
      %471 = arith.index_cast %c8192 : index to i64
      %472 = arith.index_cast %c64 : index to i64
      %473 = arith.index_cast %c128 : index to i64
      %474 = arith.index_cast %c1_1 : index to i64
      %475 = arith.index_cast %c2 : index to i64
      %476 = arith.index_cast %c2 : index to i64
      %477 = arith.index_cast %c64 : index to i64
      %478 = arith.index_cast %c64 : index to i64
      %479 = airrt.dma_memcpy_nd(%c77_i32_119, %466, %c0_i64_116, %arg1[%467, %468, %469, %470], [%475, %476, %477, %478], [%471, %472, %473, %474]) {chan_name = @QKIn_3, metadata = @air_QKIn_3_1_0_0} : (i32, i64, i64, memref<2x512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %480 = airrt.wait_all %479 : !airrt.event
      %c0_i64_120 = arith.constant 0 : i64
      %c0_121 = arith.constant 0 : index
      %c1_122 = arith.constant 1 : index
      %c89_i32_123 = arith.constant 89 : i32
      %481 = arith.index_cast %arg4 : index to i64
      %482 = arith.index_cast %c0_121 : index to i64
      %483 = arith.index_cast %c0_121 : index to i64
      %484 = arith.index_cast %c0_121 : index to i64
      %485 = arith.index_cast %c32768 : index to i64
      %486 = arith.index_cast %c0_121 : index to i64
      %487 = arith.index_cast %c0_121 : index to i64
      %488 = arith.index_cast %c0_121 : index to i64
      %489 = arith.index_cast %c1_1 : index to i64
      %490 = arith.index_cast %c1_122 : index to i64
      %491 = arith.index_cast %c1_122 : index to i64
      %492 = arith.index_cast %c1_122 : index to i64
      %493 = arith.index_cast %c8192 : index to i64
      %494 = airrt.dma_memcpy_nd(%c89_i32_123, %481, %c0_i64_120, %arg2[%482, %483, %484, %485], [%490, %491, %492, %493], [%486, %487, %488, %489]) {chan_name = @VIn_0, metadata = @air_VIn_0_1_0_0} : (i32, i64, i64, memref<2x512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %495 = airrt.wait_all %494 : !airrt.event
      %c0_i64_124 = arith.constant 0 : i64
      %c0_125 = arith.constant 0 : index
      %c1_126 = arith.constant 1 : index
      %c92_i32_127 = arith.constant 92 : i32
      %496 = arith.index_cast %arg4 : index to i64
      %497 = arith.index_cast %c0_125 : index to i64
      %498 = arith.index_cast %c0_125 : index to i64
      %499 = arith.index_cast %c0_125 : index to i64
      %500 = arith.index_cast %c40960 : index to i64
      %501 = arith.index_cast %c0_125 : index to i64
      %502 = arith.index_cast %c0_125 : index to i64
      %503 = arith.index_cast %c0_125 : index to i64
      %504 = arith.index_cast %c1_1 : index to i64
      %505 = arith.index_cast %c1_126 : index to i64
      %506 = arith.index_cast %c1_126 : index to i64
      %507 = arith.index_cast %c1_126 : index to i64
      %508 = arith.index_cast %c8192 : index to i64
      %509 = airrt.dma_memcpy_nd(%c92_i32_127, %496, %c0_i64_124, %arg2[%497, %498, %499, %500], [%505, %506, %507, %508], [%501, %502, %503, %504]) {chan_name = @VIn_1, metadata = @air_VIn_1_1_0_0} : (i32, i64, i64, memref<2x512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %510 = airrt.wait_all %509 : !airrt.event
      %c0_i64_128 = arith.constant 0 : i64
      %c0_129 = arith.constant 0 : index
      %c1_130 = arith.constant 1 : index
      %c95_i32_131 = arith.constant 95 : i32
      %511 = arith.index_cast %arg4 : index to i64
      %512 = arith.index_cast %c0_129 : index to i64
      %513 = arith.index_cast %c0_129 : index to i64
      %514 = arith.index_cast %c0_129 : index to i64
      %515 = arith.index_cast %c49152 : index to i64
      %516 = arith.index_cast %c0_129 : index to i64
      %517 = arith.index_cast %c0_129 : index to i64
      %518 = arith.index_cast %c0_129 : index to i64
      %519 = arith.index_cast %c1_1 : index to i64
      %520 = arith.index_cast %c1_130 : index to i64
      %521 = arith.index_cast %c1_130 : index to i64
      %522 = arith.index_cast %c1_130 : index to i64
      %523 = arith.index_cast %c8192 : index to i64
      %524 = airrt.dma_memcpy_nd(%c95_i32_131, %511, %c0_i64_128, %arg2[%512, %513, %514, %515], [%520, %521, %522, %523], [%516, %517, %518, %519]) {chan_name = @VIn_2, metadata = @air_VIn_2_1_0_0} : (i32, i64, i64, memref<2x512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %525 = airrt.wait_all %524 : !airrt.event
      %c0_i64_132 = arith.constant 0 : i64
      %c0_133 = arith.constant 0 : index
      %c1_134 = arith.constant 1 : index
      %c98_i32_135 = arith.constant 98 : i32
      %526 = arith.index_cast %arg4 : index to i64
      %527 = arith.index_cast %c0_133 : index to i64
      %528 = arith.index_cast %c0_133 : index to i64
      %529 = arith.index_cast %c0_133 : index to i64
      %530 = arith.index_cast %c57344 : index to i64
      %531 = arith.index_cast %c0_133 : index to i64
      %532 = arith.index_cast %c0_133 : index to i64
      %533 = arith.index_cast %c0_133 : index to i64
      %534 = arith.index_cast %c1_1 : index to i64
      %535 = arith.index_cast %c1_134 : index to i64
      %536 = arith.index_cast %c1_134 : index to i64
      %537 = arith.index_cast %c1_134 : index to i64
      %538 = arith.index_cast %c8192 : index to i64
      %539 = airrt.dma_memcpy_nd(%c98_i32_135, %526, %c0_i64_132, %arg2[%527, %528, %529, %530], [%535, %536, %537, %538], [%531, %532, %533, %534]) {chan_name = @VIn_3, metadata = @air_VIn_3_1_0_0} : (i32, i64, i64, memref<2x512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %540 = airrt.wait_all %539 : !airrt.event
      %c0_i64_136 = arith.constant 0 : i64
      %c0_137 = arith.constant 0 : index
      %c1_138 = arith.constant 1 : index
      %c105_i32_139 = arith.constant 105 : i32
      %541 = arith.index_cast %arg4 : index to i64
      %542 = arith.index_cast %c0_137 : index to i64
      %543 = arith.index_cast %c0_137 : index to i64
      %544 = arith.index_cast %c0_137 : index to i64
      %545 = arith.index_cast %c0_0 : index to i64
      %546 = arith.index_cast %c0_137 : index to i64
      %547 = arith.index_cast %c0_137 : index to i64
      %548 = arith.index_cast %c0_137 : index to i64
      %549 = arith.index_cast %c1_1 : index to i64
      %550 = arith.index_cast %c1_138 : index to i64
      %551 = arith.index_cast %c1_138 : index to i64
      %552 = arith.index_cast %c1_138 : index to i64
      %553 = arith.index_cast %c16384 : index to i64
      %554 = airrt.dma_memcpy_nd(%c105_i32_139, %541, %c0_i64_136, %arg3[%542, %543, %544, %545], [%550, %551, %552, %553], [%546, %547, %548, %549]) {chan_name = @channel_0, metadata = @air_channel_0_1_0_0} : (i32, i64, i64, memref<2x256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %555 = airrt.wait_all %554 : !airrt.event
      %c0_i64_140 = arith.constant 0 : i64
      %c0_141 = arith.constant 0 : index
      %c1_142 = arith.constant 1 : index
      %c105_i32_143 = arith.constant 105 : i32
      %556 = arith.index_cast %arg4 : index to i64
      %557 = arith.index_cast %c0_141 : index to i64
      %558 = arith.index_cast %c0_141 : index to i64
      %559 = arith.index_cast %c0_141 : index to i64
      %560 = arith.index_cast %c16384 : index to i64
      %561 = arith.index_cast %c0_141 : index to i64
      %562 = arith.index_cast %c0_141 : index to i64
      %563 = arith.index_cast %c0_141 : index to i64
      %564 = arith.index_cast %c1_1 : index to i64
      %565 = arith.index_cast %c1_142 : index to i64
      %566 = arith.index_cast %c1_142 : index to i64
      %567 = arith.index_cast %c1_142 : index to i64
      %568 = arith.index_cast %c16384 : index to i64
      %569 = airrt.dma_memcpy_nd(%c105_i32_143, %556, %c0_i64_140, %arg3[%557, %558, %559, %560], [%565, %566, %567, %568], [%561, %562, %563, %564]) {chan_name = @channel_0, metadata = @air_channel_0_1_0_1} : (i32, i64, i64, memref<2x256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %570 = airrt.wait_all %569 : !airrt.event
      %c0_i64_144 = arith.constant 0 : i64
      %c0_145 = arith.constant 0 : index
      %c1_146 = arith.constant 1 : index
      %c105_i32_147 = arith.constant 105 : i32
      %571 = arith.index_cast %arg4 : index to i64
      %572 = arith.index_cast %c0_145 : index to i64
      %573 = arith.index_cast %c0_145 : index to i64
      %574 = arith.index_cast %c0_145 : index to i64
      %575 = arith.index_cast %c32768 : index to i64
      %576 = arith.index_cast %c0_145 : index to i64
      %577 = arith.index_cast %c0_145 : index to i64
      %578 = arith.index_cast %c0_145 : index to i64
      %579 = arith.index_cast %c1_1 : index to i64
      %580 = arith.index_cast %c1_146 : index to i64
      %581 = arith.index_cast %c1_146 : index to i64
      %582 = arith.index_cast %c1_146 : index to i64
      %583 = arith.index_cast %c16384 : index to i64
      %584 = airrt.dma_memcpy_nd(%c105_i32_147, %571, %c0_i64_144, %arg3[%572, %573, %574, %575], [%580, %581, %582, %583], [%576, %577, %578, %579]) {chan_name = @channel_0, metadata = @air_channel_0_1_0_2} : (i32, i64, i64, memref<2x256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %585 = airrt.wait_all %584 : !airrt.event
      %c0_i64_148 = arith.constant 0 : i64
      %c0_149 = arith.constant 0 : index
      %c1_150 = arith.constant 1 : index
      %c105_i32_151 = arith.constant 105 : i32
      %586 = arith.index_cast %arg4 : index to i64
      %587 = arith.index_cast %c0_149 : index to i64
      %588 = arith.index_cast %c0_149 : index to i64
      %589 = arith.index_cast %c0_149 : index to i64
      %590 = arith.index_cast %c49152 : index to i64
      %591 = arith.index_cast %c0_149 : index to i64
      %592 = arith.index_cast %c0_149 : index to i64
      %593 = arith.index_cast %c0_149 : index to i64
      %594 = arith.index_cast %c1_1 : index to i64
      %595 = arith.index_cast %c1_150 : index to i64
      %596 = arith.index_cast %c1_150 : index to i64
      %597 = arith.index_cast %c1_150 : index to i64
      %598 = arith.index_cast %c16384 : index to i64
      %599 = airrt.dma_memcpy_nd(%c105_i32_151, %586, %c0_i64_148, %arg3[%587, %588, %589, %590], [%595, %596, %597, %598], [%591, %592, %593, %594]) {chan_name = @channel_0, metadata = @air_channel_0_1_0_3} : (i32, i64, i64, memref<2x256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %600 = airrt.wait_all %599 : !airrt.event
      %c0_152 = arith.constant 0 : index
      %c1_153 = arith.constant 1 : index
      %601 = airrt.wait_all : !airrt.event
      affine.for %arg5 = 0 to 2 {
        affine.for %arg6 = 0 to 1 {
          %c3_154 = arith.constant 3 : index
          %c64_155 = arith.constant 64 : index
          %c8 = arith.constant 8 : index
          %c1_156 = arith.constant 1 : index
          %c2_157 = arith.constant 2 : index
          %c0_158 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %602 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %603 = airrt.wait_all : !airrt.event
          %604 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %605 = airrt.wait_all : !airrt.event
          %606 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %607 = airrt.wait_all : !airrt.event
          %608 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %609 = airrt.wait_all : !airrt.event
          %610 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %611 = airrt.wait_all : !airrt.event
          %612 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %613 = airrt.wait_all : !airrt.event
          %614 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %615 = airrt.wait_all : !airrt.event
          %616 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %617 = airrt.wait_all : !airrt.event
          %618 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %619 = airrt.wait_all : !airrt.event
          airrt.dealloc %618 : memref<64x64xbf16, 1 : i32>
          %620 = airrt.wait_all : !airrt.event
          %621 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %622 = airrt.wait_all : !airrt.event
          airrt.dealloc %621 : memref<64x64xbf16, 1 : i32>
          %623 = airrt.wait_all : !airrt.event
          %624 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %625 = airrt.wait_all : !airrt.event
          airrt.dealloc %624 : memref<64x64xbf16, 1 : i32>
          %626 = airrt.wait_all : !airrt.event
          %627 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %628 = airrt.wait_all : !airrt.event
          airrt.dealloc %627 : memref<64x64xbf16, 1 : i32>
          %629 = airrt.wait_all : !airrt.event
          %630 = airrt.wait_all %611 : !airrt.event
          %631 = airrt.wait_all %613 : !airrt.event
          %632 = airrt.wait_all %615 : !airrt.event
          %633 = airrt.wait_all %617 : !airrt.event
          %634 = airrt.wait_all %630 : !airrt.event
          %635 = airrt.wait_all %631 : !airrt.event
          %636 = airrt.wait_all %632 : !airrt.event
          %637 = airrt.wait_all %633 : !airrt.event
          %h = airrt.herd_load "herd_0" (%arg5) {segment_name = "attn_seg"} : (index) -> i64
          %638 = airrt.wait_all : !airrt.event
          airrt.dealloc %602 : memref<64x64xbf16, 1 : i32>
          %639 = airrt.wait_all : !airrt.event
          airrt.dealloc %604 : memref<64x64xbf16, 1 : i32>
          %640 = airrt.wait_all : !airrt.event
          airrt.dealloc %606 : memref<64x64xbf16, 1 : i32>
          %641 = airrt.wait_all : !airrt.event
          airrt.dealloc %608 : memref<64x64xbf16, 1 : i32>
          %642 = airrt.wait_all : !airrt.event
          airrt.dealloc %616 : memref<64x64xbf16, 1 : i32>
          %643 = airrt.wait_all : !airrt.event
          airrt.dealloc %614 : memref<64x64xbf16, 1 : i32>
          %644 = airrt.wait_all : !airrt.event
          airrt.dealloc %612 : memref<64x64xbf16, 1 : i32>
          %645 = airrt.wait_all : !airrt.event
          airrt.dealloc %610 : memref<64x64xbf16, 1 : i32>
          %646 = airrt.wait_all : !airrt.event
          airrt.wait_all %619, %622, %625, %628, %638, %639, %640, %641, %642, %643, %644, %645, %646 {air.segment_end}
        }
      }
      airrt.wait_all %210, %240, %270, %300, %510, %540, %570, %600, %45, %30, %15, %60, %75, %90, %135, %120, %105, %150, %165, %180, %345, %330, %315, %360, %375, %390, %435, %420, %405, %450, %465, %480, %601, %585, %555, %525, %495, %285, %255, %225, %195 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
