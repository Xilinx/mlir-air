#loop_annotation = #llvm.loop_annotation<mustProgress = true>
#map = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 32768)>
#map1 = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 32768 + 64)>
#map2 = affine_map<()[s0] -> (s0 * 131072)>
#map3 = affine_map<()[s0] -> (s0 * 131072 + 16384)>
#map4 = affine_map<()[s0] -> (s0 * 131072 + 32768)>
#map5 = affine_map<()[s0] -> (s0 * 131072 + 49152)>
#map6 = affine_map<()[s0] -> (s0 * 65536)>
#map7 = affine_map<()[s0] -> (s0 * 65536 + 8192)>
#map8 = affine_map<()[s0] -> (s0 * 65536 + 16384)>
#map9 = affine_map<()[s0] -> (s0 * 65536 + 24576)>
#map10 = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 32768 + 32768)>
#map11 = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 32768 + 32832)>
#map12 = affine_map<()[s0] -> (s0 * 131072 + 65536)>
#map13 = affine_map<()[s0] -> (s0 * 131072 + 81920)>
#map14 = affine_map<()[s0] -> (s0 * 131072 + 98304)>
#map15 = affine_map<()[s0] -> (s0 * 131072 + 114688)>
#map16 = affine_map<()[s0] -> (s0 * 65536 + 32768)>
#map17 = affine_map<()[s0] -> (s0 * 65536 + 40960)>
#map18 = affine_map<()[s0] -> (s0 * 65536 + 49152)>
#map19 = affine_map<()[s0] -> (s0 * 65536 + 57344)>
#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set4 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
  aie.device(npu2_4col) @attn_seg_0_0 {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
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
    %buf255 = aie.buffer(%mem_tile_0_1) {sym_name = "buf255"} : memref<64x64xbf16, 1 : i32> 
    %buf254 = aie.buffer(%mem_tile_1_1) {sym_name = "buf254"} : memref<64x64xbf16, 1 : i32> 
    %buf253 = aie.buffer(%mem_tile_2_1) {sym_name = "buf253"} : memref<64x64xbf16, 1 : i32> 
    %buf252 = aie.buffer(%mem_tile_3_1) {sym_name = "buf252"} : memref<64x64xbf16, 1 : i32> 
    %buf251 = aie.buffer(%mem_tile_0_1) {sym_name = "buf251"} : memref<64x64xbf16, 1 : i32> 
    %buf250 = aie.buffer(%mem_tile_1_1) {sym_name = "buf250"} : memref<64x64xbf16, 1 : i32> 
    %buf249 = aie.buffer(%mem_tile_2_1) {sym_name = "buf249"} : memref<64x64xbf16, 1 : i32> 
    %buf248 = aie.buffer(%mem_tile_3_1) {sym_name = "buf248"} : memref<64x64xbf16, 1 : i32> 
    %buf247 = aie.buffer(%mem_tile_0_1) {sym_name = "buf247"} : memref<64x64xbf16, 1 : i32> 
    %buf246 = aie.buffer(%mem_tile_1_1) {sym_name = "buf246"} : memref<64x64xbf16, 1 : i32> 
    %buf245 = aie.buffer(%mem_tile_2_1) {sym_name = "buf245"} : memref<64x64xbf16, 1 : i32> 
    %buf244 = aie.buffer(%mem_tile_3_1) {sym_name = "buf244"} : memref<64x64xbf16, 1 : i32> 
    %buf243 = aie.buffer(%tile_3_5) {sym_name = "buf243"} : memref<64x1xbf16, 2 : i32> 
    %buf242 = aie.buffer(%tile_3_5) {sym_name = "buf242"} : memref<64x1xbf16, 2 : i32> 
    %buf241 = aie.buffer(%tile_3_5) {sym_name = "buf241"} : memref<64x64xbf16, 2 : i32> 
    %buf240 = aie.buffer(%tile_3_5) {sym_name = "buf240"} : memref<64x64xbf16, 2 : i32> 
    %buf239 = aie.buffer(%tile_3_5) {sym_name = "buf239"} : memref<64x64xbf16, 2 : i32> 
    %buf238 = aie.buffer(%tile_3_5) {sym_name = "buf238"} : memref<64x64xbf16, 2 : i32> 
    %buf237 = aie.buffer(%tile_3_5) {sym_name = "buf237"} : memref<64x64xbf16, 2 : i32> 
    %buf236 = aie.buffer(%tile_3_5) {sym_name = "buf236"} : memref<64x64xbf16, 2 : i32> 
    %buf235 = aie.buffer(%tile_3_5) {sym_name = "buf235"} : memref<64x1xbf16, 2 : i32> 
    %buf234 = aie.buffer(%tile_3_5) {sym_name = "buf234"} : memref<64x1xbf16, 2 : i32> 
    %buf233 = aie.buffer(%tile_2_5) {sym_name = "buf233"} : memref<64x1xbf16, 2 : i32> 
    %buf232 = aie.buffer(%tile_2_5) {sym_name = "buf232"} : memref<64x1xbf16, 2 : i32> 
    %buf231 = aie.buffer(%tile_2_5) {sym_name = "buf231"} : memref<64x64xbf16, 2 : i32> 
    %buf230 = aie.buffer(%tile_2_5) {sym_name = "buf230"} : memref<64x64xbf16, 2 : i32> 
    %buf229 = aie.buffer(%tile_2_5) {sym_name = "buf229"} : memref<64x64xbf16, 2 : i32> 
    %buf228 = aie.buffer(%tile_2_5) {sym_name = "buf228"} : memref<64x64xbf16, 2 : i32> 
    %buf227 = aie.buffer(%tile_2_5) {sym_name = "buf227"} : memref<64x64xbf16, 2 : i32> 
    %buf226 = aie.buffer(%tile_2_5) {sym_name = "buf226"} : memref<64x64xbf16, 2 : i32> 
    %buf225 = aie.buffer(%tile_2_5) {sym_name = "buf225"} : memref<64x1xbf16, 2 : i32> 
    %buf224 = aie.buffer(%tile_2_5) {sym_name = "buf224"} : memref<64x1xbf16, 2 : i32> 
    %buf223 = aie.buffer(%tile_1_5) {sym_name = "buf223"} : memref<64x1xbf16, 2 : i32> 
    %buf222 = aie.buffer(%tile_1_5) {sym_name = "buf222"} : memref<64x1xbf16, 2 : i32> 
    %buf221 = aie.buffer(%tile_1_5) {sym_name = "buf221"} : memref<64x64xbf16, 2 : i32> 
    %buf220 = aie.buffer(%tile_1_5) {sym_name = "buf220"} : memref<64x64xbf16, 2 : i32> 
    %buf219 = aie.buffer(%tile_1_5) {sym_name = "buf219"} : memref<64x64xbf16, 2 : i32> 
    %buf218 = aie.buffer(%tile_1_5) {sym_name = "buf218"} : memref<64x64xbf16, 2 : i32> 
    %buf217 = aie.buffer(%tile_1_5) {sym_name = "buf217"} : memref<64x64xbf16, 2 : i32> 
    %buf216 = aie.buffer(%tile_1_5) {sym_name = "buf216"} : memref<64x64xbf16, 2 : i32> 
    %buf215 = aie.buffer(%tile_1_5) {sym_name = "buf215"} : memref<64x1xbf16, 2 : i32> 
    %buf214 = aie.buffer(%tile_1_5) {sym_name = "buf214"} : memref<64x1xbf16, 2 : i32> 
    %buf213 = aie.buffer(%tile_0_5) {sym_name = "buf213"} : memref<64x1xbf16, 2 : i32> 
    %buf212 = aie.buffer(%tile_0_5) {sym_name = "buf212"} : memref<64x1xbf16, 2 : i32> 
    %buf211 = aie.buffer(%tile_0_5) {sym_name = "buf211"} : memref<64x64xbf16, 2 : i32> 
    %buf210 = aie.buffer(%tile_0_5) {sym_name = "buf210"} : memref<64x64xbf16, 2 : i32> 
    %buf209 = aie.buffer(%tile_0_5) {sym_name = "buf209"} : memref<64x64xbf16, 2 : i32> 
    %buf208 = aie.buffer(%tile_0_5) {sym_name = "buf208"} : memref<64x64xbf16, 2 : i32> 
    %buf207 = aie.buffer(%tile_0_5) {sym_name = "buf207"} : memref<64x64xbf16, 2 : i32> 
    %buf206 = aie.buffer(%tile_0_5) {sym_name = "buf206"} : memref<64x64xbf16, 2 : i32> 
    %buf205 = aie.buffer(%tile_0_5) {sym_name = "buf205"} : memref<64x1xbf16, 2 : i32> 
    %buf204 = aie.buffer(%tile_0_5) {sym_name = "buf204"} : memref<64x1xbf16, 2 : i32> 
    %buf203 = aie.buffer(%tile_3_4) {sym_name = "buf203"} : memref<64x1xbf16, 2 : i32> 
    %buf202 = aie.buffer(%tile_3_4) {sym_name = "buf202"} : memref<64x1xbf16, 2 : i32> 
    %buf201 = aie.buffer(%tile_3_4) {sym_name = "buf201"} : memref<64x64xbf16, 2 : i32> 
    %buf200 = aie.buffer(%tile_3_4) {sym_name = "buf200"} : memref<64x64xbf16, 2 : i32> 
    %buf199 = aie.buffer(%tile_3_4) {sym_name = "buf199"} : memref<64x64xbf16, 2 : i32> 
    %buf198 = aie.buffer(%tile_3_4) {sym_name = "buf198"} : memref<64x64xbf16, 2 : i32> 
    %buf197 = aie.buffer(%tile_3_4) {sym_name = "buf197"} : memref<64x64xbf16, 2 : i32> 
    %buf196 = aie.buffer(%tile_3_4) {sym_name = "buf196"} : memref<64x64xbf16, 2 : i32> 
    %buf195 = aie.buffer(%tile_3_4) {sym_name = "buf195"} : memref<64x1xbf16, 2 : i32> 
    %buf194 = aie.buffer(%tile_3_4) {sym_name = "buf194"} : memref<64x1xbf16, 2 : i32> 
    %buf193 = aie.buffer(%tile_3_4) {sym_name = "buf193"} : memref<64x64xbf16, 2 : i32> 
    %buf192 = aie.buffer(%tile_3_4) {sym_name = "buf192"} : memref<64x1xbf16, 2 : i32> 
    %buf191 = aie.buffer(%tile_3_4) {sym_name = "buf191"} : memref<64x1xbf16, 2 : i32> 
    %buf190 = aie.buffer(%tile_3_4) {sym_name = "buf190"} : memref<64x1xbf16, 2 : i32> 
    %buf189 = aie.buffer(%tile_3_4) {sym_name = "buf189"} : memref<64x1xbf16, 2 : i32> 
    %buf188 = aie.buffer(%tile_3_4) {sym_name = "buf188"} : memref<64x1xbf16, 2 : i32> 
    %buf187 = aie.buffer(%tile_3_4) {sym_name = "buf187"} : memref<64x1xbf16, 2 : i32> 
    %buf186 = aie.buffer(%tile_2_4) {sym_name = "buf186"} : memref<64x1xbf16, 2 : i32> 
    %buf185 = aie.buffer(%tile_2_4) {sym_name = "buf185"} : memref<64x1xbf16, 2 : i32> 
    %buf184 = aie.buffer(%tile_2_4) {sym_name = "buf184"} : memref<64x64xbf16, 2 : i32> 
    %buf183 = aie.buffer(%tile_2_4) {sym_name = "buf183"} : memref<64x64xbf16, 2 : i32> 
    %buf182 = aie.buffer(%tile_2_4) {sym_name = "buf182"} : memref<64x64xbf16, 2 : i32> 
    %buf181 = aie.buffer(%tile_2_4) {sym_name = "buf181"} : memref<64x64xbf16, 2 : i32> 
    %buf180 = aie.buffer(%tile_2_4) {sym_name = "buf180"} : memref<64x64xbf16, 2 : i32> 
    %buf179 = aie.buffer(%tile_2_4) {sym_name = "buf179"} : memref<64x64xbf16, 2 : i32> 
    %buf178 = aie.buffer(%tile_2_4) {sym_name = "buf178"} : memref<64x1xbf16, 2 : i32> 
    %buf177 = aie.buffer(%tile_2_4) {sym_name = "buf177"} : memref<64x1xbf16, 2 : i32> 
    %buf176 = aie.buffer(%tile_2_4) {sym_name = "buf176"} : memref<64x64xbf16, 2 : i32> 
    %buf175 = aie.buffer(%tile_2_4) {sym_name = "buf175"} : memref<64x1xbf16, 2 : i32> 
    %buf174 = aie.buffer(%tile_2_4) {sym_name = "buf174"} : memref<64x1xbf16, 2 : i32> 
    %buf173 = aie.buffer(%tile_2_4) {sym_name = "buf173"} : memref<64x1xbf16, 2 : i32> 
    %buf172 = aie.buffer(%tile_2_4) {sym_name = "buf172"} : memref<64x1xbf16, 2 : i32> 
    %buf171 = aie.buffer(%tile_2_4) {sym_name = "buf171"} : memref<64x1xbf16, 2 : i32> 
    %buf170 = aie.buffer(%tile_2_4) {sym_name = "buf170"} : memref<64x1xbf16, 2 : i32> 
    %buf169 = aie.buffer(%tile_1_4) {sym_name = "buf169"} : memref<64x1xbf16, 2 : i32> 
    %buf168 = aie.buffer(%tile_1_4) {sym_name = "buf168"} : memref<64x1xbf16, 2 : i32> 
    %buf167 = aie.buffer(%tile_1_4) {sym_name = "buf167"} : memref<64x64xbf16, 2 : i32> 
    %buf166 = aie.buffer(%tile_1_4) {sym_name = "buf166"} : memref<64x64xbf16, 2 : i32> 
    %buf165 = aie.buffer(%tile_1_4) {sym_name = "buf165"} : memref<64x64xbf16, 2 : i32> 
    %buf164 = aie.buffer(%tile_1_4) {sym_name = "buf164"} : memref<64x64xbf16, 2 : i32> 
    %buf163 = aie.buffer(%tile_1_4) {sym_name = "buf163"} : memref<64x64xbf16, 2 : i32> 
    %buf162 = aie.buffer(%tile_1_4) {sym_name = "buf162"} : memref<64x64xbf16, 2 : i32> 
    %buf161 = aie.buffer(%tile_1_4) {sym_name = "buf161"} : memref<64x1xbf16, 2 : i32> 
    %buf160 = aie.buffer(%tile_1_4) {sym_name = "buf160"} : memref<64x1xbf16, 2 : i32> 
    %buf159 = aie.buffer(%tile_1_4) {sym_name = "buf159"} : memref<64x64xbf16, 2 : i32> 
    %buf158 = aie.buffer(%tile_1_4) {sym_name = "buf158"} : memref<64x1xbf16, 2 : i32> 
    %buf157 = aie.buffer(%tile_1_4) {sym_name = "buf157"} : memref<64x1xbf16, 2 : i32> 
    %buf156 = aie.buffer(%tile_1_4) {sym_name = "buf156"} : memref<64x1xbf16, 2 : i32> 
    %buf155 = aie.buffer(%tile_1_4) {sym_name = "buf155"} : memref<64x1xbf16, 2 : i32> 
    %buf154 = aie.buffer(%tile_1_4) {sym_name = "buf154"} : memref<64x1xbf16, 2 : i32> 
    %buf153 = aie.buffer(%tile_1_4) {sym_name = "buf153"} : memref<64x1xbf16, 2 : i32> 
    %buf152 = aie.buffer(%tile_0_4) {sym_name = "buf152"} : memref<64x1xbf16, 2 : i32> 
    %buf151 = aie.buffer(%tile_0_4) {sym_name = "buf151"} : memref<64x1xbf16, 2 : i32> 
    %buf150 = aie.buffer(%tile_0_4) {sym_name = "buf150"} : memref<64x64xbf16, 2 : i32> 
    %buf149 = aie.buffer(%tile_0_4) {sym_name = "buf149"} : memref<64x64xbf16, 2 : i32> 
    %buf148 = aie.buffer(%tile_0_4) {sym_name = "buf148"} : memref<64x64xbf16, 2 : i32> 
    %buf147 = aie.buffer(%tile_0_4) {sym_name = "buf147"} : memref<64x64xbf16, 2 : i32> 
    %buf146 = aie.buffer(%tile_0_4) {sym_name = "buf146"} : memref<64x64xbf16, 2 : i32> 
    %buf145 = aie.buffer(%tile_0_4) {sym_name = "buf145"} : memref<64x64xbf16, 2 : i32> 
    %buf144 = aie.buffer(%tile_0_4) {sym_name = "buf144"} : memref<64x1xbf16, 2 : i32> 
    %buf143 = aie.buffer(%tile_0_4) {sym_name = "buf143"} : memref<64x1xbf16, 2 : i32> 
    %buf142 = aie.buffer(%tile_0_4) {sym_name = "buf142"} : memref<64x64xbf16, 2 : i32> 
    %buf141 = aie.buffer(%tile_0_4) {sym_name = "buf141"} : memref<64x1xbf16, 2 : i32> 
    %buf140 = aie.buffer(%tile_0_4) {sym_name = "buf140"} : memref<64x1xbf16, 2 : i32> 
    %buf139 = aie.buffer(%tile_0_4) {sym_name = "buf139"} : memref<64x1xbf16, 2 : i32> 
    %buf138 = aie.buffer(%tile_0_4) {sym_name = "buf138"} : memref<64x1xbf16, 2 : i32> 
    %buf137 = aie.buffer(%tile_0_4) {sym_name = "buf137"} : memref<64x1xbf16, 2 : i32> 
    %buf136 = aie.buffer(%tile_0_4) {sym_name = "buf136"} : memref<64x1xbf16, 2 : i32> 
    %buf135 = aie.buffer(%tile_3_3) {sym_name = "buf135"} : memref<64x1xbf16, 2 : i32> 
    %buf134 = aie.buffer(%tile_3_3) {sym_name = "buf134"} : memref<64x1xbf16, 2 : i32> 
    %buf133 = aie.buffer(%tile_3_3) {sym_name = "buf133"} : memref<64x64xbf16, 2 : i32> 
    %buf132 = aie.buffer(%tile_3_3) {sym_name = "buf132"} : memref<64x64xbf16, 2 : i32> 
    %buf131 = aie.buffer(%tile_3_3) {sym_name = "buf131"} : memref<64x64xbf16, 2 : i32> 
    %buf130 = aie.buffer(%tile_3_3) {sym_name = "buf130"} : memref<64x64xbf16, 2 : i32> 
    %buf129 = aie.buffer(%tile_3_3) {sym_name = "buf129"} : memref<64x64xbf16, 2 : i32> 
    %buf128 = aie.buffer(%tile_3_3) {sym_name = "buf128"} : memref<64x64xbf16, 2 : i32> 
    %buf127 = aie.buffer(%tile_3_3) {sym_name = "buf127"} : memref<64x1xbf16, 2 : i32> 
    %buf126 = aie.buffer(%tile_3_3) {sym_name = "buf126"} : memref<64x1xbf16, 2 : i32> 
    %buf125 = aie.buffer(%tile_3_3) {sym_name = "buf125"} : memref<64x64xbf16, 2 : i32> 
    %buf124 = aie.buffer(%tile_3_3) {sym_name = "buf124"} : memref<64x1xbf16, 2 : i32> 
    %buf123 = aie.buffer(%tile_3_3) {sym_name = "buf123"} : memref<64x1xbf16, 2 : i32> 
    %buf122 = aie.buffer(%tile_3_3) {sym_name = "buf122"} : memref<64x1xbf16, 2 : i32> 
    %buf121 = aie.buffer(%tile_3_3) {sym_name = "buf121"} : memref<64x1xbf16, 2 : i32> 
    %buf120 = aie.buffer(%tile_3_3) {sym_name = "buf120"} : memref<64x1xbf16, 2 : i32> 
    %buf119 = aie.buffer(%tile_3_3) {sym_name = "buf119"} : memref<64x1xbf16, 2 : i32> 
    %buf118 = aie.buffer(%tile_2_3) {sym_name = "buf118"} : memref<64x1xbf16, 2 : i32> 
    %buf117 = aie.buffer(%tile_2_3) {sym_name = "buf117"} : memref<64x1xbf16, 2 : i32> 
    %buf116 = aie.buffer(%tile_2_3) {sym_name = "buf116"} : memref<64x64xbf16, 2 : i32> 
    %buf115 = aie.buffer(%tile_2_3) {sym_name = "buf115"} : memref<64x64xbf16, 2 : i32> 
    %buf114 = aie.buffer(%tile_2_3) {sym_name = "buf114"} : memref<64x64xbf16, 2 : i32> 
    %buf113 = aie.buffer(%tile_2_3) {sym_name = "buf113"} : memref<64x64xbf16, 2 : i32> 
    %buf112 = aie.buffer(%tile_2_3) {sym_name = "buf112"} : memref<64x64xbf16, 2 : i32> 
    %buf111 = aie.buffer(%tile_2_3) {sym_name = "buf111"} : memref<64x64xbf16, 2 : i32> 
    %buf110 = aie.buffer(%tile_2_3) {sym_name = "buf110"} : memref<64x1xbf16, 2 : i32> 
    %buf109 = aie.buffer(%tile_2_3) {sym_name = "buf109"} : memref<64x1xbf16, 2 : i32> 
    %buf108 = aie.buffer(%tile_2_3) {sym_name = "buf108"} : memref<64x64xbf16, 2 : i32> 
    %buf107 = aie.buffer(%tile_2_3) {sym_name = "buf107"} : memref<64x1xbf16, 2 : i32> 
    %buf106 = aie.buffer(%tile_2_3) {sym_name = "buf106"} : memref<64x1xbf16, 2 : i32> 
    %buf105 = aie.buffer(%tile_2_3) {sym_name = "buf105"} : memref<64x1xbf16, 2 : i32> 
    %buf104 = aie.buffer(%tile_2_3) {sym_name = "buf104"} : memref<64x1xbf16, 2 : i32> 
    %buf103 = aie.buffer(%tile_2_3) {sym_name = "buf103"} : memref<64x1xbf16, 2 : i32> 
    %buf102 = aie.buffer(%tile_2_3) {sym_name = "buf102"} : memref<64x1xbf16, 2 : i32> 
    %buf101 = aie.buffer(%tile_1_3) {sym_name = "buf101"} : memref<64x1xbf16, 2 : i32> 
    %buf100 = aie.buffer(%tile_1_3) {sym_name = "buf100"} : memref<64x1xbf16, 2 : i32> 
    %buf99 = aie.buffer(%tile_1_3) {sym_name = "buf99"} : memref<64x64xbf16, 2 : i32> 
    %buf98 = aie.buffer(%tile_1_3) {sym_name = "buf98"} : memref<64x64xbf16, 2 : i32> 
    %buf97 = aie.buffer(%tile_1_3) {sym_name = "buf97"} : memref<64x64xbf16, 2 : i32> 
    %buf96 = aie.buffer(%tile_1_3) {sym_name = "buf96"} : memref<64x64xbf16, 2 : i32> 
    %buf95 = aie.buffer(%tile_1_3) {sym_name = "buf95"} : memref<64x64xbf16, 2 : i32> 
    %buf94 = aie.buffer(%tile_1_3) {sym_name = "buf94"} : memref<64x64xbf16, 2 : i32> 
    %buf93 = aie.buffer(%tile_1_3) {sym_name = "buf93"} : memref<64x1xbf16, 2 : i32> 
    %buf92 = aie.buffer(%tile_1_3) {sym_name = "buf92"} : memref<64x1xbf16, 2 : i32> 
    %buf91 = aie.buffer(%tile_1_3) {sym_name = "buf91"} : memref<64x64xbf16, 2 : i32> 
    %buf90 = aie.buffer(%tile_1_3) {sym_name = "buf90"} : memref<64x1xbf16, 2 : i32> 
    %buf89 = aie.buffer(%tile_1_3) {sym_name = "buf89"} : memref<64x1xbf16, 2 : i32> 
    %buf88 = aie.buffer(%tile_1_3) {sym_name = "buf88"} : memref<64x1xbf16, 2 : i32> 
    %buf87 = aie.buffer(%tile_1_3) {sym_name = "buf87"} : memref<64x1xbf16, 2 : i32> 
    %buf86 = aie.buffer(%tile_1_3) {sym_name = "buf86"} : memref<64x1xbf16, 2 : i32> 
    %buf85 = aie.buffer(%tile_1_3) {sym_name = "buf85"} : memref<64x1xbf16, 2 : i32> 
    %buf84 = aie.buffer(%tile_0_3) {sym_name = "buf84"} : memref<64x1xbf16, 2 : i32> 
    %buf83 = aie.buffer(%tile_0_3) {sym_name = "buf83"} : memref<64x1xbf16, 2 : i32> 
    %buf82 = aie.buffer(%tile_0_3) {sym_name = "buf82"} : memref<64x64xbf16, 2 : i32> 
    %buf81 = aie.buffer(%tile_0_3) {sym_name = "buf81"} : memref<64x64xbf16, 2 : i32> 
    %buf80 = aie.buffer(%tile_0_3) {sym_name = "buf80"} : memref<64x64xbf16, 2 : i32> 
    %buf79 = aie.buffer(%tile_0_3) {sym_name = "buf79"} : memref<64x64xbf16, 2 : i32> 
    %buf78 = aie.buffer(%tile_0_3) {sym_name = "buf78"} : memref<64x64xbf16, 2 : i32> 
    %buf77 = aie.buffer(%tile_0_3) {sym_name = "buf77"} : memref<64x64xbf16, 2 : i32> 
    %buf76 = aie.buffer(%tile_0_3) {sym_name = "buf76"} : memref<64x1xbf16, 2 : i32> 
    %buf75 = aie.buffer(%tile_0_3) {sym_name = "buf75"} : memref<64x1xbf16, 2 : i32> 
    %buf74 = aie.buffer(%tile_0_3) {sym_name = "buf74"} : memref<64x64xbf16, 2 : i32> 
    %buf73 = aie.buffer(%tile_0_3) {sym_name = "buf73"} : memref<64x1xbf16, 2 : i32> 
    %buf72 = aie.buffer(%tile_0_3) {sym_name = "buf72"} : memref<64x1xbf16, 2 : i32> 
    %buf71 = aie.buffer(%tile_0_3) {sym_name = "buf71"} : memref<64x1xbf16, 2 : i32> 
    %buf70 = aie.buffer(%tile_0_3) {sym_name = "buf70"} : memref<64x1xbf16, 2 : i32> 
    %buf69 = aie.buffer(%tile_0_3) {sym_name = "buf69"} : memref<64x1xbf16, 2 : i32> 
    %buf68 = aie.buffer(%tile_0_3) {sym_name = "buf68"} : memref<64x1xbf16, 2 : i32> 
    %buf67 = aie.buffer(%tile_3_2) {sym_name = "buf67"} : memref<64x1xbf16, 2 : i32> 
    %buf66 = aie.buffer(%tile_3_2) {sym_name = "buf66"} : memref<64x1xbf16, 2 : i32> 
    %buf65 = aie.buffer(%tile_3_2) {sym_name = "buf65"} : memref<64x64xbf16, 2 : i32> 
    %buf64 = aie.buffer(%tile_3_2) {sym_name = "buf64"} : memref<64x64xbf16, 2 : i32> 
    %buf63 = aie.buffer(%tile_3_2) {sym_name = "buf63"} : memref<64x64xbf16, 2 : i32> 
    %buf62 = aie.buffer(%tile_3_2) {sym_name = "buf62"} : memref<64x64xbf16, 2 : i32> 
    %buf61 = aie.buffer(%tile_3_2) {sym_name = "buf61"} : memref<64x64xbf16, 2 : i32> 
    %buf60 = aie.buffer(%tile_3_2) {sym_name = "buf60"} : memref<64x64xbf16, 2 : i32> 
    %buf59 = aie.buffer(%tile_3_2) {sym_name = "buf59"} : memref<64x1xbf16, 2 : i32> 
    %buf58 = aie.buffer(%tile_3_2) {sym_name = "buf58"} : memref<64x1xbf16, 2 : i32> 
    %buf57 = aie.buffer(%tile_3_2) {sym_name = "buf57"} : memref<64x64xbf16, 2 : i32> 
    %buf56 = aie.buffer(%tile_3_2) {sym_name = "buf56"} : memref<64x1xbf16, 2 : i32> 
    %buf55 = aie.buffer(%tile_3_2) {sym_name = "buf55"} : memref<64x1xbf16, 2 : i32> 
    %buf54 = aie.buffer(%tile_3_2) {sym_name = "buf54"} : memref<64x1xbf16, 2 : i32> 
    %buf53 = aie.buffer(%tile_3_2) {sym_name = "buf53"} : memref<64x1xbf16, 2 : i32> 
    %buf52 = aie.buffer(%tile_3_2) {sym_name = "buf52"} : memref<64x1xbf16, 2 : i32> 
    %buf51 = aie.buffer(%tile_3_2) {sym_name = "buf51"} : memref<64x1xbf16, 2 : i32> 
    %buf50 = aie.buffer(%tile_2_2) {sym_name = "buf50"} : memref<64x1xbf16, 2 : i32> 
    %buf49 = aie.buffer(%tile_2_2) {sym_name = "buf49"} : memref<64x1xbf16, 2 : i32> 
    %buf48 = aie.buffer(%tile_2_2) {sym_name = "buf48"} : memref<64x64xbf16, 2 : i32> 
    %buf47 = aie.buffer(%tile_2_2) {sym_name = "buf47"} : memref<64x64xbf16, 2 : i32> 
    %buf46 = aie.buffer(%tile_2_2) {sym_name = "buf46"} : memref<64x64xbf16, 2 : i32> 
    %buf45 = aie.buffer(%tile_2_2) {sym_name = "buf45"} : memref<64x64xbf16, 2 : i32> 
    %buf44 = aie.buffer(%tile_2_2) {sym_name = "buf44"} : memref<64x64xbf16, 2 : i32> 
    %buf43 = aie.buffer(%tile_2_2) {sym_name = "buf43"} : memref<64x64xbf16, 2 : i32> 
    %buf42 = aie.buffer(%tile_2_2) {sym_name = "buf42"} : memref<64x1xbf16, 2 : i32> 
    %buf41 = aie.buffer(%tile_2_2) {sym_name = "buf41"} : memref<64x1xbf16, 2 : i32> 
    %buf40 = aie.buffer(%tile_2_2) {sym_name = "buf40"} : memref<64x64xbf16, 2 : i32> 
    %buf39 = aie.buffer(%tile_2_2) {sym_name = "buf39"} : memref<64x1xbf16, 2 : i32> 
    %buf38 = aie.buffer(%tile_2_2) {sym_name = "buf38"} : memref<64x1xbf16, 2 : i32> 
    %buf37 = aie.buffer(%tile_2_2) {sym_name = "buf37"} : memref<64x1xbf16, 2 : i32> 
    %buf36 = aie.buffer(%tile_2_2) {sym_name = "buf36"} : memref<64x1xbf16, 2 : i32> 
    %buf35 = aie.buffer(%tile_2_2) {sym_name = "buf35"} : memref<64x1xbf16, 2 : i32> 
    %buf34 = aie.buffer(%tile_2_2) {sym_name = "buf34"} : memref<64x1xbf16, 2 : i32> 
    %buf33 = aie.buffer(%tile_1_2) {sym_name = "buf33"} : memref<64x1xbf16, 2 : i32> 
    %buf32 = aie.buffer(%tile_1_2) {sym_name = "buf32"} : memref<64x1xbf16, 2 : i32> 
    %buf31 = aie.buffer(%tile_1_2) {sym_name = "buf31"} : memref<64x64xbf16, 2 : i32> 
    %buf30 = aie.buffer(%tile_1_2) {sym_name = "buf30"} : memref<64x64xbf16, 2 : i32> 
    %buf29 = aie.buffer(%tile_1_2) {sym_name = "buf29"} : memref<64x64xbf16, 2 : i32> 
    %buf28 = aie.buffer(%tile_1_2) {sym_name = "buf28"} : memref<64x64xbf16, 2 : i32> 
    %buf27 = aie.buffer(%tile_1_2) {sym_name = "buf27"} : memref<64x64xbf16, 2 : i32> 
    %buf26 = aie.buffer(%tile_1_2) {sym_name = "buf26"} : memref<64x64xbf16, 2 : i32> 
    %buf25 = aie.buffer(%tile_1_2) {sym_name = "buf25"} : memref<64x1xbf16, 2 : i32> 
    %buf24 = aie.buffer(%tile_1_2) {sym_name = "buf24"} : memref<64x1xbf16, 2 : i32> 
    %buf23 = aie.buffer(%tile_1_2) {sym_name = "buf23"} : memref<64x64xbf16, 2 : i32> 
    %buf22 = aie.buffer(%tile_1_2) {sym_name = "buf22"} : memref<64x1xbf16, 2 : i32> 
    %buf21 = aie.buffer(%tile_1_2) {sym_name = "buf21"} : memref<64x1xbf16, 2 : i32> 
    %buf20 = aie.buffer(%tile_1_2) {sym_name = "buf20"} : memref<64x1xbf16, 2 : i32> 
    %buf19 = aie.buffer(%tile_1_2) {sym_name = "buf19"} : memref<64x1xbf16, 2 : i32> 
    %buf18 = aie.buffer(%tile_1_2) {sym_name = "buf18"} : memref<64x1xbf16, 2 : i32> 
    %buf17 = aie.buffer(%tile_1_2) {sym_name = "buf17"} : memref<64x1xbf16, 2 : i32> 
    %buf16 = aie.buffer(%tile_0_2) {sym_name = "buf16"} : memref<64x1xbf16, 2 : i32> 
    %buf15 = aie.buffer(%tile_0_2) {sym_name = "buf15"} : memref<64x1xbf16, 2 : i32> 
    %buf14 = aie.buffer(%tile_0_2) {sym_name = "buf14"} : memref<64x64xbf16, 2 : i32> 
    %buf13 = aie.buffer(%tile_0_2) {sym_name = "buf13"} : memref<64x64xbf16, 2 : i32> 
    %buf12 = aie.buffer(%tile_0_2) {sym_name = "buf12"} : memref<64x64xbf16, 2 : i32> 
    %buf11 = aie.buffer(%tile_0_2) {sym_name = "buf11"} : memref<64x64xbf16, 2 : i32> 
    %buf10 = aie.buffer(%tile_0_2) {sym_name = "buf10"} : memref<64x64xbf16, 2 : i32> 
    %buf9 = aie.buffer(%tile_0_2) {sym_name = "buf9"} : memref<64x64xbf16, 2 : i32> 
    %buf8 = aie.buffer(%tile_0_2) {sym_name = "buf8"} : memref<64x1xbf16, 2 : i32> 
    %buf7 = aie.buffer(%tile_0_2) {sym_name = "buf7"} : memref<64x1xbf16, 2 : i32> 
    %buf6 = aie.buffer(%tile_0_2) {sym_name = "buf6"} : memref<64x64xbf16, 2 : i32> 
    %buf5 = aie.buffer(%tile_0_2) {sym_name = "buf5"} : memref<64x1xbf16, 2 : i32> 
    %buf4 = aie.buffer(%tile_0_2) {sym_name = "buf4"} : memref<64x1xbf16, 2 : i32> 
    %buf3 = aie.buffer(%tile_0_2) {sym_name = "buf3"} : memref<64x1xbf16, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<64x1xbf16, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<64x1xbf16, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2x256x128xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2x512x128xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2x512x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<2x256x64xbf16>
    scf.for %arg0 = %c0 to %c2 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c2 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c2 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c2 step %c1 {
    } {loop_annotation = #loop_annotation}
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_74, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf240 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_75, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf237 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_73, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_76 = arith.constant 1 : index
      %c2_77 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf241) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf243) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf242) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf240, %buf238) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf240, %buf239) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_77 step %c1_76 {
        %collapse_shape_81 = memref.collapse_shape %buf236 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf236 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf238, %buf240, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_74, Release, 1)
        aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf236 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf239, %buf240, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_74, Release, 1)
        aie.use_lock(%lock_3_5_73, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf236 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf242, %buf235, %buf234) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf234, %buf241) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf236 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf237, %buf241) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf243, %buf234, %buf235) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf235, %buf243) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf241 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_79 = memref.collapse_shape %buf242 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf243 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_71, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf227 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_70, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_76 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf231) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf233) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf232) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf230, %buf228) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf230, %buf229) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_78 step %c1_76 {
        %collapse_shape_81 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf228, %buf230, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_71, Release, 1)
        aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf229, %buf230, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_71, Release, 1)
        aie.use_lock(%lock_2_5_70, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf232, %buf225, %buf224) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf224, %buf231) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf227, %buf231) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf233, %buf224, %buf225) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf225, %buf233) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf231 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_79 = memref.collapse_shape %buf232 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf233 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_68, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf220 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_69, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf217 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_67, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf221) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf223) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf222) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf220, %buf218) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf220, %buf219) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_76 step %c1_78 {
        %collapse_shape_81 = memref.collapse_shape %buf216 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf216 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf218, %buf220, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_68, Release, 1)
        aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf216 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf219, %buf220, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_68, Release, 1)
        aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf216 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf222, %buf215, %buf214) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf214, %buf221) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf216 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf217, %buf221) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf223, %buf214, %buf215) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf215, %buf223) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf221 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_79 = memref.collapse_shape %buf222 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf223 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_65, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf210 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_66, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf207 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_64, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_76 = arith.constant 1 : index
      %c2_77 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf211) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf213) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf212) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf210, %buf208) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf210, %buf209) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_77 step %c1_76 {
        %collapse_shape_81 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf208, %buf210, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_65, Release, 1)
        aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf209, %buf210, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_65, Release, 1)
        aie.use_lock(%lock_0_5_64, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf212, %buf205, %buf204) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf204, %buf211) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf207, %buf211) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf213, %buf204, %buf205) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf205, %buf213) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf211 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_79 = memref.collapse_shape %buf212 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf213 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf200 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_63, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf197 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_61, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_76 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf201) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf203) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf202) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf200, %buf198) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf200, %buf199) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_78 step %c1_76 {
        %collapse_shape_84 = memref.collapse_shape %buf196 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf196 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf198, %buf200, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_62, Release, 1)
        aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf196 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf199, %buf200, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_62, Release, 1)
        aie.use_lock(%lock_3_4_61, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf196 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf202, %buf195, %buf194) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf194, %buf201) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf196 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf197, %buf201) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf203, %buf194, %buf195) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf195, %buf203) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf193 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf192 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf191 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf202, %buf190) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf192, %buf202) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf192, %buf202, %buf189) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf190, %buf202, %buf188) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf189, %buf193) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf188, %buf201) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf201, %buf193) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf187) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf191, %buf189, %buf187) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf203, %buf188, %buf187) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf187, %buf191) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf193 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf202 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf191 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_59, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf183 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf180 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_58, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_76 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf184) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf186) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf185) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf183, %buf181) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf183, %buf182) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_78 step %c1_76 {
        %collapse_shape_84 = memref.collapse_shape %buf179 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf179 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf181, %buf183, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_59, Release, 1)
        aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf179 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf182, %buf183, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_59, Release, 1)
        aie.use_lock(%lock_2_4_58, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf179 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf185, %buf178, %buf177) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf177, %buf184) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf179 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf180, %buf184) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf186, %buf177, %buf178) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf178, %buf186) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf176 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf175 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf174 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf185, %buf173) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf175, %buf185) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf175, %buf185, %buf172) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf173, %buf185, %buf171) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf172, %buf176) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf171, %buf184) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf184, %buf176) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf170) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf174, %buf172, %buf170) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf186, %buf171, %buf170) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf170, %buf174) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf176 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf174 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_56, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf166 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf163 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_55, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_76 = arith.constant 0 : index
      %c2_77 = arith.constant 2 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf167) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf169) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf168) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf166, %buf164) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf166, %buf165) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      scf.for %arg0 = %c0_76 to %c2_77 step %c1_78 {
        %collapse_shape_84 = memref.collapse_shape %buf162 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf162 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf164, %buf166, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_56, Release, 1)
        aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf162 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf165, %buf166, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_56, Release, 1)
        aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf162 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf168, %buf161, %buf160) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf160, %buf167) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf162 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf163, %buf167) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf169, %buf160, %buf161) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf161, %buf169) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf159 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf158 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf157 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf168, %buf156) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf158, %buf168) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf158, %buf168, %buf155) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf156, %buf168, %buf154) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf155, %buf159) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf154, %buf167) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf167, %buf159) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf153) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf157, %buf155, %buf153) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf169, %buf154, %buf153) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf153, %buf157) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf159 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf168 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf157 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf149 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf146 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_52, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_76 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c2_77 = arith.constant 2 : index
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf150) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf152) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf151) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf149, %buf147) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf149, %buf148) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_77 step %c1_76 {
        %collapse_shape_84 = memref.collapse_shape %buf145 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf145 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf147, %buf149, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_53, Release, 1)
        aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf145 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf148, %buf149, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_53, Release, 1)
        aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf145 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf151, %buf144, %buf143) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf143, %buf150) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf145 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf146, %buf150) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf152, %buf143, %buf144) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf144, %buf152) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf142 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf141 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf140 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf151, %buf139) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf141, %buf151) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf141, %buf151, %buf138) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf139, %buf151, %buf137) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf138, %buf142) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf137, %buf150) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf150, %buf142) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf136) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf140, %buf138, %buf136) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf152, %buf137, %buf136) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf136, %buf140) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf142 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf151 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf140 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf132 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_51, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf129 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_49, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf133) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf135) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf134) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf132, %buf130) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf132, %buf131) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_76 step %c1_78 {
        %collapse_shape_84 = memref.collapse_shape %buf128 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf128 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf130, %buf132, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_50, Release, 1)
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf128 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf131, %buf132, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_50, Release, 1)
        aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf128 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf134, %buf127, %buf126) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf126, %buf133) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf128 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf129, %buf133) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf135, %buf126, %buf127) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf127, %buf135) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf125 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf124 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf123 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf134, %buf122) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf124, %buf134) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf124, %buf134, %buf121) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf122, %buf134, %buf120) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf121, %buf125) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf120, %buf133) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf133, %buf125) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf119) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf123, %buf121, %buf119) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf135, %buf120, %buf119) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf119, %buf123) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf125 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf134 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf123 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf115 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf112 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_46, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_76 = arith.constant 0 : index
      %c1_77 = arith.constant 1 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf116) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf118) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf117) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf115, %buf113) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf115, %buf114) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      scf.for %arg0 = %c0_76 to %c2_78 step %c1_77 {
        %collapse_shape_84 = memref.collapse_shape %buf111 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf111 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf113, %buf115, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_47, Release, 1)
        aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf111 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf114, %buf115, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_47, Release, 1)
        aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf111 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf117, %buf110, %buf109) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf109, %buf116) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf111 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf112, %buf116) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf118, %buf109, %buf110) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf110, %buf118) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf108 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf107 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf106 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf117, %buf105) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf107, %buf117) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf107, %buf117, %buf104) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf105, %buf117, %buf103) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf104, %buf108) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf103, %buf116) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf116, %buf108) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf102) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf106, %buf104, %buf102) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf118, %buf103, %buf102) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf102, %buf106) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf108 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf117 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf106 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_44, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf98 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf95 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_43, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf99) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf101) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf100) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf98, %buf96) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf98, %buf97) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_76 step %c1_78 {
        %collapse_shape_84 = memref.collapse_shape %buf94 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf94 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf96, %buf98, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_44, Release, 1)
        aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf94 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf97, %buf98, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_44, Release, 1)
        aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf94 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf100, %buf93, %buf92) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf92, %buf99) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf94 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf95, %buf99) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf101, %buf92, %buf93) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf93, %buf101) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf91 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf90 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf100, %buf88) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf90, %buf100) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf90, %buf100, %buf87) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf88, %buf100, %buf86) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf87, %buf91) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf86, %buf99) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf99, %buf91) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf85) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf89, %buf87, %buf85) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf101, %buf86, %buf85) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf85, %buf89) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf91 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf100 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf81 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf78 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_40, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_77 = arith.constant 1 : index
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf82) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf84) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf83) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf81, %buf79) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf81, %buf80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_76 step %c1_77 {
        %collapse_shape_84 = memref.collapse_shape %buf77 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf77 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf79, %buf81, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_41, Release, 1)
        aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf77 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf80, %buf81, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_41, Release, 1)
        aie.use_lock(%lock_0_3_40, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf77 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf83, %buf76, %buf75) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf75, %buf82) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf77 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf78, %buf82) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf84, %buf75, %buf76) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf76, %buf84) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf74 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf73 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf72 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf83, %buf71) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf73, %buf83) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf73, %buf83, %buf70) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf71, %buf83, %buf69) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf70, %buf74) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf69, %buf82) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf82, %buf74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf68) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf72, %buf70, %buf68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf84, %buf69, %buf68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf68, %buf72) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf74 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf83 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf72 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf57 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf64 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf61 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_35, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_76 = arith.constant 1 : index
      %c2_77 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_2_38, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf65) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf67) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf66) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf64, %buf62) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf64, %buf63) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_77 step %c1_76 {
        %collapse_shape_81 = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf62, %buf64, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_36, Release, 1)
        aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf63, %buf64, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_36, Release, 1)
        aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf66, %buf59, %buf58) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf58, %buf65) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf61, %buf65) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf67, %buf58, %buf59) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf59, %buf67) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf56 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf55 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf66, %buf54) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf56, %buf66) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf56, %buf66, %buf53) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf54, %buf66, %buf52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf53, %buf57) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf52, %buf65) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf65, %buf57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf51) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf55, %buf53, %buf51) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf67, %buf52, %buf51) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf51, %buf55) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf55, %buf57) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_39, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf40 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf47 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf44 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_30, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_76 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_77 = arith.constant 0 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_2_33, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf48) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf50) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf49) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf47, %buf45) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf47, %buf46) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_78 step %c1_76 {
        %collapse_shape_81 = memref.collapse_shape %buf43 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf43 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf45, %buf47, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_31, Release, 1)
        aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf43 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf46, %buf47, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_31, Release, 1)
        aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf43 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf49, %buf42, %buf41) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf41, %buf48) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf43 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf44, %buf48) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf50, %buf41, %buf42) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf42, %buf50) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf40 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf39 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf38 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf49, %buf37) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf39, %buf49) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf39, %buf49, %buf36) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf37, %buf49, %buf35) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf36, %buf40) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf35, %buf48) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf48, %buf40) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf34) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf38, %buf36, %buf34) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf50, %buf35, %buf34) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf34, %buf38) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf38, %buf40) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_34, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf23 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_27, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf27 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_25, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_28, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf31) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf33) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf32) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf30, %buf28) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf30, %buf29) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_76 step %c1_78 {
        %collapse_shape_81 = memref.collapse_shape %buf26 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf26 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf28, %buf30, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_26, Release, 1)
        aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf26 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf29, %buf30, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_26, Release, 1)
        aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf26 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf32, %buf25, %buf24) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf24, %buf31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf26 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf27, %buf31) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf33, %buf24, %buf25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf25, %buf33) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf22 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf21 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf32, %buf20) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf22, %buf32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf22, %buf32, %buf19) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf20, %buf32, %buf18) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf19, %buf23) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf18, %buf31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf31, %buf23) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf17) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf21, %buf19, %buf17) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf33, %buf18, %buf17) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf17, %buf21) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf21, %buf23) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_29, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_23, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_20, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_76 = arith.constant 1 : index
      %c2_77 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_23, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf14) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf16) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf15) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf13, %buf11) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf13, %buf12) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_77 step %c1_76 {
        %collapse_shape_81 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf11, %buf13, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_21, Release, 1)
        aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf12, %buf13, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_21, Release, 1)
        aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf15, %buf8, %buf7) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf7, %buf14) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf10, %buf14) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf16, %buf7, %buf8) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf8, %buf16) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf6 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf5 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf4 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf15, %buf3) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf5, %buf15) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf5, %buf15, %buf2) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf3, %buf15, %buf1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf2, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf1, %buf14) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf14, %buf6) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf4, %buf2, %buf0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf16, %buf1, %buf0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf0, %buf4) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf4, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_24, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    air.channel @channel_62 [1, 1]
    air.channel @QK2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @QK2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_60 [1, 1]
    air.channel @QK2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @QK2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_58 [1, 1]
    air.channel @QK2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @QK2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_56 [1, 1]
    air.channel @QK2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @QK2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_54 [1, 1]
    air.channel @V2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @V2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_52 [1, 1]
    air.channel @V2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @V2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_50 [1, 1]
    air.channel @V2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @V2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_48 [1, 1]
    air.channel @V2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @V2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_0 [1, 1]
    air.channel @channel_45 [1, 1]
    air.channel @channel_46 [1, 1]
    air.channel @channel_47 [1, 1]
    air.channel @channel_37 [1, 1]
    air.channel @channel_39 [1, 1]
    air.channel @channel_41 [1, 1]
    air.channel @channel_43 [1, 1]
    air.channel @channel_25 [1, 1] {channel_type = "cascade"}
    air.channel @channel_26 [1, 1] {channel_type = "cascade"}
    air.channel @channel_27 [1, 1] {channel_type = "cascade"}
    air.channel @channel_28 [1, 1] {channel_type = "cascade"}
    air.channel @channel_29 [1, 1] {channel_type = "cascade"}
    air.channel @channel_30 [1, 1] {channel_type = "cascade"}
    air.channel @channel_31 [1, 1] {channel_type = "cascade"}
    air.channel @channel_32 [1, 1] {channel_type = "cascade"}
    air.channel @channel_33 [1, 1] {channel_type = "cascade"}
    air.channel @channel_34 [1, 1] {channel_type = "cascade"}
    air.channel @channel_35 [1, 1] {channel_type = "cascade"}
    air.channel @channel_36 [1, 1] {channel_type = "cascade"}
    air.channel @channel_13 [1, 1] {channel_type = "cascade"}
    air.channel @channel_14 [1, 1] {channel_type = "cascade"}
    air.channel @channel_15 [1, 1] {channel_type = "cascade"}
    air.channel @channel_16 [1, 1] {channel_type = "cascade"}
    air.channel @channel_17 [1, 1] {channel_type = "cascade"}
    air.channel @channel_18 [1, 1] {channel_type = "cascade"}
    air.channel @channel_19 [1, 1] {channel_type = "cascade"}
    air.channel @channel_20 [1, 1] {channel_type = "cascade"}
    air.channel @channel_21 [1, 1] {channel_type = "cascade"}
    air.channel @channel_22 [1, 1] {channel_type = "cascade"}
    air.channel @channel_23 [1, 1] {channel_type = "cascade"}
    air.channel @channel_24 [1, 1] {channel_type = "cascade"}
    air.channel @channel_1 [1, 1] {channel_type = "cascade"}
    air.channel @channel_2 [1, 1] {channel_type = "cascade"}
    air.channel @channel_3 [1, 1] {channel_type = "cascade"}
    air.channel @channel_4 [1, 1] {channel_type = "cascade"}
    air.channel @channel_5 [1, 1] {channel_type = "cascade"}
    air.channel @channel_6 [1, 1] {channel_type = "cascade"}
    air.channel @channel_7 [1, 1] {channel_type = "cascade"}
    air.channel @channel_8 [1, 1] {channel_type = "cascade"}
    air.channel @channel_9 [1, 1] {channel_type = "cascade"}
    air.channel @channel_10 [1, 1] {channel_type = "cascade"}
    air.channel @channel_11 [1, 1] {channel_type = "cascade"}
    air.channel @channel_12 [1, 1] {channel_type = "cascade"}
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
      aie.dma_bd(%buf251 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf255 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_16, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf247 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf255 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_17, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf247 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_15, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_0_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf251 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_19, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf250 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf254 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_11, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf246 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf254 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_12, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf246 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_10, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_1_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf250 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_14, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf249 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf253 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_6, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf245 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf253 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_7, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf245 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_2_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf249 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf248 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf252 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf244 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf252 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf244 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_3_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf248 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
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
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 0 : i64, segment_unroll_y = 0 : i64}
  aie.device(npu2_4col) @attn_seg_1_0 {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
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
    %buf511 = aie.buffer(%mem_tile_0_1) {sym_name = "buf511"} : memref<64x64xbf16, 1 : i32> 
    %buf510 = aie.buffer(%mem_tile_1_1) {sym_name = "buf510"} : memref<64x64xbf16, 1 : i32> 
    %buf509 = aie.buffer(%mem_tile_2_1) {sym_name = "buf509"} : memref<64x64xbf16, 1 : i32> 
    %buf508 = aie.buffer(%mem_tile_3_1) {sym_name = "buf508"} : memref<64x64xbf16, 1 : i32> 
    %buf507 = aie.buffer(%mem_tile_0_1) {sym_name = "buf507"} : memref<64x64xbf16, 1 : i32> 
    %buf506 = aie.buffer(%mem_tile_1_1) {sym_name = "buf506"} : memref<64x64xbf16, 1 : i32> 
    %buf505 = aie.buffer(%mem_tile_2_1) {sym_name = "buf505"} : memref<64x64xbf16, 1 : i32> 
    %buf504 = aie.buffer(%mem_tile_3_1) {sym_name = "buf504"} : memref<64x64xbf16, 1 : i32> 
    %buf503 = aie.buffer(%mem_tile_0_1) {sym_name = "buf503"} : memref<64x64xbf16, 1 : i32> 
    %buf502 = aie.buffer(%mem_tile_1_1) {sym_name = "buf502"} : memref<64x64xbf16, 1 : i32> 
    %buf501 = aie.buffer(%mem_tile_2_1) {sym_name = "buf501"} : memref<64x64xbf16, 1 : i32> 
    %buf500 = aie.buffer(%mem_tile_3_1) {sym_name = "buf500"} : memref<64x64xbf16, 1 : i32> 
    %buf499 = aie.buffer(%tile_3_5) {sym_name = "buf499"} : memref<64x1xbf16, 2 : i32> 
    %buf498 = aie.buffer(%tile_3_5) {sym_name = "buf498"} : memref<64x1xbf16, 2 : i32> 
    %buf497 = aie.buffer(%tile_3_5) {sym_name = "buf497"} : memref<64x64xbf16, 2 : i32> 
    %buf496 = aie.buffer(%tile_3_5) {sym_name = "buf496"} : memref<64x64xbf16, 2 : i32> 
    %buf495 = aie.buffer(%tile_3_5) {sym_name = "buf495"} : memref<64x64xbf16, 2 : i32> 
    %buf494 = aie.buffer(%tile_3_5) {sym_name = "buf494"} : memref<64x64xbf16, 2 : i32> 
    %buf493 = aie.buffer(%tile_3_5) {sym_name = "buf493"} : memref<64x64xbf16, 2 : i32> 
    %buf492 = aie.buffer(%tile_3_5) {sym_name = "buf492"} : memref<64x64xbf16, 2 : i32> 
    %buf491 = aie.buffer(%tile_3_5) {sym_name = "buf491"} : memref<64x1xbf16, 2 : i32> 
    %buf490 = aie.buffer(%tile_3_5) {sym_name = "buf490"} : memref<64x1xbf16, 2 : i32> 
    %buf489 = aie.buffer(%tile_2_5) {sym_name = "buf489"} : memref<64x1xbf16, 2 : i32> 
    %buf488 = aie.buffer(%tile_2_5) {sym_name = "buf488"} : memref<64x1xbf16, 2 : i32> 
    %buf487 = aie.buffer(%tile_2_5) {sym_name = "buf487"} : memref<64x64xbf16, 2 : i32> 
    %buf486 = aie.buffer(%tile_2_5) {sym_name = "buf486"} : memref<64x64xbf16, 2 : i32> 
    %buf485 = aie.buffer(%tile_2_5) {sym_name = "buf485"} : memref<64x64xbf16, 2 : i32> 
    %buf484 = aie.buffer(%tile_2_5) {sym_name = "buf484"} : memref<64x64xbf16, 2 : i32> 
    %buf483 = aie.buffer(%tile_2_5) {sym_name = "buf483"} : memref<64x64xbf16, 2 : i32> 
    %buf482 = aie.buffer(%tile_2_5) {sym_name = "buf482"} : memref<64x64xbf16, 2 : i32> 
    %buf481 = aie.buffer(%tile_2_5) {sym_name = "buf481"} : memref<64x1xbf16, 2 : i32> 
    %buf480 = aie.buffer(%tile_2_5) {sym_name = "buf480"} : memref<64x1xbf16, 2 : i32> 
    %buf479 = aie.buffer(%tile_1_5) {sym_name = "buf479"} : memref<64x1xbf16, 2 : i32> 
    %buf478 = aie.buffer(%tile_1_5) {sym_name = "buf478"} : memref<64x1xbf16, 2 : i32> 
    %buf477 = aie.buffer(%tile_1_5) {sym_name = "buf477"} : memref<64x64xbf16, 2 : i32> 
    %buf476 = aie.buffer(%tile_1_5) {sym_name = "buf476"} : memref<64x64xbf16, 2 : i32> 
    %buf475 = aie.buffer(%tile_1_5) {sym_name = "buf475"} : memref<64x64xbf16, 2 : i32> 
    %buf474 = aie.buffer(%tile_1_5) {sym_name = "buf474"} : memref<64x64xbf16, 2 : i32> 
    %buf473 = aie.buffer(%tile_1_5) {sym_name = "buf473"} : memref<64x64xbf16, 2 : i32> 
    %buf472 = aie.buffer(%tile_1_5) {sym_name = "buf472"} : memref<64x64xbf16, 2 : i32> 
    %buf471 = aie.buffer(%tile_1_5) {sym_name = "buf471"} : memref<64x1xbf16, 2 : i32> 
    %buf470 = aie.buffer(%tile_1_5) {sym_name = "buf470"} : memref<64x1xbf16, 2 : i32> 
    %buf469 = aie.buffer(%tile_0_5) {sym_name = "buf469"} : memref<64x1xbf16, 2 : i32> 
    %buf468 = aie.buffer(%tile_0_5) {sym_name = "buf468"} : memref<64x1xbf16, 2 : i32> 
    %buf467 = aie.buffer(%tile_0_5) {sym_name = "buf467"} : memref<64x64xbf16, 2 : i32> 
    %buf466 = aie.buffer(%tile_0_5) {sym_name = "buf466"} : memref<64x64xbf16, 2 : i32> 
    %buf465 = aie.buffer(%tile_0_5) {sym_name = "buf465"} : memref<64x64xbf16, 2 : i32> 
    %buf464 = aie.buffer(%tile_0_5) {sym_name = "buf464"} : memref<64x64xbf16, 2 : i32> 
    %buf463 = aie.buffer(%tile_0_5) {sym_name = "buf463"} : memref<64x64xbf16, 2 : i32> 
    %buf462 = aie.buffer(%tile_0_5) {sym_name = "buf462"} : memref<64x64xbf16, 2 : i32> 
    %buf461 = aie.buffer(%tile_0_5) {sym_name = "buf461"} : memref<64x1xbf16, 2 : i32> 
    %buf460 = aie.buffer(%tile_0_5) {sym_name = "buf460"} : memref<64x1xbf16, 2 : i32> 
    %buf459 = aie.buffer(%tile_3_4) {sym_name = "buf459"} : memref<64x1xbf16, 2 : i32> 
    %buf458 = aie.buffer(%tile_3_4) {sym_name = "buf458"} : memref<64x1xbf16, 2 : i32> 
    %buf457 = aie.buffer(%tile_3_4) {sym_name = "buf457"} : memref<64x64xbf16, 2 : i32> 
    %buf456 = aie.buffer(%tile_3_4) {sym_name = "buf456"} : memref<64x64xbf16, 2 : i32> 
    %buf455 = aie.buffer(%tile_3_4) {sym_name = "buf455"} : memref<64x64xbf16, 2 : i32> 
    %buf454 = aie.buffer(%tile_3_4) {sym_name = "buf454"} : memref<64x64xbf16, 2 : i32> 
    %buf453 = aie.buffer(%tile_3_4) {sym_name = "buf453"} : memref<64x64xbf16, 2 : i32> 
    %buf452 = aie.buffer(%tile_3_4) {sym_name = "buf452"} : memref<64x64xbf16, 2 : i32> 
    %buf451 = aie.buffer(%tile_3_4) {sym_name = "buf451"} : memref<64x1xbf16, 2 : i32> 
    %buf450 = aie.buffer(%tile_3_4) {sym_name = "buf450"} : memref<64x1xbf16, 2 : i32> 
    %buf449 = aie.buffer(%tile_3_4) {sym_name = "buf449"} : memref<64x64xbf16, 2 : i32> 
    %buf448 = aie.buffer(%tile_3_4) {sym_name = "buf448"} : memref<64x1xbf16, 2 : i32> 
    %buf447 = aie.buffer(%tile_3_4) {sym_name = "buf447"} : memref<64x1xbf16, 2 : i32> 
    %buf446 = aie.buffer(%tile_3_4) {sym_name = "buf446"} : memref<64x1xbf16, 2 : i32> 
    %buf445 = aie.buffer(%tile_3_4) {sym_name = "buf445"} : memref<64x1xbf16, 2 : i32> 
    %buf444 = aie.buffer(%tile_3_4) {sym_name = "buf444"} : memref<64x1xbf16, 2 : i32> 
    %buf443 = aie.buffer(%tile_3_4) {sym_name = "buf443"} : memref<64x1xbf16, 2 : i32> 
    %buf442 = aie.buffer(%tile_2_4) {sym_name = "buf442"} : memref<64x1xbf16, 2 : i32> 
    %buf441 = aie.buffer(%tile_2_4) {sym_name = "buf441"} : memref<64x1xbf16, 2 : i32> 
    %buf440 = aie.buffer(%tile_2_4) {sym_name = "buf440"} : memref<64x64xbf16, 2 : i32> 
    %buf439 = aie.buffer(%tile_2_4) {sym_name = "buf439"} : memref<64x64xbf16, 2 : i32> 
    %buf438 = aie.buffer(%tile_2_4) {sym_name = "buf438"} : memref<64x64xbf16, 2 : i32> 
    %buf437 = aie.buffer(%tile_2_4) {sym_name = "buf437"} : memref<64x64xbf16, 2 : i32> 
    %buf436 = aie.buffer(%tile_2_4) {sym_name = "buf436"} : memref<64x64xbf16, 2 : i32> 
    %buf435 = aie.buffer(%tile_2_4) {sym_name = "buf435"} : memref<64x64xbf16, 2 : i32> 
    %buf434 = aie.buffer(%tile_2_4) {sym_name = "buf434"} : memref<64x1xbf16, 2 : i32> 
    %buf433 = aie.buffer(%tile_2_4) {sym_name = "buf433"} : memref<64x1xbf16, 2 : i32> 
    %buf432 = aie.buffer(%tile_2_4) {sym_name = "buf432"} : memref<64x64xbf16, 2 : i32> 
    %buf431 = aie.buffer(%tile_2_4) {sym_name = "buf431"} : memref<64x1xbf16, 2 : i32> 
    %buf430 = aie.buffer(%tile_2_4) {sym_name = "buf430"} : memref<64x1xbf16, 2 : i32> 
    %buf429 = aie.buffer(%tile_2_4) {sym_name = "buf429"} : memref<64x1xbf16, 2 : i32> 
    %buf428 = aie.buffer(%tile_2_4) {sym_name = "buf428"} : memref<64x1xbf16, 2 : i32> 
    %buf427 = aie.buffer(%tile_2_4) {sym_name = "buf427"} : memref<64x1xbf16, 2 : i32> 
    %buf426 = aie.buffer(%tile_2_4) {sym_name = "buf426"} : memref<64x1xbf16, 2 : i32> 
    %buf425 = aie.buffer(%tile_1_4) {sym_name = "buf425"} : memref<64x1xbf16, 2 : i32> 
    %buf424 = aie.buffer(%tile_1_4) {sym_name = "buf424"} : memref<64x1xbf16, 2 : i32> 
    %buf423 = aie.buffer(%tile_1_4) {sym_name = "buf423"} : memref<64x64xbf16, 2 : i32> 
    %buf422 = aie.buffer(%tile_1_4) {sym_name = "buf422"} : memref<64x64xbf16, 2 : i32> 
    %buf421 = aie.buffer(%tile_1_4) {sym_name = "buf421"} : memref<64x64xbf16, 2 : i32> 
    %buf420 = aie.buffer(%tile_1_4) {sym_name = "buf420"} : memref<64x64xbf16, 2 : i32> 
    %buf419 = aie.buffer(%tile_1_4) {sym_name = "buf419"} : memref<64x64xbf16, 2 : i32> 
    %buf418 = aie.buffer(%tile_1_4) {sym_name = "buf418"} : memref<64x64xbf16, 2 : i32> 
    %buf417 = aie.buffer(%tile_1_4) {sym_name = "buf417"} : memref<64x1xbf16, 2 : i32> 
    %buf416 = aie.buffer(%tile_1_4) {sym_name = "buf416"} : memref<64x1xbf16, 2 : i32> 
    %buf415 = aie.buffer(%tile_1_4) {sym_name = "buf415"} : memref<64x64xbf16, 2 : i32> 
    %buf414 = aie.buffer(%tile_1_4) {sym_name = "buf414"} : memref<64x1xbf16, 2 : i32> 
    %buf413 = aie.buffer(%tile_1_4) {sym_name = "buf413"} : memref<64x1xbf16, 2 : i32> 
    %buf412 = aie.buffer(%tile_1_4) {sym_name = "buf412"} : memref<64x1xbf16, 2 : i32> 
    %buf411 = aie.buffer(%tile_1_4) {sym_name = "buf411"} : memref<64x1xbf16, 2 : i32> 
    %buf410 = aie.buffer(%tile_1_4) {sym_name = "buf410"} : memref<64x1xbf16, 2 : i32> 
    %buf409 = aie.buffer(%tile_1_4) {sym_name = "buf409"} : memref<64x1xbf16, 2 : i32> 
    %buf408 = aie.buffer(%tile_0_4) {sym_name = "buf408"} : memref<64x1xbf16, 2 : i32> 
    %buf407 = aie.buffer(%tile_0_4) {sym_name = "buf407"} : memref<64x1xbf16, 2 : i32> 
    %buf406 = aie.buffer(%tile_0_4) {sym_name = "buf406"} : memref<64x64xbf16, 2 : i32> 
    %buf405 = aie.buffer(%tile_0_4) {sym_name = "buf405"} : memref<64x64xbf16, 2 : i32> 
    %buf404 = aie.buffer(%tile_0_4) {sym_name = "buf404"} : memref<64x64xbf16, 2 : i32> 
    %buf403 = aie.buffer(%tile_0_4) {sym_name = "buf403"} : memref<64x64xbf16, 2 : i32> 
    %buf402 = aie.buffer(%tile_0_4) {sym_name = "buf402"} : memref<64x64xbf16, 2 : i32> 
    %buf401 = aie.buffer(%tile_0_4) {sym_name = "buf401"} : memref<64x64xbf16, 2 : i32> 
    %buf400 = aie.buffer(%tile_0_4) {sym_name = "buf400"} : memref<64x1xbf16, 2 : i32> 
    %buf399 = aie.buffer(%tile_0_4) {sym_name = "buf399"} : memref<64x1xbf16, 2 : i32> 
    %buf398 = aie.buffer(%tile_0_4) {sym_name = "buf398"} : memref<64x64xbf16, 2 : i32> 
    %buf397 = aie.buffer(%tile_0_4) {sym_name = "buf397"} : memref<64x1xbf16, 2 : i32> 
    %buf396 = aie.buffer(%tile_0_4) {sym_name = "buf396"} : memref<64x1xbf16, 2 : i32> 
    %buf395 = aie.buffer(%tile_0_4) {sym_name = "buf395"} : memref<64x1xbf16, 2 : i32> 
    %buf394 = aie.buffer(%tile_0_4) {sym_name = "buf394"} : memref<64x1xbf16, 2 : i32> 
    %buf393 = aie.buffer(%tile_0_4) {sym_name = "buf393"} : memref<64x1xbf16, 2 : i32> 
    %buf392 = aie.buffer(%tile_0_4) {sym_name = "buf392"} : memref<64x1xbf16, 2 : i32> 
    %buf391 = aie.buffer(%tile_3_3) {sym_name = "buf391"} : memref<64x1xbf16, 2 : i32> 
    %buf390 = aie.buffer(%tile_3_3) {sym_name = "buf390"} : memref<64x1xbf16, 2 : i32> 
    %buf389 = aie.buffer(%tile_3_3) {sym_name = "buf389"} : memref<64x64xbf16, 2 : i32> 
    %buf388 = aie.buffer(%tile_3_3) {sym_name = "buf388"} : memref<64x64xbf16, 2 : i32> 
    %buf387 = aie.buffer(%tile_3_3) {sym_name = "buf387"} : memref<64x64xbf16, 2 : i32> 
    %buf386 = aie.buffer(%tile_3_3) {sym_name = "buf386"} : memref<64x64xbf16, 2 : i32> 
    %buf385 = aie.buffer(%tile_3_3) {sym_name = "buf385"} : memref<64x64xbf16, 2 : i32> 
    %buf384 = aie.buffer(%tile_3_3) {sym_name = "buf384"} : memref<64x64xbf16, 2 : i32> 
    %buf383 = aie.buffer(%tile_3_3) {sym_name = "buf383"} : memref<64x1xbf16, 2 : i32> 
    %buf382 = aie.buffer(%tile_3_3) {sym_name = "buf382"} : memref<64x1xbf16, 2 : i32> 
    %buf381 = aie.buffer(%tile_3_3) {sym_name = "buf381"} : memref<64x64xbf16, 2 : i32> 
    %buf380 = aie.buffer(%tile_3_3) {sym_name = "buf380"} : memref<64x1xbf16, 2 : i32> 
    %buf379 = aie.buffer(%tile_3_3) {sym_name = "buf379"} : memref<64x1xbf16, 2 : i32> 
    %buf378 = aie.buffer(%tile_3_3) {sym_name = "buf378"} : memref<64x1xbf16, 2 : i32> 
    %buf377 = aie.buffer(%tile_3_3) {sym_name = "buf377"} : memref<64x1xbf16, 2 : i32> 
    %buf376 = aie.buffer(%tile_3_3) {sym_name = "buf376"} : memref<64x1xbf16, 2 : i32> 
    %buf375 = aie.buffer(%tile_3_3) {sym_name = "buf375"} : memref<64x1xbf16, 2 : i32> 
    %buf374 = aie.buffer(%tile_2_3) {sym_name = "buf374"} : memref<64x1xbf16, 2 : i32> 
    %buf373 = aie.buffer(%tile_2_3) {sym_name = "buf373"} : memref<64x1xbf16, 2 : i32> 
    %buf372 = aie.buffer(%tile_2_3) {sym_name = "buf372"} : memref<64x64xbf16, 2 : i32> 
    %buf371 = aie.buffer(%tile_2_3) {sym_name = "buf371"} : memref<64x64xbf16, 2 : i32> 
    %buf370 = aie.buffer(%tile_2_3) {sym_name = "buf370"} : memref<64x64xbf16, 2 : i32> 
    %buf369 = aie.buffer(%tile_2_3) {sym_name = "buf369"} : memref<64x64xbf16, 2 : i32> 
    %buf368 = aie.buffer(%tile_2_3) {sym_name = "buf368"} : memref<64x64xbf16, 2 : i32> 
    %buf367 = aie.buffer(%tile_2_3) {sym_name = "buf367"} : memref<64x64xbf16, 2 : i32> 
    %buf366 = aie.buffer(%tile_2_3) {sym_name = "buf366"} : memref<64x1xbf16, 2 : i32> 
    %buf365 = aie.buffer(%tile_2_3) {sym_name = "buf365"} : memref<64x1xbf16, 2 : i32> 
    %buf364 = aie.buffer(%tile_2_3) {sym_name = "buf364"} : memref<64x64xbf16, 2 : i32> 
    %buf363 = aie.buffer(%tile_2_3) {sym_name = "buf363"} : memref<64x1xbf16, 2 : i32> 
    %buf362 = aie.buffer(%tile_2_3) {sym_name = "buf362"} : memref<64x1xbf16, 2 : i32> 
    %buf361 = aie.buffer(%tile_2_3) {sym_name = "buf361"} : memref<64x1xbf16, 2 : i32> 
    %buf360 = aie.buffer(%tile_2_3) {sym_name = "buf360"} : memref<64x1xbf16, 2 : i32> 
    %buf359 = aie.buffer(%tile_2_3) {sym_name = "buf359"} : memref<64x1xbf16, 2 : i32> 
    %buf358 = aie.buffer(%tile_2_3) {sym_name = "buf358"} : memref<64x1xbf16, 2 : i32> 
    %buf357 = aie.buffer(%tile_1_3) {sym_name = "buf357"} : memref<64x1xbf16, 2 : i32> 
    %buf356 = aie.buffer(%tile_1_3) {sym_name = "buf356"} : memref<64x1xbf16, 2 : i32> 
    %buf355 = aie.buffer(%tile_1_3) {sym_name = "buf355"} : memref<64x64xbf16, 2 : i32> 
    %buf354 = aie.buffer(%tile_1_3) {sym_name = "buf354"} : memref<64x64xbf16, 2 : i32> 
    %buf353 = aie.buffer(%tile_1_3) {sym_name = "buf353"} : memref<64x64xbf16, 2 : i32> 
    %buf352 = aie.buffer(%tile_1_3) {sym_name = "buf352"} : memref<64x64xbf16, 2 : i32> 
    %buf351 = aie.buffer(%tile_1_3) {sym_name = "buf351"} : memref<64x64xbf16, 2 : i32> 
    %buf350 = aie.buffer(%tile_1_3) {sym_name = "buf350"} : memref<64x64xbf16, 2 : i32> 
    %buf349 = aie.buffer(%tile_1_3) {sym_name = "buf349"} : memref<64x1xbf16, 2 : i32> 
    %buf348 = aie.buffer(%tile_1_3) {sym_name = "buf348"} : memref<64x1xbf16, 2 : i32> 
    %buf347 = aie.buffer(%tile_1_3) {sym_name = "buf347"} : memref<64x64xbf16, 2 : i32> 
    %buf346 = aie.buffer(%tile_1_3) {sym_name = "buf346"} : memref<64x1xbf16, 2 : i32> 
    %buf345 = aie.buffer(%tile_1_3) {sym_name = "buf345"} : memref<64x1xbf16, 2 : i32> 
    %buf344 = aie.buffer(%tile_1_3) {sym_name = "buf344"} : memref<64x1xbf16, 2 : i32> 
    %buf343 = aie.buffer(%tile_1_3) {sym_name = "buf343"} : memref<64x1xbf16, 2 : i32> 
    %buf342 = aie.buffer(%tile_1_3) {sym_name = "buf342"} : memref<64x1xbf16, 2 : i32> 
    %buf341 = aie.buffer(%tile_1_3) {sym_name = "buf341"} : memref<64x1xbf16, 2 : i32> 
    %buf340 = aie.buffer(%tile_0_3) {sym_name = "buf340"} : memref<64x1xbf16, 2 : i32> 
    %buf339 = aie.buffer(%tile_0_3) {sym_name = "buf339"} : memref<64x1xbf16, 2 : i32> 
    %buf338 = aie.buffer(%tile_0_3) {sym_name = "buf338"} : memref<64x64xbf16, 2 : i32> 
    %buf337 = aie.buffer(%tile_0_3) {sym_name = "buf337"} : memref<64x64xbf16, 2 : i32> 
    %buf336 = aie.buffer(%tile_0_3) {sym_name = "buf336"} : memref<64x64xbf16, 2 : i32> 
    %buf335 = aie.buffer(%tile_0_3) {sym_name = "buf335"} : memref<64x64xbf16, 2 : i32> 
    %buf334 = aie.buffer(%tile_0_3) {sym_name = "buf334"} : memref<64x64xbf16, 2 : i32> 
    %buf333 = aie.buffer(%tile_0_3) {sym_name = "buf333"} : memref<64x64xbf16, 2 : i32> 
    %buf332 = aie.buffer(%tile_0_3) {sym_name = "buf332"} : memref<64x1xbf16, 2 : i32> 
    %buf331 = aie.buffer(%tile_0_3) {sym_name = "buf331"} : memref<64x1xbf16, 2 : i32> 
    %buf330 = aie.buffer(%tile_0_3) {sym_name = "buf330"} : memref<64x64xbf16, 2 : i32> 
    %buf329 = aie.buffer(%tile_0_3) {sym_name = "buf329"} : memref<64x1xbf16, 2 : i32> 
    %buf328 = aie.buffer(%tile_0_3) {sym_name = "buf328"} : memref<64x1xbf16, 2 : i32> 
    %buf327 = aie.buffer(%tile_0_3) {sym_name = "buf327"} : memref<64x1xbf16, 2 : i32> 
    %buf326 = aie.buffer(%tile_0_3) {sym_name = "buf326"} : memref<64x1xbf16, 2 : i32> 
    %buf325 = aie.buffer(%tile_0_3) {sym_name = "buf325"} : memref<64x1xbf16, 2 : i32> 
    %buf324 = aie.buffer(%tile_0_3) {sym_name = "buf324"} : memref<64x1xbf16, 2 : i32> 
    %buf323 = aie.buffer(%tile_3_2) {sym_name = "buf323"} : memref<64x1xbf16, 2 : i32> 
    %buf322 = aie.buffer(%tile_3_2) {sym_name = "buf322"} : memref<64x1xbf16, 2 : i32> 
    %buf321 = aie.buffer(%tile_3_2) {sym_name = "buf321"} : memref<64x64xbf16, 2 : i32> 
    %buf320 = aie.buffer(%tile_3_2) {sym_name = "buf320"} : memref<64x64xbf16, 2 : i32> 
    %buf319 = aie.buffer(%tile_3_2) {sym_name = "buf319"} : memref<64x64xbf16, 2 : i32> 
    %buf318 = aie.buffer(%tile_3_2) {sym_name = "buf318"} : memref<64x64xbf16, 2 : i32> 
    %buf317 = aie.buffer(%tile_3_2) {sym_name = "buf317"} : memref<64x64xbf16, 2 : i32> 
    %buf316 = aie.buffer(%tile_3_2) {sym_name = "buf316"} : memref<64x64xbf16, 2 : i32> 
    %buf315 = aie.buffer(%tile_3_2) {sym_name = "buf315"} : memref<64x1xbf16, 2 : i32> 
    %buf314 = aie.buffer(%tile_3_2) {sym_name = "buf314"} : memref<64x1xbf16, 2 : i32> 
    %buf313 = aie.buffer(%tile_3_2) {sym_name = "buf313"} : memref<64x64xbf16, 2 : i32> 
    %buf312 = aie.buffer(%tile_3_2) {sym_name = "buf312"} : memref<64x1xbf16, 2 : i32> 
    %buf311 = aie.buffer(%tile_3_2) {sym_name = "buf311"} : memref<64x1xbf16, 2 : i32> 
    %buf310 = aie.buffer(%tile_3_2) {sym_name = "buf310"} : memref<64x1xbf16, 2 : i32> 
    %buf309 = aie.buffer(%tile_3_2) {sym_name = "buf309"} : memref<64x1xbf16, 2 : i32> 
    %buf308 = aie.buffer(%tile_3_2) {sym_name = "buf308"} : memref<64x1xbf16, 2 : i32> 
    %buf307 = aie.buffer(%tile_3_2) {sym_name = "buf307"} : memref<64x1xbf16, 2 : i32> 
    %buf306 = aie.buffer(%tile_2_2) {sym_name = "buf306"} : memref<64x1xbf16, 2 : i32> 
    %buf305 = aie.buffer(%tile_2_2) {sym_name = "buf305"} : memref<64x1xbf16, 2 : i32> 
    %buf304 = aie.buffer(%tile_2_2) {sym_name = "buf304"} : memref<64x64xbf16, 2 : i32> 
    %buf303 = aie.buffer(%tile_2_2) {sym_name = "buf303"} : memref<64x64xbf16, 2 : i32> 
    %buf302 = aie.buffer(%tile_2_2) {sym_name = "buf302"} : memref<64x64xbf16, 2 : i32> 
    %buf301 = aie.buffer(%tile_2_2) {sym_name = "buf301"} : memref<64x64xbf16, 2 : i32> 
    %buf300 = aie.buffer(%tile_2_2) {sym_name = "buf300"} : memref<64x64xbf16, 2 : i32> 
    %buf299 = aie.buffer(%tile_2_2) {sym_name = "buf299"} : memref<64x64xbf16, 2 : i32> 
    %buf298 = aie.buffer(%tile_2_2) {sym_name = "buf298"} : memref<64x1xbf16, 2 : i32> 
    %buf297 = aie.buffer(%tile_2_2) {sym_name = "buf297"} : memref<64x1xbf16, 2 : i32> 
    %buf296 = aie.buffer(%tile_2_2) {sym_name = "buf296"} : memref<64x64xbf16, 2 : i32> 
    %buf295 = aie.buffer(%tile_2_2) {sym_name = "buf295"} : memref<64x1xbf16, 2 : i32> 
    %buf294 = aie.buffer(%tile_2_2) {sym_name = "buf294"} : memref<64x1xbf16, 2 : i32> 
    %buf293 = aie.buffer(%tile_2_2) {sym_name = "buf293"} : memref<64x1xbf16, 2 : i32> 
    %buf292 = aie.buffer(%tile_2_2) {sym_name = "buf292"} : memref<64x1xbf16, 2 : i32> 
    %buf291 = aie.buffer(%tile_2_2) {sym_name = "buf291"} : memref<64x1xbf16, 2 : i32> 
    %buf290 = aie.buffer(%tile_2_2) {sym_name = "buf290"} : memref<64x1xbf16, 2 : i32> 
    %buf289 = aie.buffer(%tile_1_2) {sym_name = "buf289"} : memref<64x1xbf16, 2 : i32> 
    %buf288 = aie.buffer(%tile_1_2) {sym_name = "buf288"} : memref<64x1xbf16, 2 : i32> 
    %buf287 = aie.buffer(%tile_1_2) {sym_name = "buf287"} : memref<64x64xbf16, 2 : i32> 
    %buf286 = aie.buffer(%tile_1_2) {sym_name = "buf286"} : memref<64x64xbf16, 2 : i32> 
    %buf285 = aie.buffer(%tile_1_2) {sym_name = "buf285"} : memref<64x64xbf16, 2 : i32> 
    %buf284 = aie.buffer(%tile_1_2) {sym_name = "buf284"} : memref<64x64xbf16, 2 : i32> 
    %buf283 = aie.buffer(%tile_1_2) {sym_name = "buf283"} : memref<64x64xbf16, 2 : i32> 
    %buf282 = aie.buffer(%tile_1_2) {sym_name = "buf282"} : memref<64x64xbf16, 2 : i32> 
    %buf281 = aie.buffer(%tile_1_2) {sym_name = "buf281"} : memref<64x1xbf16, 2 : i32> 
    %buf280 = aie.buffer(%tile_1_2) {sym_name = "buf280"} : memref<64x1xbf16, 2 : i32> 
    %buf279 = aie.buffer(%tile_1_2) {sym_name = "buf279"} : memref<64x64xbf16, 2 : i32> 
    %buf278 = aie.buffer(%tile_1_2) {sym_name = "buf278"} : memref<64x1xbf16, 2 : i32> 
    %buf277 = aie.buffer(%tile_1_2) {sym_name = "buf277"} : memref<64x1xbf16, 2 : i32> 
    %buf276 = aie.buffer(%tile_1_2) {sym_name = "buf276"} : memref<64x1xbf16, 2 : i32> 
    %buf275 = aie.buffer(%tile_1_2) {sym_name = "buf275"} : memref<64x1xbf16, 2 : i32> 
    %buf274 = aie.buffer(%tile_1_2) {sym_name = "buf274"} : memref<64x1xbf16, 2 : i32> 
    %buf273 = aie.buffer(%tile_1_2) {sym_name = "buf273"} : memref<64x1xbf16, 2 : i32> 
    %buf272 = aie.buffer(%tile_0_2) {sym_name = "buf272"} : memref<64x1xbf16, 2 : i32> 
    %buf271 = aie.buffer(%tile_0_2) {sym_name = "buf271"} : memref<64x1xbf16, 2 : i32> 
    %buf270 = aie.buffer(%tile_0_2) {sym_name = "buf270"} : memref<64x64xbf16, 2 : i32> 
    %buf269 = aie.buffer(%tile_0_2) {sym_name = "buf269"} : memref<64x64xbf16, 2 : i32> 
    %buf268 = aie.buffer(%tile_0_2) {sym_name = "buf268"} : memref<64x64xbf16, 2 : i32> 
    %buf267 = aie.buffer(%tile_0_2) {sym_name = "buf267"} : memref<64x64xbf16, 2 : i32> 
    %buf266 = aie.buffer(%tile_0_2) {sym_name = "buf266"} : memref<64x64xbf16, 2 : i32> 
    %buf265 = aie.buffer(%tile_0_2) {sym_name = "buf265"} : memref<64x64xbf16, 2 : i32> 
    %buf264 = aie.buffer(%tile_0_2) {sym_name = "buf264"} : memref<64x1xbf16, 2 : i32> 
    %buf263 = aie.buffer(%tile_0_2) {sym_name = "buf263"} : memref<64x1xbf16, 2 : i32> 
    %buf262 = aie.buffer(%tile_0_2) {sym_name = "buf262"} : memref<64x64xbf16, 2 : i32> 
    %buf261 = aie.buffer(%tile_0_2) {sym_name = "buf261"} : memref<64x1xbf16, 2 : i32> 
    %buf260 = aie.buffer(%tile_0_2) {sym_name = "buf260"} : memref<64x1xbf16, 2 : i32> 
    %buf259 = aie.buffer(%tile_0_2) {sym_name = "buf259"} : memref<64x1xbf16, 2 : i32> 
    %buf258 = aie.buffer(%tile_0_2) {sym_name = "buf258"} : memref<64x1xbf16, 2 : i32> 
    %buf257 = aie.buffer(%tile_0_2) {sym_name = "buf257"} : memref<64x1xbf16, 2 : i32> 
    %buf256 = aie.buffer(%tile_0_2) {sym_name = "buf256"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2x256x128xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2x512x128xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2x512x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<2x256x64xbf16>
    scf.for %arg0 = %c0 to %c2 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c2 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c2 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c2 step %c1 {
    } {loop_annotation = #loop_annotation}
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_74, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf496 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_75, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf493 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_73, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf497) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf499) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf498) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf496, %buf494) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf496, %buf495) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_76 step %c1_78 {
        %collapse_shape_81 = memref.collapse_shape %buf492 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf492 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf494, %buf496, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_74, Release, 1)
        aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf492 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf495, %buf496, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_74, Release, 1)
        aie.use_lock(%lock_3_5_73, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf492 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf498, %buf491, %buf490) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf490, %buf497) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf492 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf493, %buf497) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf499, %buf490, %buf491) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf491, %buf499) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf497 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_79 = memref.collapse_shape %buf498 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf499 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_71, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf486 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf483 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_70, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_76 = arith.constant 0 : index
      %c1_77 = arith.constant 1 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf487) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf489) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf488) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf486, %buf484) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf486, %buf485) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      scf.for %arg0 = %c0_76 to %c2_78 step %c1_77 {
        %collapse_shape_81 = memref.collapse_shape %buf482 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf482 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf484, %buf486, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_71, Release, 1)
        aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf482 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf485, %buf486, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_71, Release, 1)
        aie.use_lock(%lock_2_5_70, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf482 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf488, %buf481, %buf480) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf480, %buf487) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf482 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf483, %buf487) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf489, %buf480, %buf481) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf481, %buf489) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf487 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_79 = memref.collapse_shape %buf488 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf489 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_68, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf476 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_69, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf473 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_67, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf477) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf479) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf478) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf476, %buf474) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf476, %buf475) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_76 step %c1_78 {
        %collapse_shape_81 = memref.collapse_shape %buf472 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf472 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf474, %buf476, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_68, Release, 1)
        aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf472 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf475, %buf476, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_68, Release, 1)
        aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf472 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf478, %buf471, %buf470) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf470, %buf477) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf472 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf473, %buf477) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf479, %buf470, %buf471) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf471, %buf479) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf477 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_79 = memref.collapse_shape %buf478 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf479 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_65, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf466 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_66, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf463 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_64, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_77 = arith.constant 1 : index
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf467) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf469) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf468) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf466, %buf464) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf466, %buf465) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_76 step %c1_77 {
        %collapse_shape_81 = memref.collapse_shape %buf462 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf462 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf464, %buf466, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_65, Release, 1)
        aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf462 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf465, %buf466, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_65, Release, 1)
        aie.use_lock(%lock_0_5_64, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf462 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf468, %buf461, %buf460) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf460, %buf467) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf462 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf463, %buf467) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf469, %buf460, %buf461) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf461, %buf469) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf467 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_79 = memref.collapse_shape %buf468 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf469 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf456 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_63, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf453 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_61, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_76 = arith.constant 0 : index
      %c1_77 = arith.constant 1 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf457) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf459) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf458) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf456, %buf454) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf456, %buf455) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      scf.for %arg0 = %c0_76 to %c2_78 step %c1_77 {
        %collapse_shape_84 = memref.collapse_shape %buf452 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf452 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf454, %buf456, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_62, Release, 1)
        aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf452 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf455, %buf456, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_62, Release, 1)
        aie.use_lock(%lock_3_4_61, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf452 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf458, %buf451, %buf450) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf450, %buf457) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf452 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf453, %buf457) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf459, %buf450, %buf451) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf451, %buf459) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf449 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf448 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf447 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf458, %buf446) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf448, %buf458) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf448, %buf458, %buf445) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf446, %buf458, %buf444) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf445, %buf449) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf444, %buf457) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf457, %buf449) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf443) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf447, %buf445, %buf443) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf459, %buf444, %buf443) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf443, %buf447) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf449 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf458 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf447 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_59, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf439 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf436 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_58, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_76 = arith.constant 0 : index
      %c1_77 = arith.constant 1 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf440) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf442) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf441) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf439, %buf437) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf439, %buf438) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      scf.for %arg0 = %c0_76 to %c2_78 step %c1_77 {
        %collapse_shape_84 = memref.collapse_shape %buf435 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf435 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf437, %buf439, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_59, Release, 1)
        aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf435 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf438, %buf439, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_59, Release, 1)
        aie.use_lock(%lock_2_4_58, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf435 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf441, %buf434, %buf433) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf433, %buf440) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf435 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf436, %buf440) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf442, %buf433, %buf434) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf434, %buf442) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf432 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf431 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf430 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf441, %buf429) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf431, %buf441) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf431, %buf441, %buf428) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf429, %buf441, %buf427) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf428, %buf432) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf427, %buf440) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf440, %buf432) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf426) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf430, %buf428, %buf426) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf442, %buf427, %buf426) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf426, %buf430) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf432 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf441 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf430 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_56, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf422 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf419 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_55, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_76 = arith.constant 0 : index
      %c2_77 = arith.constant 2 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf423) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf425) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf424) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf422, %buf420) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf422, %buf421) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      scf.for %arg0 = %c0_76 to %c2_77 step %c1_78 {
        %collapse_shape_84 = memref.collapse_shape %buf418 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf418 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf420, %buf422, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_56, Release, 1)
        aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf418 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf421, %buf422, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_56, Release, 1)
        aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf418 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf424, %buf417, %buf416) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf416, %buf423) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf418 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf419, %buf423) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf425, %buf416, %buf417) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf417, %buf425) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf415 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf414 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf413 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf424, %buf412) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf414, %buf424) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf414, %buf424, %buf411) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf412, %buf424, %buf410) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf411, %buf415) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf410, %buf423) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf423, %buf415) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf409) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf413, %buf411, %buf409) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf425, %buf410, %buf409) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf409, %buf413) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf415 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf424 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf413 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf405 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf402 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_52, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_76 = arith.constant 1 : index
      %c2_77 = arith.constant 2 : index
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf406) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf408) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf407) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf405, %buf403) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf405, %buf404) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_77 step %c1_76 {
        %collapse_shape_84 = memref.collapse_shape %buf401 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf401 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf403, %buf405, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_53, Release, 1)
        aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf401 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf404, %buf405, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_53, Release, 1)
        aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf401 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf407, %buf400, %buf399) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf399, %buf406) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf401 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf402, %buf406) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf408, %buf399, %buf400) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf400, %buf408) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf398 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf397 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf396 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf407, %buf395) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf397, %buf407) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf397, %buf407, %buf394) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf395, %buf407, %buf393) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf394, %buf398) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf393, %buf406) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf406, %buf398) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf392) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf396, %buf394, %buf392) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf408, %buf393, %buf392) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf392, %buf396) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf398 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf407 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf396 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf388 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_51, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf385 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_49, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf389) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf391) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf390) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf388, %buf386) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf388, %buf387) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_76 step %c1_78 {
        %collapse_shape_84 = memref.collapse_shape %buf384 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf384 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf386, %buf388, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_50, Release, 1)
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf384 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf387, %buf388, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_50, Release, 1)
        aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf384 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf390, %buf383, %buf382) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf382, %buf389) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf384 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf385, %buf389) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf391, %buf382, %buf383) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf383, %buf391) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf381 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf380 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf379 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf390, %buf378) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf380, %buf390) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf380, %buf390, %buf377) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf378, %buf390, %buf376) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf377, %buf381) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf376, %buf389) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf389, %buf381) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf375) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf379, %buf377, %buf375) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf391, %buf376, %buf375) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf375, %buf379) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf381 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf390 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf379 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf371 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf368 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_46, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_76 = arith.constant 0 : index
      %c1_77 = arith.constant 1 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf372) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf374) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf373) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf371, %buf369) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf371, %buf370) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      scf.for %arg0 = %c0_76 to %c2_78 step %c1_77 {
        %collapse_shape_84 = memref.collapse_shape %buf367 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf367 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf369, %buf371, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_47, Release, 1)
        aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf367 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf370, %buf371, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_47, Release, 1)
        aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf367 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf373, %buf366, %buf365) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf365, %buf372) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf367 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf368, %buf372) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf374, %buf365, %buf366) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf366, %buf374) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf364 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf363 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf362 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_76] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf373, %buf361) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf363, %buf373) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf363, %buf373, %buf360) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf361, %buf373, %buf359) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf360, %buf364) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf359, %buf372) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf372, %buf364) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf358) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf362, %buf360, %buf358) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf374, %buf359, %buf358) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf358, %buf362) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf364 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf362 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_76 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_76], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_44, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf354 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf351 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_43, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf355) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf357) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf356) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf354, %buf352) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf354, %buf353) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_76 step %c1_78 {
        %collapse_shape_84 = memref.collapse_shape %buf350 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf350 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf352, %buf354, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_44, Release, 1)
        aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf350 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf353, %buf354, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_44, Release, 1)
        aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf350 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf356, %buf349, %buf348) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf348, %buf355) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf350 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf351, %buf355) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf357, %buf348, %buf349) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf349, %buf357) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf347 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf346 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf345 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf356, %buf344) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf346, %buf356) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf346, %buf356, %buf343) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf344, %buf356, %buf342) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf343, %buf347) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf342, %buf355) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf355, %buf347) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf341) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf345, %buf343, %buf341) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf357, %buf342, %buf341) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf341, %buf345) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf347 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf356 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf345 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_77], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf337 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf334 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_40, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_77 = arith.constant 1 : index
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf338) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf340) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf339) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf337, %buf335) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf337, %buf336) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_76 step %c1_77 {
        %collapse_shape_84 = memref.collapse_shape %buf333 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_84) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
        %collapse_shape_85 = memref.collapse_shape %buf333 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf335, %buf337, %collapse_shape_85) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_41, Release, 1)
        aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
        %collapse_shape_86 = memref.collapse_shape %buf333 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf336, %buf337, %collapse_shape_86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_41, Release, 1)
        aie.use_lock(%lock_0_3_40, AcquireGreaterEqual, 1)
        %collapse_shape_87 = memref.collapse_shape %buf333 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_87, %buf339, %buf332, %buf331) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf331, %buf338) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_88 = memref.collapse_shape %buf333 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_88, %buf334, %buf338) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf340, %buf331, %buf332) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf332, %buf340) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf330 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf329 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf328 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf339, %buf327) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf329, %buf339) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf329, %buf339, %buf326) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf327, %buf339, %buf325) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf326, %buf330) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf325, %buf338) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf338, %buf330) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf324) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf328, %buf326, %buf324) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf340, %buf325, %buf324) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf324, %buf328) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf330 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_82 = memref.collapse_shape %buf339 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf328 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_78], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf313 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf320 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf317 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_35, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c1_77 = arith.constant 1 : index
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_2_38, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf321) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf323) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf322) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf320, %buf318) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf320, %buf319) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_76 step %c1_77 {
        %collapse_shape_81 = memref.collapse_shape %buf316 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf316 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf318, %buf320, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_36, Release, 1)
        aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf316 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf319, %buf320, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_36, Release, 1)
        aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf316 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf322, %buf315, %buf314) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf314, %buf321) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf316 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf317, %buf321) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf323, %buf314, %buf315) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf315, %buf323) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf313 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf312 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf311 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf322, %buf310) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf312, %buf322) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf312, %buf322, %buf309) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf310, %buf322, %buf308) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf309, %buf313) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf308, %buf321) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf321, %buf313) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf307) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf311, %buf309, %buf307) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf323, %buf308, %buf307) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf307, %buf311) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf311, %buf313) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_39, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf296 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf303 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf300 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_30, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c1_76 = arith.constant 1 : index
      %c0_77 = arith.constant 0 : index
      %c2_78 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_2_33, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf304) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf306) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf305) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf303, %buf301) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf303, %buf302) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_78 step %c1_76 {
        %collapse_shape_81 = memref.collapse_shape %buf299 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf299 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf301, %buf303, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_31, Release, 1)
        aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf299 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf302, %buf303, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_31, Release, 1)
        aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf299 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf305, %buf298, %buf297) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf297, %buf304) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf299 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf300, %buf304) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf306, %buf297, %buf298) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf298, %buf306) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf296 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf295 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf294 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf305, %buf293) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf295, %buf305) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf295, %buf305, %buf292) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf293, %buf305, %buf291) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf292, %buf296) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf291, %buf304) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf304, %buf296) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf290) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf294, %buf292, %buf290) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf306, %buf291, %buf290) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf290, %buf294) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf294, %buf296) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_34, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf279 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf286 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_27, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf283 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_25, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_77 = arith.constant 0 : index
      %c1_78 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_28, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf287) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf289) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf288) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf286, %buf284) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf286, %buf285) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      scf.for %arg0 = %c0_77 to %c2_76 step %c1_78 {
        %collapse_shape_81 = memref.collapse_shape %buf282 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf282 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf284, %buf286, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_26, Release, 1)
        aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf282 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf285, %buf286, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_26, Release, 1)
        aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf282 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf288, %buf281, %buf280) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf280, %buf287) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf282 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf283, %buf287) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf289, %buf280, %buf281) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf281, %buf289) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf279 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf278 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf277 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_77 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_77] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf288, %buf276) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf278, %buf288) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf278, %buf288, %buf275) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf276, %buf288, %buf274) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf275, %buf279) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf274, %buf287) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf287, %buf279) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf273) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf277, %buf275, %buf273) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf289, %buf274, %buf273) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf273, %buf277) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf277, %buf279) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_29, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf262 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_23, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf269 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf266 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_20, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c2_76 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c1_77 = arith.constant 1 : index
      %c0_78 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_23, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf270) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf272) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf271) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf269, %buf267) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf269, %buf268) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      scf.for %arg0 = %c0_78 to %c2_76 step %c1_77 {
        %collapse_shape_81 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_81) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
        %collapse_shape_82 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf267, %buf269, %collapse_shape_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_21, Release, 1)
        aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
        %collapse_shape_83 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf268, %buf269, %collapse_shape_83) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_21, Release, 1)
        aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
        %collapse_shape_84 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_84, %buf271, %buf264, %buf263) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf263, %buf270) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_85 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_85, %buf266, %buf270) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf272, %buf263, %buf264) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf264, %buf272) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf262 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_79 = memref.collapse_shape %buf261 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf260 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_78 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_78] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf271, %buf259) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf261, %buf271) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf261, %buf271, %buf258) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf259, %buf271, %buf257) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf258, %buf262) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf257, %buf270) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf270, %buf262) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf256) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf260, %buf258, %buf256) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf272, %buf257, %buf256) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf256, %buf260) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf260, %buf262) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_24, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    air.channel @channel_63 [1, 1]
    air.channel @QK2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @QK2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_61 [1, 1]
    air.channel @QK2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @QK2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_59 [1, 1]
    air.channel @QK2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @QK2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_57 [1, 1]
    air.channel @QK2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @QK2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_55 [1, 1]
    air.channel @V2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @V2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_53 [1, 1]
    air.channel @V2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @V2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_51 [1, 1]
    air.channel @V2L1_2_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @V2L1_2_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_49 [1, 1]
    air.channel @V2L1_3_0 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @V2L1_3_1 [1, 1, 1] {broadcast_shape = [1, 1 : index, 4 : index]}
    air.channel @channel_0 [1, 1]
    air.channel @channel_45 [1, 1]
    air.channel @channel_46 [1, 1]
    air.channel @channel_47 [1, 1]
    air.channel @channel_38 [1, 1]
    air.channel @channel_40 [1, 1]
    air.channel @channel_42 [1, 1]
    air.channel @channel_44 [1, 1]
    air.channel @channel_25 [1, 1] {channel_type = "cascade"}
    air.channel @channel_26 [1, 1] {channel_type = "cascade"}
    air.channel @channel_27 [1, 1] {channel_type = "cascade"}
    air.channel @channel_28 [1, 1] {channel_type = "cascade"}
    air.channel @channel_29 [1, 1] {channel_type = "cascade"}
    air.channel @channel_30 [1, 1] {channel_type = "cascade"}
    air.channel @channel_31 [1, 1] {channel_type = "cascade"}
    air.channel @channel_32 [1, 1] {channel_type = "cascade"}
    air.channel @channel_33 [1, 1] {channel_type = "cascade"}
    air.channel @channel_34 [1, 1] {channel_type = "cascade"}
    air.channel @channel_35 [1, 1] {channel_type = "cascade"}
    air.channel @channel_36 [1, 1] {channel_type = "cascade"}
    air.channel @channel_13 [1, 1] {channel_type = "cascade"}
    air.channel @channel_14 [1, 1] {channel_type = "cascade"}
    air.channel @channel_15 [1, 1] {channel_type = "cascade"}
    air.channel @channel_16 [1, 1] {channel_type = "cascade"}
    air.channel @channel_17 [1, 1] {channel_type = "cascade"}
    air.channel @channel_18 [1, 1] {channel_type = "cascade"}
    air.channel @channel_19 [1, 1] {channel_type = "cascade"}
    air.channel @channel_20 [1, 1] {channel_type = "cascade"}
    air.channel @channel_21 [1, 1] {channel_type = "cascade"}
    air.channel @channel_22 [1, 1] {channel_type = "cascade"}
    air.channel @channel_23 [1, 1] {channel_type = "cascade"}
    air.channel @channel_24 [1, 1] {channel_type = "cascade"}
    air.channel @channel_1 [1, 1] {channel_type = "cascade"}
    air.channel @channel_2 [1, 1] {channel_type = "cascade"}
    air.channel @channel_3 [1, 1] {channel_type = "cascade"}
    air.channel @channel_4 [1, 1] {channel_type = "cascade"}
    air.channel @channel_5 [1, 1] {channel_type = "cascade"}
    air.channel @channel_6 [1, 1] {channel_type = "cascade"}
    air.channel @channel_7 [1, 1] {channel_type = "cascade"}
    air.channel @channel_8 [1, 1] {channel_type = "cascade"}
    air.channel @channel_9 [1, 1] {channel_type = "cascade"}
    air.channel @channel_10 [1, 1] {channel_type = "cascade"}
    air.channel @channel_11 [1, 1] {channel_type = "cascade"}
    air.channel @channel_12 [1, 1] {channel_type = "cascade"}
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
      aie.dma_bd(%buf507 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf511 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_16, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf503 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf511 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_17, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf503 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_15, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_0_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf507 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_19, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf506 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf510 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_11, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf502 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf510 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_12, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf502 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_10, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_1_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf506 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_14, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf505 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf509 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_6, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf501 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf509 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_7, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf501 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_2_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf505 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf504 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf508 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf500 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf508 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf500 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_3_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf504 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_4, Release, 1)
      aie.next_bd ^bb12
    }
    aie.shim_dma_allocation @air_channel_0_1_0_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_QKIn_0_1_0_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_1_1_0_0(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_2_1_0_0(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_3_1_0_0(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_0_1_0_0(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_1_1_0_0(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_2_1_0_0(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_3_1_0_0(%shim_noc_tile_3_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 1 : i64, segment_unroll_y = 0 : i64}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 41 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 44 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 47 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 50 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 53 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 56 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 59 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 62 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 65 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 68 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 71 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 74 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 77 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 80 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 83 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 86 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 89 : i64, location = 0 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 92 : i64, location = 1 : i64, row = -1 : i64}, {channel = 3 : i64, col = 2 : i64, id = 95 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 98 : i64, location = 3 : i64, row = -1 : i64}], sym_name = "attn_seg_0_0"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 41 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 44 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 47 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 50 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 53 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 56 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 59 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 62 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 65 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 68 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 71 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 74 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 77 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 80 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 83 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 86 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 89 : i64, location = 0 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 92 : i64, location = 1 : i64, row = -1 : i64}, {channel = 3 : i64, col = 2 : i64, id = 95 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 98 : i64, location = 3 : i64, row = -1 : i64}], sym_name = "attn_seg_1_0"}{
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
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x128xbf16>, memref<2x512x128xbf16>, memref<2x512x64xbf16>, memref<2x256x64xbf16> attributes {id = 1 : i32} {
      %c3 = arith.constant 3 : index
      %c16384 = arith.constant 16384 : index
      %c4096 = arith.constant 4096 : index
      %c8192 = arith.constant 8192 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c128, %c1_0]) {id = 1 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %3 = affine.apply #map1()[%arg5, %arg4]
      %4 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %3] [%c256, %c64] [%c128, %c1_0]) {id = 2 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %5 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c128, %c1_0]) {id = 3 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %6 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %3] [%c256, %c64] [%c128, %c1_0]) {id = 4 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %7 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c128, %c1_0]) {id = 5 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %8 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %3] [%c256, %c64] [%c128, %c1_0]) {id = 6 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %9 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c128, %c1_0]) {id = 7 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %10 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %3] [%c256, %c64] [%c128, %c1_0]) {id = 8 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %11 = affine.apply #map2()[%arg5]
      %12 = air.channel.put async  @QKIn_0[%c0] (%arg9[%c0, %c0, %c0, %11] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 9 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x512x128xbf16>)
      %13 = affine.apply #map3()[%arg5]
      %14 = air.channel.put async  @QKIn_1[%c0] (%arg9[%c0, %c0, %c0, %13] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 10 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x512x128xbf16>)
      %15 = affine.apply #map4()[%arg5]
      %16 = air.channel.put async  @QKIn_2[%c0] (%arg9[%c0, %c0, %c0, %15] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 11 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x512x128xbf16>)
      %17 = affine.apply #map5()[%arg5]
      %18 = air.channel.put async  @QKIn_3[%c0] (%arg9[%c0, %c0, %c0, %17] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 12 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x512x128xbf16>)
      %19 = affine.apply #map6()[%arg5]
      %20 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %19] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 13 : i32, metadataArray = [{base = "air_VIn_0_0_0", index = 0 : i32}, {base = "air_VIn_0_1_0_0", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %21 = affine.apply #map7()[%arg5]
      %22 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %21] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 14 : i32, metadataArray = [{base = "air_VIn_1_0_0", index = 0 : i32}, {base = "air_VIn_1_1_0_0", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %23 = affine.apply #map8()[%arg5]
      %24 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %23] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 15 : i32, metadataArray = [{base = "air_VIn_2_0_0", index = 0 : i32}, {base = "air_VIn_2_1_0_0", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %25 = affine.apply #map9()[%arg5]
      %26 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %25] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 16 : i32, metadataArray = [{base = "air_VIn_3_0_0", index = 0 : i32}, {base = "air_VIn_3_1_0_0", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %27 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %c0, %c0] [%c1_0, %c256, %c64] [%c16384, %c64, %c1_0]) {id = 17 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %28 = air.channel.get async  @channel_0[%c1_0, %c0] (%arg11[%c1_0, %c0, %c0] [%c1_0, %c256, %c64] [%c16384, %c64, %c1_0]) {id = 18 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %29 = air.channel.get async  @channel_0[%c2, %c0] (%arg11[%c2, %c0, %c0] [%c1_0, %c256, %c64] [%c16384, %c64, %c1_0]) {id = 19 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %30 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c3, %c0, %c0] [%c1_0, %c256, %c64] [%c16384, %c64, %c1_0]) {id = 20 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %31 = affine.apply #map10()[%arg5, %arg4]
      %32 = air.channel.put async  @QKIn_0[%c1_0] (%arg8[%c0, %31] [%c256, %c64] [%c128, %c1_0]) {id = 21 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %33 = affine.apply #map11()[%arg5, %arg4]
      %34 = air.channel.put async  @QKIn_0[%c1_0] (%arg8[%c0, %33] [%c256, %c64] [%c128, %c1_0]) {id = 22 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %35 = air.channel.put async  @QKIn_1[%c1_0] (%arg8[%c0, %31] [%c256, %c64] [%c128, %c1_0]) {id = 23 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %36 = air.channel.put async  @QKIn_1[%c1_0] (%arg8[%c0, %33] [%c256, %c64] [%c128, %c1_0]) {id = 24 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %37 = air.channel.put async  @QKIn_2[%c1_0] (%arg8[%c0, %31] [%c256, %c64] [%c128, %c1_0]) {id = 25 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %38 = air.channel.put async  @QKIn_2[%c1_0] (%arg8[%c0, %33] [%c256, %c64] [%c128, %c1_0]) {id = 26 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %39 = air.channel.put async  @QKIn_3[%c1_0] (%arg8[%c0, %31] [%c256, %c64] [%c128, %c1_0]) {id = 27 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %40 = air.channel.put async  @QKIn_3[%c1_0] (%arg8[%c0, %33] [%c256, %c64] [%c128, %c1_0]) {id = 28 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x128xbf16>)
      %41 = affine.apply #map12()[%arg5]
      %42 = air.channel.put async  @QKIn_0[%c1_0] (%arg9[%c0, %c0, %c0, %41] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 29 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x512x128xbf16>)
      %43 = affine.apply #map13()[%arg5]
      %44 = air.channel.put async  @QKIn_1[%c1_0] (%arg9[%c0, %c0, %c0, %43] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 30 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x512x128xbf16>)
      %45 = affine.apply #map14()[%arg5]
      %46 = air.channel.put async  @QKIn_2[%c1_0] (%arg9[%c0, %c0, %c0, %45] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 31 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x512x128xbf16>)
      %47 = affine.apply #map15()[%arg5]
      %48 = air.channel.put async  @QKIn_3[%c1_0] (%arg9[%c0, %c0, %c0, %47] [%c2, %c2, %c64, %c64] [%c8192, %c64, %c128, %c1_0]) {id = 32 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x512x128xbf16>)
      %49 = affine.apply #map16()[%arg5]
      %50 = air.channel.put async  @VIn_0[%c1_0] (%arg10[%c0, %c0, %49] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 33 : i32, metadataArray = [{base = "air_VIn_0_0_0", index = 0 : i32}, {base = "air_VIn_0_1_0_0", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %51 = affine.apply #map17()[%arg5]
      %52 = air.channel.put async  @VIn_1[%c1_0] (%arg10[%c0, %c0, %51] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 34 : i32, metadataArray = [{base = "air_VIn_1_0_0", index = 0 : i32}, {base = "air_VIn_1_1_0_0", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %53 = affine.apply #map18()[%arg5]
      %54 = air.channel.put async  @VIn_2[%c1_0] (%arg10[%c0, %c0, %53] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 35 : i32, metadataArray = [{base = "air_VIn_2_0_0", index = 0 : i32}, {base = "air_VIn_2_1_0_0", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %55 = affine.apply #map19()[%arg5]
      %56 = air.channel.put async  @VIn_3[%c1_0] (%arg10[%c0, %c0, %55] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 36 : i32, metadataArray = [{base = "air_VIn_3_0_0", index = 0 : i32}, {base = "air_VIn_3_1_0_0", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %57 = air.channel.get async  @channel_0[%c0, %c1_0] (%arg11[%c0, %c0, %c0] [%c1_0, %c256, %c64] [%c16384, %c64, %c1_0]) {id = 37 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %58 = air.channel.get async  @channel_0[%c1_0, %c1_0] (%arg11[%c1_0, %c0, %c0] [%c1_0, %c256, %c64] [%c16384, %c64, %c1_0]) {id = 38 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %59 = air.channel.get async  @channel_0[%c2, %c1_0] (%arg11[%c2, %c0, %c0] [%c1_0, %c256, %c64] [%c16384, %c64, %c1_0]) {id = 39 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %60 = air.channel.get async  @channel_0[%c3, %c1_0] (%arg11[%c3, %c0, %c0] [%c1_0, %c256, %c64] [%c16384, %c64, %c1_0]) {id = 40 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %61 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c3_1 = arith.constant 3 : index
        %c64_2 = arith.constant 64 : index
        %c8 = arith.constant 8 : index
        %c1_3 = arith.constant 1 : index
        %c2_4 = arith.constant 2 : index
        %c0_5 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_6, %results_7 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_8, %results_9 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_10, %results_11 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %62 = air.wait_all async 
        %63 = air.wait_all async 
        %64 = air.wait_all async 
        %65 = air.wait_all async 
        %async_token_12, %results_13 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_14, %results_15 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_16, %results_17 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %66 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @QK2L1_0_0[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @QK2L1_0_1[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          scf.yield %93 : !air.async.token
        }
        %67 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %66) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @QK2L1_0_0[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @QK2L1_0_1[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          scf.yield %93 : !air.async.token
        }
        %68 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %67) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %96 = air.channel.put async [%91]  @QK2L1_0_0[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          } else {
            %96 = air.channel.put async [%91]  @QK2L1_0_1[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          }
          %94 = air.channel.get async [%93]  @QKIn_0[%arg12] (%results[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
          %95 = scf.if %92 -> (!air.async.token) {
            %96 = air.channel.put async [%94]  @QK2L1_0_0[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          } else {
            %96 = air.channel.put async [%94]  @QK2L1_0_1[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          }
          scf.yield %95 : !air.async.token
        }
        %69 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %async_token_6) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @QK2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 54 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @QK2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 55 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          scf.yield %93 : !air.async.token
        }
        %70 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %69) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @QK2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 57 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @QK2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 58 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          scf.yield %93 : !air.async.token
        }
        %71 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %70) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %96 = air.channel.put async [%91]  @QK2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 60 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          } else {
            %96 = air.channel.put async [%91]  @QK2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 61 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          }
          %94 = air.channel.get async [%93]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 1 : i32>)
          %95 = scf.if %92 -> (!air.async.token) {
            %96 = air.channel.put async [%94]  @QK2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 63 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          } else {
            %96 = air.channel.put async [%94]  @QK2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 64 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          }
          scf.yield %95 : !air.async.token
        }
        %72 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %async_token_8) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @QK2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 66 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @QK2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 67 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          scf.yield %93 : !air.async.token
        }
        %73 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %72) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @QK2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 69 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @QK2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 70 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          scf.yield %93 : !air.async.token
        }
        %74 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %73) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %96 = air.channel.put async [%91]  @QK2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 72 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          } else {
            %96 = air.channel.put async [%91]  @QK2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 73 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          }
          %94 = air.channel.get async [%93]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 1 : i32>)
          %95 = scf.if %92 -> (!air.async.token) {
            %96 = air.channel.put async [%94]  @QK2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 75 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          } else {
            %96 = air.channel.put async [%94]  @QK2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 76 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          }
          scf.yield %95 : !air.async.token
        }
        %75 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %async_token_10) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @QK2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 78 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @QK2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 79 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          scf.yield %93 : !air.async.token
        }
        %76 = scf.for %arg16 = %c0_5 to %c4 step %c1_3 iter_args(%arg17 = %75) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @QK2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 81 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @QK2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 82 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          scf.yield %93 : !air.async.token
        }
        %77 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %76) -> (!air.async.token) {
          %91 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %96 = air.channel.put async [%91]  @QK2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 84 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          } else {
            %96 = air.channel.put async [%91]  @QK2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 85 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          }
          %94 = air.channel.get async [%93]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 1 : i32>)
          %95 = scf.if %92 -> (!air.async.token) {
            %96 = air.channel.put async [%94]  @QK2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 87 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          } else {
            %96 = air.channel.put async [%94]  @QK2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 88 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %96 : !air.async.token
          }
          scf.yield %95 : !air.async.token
        }
        %78 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %62) -> (!air.async.token) {
          %async_token_28, %results_29 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %91 = air.channel.get async [%async_token_28, %arg17]  @VIn_0[%arg12] (%results_29[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @V2L1_0_0[%c0_5, %c0_5, %c0_5] (%results_29[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 90 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @V2L1_0_1[%c0_5, %c0_5, %c0_5] (%results_29[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 91 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          %async_token_30 = air.execute [%93, %91] {
            memref.dealloc %results_29 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %93 : !air.async.token
        }
        %79 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %63) -> (!air.async.token) {
          %async_token_28, %results_29 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %91 = air.channel.get async [%async_token_28, %arg17]  @VIn_1[%arg12] (%results_29[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @V2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_29[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 93 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @V2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_29[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 94 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          %async_token_30 = air.execute [%93, %91] {
            memref.dealloc %results_29 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %93 : !air.async.token
        }
        %80 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %64) -> (!air.async.token) {
          %async_token_28, %results_29 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %91 = air.channel.get async [%async_token_28, %arg17]  @VIn_2[%arg12] (%results_29[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @V2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_29[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 96 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @V2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_29[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 97 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          %async_token_30 = air.execute [%93, %91] {
            memref.dealloc %results_29 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %93 : !air.async.token
        }
        %81 = scf.for %arg16 = %c0_5 to %c2_4 step %c1_3 iter_args(%arg17 = %65) -> (!air.async.token) {
          %async_token_28, %results_29 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %91 = air.channel.get async [%async_token_28, %arg17]  @VIn_3[%arg12] (%results_29[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 1 : i32>)
          %92 = arith.cmpi eq, %arg12, %c0_5 : index
          %93 = scf.if %92 -> (!air.async.token) {
            %94 = air.channel.put async [%91]  @V2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_29[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 99 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          } else {
            %94 = air.channel.put async [%91]  @V2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_29[%c0_5, %c0_5, %c0_5] [%c8, %c64_2, %c8] [%c8, %c64_2, %c1_3]) {id = 100 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %94 : !air.async.token
          }
          %async_token_30 = air.execute [%93, %91] {
            memref.dealloc %results_29 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %93 : !air.async.token
        }
        %82 = air.channel.get async [%async_token_12]  @Gp2L2[%c0_5, %c0_5] (%results_13[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 1 : i32>)
        %83 = air.channel.get async [%async_token_14]  @Gp2L2[%c1_3, %c0_5] (%results_15[] [] []) {id = 102 : i32} : (memref<64x64xbf16, 1 : i32>)
        %84 = air.channel.get async [%async_token_16]  @Gp2L2[%c2_4, %c0_5] (%results_17[] [] []) {id = 103 : i32} : (memref<64x64xbf16, 1 : i32>)
        %85 = air.channel.get async [%async_token_18]  @Gp2L2[%c3_1, %c0_5] (%results_19[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 1 : i32>)
        %86 = air.channel.put async [%82]  @channel_0[%c0_5, %arg12] (%results_13[] [] []) {id = 105 : i32} : (memref<64x64xbf16, 1 : i32>)
        %87 = air.channel.put async [%83]  @channel_0[%c1_3, %arg12] (%results_15[] [] []) {id = 106 : i32} : (memref<64x64xbf16, 1 : i32>)
        %88 = air.channel.put async [%84]  @channel_0[%c2_4, %arg12] (%results_17[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 1 : i32>)
        %89 = air.channel.put async [%85]  @channel_0[%c3_1, %arg12] (%results_19[] [] []) {id = 108 : i32} : (memref<64x64xbf16, 1 : i32>)
        %90 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c64_28 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2_29 = arith.constant 2 : index
          %c0_30 = arith.constant 0 : index
          %c1_31 = arith.constant 1 : index
          %c8_32 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %async_token_33, %results_34 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_35, %results_36 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_37, %results_38 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_39, %results_40 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_41, %results_42 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_43, %results_44 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_45 = air.execute [%async_token_37] {
            func.call @zero_fill_gp_bf16(%results_38) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_46 = air.execute [%async_token_33] {
            func.call @zero_fill_sp_bf16(%results_34) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_47 = air.execute [%async_token_35] {
            func.call @neg_inf_fill_up_bf16(%results_36) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %91 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 109 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %92 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %91]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 111 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %91]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 112 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %93 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %92]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %92]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 114 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %94 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %93]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 115 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %93]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 116 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %95 = arith.index_cast %arg16 : index to i32
          %96 = arith.cmpi eq, %95, %c0_i32 : i32
          scf.if %96 {
            %async_token_54 = air.execute [%async_token_39, %async_token_43, %94] {
              func.call @copy_tile(%results_40, %results_44) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 117 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 118 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %97]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 119 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %97]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 120 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %98]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 121 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %98]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 122 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %100 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %99]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 123 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %99]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 124 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %101 = arith.cmpi eq, %95, %c1_i32 : i32
          scf.if %101 {
            %async_token_54 = air.execute [%async_token_39, %async_token_43, %100] {
              func.call @copy_tile(%results_40, %results_44) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %102 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 125 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 126 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %103 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %102]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 127 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %102]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 128 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %104 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %103]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 129 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %103]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 130 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %105 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %104]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 131 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %104]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 132 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %106 = arith.cmpi eq, %95, %c2_i32 : i32
          scf.if %106 {
            %async_token_54 = air.execute [%async_token_39, %async_token_43, %105] {
              func.call @copy_tile(%results_40, %results_44) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %107 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 133 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 134 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %108 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %107]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 135 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %107]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 136 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %109 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %108]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 137 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %108]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 138 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %110 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %109]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 139 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %109]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 140 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %111 = arith.cmpi eq, %95, %c3_i32 : i32
          scf.if %111 {
            %async_token_54 = air.execute [%async_token_39, %async_token_43, %110] {
              func.call @copy_tile(%results_40, %results_44) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %112 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 141 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 142 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %113 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %112]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 143 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %112]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 144 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %114 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %113]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 145 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %113]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 146 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %115 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %114]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 147 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %114]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 148 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          scf.if %96 {
            %async_token_54 = air.execute [%async_token_39, %async_token_41, %115] {
              func.call @copy_tile(%results_40, %results_42) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %116 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 149 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 150 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %117 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %116]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 151 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %116]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 152 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %118 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %117]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 153 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %117]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 154 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %119 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %118]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 155 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %118]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 156 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          scf.if %101 {
            %async_token_54 = air.execute [%async_token_39, %async_token_41, %119] {
              func.call @copy_tile(%results_40, %results_42) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %120 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 157 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 158 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %121 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %120]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 159 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %120]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 160 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %122 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %121]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 161 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %121]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 162 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %123 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %122]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 163 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %122]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 164 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          scf.if %106 {
            %async_token_54 = air.execute [%async_token_39, %async_token_41, %123] {
              func.call @copy_tile(%results_40, %results_42) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %124 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 165 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 166 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %125 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %124]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 167 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %124]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 168 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %126 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %125]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 169 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %125]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 170 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          %127 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.cmpi eq, %arg20, %c0_30 : index
            %132 = scf.if %131 -> (!air.async.token) {
              %133 = air.channel.get async [%async_token_39, %126]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 171 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            } else {
              %133 = air.channel.get async [%async_token_39, %126]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 172 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %133 : !air.async.token
            }
            affine.yield %132 : !air.async.token
          } else {
            %131 = air.wait_all async 
            affine.yield %131 : !air.async.token
          }
          scf.if %111 {
            %async_token_54 = air.execute [%async_token_39, %async_token_41, %127] {
              func.call @copy_tile(%results_40, %results_42) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %128 = air.wait_all async [%async_token_39, %async_token_41, %async_token_43, %async_token_45, %async_token_46, %async_token_47] 
          %129 = scf.for %arg21 = %c0_30 to %c2_29 step %c1_31 iter_args(%arg22 = %128) -> (!air.async.token) {
            %async_token_54, %results_55 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_56, %results_57 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_58 = air.execute [%async_token_56, %arg22] {
              %collapse_shape = memref.collapse_shape %results_57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %131 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%arg22]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 173 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%arg22]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 174 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %132 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%arg22, %131]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 175 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%arg22, %131]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 176 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %133 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%arg22, %132]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 177 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%arg22, %132]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 178 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %134 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%arg22, %133]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 179 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%arg22, %133]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 180 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %async_token_59 = air.execute [%async_token_58, %134] {
              %collapse_shape = memref.collapse_shape %results_57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_44, %results_40, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %135 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%arg22, %async_token_59]  @QK2L1_0_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 181 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%arg22, %async_token_59]  @QK2L1_0_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 182 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %136 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%arg22, %135]  @QK2L1_1_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 183 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%arg22, %135]  @QK2L1_1_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 184 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %137 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%arg22, %136]  @QK2L1_2_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 185 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%arg22, %136]  @QK2L1_2_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 186 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %138 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%arg22, %137]  @QK2L1_3_0[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 187 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%arg22, %137]  @QK2L1_3_1[%c0_30, %arg17, %arg16] (%results_40[] [] []) {id = 188 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %async_token_60 = air.execute [%138, %arg22, %async_token_56] {
              %collapse_shape = memref.collapse_shape %results_57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_42, %results_40, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %139 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%async_token_54]  @V2L1_0_0[%c0_30, %arg17, %arg16] (%results_55[] [] []) {id = 189 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%async_token_54]  @V2L1_0_1[%c0_30, %arg17, %arg16] (%results_55[] [] []) {id = 190 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %140 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%async_token_54, %arg22, %139]  @V2L1_1_0[%c0_30, %arg17, %arg16] (%results_55[] [] []) {id = 191 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%async_token_54, %arg22, %139]  @V2L1_1_1[%c0_30, %arg17, %arg16] (%results_55[] [] []) {id = 192 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %141 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%async_token_54, %arg22, %140]  @V2L1_2_0[%c0_30, %arg17, %arg16] (%results_55[] [] []) {id = 193 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%async_token_54, %arg22, %140]  @V2L1_2_1[%c0_30, %arg17, %arg16] (%results_55[] [] []) {id = 194 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %142 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %144 = arith.cmpi eq, %arg20, %c0_30 : index
              %145 = scf.if %144 -> (!air.async.token) {
                %146 = air.channel.get async [%async_token_54, %arg22, %141]  @V2L1_3_0[%c0_30, %arg17, %arg16] (%results_55[] [] []) {id = 195 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              } else {
                %146 = air.channel.get async [%async_token_54, %arg22, %141]  @V2L1_3_1[%c0_30, %arg17, %arg16] (%results_55[] [] []) {id = 196 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %146 : !air.async.token
              }
              affine.yield %145 : !air.async.token
            } else {
              %144 = air.wait_all async 
              affine.yield %144 : !air.async.token
            }
            %async_token_61, %results_62 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_65 = air.execute [%async_token_60, %async_token_61, %async_token_63] {
              %collapse_shape = memref.collapse_shape %results_57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_36, %results_62, %results_64) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_66 = air.execute [%async_token_65] {
              func.call @mul_r_gp(%results_64, %results_38) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_67 = air.execute [%142, %async_token_66, %async_token_54, %async_token_56] {
              %collapse_shape = memref.collapse_shape %results_57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_55, %results_38) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_68 = air.execute [%async_token_66] {
              func.call @accum_sp_r_s(%results_34, %results_64, %results_62) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_69 = air.execute [%async_token_68] {
              func.call @vector_copy_32elems(%c0_i32, %results_62, %results_34) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_70 = air.execute [%async_token_69] {
              memref.dealloc %results_62 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_71 = air.execute [%async_token_68] {
              memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
            }
            %143 = air.wait_all async [%131, %132, %133, %async_token_59, %135, %136, %137, %139, %140, %141, %async_token_67, %async_token_69] 
            %async_token_72 = air.execute [%async_token_59, %async_token_65, %async_token_67] {
              memref.dealloc %results_57 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_73 = air.execute [%139, %140, %141, %async_token_67] {
              memref.dealloc %results_55 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %143 : !air.async.token
          }
          %130 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %131 = arith.subi %arg17, %c1_31 : index
            %132 = air.channel.put async [%async_token_37, %129]  @cascade_gp[%arg16, %131] (%results_38[] [] []) {id = 197 : i32} : (memref<64x64xbf16, 2 : i32>)
            %133 = air.channel.put async [%async_token_35, %129]  @cascade_up[%arg16, %131] (%results_36[] [] []) {id = 198 : i32} : (memref<64x1xbf16, 2 : i32>)
            %134 = air.channel.put async [%async_token_33, %129]  @cascade_sp[%arg16, %131] (%results_34[] [] []) {id = 199 : i32} : (memref<64x1xbf16, 2 : i32>)
            %135 = air.wait_all async [%132, %133, %134] 
            affine.yield %135 : !air.async.token
          } else {
            %131 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_54, %results_55 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_58, %results_59 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %132 = air.channel.get async [%async_token_54]  @cascade_gp[%arg16, %arg17] (%results_55[] [] []) {id = 200 : i32} : (memref<64x64xbf16, 2 : i32>)
              %133 = air.channel.get async [%async_token_56]  @cascade_up[%arg16, %arg17] (%results_57[] [] []) {id = 201 : i32} : (memref<64x1xbf16, 2 : i32>)
              %134 = air.channel.get async [%async_token_58]  @cascade_sp[%arg16, %arg17] (%results_59[] [] []) {id = 202 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_62 = air.execute [%async_token_35, %async_token_60, %129] {
                func.call @vector_copy_32elems(%c0_i32, %results_36, %results_61) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_63 = air.execute [%133, %async_token_62] {
                func.call @maximum_up_u_bf16(%results_57, %results_36) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_66 = air.execute [%async_token_63, %async_token_64] {
                func.call @exp_up_minus_u(%results_57, %results_36, %results_65) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_66, %async_token_67] {
                func.call @exp_up_minus_u(%results_61, %results_36, %results_68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_66, %132] {
                func.call @mul_r_gp(%results_65, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_71 = air.execute [%async_token_37, %async_token_69] {
                func.call @mul_r_gp(%results_68, %results_38) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_72 = air.execute [%async_token_70, %async_token_71] {
                func.call @add_gp_g(%results_38, %results_55) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_73] {
                func.call @zero_fill_sp_bf16(%results_74) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_75, %async_token_70, %134] {
                func.call @accum_sp_r_s(%results_59, %results_65, %results_74) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_33, %async_token_76, %async_token_71] {
                func.call @accum_sp_r_s(%results_34, %results_68, %results_74) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_78 = air.execute [%async_token_77] {
                func.call @vector_copy_32elems(%c0_i32, %results_74, %results_59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %135 = arith.subi %arg17, %c1_31 : index
              %136 = air.channel.put async [%async_token_72]  @cascade_gp[%arg16, %135] (%results_55[] [] []) {id = 203 : i32} : (memref<64x64xbf16, 2 : i32>)
              %137 = air.channel.put async [%async_token_35, %async_token_69]  @cascade_up[%arg16, %135] (%results_36[] [] []) {id = 204 : i32} : (memref<64x1xbf16, 2 : i32>)
              %138 = air.channel.put async [%async_token_78]  @cascade_sp[%arg16, %135] (%results_59[] [] []) {id = 205 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_79 = air.execute [%136] {
                memref.dealloc %results_55 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_80 = air.execute [%async_token_66] {
                memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_81 = air.execute [%138] {
                memref.dealloc %results_59 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_82 = air.execute [%async_token_69] {
                memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_76] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_84 = air.execute [%async_token_77] {
                memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_85 = air.execute [%async_token_78] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              }
              %139 = air.wait_all async [%136, %137, %138] 
              affine.yield %139 : !air.async.token
            } else {
              %async_token_54, %results_55 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_58, %results_59 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %132 = air.channel.get async [%async_token_54]  @cascade_gp[%arg16, %arg17] (%results_55[] [] []) {id = 206 : i32} : (memref<64x64xbf16, 2 : i32>)
              %133 = air.channel.get async [%async_token_56]  @cascade_up[%arg16, %arg17] (%results_57[] [] []) {id = 207 : i32} : (memref<64x1xbf16, 2 : i32>)
              %134 = air.channel.get async [%async_token_58]  @cascade_sp[%arg16, %arg17] (%results_59[] [] []) {id = 208 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_62 = air.execute [%async_token_35, %async_token_60, %129] {
                func.call @vector_copy_32elems(%c0_i32, %results_36, %results_61) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_63 = air.execute [%133, %async_token_62] {
                func.call @maximum_up_u_bf16(%results_57, %results_36) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_66 = air.execute [%async_token_63, %async_token_64] {
                func.call @exp_up_minus_u(%results_57, %results_36, %results_65) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_66, %async_token_67] {
                func.call @exp_up_minus_u(%results_61, %results_36, %results_68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_66, %132] {
                func.call @mul_r_gp(%results_65, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_71 = air.execute [%async_token_37, %async_token_69] {
                func.call @mul_r_gp(%results_68, %results_38) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_72 = air.execute [%async_token_70, %async_token_71] {
                func.call @add_gp_g(%results_38, %results_55) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_73, %results_74 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_73] {
                func.call @zero_fill_sp_bf16(%results_74) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_75, %async_token_70, %134] {
                func.call @accum_sp_r_s(%results_59, %results_65, %results_74) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_33, %async_token_76, %async_token_71] {
                func.call @accum_sp_r_s(%results_34, %results_68, %results_74) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_78 = air.execute [%async_token_77] {
                func.call @vector_copy_32elems(%c0_i32, %results_74, %results_59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_79 = air.execute [%async_token_78, %async_token_72] {
                func.call @div_gp_sp(%results_59, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %135 = air.channel.put async [%async_token_79]  @Gp2L2[%arg16, %c0_30] (%results_55[%c0_30, %c0_30, %c0_30] [%c64_28, %c8_32, %c8_32] [%c8_32, %c512, %c1_31]) {id = 209 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_80 = air.execute [%135] {
                memref.dealloc %results_55 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_81 = air.execute [%async_token_66] {
                memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_82 = air.execute [%async_token_79] {
                memref.dealloc %results_59 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_69] {
                memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_84 = air.execute [%async_token_76] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_85 = air.execute [%async_token_77] {
                memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_86 = air.execute [%async_token_78] {
                memref.dealloc %results_74 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %135 : !air.async.token
            }
            affine.yield %129 : !air.async.token
          }
          %async_token_48 = air.execute [%129] {
            memref.dealloc %results_44 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_49 = air.execute [%129] {
            memref.dealloc %results_42 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_50 = air.execute [%129, %127, %126, %125, %124, %123, %122, %121, %120, %119, %118, %117, %116, %115, %114, %113, %112, %110, %109, %108, %107, %105, %104, %103, %102, %100, %99, %98, %97, %94, %93, %92, %91] {
            memref.dealloc %results_40 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_51 = air.execute [%130, %129, %async_token_45] {
            memref.dealloc %results_38 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_52 = air.execute [%130, %129, %async_token_47] {
            memref.dealloc %results_36 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_53 = air.execute [%130, %129, %async_token_46] {
            memref.dealloc %results_34 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_20 = air.execute [%68] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_21 = air.execute [%71] {
          memref.dealloc %results_7 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_22 = air.execute [%74] {
          memref.dealloc %results_9 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_23 = air.execute [%77] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_24 = air.execute [%89] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_25 = air.execute [%88] {
          memref.dealloc %results_17 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_26 = air.execute [%87] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_27 = air.execute [%86] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        air.wait_all [%78, %79, %80, %81, %90, %async_token_20, %async_token_21, %async_token_22, %async_token_23, %async_token_24, %async_token_25, %async_token_26, %async_token_27]  {air.segment_end}
      }
    }
    return
  }
}
