#map = affine_map<()[s0, s1] -> (s0 * 32768 + s1 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 32768)>
#map2 = affine_map<()[s0] -> (s0 * 32768 + 4096)>
#map3 = affine_map<()[s0] -> (s0 * 32768 + 8192)>
#map4 = affine_map<()[s0] -> (s0 * 32768 + 12288)>
#map5 = affine_map<()[s0, s1] -> (s0 * 32768 + s1 * 16384 + 16384)>
#map6 = affine_map<()[s0] -> (s0 * 32768 + 16384)>
#map7 = affine_map<()[s0] -> (s0 * 32768 + 20480)>
#map8 = affine_map<()[s0] -> (s0 * 32768 + 24576)>
#map9 = affine_map<()[s0] -> (s0 * 32768 + 28672)>
#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set4 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
  aie.device(npu2_4col) @attn_seg_0_0 {
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
    %buf239 = aie.buffer(%mem_tile_0_1) {sym_name = "buf239"} : memref<64x64xbf16, 1 : i32> 
    %buf238 = aie.buffer(%mem_tile_1_1) {sym_name = "buf238"} : memref<64x64xbf16, 1 : i32> 
    %buf237 = aie.buffer(%mem_tile_2_1) {sym_name = "buf237"} : memref<64x64xbf16, 1 : i32> 
    %buf236 = aie.buffer(%mem_tile_3_1) {sym_name = "buf236"} : memref<64x64xbf16, 1 : i32> 
    %buf235 = aie.buffer(%mem_tile_0_1) {sym_name = "buf235"} : memref<64x64xbf16, 1 : i32> 
    %buf234 = aie.buffer(%mem_tile_1_1) {sym_name = "buf234"} : memref<64x64xbf16, 1 : i32> 
    %buf233 = aie.buffer(%mem_tile_2_1) {sym_name = "buf233"} : memref<64x64xbf16, 1 : i32> 
    %buf232 = aie.buffer(%mem_tile_3_1) {sym_name = "buf232"} : memref<64x64xbf16, 1 : i32> 
    %buf231 = aie.buffer(%mem_tile_0_1) {sym_name = "buf231"} : memref<64x64xbf16, 1 : i32> 
    %buf230 = aie.buffer(%mem_tile_1_1) {sym_name = "buf230"} : memref<64x64xbf16, 1 : i32> 
    %buf229 = aie.buffer(%mem_tile_2_1) {sym_name = "buf229"} : memref<64x64xbf16, 1 : i32> 
    %buf228 = aie.buffer(%mem_tile_3_1) {sym_name = "buf228"} : memref<64x64xbf16, 1 : i32> 
    %buf227 = aie.buffer(%tile_3_5) {sym_name = "buf227"} : memref<64x1xbf16, 2 : i32> 
    %buf226 = aie.buffer(%tile_3_5) {sym_name = "buf226"} : memref<64x1xbf16, 2 : i32> 
    %buf225 = aie.buffer(%tile_3_5) {sym_name = "buf225"} : memref<64x64xbf16, 2 : i32> 
    %buf224 = aie.buffer(%tile_3_5) {sym_name = "buf224"} : memref<64x64xbf16, 2 : i32> 
    %buf223 = aie.buffer(%tile_3_5) {sym_name = "buf223"} : memref<64x64xbf16, 2 : i32> 
    %buf222 = aie.buffer(%tile_3_5) {sym_name = "buf222"} : memref<64x64xbf16, 2 : i32> 
    %buf221 = aie.buffer(%tile_3_5) {sym_name = "buf221"} : memref<64x64xbf16, 2 : i32> 
    %buf220 = aie.buffer(%tile_3_5) {sym_name = "buf220"} : memref<64x1xbf16, 2 : i32> 
    %buf219 = aie.buffer(%tile_3_5) {sym_name = "buf219"} : memref<64x1xbf16, 2 : i32> 
    %buf218 = aie.buffer(%tile_2_5) {sym_name = "buf218"} : memref<64x1xbf16, 2 : i32> 
    %buf217 = aie.buffer(%tile_2_5) {sym_name = "buf217"} : memref<64x1xbf16, 2 : i32> 
    %buf216 = aie.buffer(%tile_2_5) {sym_name = "buf216"} : memref<64x64xbf16, 2 : i32> 
    %buf215 = aie.buffer(%tile_2_5) {sym_name = "buf215"} : memref<64x64xbf16, 2 : i32> 
    %buf214 = aie.buffer(%tile_2_5) {sym_name = "buf214"} : memref<64x64xbf16, 2 : i32> 
    %buf213 = aie.buffer(%tile_2_5) {sym_name = "buf213"} : memref<64x64xbf16, 2 : i32> 
    %buf212 = aie.buffer(%tile_2_5) {sym_name = "buf212"} : memref<64x64xbf16, 2 : i32> 
    %buf211 = aie.buffer(%tile_2_5) {sym_name = "buf211"} : memref<64x1xbf16, 2 : i32> 
    %buf210 = aie.buffer(%tile_2_5) {sym_name = "buf210"} : memref<64x1xbf16, 2 : i32> 
    %buf209 = aie.buffer(%tile_1_5) {sym_name = "buf209"} : memref<64x1xbf16, 2 : i32> 
    %buf208 = aie.buffer(%tile_1_5) {sym_name = "buf208"} : memref<64x1xbf16, 2 : i32> 
    %buf207 = aie.buffer(%tile_1_5) {sym_name = "buf207"} : memref<64x64xbf16, 2 : i32> 
    %buf206 = aie.buffer(%tile_1_5) {sym_name = "buf206"} : memref<64x64xbf16, 2 : i32> 
    %buf205 = aie.buffer(%tile_1_5) {sym_name = "buf205"} : memref<64x64xbf16, 2 : i32> 
    %buf204 = aie.buffer(%tile_1_5) {sym_name = "buf204"} : memref<64x64xbf16, 2 : i32> 
    %buf203 = aie.buffer(%tile_1_5) {sym_name = "buf203"} : memref<64x64xbf16, 2 : i32> 
    %buf202 = aie.buffer(%tile_1_5) {sym_name = "buf202"} : memref<64x1xbf16, 2 : i32> 
    %buf201 = aie.buffer(%tile_1_5) {sym_name = "buf201"} : memref<64x1xbf16, 2 : i32> 
    %buf200 = aie.buffer(%tile_0_5) {sym_name = "buf200"} : memref<64x1xbf16, 2 : i32> 
    %buf199 = aie.buffer(%tile_0_5) {sym_name = "buf199"} : memref<64x1xbf16, 2 : i32> 
    %buf198 = aie.buffer(%tile_0_5) {sym_name = "buf198"} : memref<64x64xbf16, 2 : i32> 
    %buf197 = aie.buffer(%tile_0_5) {sym_name = "buf197"} : memref<64x64xbf16, 2 : i32> 
    %buf196 = aie.buffer(%tile_0_5) {sym_name = "buf196"} : memref<64x64xbf16, 2 : i32> 
    %buf195 = aie.buffer(%tile_0_5) {sym_name = "buf195"} : memref<64x64xbf16, 2 : i32> 
    %buf194 = aie.buffer(%tile_0_5) {sym_name = "buf194"} : memref<64x64xbf16, 2 : i32> 
    %buf193 = aie.buffer(%tile_0_5) {sym_name = "buf193"} : memref<64x1xbf16, 2 : i32> 
    %buf192 = aie.buffer(%tile_0_5) {sym_name = "buf192"} : memref<64x1xbf16, 2 : i32> 
    %buf191 = aie.buffer(%tile_3_4) {sym_name = "buf191"} : memref<64x1xbf16, 2 : i32> 
    %buf190 = aie.buffer(%tile_3_4) {sym_name = "buf190"} : memref<64x1xbf16, 2 : i32> 
    %buf189 = aie.buffer(%tile_3_4) {sym_name = "buf189"} : memref<64x64xbf16, 2 : i32> 
    %buf188 = aie.buffer(%tile_3_4) {sym_name = "buf188"} : memref<64x64xbf16, 2 : i32> 
    %buf187 = aie.buffer(%tile_3_4) {sym_name = "buf187"} : memref<64x64xbf16, 2 : i32> 
    %buf186 = aie.buffer(%tile_3_4) {sym_name = "buf186"} : memref<64x64xbf16, 2 : i32> 
    %buf185 = aie.buffer(%tile_3_4) {sym_name = "buf185"} : memref<64x64xbf16, 2 : i32> 
    %buf184 = aie.buffer(%tile_3_4) {sym_name = "buf184"} : memref<64x1xbf16, 2 : i32> 
    %buf183 = aie.buffer(%tile_3_4) {sym_name = "buf183"} : memref<64x1xbf16, 2 : i32> 
    %buf182 = aie.buffer(%tile_3_4) {sym_name = "buf182"} : memref<64x64xbf16, 2 : i32> 
    %buf181 = aie.buffer(%tile_3_4) {sym_name = "buf181"} : memref<64x1xbf16, 2 : i32> 
    %buf180 = aie.buffer(%tile_3_4) {sym_name = "buf180"} : memref<64x1xbf16, 2 : i32> 
    %buf179 = aie.buffer(%tile_3_4) {sym_name = "buf179"} : memref<64x1xbf16, 2 : i32> 
    %buf178 = aie.buffer(%tile_3_4) {sym_name = "buf178"} : memref<64x1xbf16, 2 : i32> 
    %buf177 = aie.buffer(%tile_3_4) {sym_name = "buf177"} : memref<64x1xbf16, 2 : i32> 
    %buf176 = aie.buffer(%tile_3_4) {sym_name = "buf176"} : memref<64x1xbf16, 2 : i32> 
    %buf175 = aie.buffer(%tile_2_4) {sym_name = "buf175"} : memref<64x1xbf16, 2 : i32> 
    %buf174 = aie.buffer(%tile_2_4) {sym_name = "buf174"} : memref<64x1xbf16, 2 : i32> 
    %buf173 = aie.buffer(%tile_2_4) {sym_name = "buf173"} : memref<64x64xbf16, 2 : i32> 
    %buf172 = aie.buffer(%tile_2_4) {sym_name = "buf172"} : memref<64x64xbf16, 2 : i32> 
    %buf171 = aie.buffer(%tile_2_4) {sym_name = "buf171"} : memref<64x64xbf16, 2 : i32> 
    %buf170 = aie.buffer(%tile_2_4) {sym_name = "buf170"} : memref<64x64xbf16, 2 : i32> 
    %buf169 = aie.buffer(%tile_2_4) {sym_name = "buf169"} : memref<64x64xbf16, 2 : i32> 
    %buf168 = aie.buffer(%tile_2_4) {sym_name = "buf168"} : memref<64x1xbf16, 2 : i32> 
    %buf167 = aie.buffer(%tile_2_4) {sym_name = "buf167"} : memref<64x1xbf16, 2 : i32> 
    %buf166 = aie.buffer(%tile_2_4) {sym_name = "buf166"} : memref<64x64xbf16, 2 : i32> 
    %buf165 = aie.buffer(%tile_2_4) {sym_name = "buf165"} : memref<64x1xbf16, 2 : i32> 
    %buf164 = aie.buffer(%tile_2_4) {sym_name = "buf164"} : memref<64x1xbf16, 2 : i32> 
    %buf163 = aie.buffer(%tile_2_4) {sym_name = "buf163"} : memref<64x1xbf16, 2 : i32> 
    %buf162 = aie.buffer(%tile_2_4) {sym_name = "buf162"} : memref<64x1xbf16, 2 : i32> 
    %buf161 = aie.buffer(%tile_2_4) {sym_name = "buf161"} : memref<64x1xbf16, 2 : i32> 
    %buf160 = aie.buffer(%tile_2_4) {sym_name = "buf160"} : memref<64x1xbf16, 2 : i32> 
    %buf159 = aie.buffer(%tile_1_4) {sym_name = "buf159"} : memref<64x1xbf16, 2 : i32> 
    %buf158 = aie.buffer(%tile_1_4) {sym_name = "buf158"} : memref<64x1xbf16, 2 : i32> 
    %buf157 = aie.buffer(%tile_1_4) {sym_name = "buf157"} : memref<64x64xbf16, 2 : i32> 
    %buf156 = aie.buffer(%tile_1_4) {sym_name = "buf156"} : memref<64x64xbf16, 2 : i32> 
    %buf155 = aie.buffer(%tile_1_4) {sym_name = "buf155"} : memref<64x64xbf16, 2 : i32> 
    %buf154 = aie.buffer(%tile_1_4) {sym_name = "buf154"} : memref<64x64xbf16, 2 : i32> 
    %buf153 = aie.buffer(%tile_1_4) {sym_name = "buf153"} : memref<64x64xbf16, 2 : i32> 
    %buf152 = aie.buffer(%tile_1_4) {sym_name = "buf152"} : memref<64x1xbf16, 2 : i32> 
    %buf151 = aie.buffer(%tile_1_4) {sym_name = "buf151"} : memref<64x1xbf16, 2 : i32> 
    %buf150 = aie.buffer(%tile_1_4) {sym_name = "buf150"} : memref<64x64xbf16, 2 : i32> 
    %buf149 = aie.buffer(%tile_1_4) {sym_name = "buf149"} : memref<64x1xbf16, 2 : i32> 
    %buf148 = aie.buffer(%tile_1_4) {sym_name = "buf148"} : memref<64x1xbf16, 2 : i32> 
    %buf147 = aie.buffer(%tile_1_4) {sym_name = "buf147"} : memref<64x1xbf16, 2 : i32> 
    %buf146 = aie.buffer(%tile_1_4) {sym_name = "buf146"} : memref<64x1xbf16, 2 : i32> 
    %buf145 = aie.buffer(%tile_1_4) {sym_name = "buf145"} : memref<64x1xbf16, 2 : i32> 
    %buf144 = aie.buffer(%tile_1_4) {sym_name = "buf144"} : memref<64x1xbf16, 2 : i32> 
    %buf143 = aie.buffer(%tile_0_4) {sym_name = "buf143"} : memref<64x1xbf16, 2 : i32> 
    %buf142 = aie.buffer(%tile_0_4) {sym_name = "buf142"} : memref<64x1xbf16, 2 : i32> 
    %buf141 = aie.buffer(%tile_0_4) {sym_name = "buf141"} : memref<64x64xbf16, 2 : i32> 
    %buf140 = aie.buffer(%tile_0_4) {sym_name = "buf140"} : memref<64x64xbf16, 2 : i32> 
    %buf139 = aie.buffer(%tile_0_4) {sym_name = "buf139"} : memref<64x64xbf16, 2 : i32> 
    %buf138 = aie.buffer(%tile_0_4) {sym_name = "buf138"} : memref<64x64xbf16, 2 : i32> 
    %buf137 = aie.buffer(%tile_0_4) {sym_name = "buf137"} : memref<64x64xbf16, 2 : i32> 
    %buf136 = aie.buffer(%tile_0_4) {sym_name = "buf136"} : memref<64x1xbf16, 2 : i32> 
    %buf135 = aie.buffer(%tile_0_4) {sym_name = "buf135"} : memref<64x1xbf16, 2 : i32> 
    %buf134 = aie.buffer(%tile_0_4) {sym_name = "buf134"} : memref<64x64xbf16, 2 : i32> 
    %buf133 = aie.buffer(%tile_0_4) {sym_name = "buf133"} : memref<64x1xbf16, 2 : i32> 
    %buf132 = aie.buffer(%tile_0_4) {sym_name = "buf132"} : memref<64x1xbf16, 2 : i32> 
    %buf131 = aie.buffer(%tile_0_4) {sym_name = "buf131"} : memref<64x1xbf16, 2 : i32> 
    %buf130 = aie.buffer(%tile_0_4) {sym_name = "buf130"} : memref<64x1xbf16, 2 : i32> 
    %buf129 = aie.buffer(%tile_0_4) {sym_name = "buf129"} : memref<64x1xbf16, 2 : i32> 
    %buf128 = aie.buffer(%tile_0_4) {sym_name = "buf128"} : memref<64x1xbf16, 2 : i32> 
    %buf127 = aie.buffer(%tile_3_3) {sym_name = "buf127"} : memref<64x1xbf16, 2 : i32> 
    %buf126 = aie.buffer(%tile_3_3) {sym_name = "buf126"} : memref<64x1xbf16, 2 : i32> 
    %buf125 = aie.buffer(%tile_3_3) {sym_name = "buf125"} : memref<64x64xbf16, 2 : i32> 
    %buf124 = aie.buffer(%tile_3_3) {sym_name = "buf124"} : memref<64x64xbf16, 2 : i32> 
    %buf123 = aie.buffer(%tile_3_3) {sym_name = "buf123"} : memref<64x64xbf16, 2 : i32> 
    %buf122 = aie.buffer(%tile_3_3) {sym_name = "buf122"} : memref<64x64xbf16, 2 : i32> 
    %buf121 = aie.buffer(%tile_3_3) {sym_name = "buf121"} : memref<64x64xbf16, 2 : i32> 
    %buf120 = aie.buffer(%tile_3_3) {sym_name = "buf120"} : memref<64x1xbf16, 2 : i32> 
    %buf119 = aie.buffer(%tile_3_3) {sym_name = "buf119"} : memref<64x1xbf16, 2 : i32> 
    %buf118 = aie.buffer(%tile_3_3) {sym_name = "buf118"} : memref<64x64xbf16, 2 : i32> 
    %buf117 = aie.buffer(%tile_3_3) {sym_name = "buf117"} : memref<64x1xbf16, 2 : i32> 
    %buf116 = aie.buffer(%tile_3_3) {sym_name = "buf116"} : memref<64x1xbf16, 2 : i32> 
    %buf115 = aie.buffer(%tile_3_3) {sym_name = "buf115"} : memref<64x1xbf16, 2 : i32> 
    %buf114 = aie.buffer(%tile_3_3) {sym_name = "buf114"} : memref<64x1xbf16, 2 : i32> 
    %buf113 = aie.buffer(%tile_3_3) {sym_name = "buf113"} : memref<64x1xbf16, 2 : i32> 
    %buf112 = aie.buffer(%tile_3_3) {sym_name = "buf112"} : memref<64x1xbf16, 2 : i32> 
    %buf111 = aie.buffer(%tile_2_3) {sym_name = "buf111"} : memref<64x1xbf16, 2 : i32> 
    %buf110 = aie.buffer(%tile_2_3) {sym_name = "buf110"} : memref<64x1xbf16, 2 : i32> 
    %buf109 = aie.buffer(%tile_2_3) {sym_name = "buf109"} : memref<64x64xbf16, 2 : i32> 
    %buf108 = aie.buffer(%tile_2_3) {sym_name = "buf108"} : memref<64x64xbf16, 2 : i32> 
    %buf107 = aie.buffer(%tile_2_3) {sym_name = "buf107"} : memref<64x64xbf16, 2 : i32> 
    %buf106 = aie.buffer(%tile_2_3) {sym_name = "buf106"} : memref<64x64xbf16, 2 : i32> 
    %buf105 = aie.buffer(%tile_2_3) {sym_name = "buf105"} : memref<64x64xbf16, 2 : i32> 
    %buf104 = aie.buffer(%tile_2_3) {sym_name = "buf104"} : memref<64x1xbf16, 2 : i32> 
    %buf103 = aie.buffer(%tile_2_3) {sym_name = "buf103"} : memref<64x1xbf16, 2 : i32> 
    %buf102 = aie.buffer(%tile_2_3) {sym_name = "buf102"} : memref<64x64xbf16, 2 : i32> 
    %buf101 = aie.buffer(%tile_2_3) {sym_name = "buf101"} : memref<64x1xbf16, 2 : i32> 
    %buf100 = aie.buffer(%tile_2_3) {sym_name = "buf100"} : memref<64x1xbf16, 2 : i32> 
    %buf99 = aie.buffer(%tile_2_3) {sym_name = "buf99"} : memref<64x1xbf16, 2 : i32> 
    %buf98 = aie.buffer(%tile_2_3) {sym_name = "buf98"} : memref<64x1xbf16, 2 : i32> 
    %buf97 = aie.buffer(%tile_2_3) {sym_name = "buf97"} : memref<64x1xbf16, 2 : i32> 
    %buf96 = aie.buffer(%tile_2_3) {sym_name = "buf96"} : memref<64x1xbf16, 2 : i32> 
    %buf95 = aie.buffer(%tile_1_3) {sym_name = "buf95"} : memref<64x1xbf16, 2 : i32> 
    %buf94 = aie.buffer(%tile_1_3) {sym_name = "buf94"} : memref<64x1xbf16, 2 : i32> 
    %buf93 = aie.buffer(%tile_1_3) {sym_name = "buf93"} : memref<64x64xbf16, 2 : i32> 
    %buf92 = aie.buffer(%tile_1_3) {sym_name = "buf92"} : memref<64x64xbf16, 2 : i32> 
    %buf91 = aie.buffer(%tile_1_3) {sym_name = "buf91"} : memref<64x64xbf16, 2 : i32> 
    %buf90 = aie.buffer(%tile_1_3) {sym_name = "buf90"} : memref<64x64xbf16, 2 : i32> 
    %buf89 = aie.buffer(%tile_1_3) {sym_name = "buf89"} : memref<64x64xbf16, 2 : i32> 
    %buf88 = aie.buffer(%tile_1_3) {sym_name = "buf88"} : memref<64x1xbf16, 2 : i32> 
    %buf87 = aie.buffer(%tile_1_3) {sym_name = "buf87"} : memref<64x1xbf16, 2 : i32> 
    %buf86 = aie.buffer(%tile_1_3) {sym_name = "buf86"} : memref<64x64xbf16, 2 : i32> 
    %buf85 = aie.buffer(%tile_1_3) {sym_name = "buf85"} : memref<64x1xbf16, 2 : i32> 
    %buf84 = aie.buffer(%tile_1_3) {sym_name = "buf84"} : memref<64x1xbf16, 2 : i32> 
    %buf83 = aie.buffer(%tile_1_3) {sym_name = "buf83"} : memref<64x1xbf16, 2 : i32> 
    %buf82 = aie.buffer(%tile_1_3) {sym_name = "buf82"} : memref<64x1xbf16, 2 : i32> 
    %buf81 = aie.buffer(%tile_1_3) {sym_name = "buf81"} : memref<64x1xbf16, 2 : i32> 
    %buf80 = aie.buffer(%tile_1_3) {sym_name = "buf80"} : memref<64x1xbf16, 2 : i32> 
    %buf79 = aie.buffer(%tile_0_3) {sym_name = "buf79"} : memref<64x1xbf16, 2 : i32> 
    %buf78 = aie.buffer(%tile_0_3) {sym_name = "buf78"} : memref<64x1xbf16, 2 : i32> 
    %buf77 = aie.buffer(%tile_0_3) {sym_name = "buf77"} : memref<64x64xbf16, 2 : i32> 
    %buf76 = aie.buffer(%tile_0_3) {sym_name = "buf76"} : memref<64x64xbf16, 2 : i32> 
    %buf75 = aie.buffer(%tile_0_3) {sym_name = "buf75"} : memref<64x64xbf16, 2 : i32> 
    %buf74 = aie.buffer(%tile_0_3) {sym_name = "buf74"} : memref<64x64xbf16, 2 : i32> 
    %buf73 = aie.buffer(%tile_0_3) {sym_name = "buf73"} : memref<64x64xbf16, 2 : i32> 
    %buf72 = aie.buffer(%tile_0_3) {sym_name = "buf72"} : memref<64x1xbf16, 2 : i32> 
    %buf71 = aie.buffer(%tile_0_3) {sym_name = "buf71"} : memref<64x1xbf16, 2 : i32> 
    %buf70 = aie.buffer(%tile_0_3) {sym_name = "buf70"} : memref<64x64xbf16, 2 : i32> 
    %buf69 = aie.buffer(%tile_0_3) {sym_name = "buf69"} : memref<64x1xbf16, 2 : i32> 
    %buf68 = aie.buffer(%tile_0_3) {sym_name = "buf68"} : memref<64x1xbf16, 2 : i32> 
    %buf67 = aie.buffer(%tile_0_3) {sym_name = "buf67"} : memref<64x1xbf16, 2 : i32> 
    %buf66 = aie.buffer(%tile_0_3) {sym_name = "buf66"} : memref<64x1xbf16, 2 : i32> 
    %buf65 = aie.buffer(%tile_0_3) {sym_name = "buf65"} : memref<64x1xbf16, 2 : i32> 
    %buf64 = aie.buffer(%tile_0_3) {sym_name = "buf64"} : memref<64x1xbf16, 2 : i32> 
    %buf63 = aie.buffer(%tile_3_2) {sym_name = "buf63"} : memref<64x1xbf16, 2 : i32> 
    %buf62 = aie.buffer(%tile_3_2) {sym_name = "buf62"} : memref<64x1xbf16, 2 : i32> 
    %buf61 = aie.buffer(%tile_3_2) {sym_name = "buf61"} : memref<64x64xbf16, 2 : i32> 
    %buf60 = aie.buffer(%tile_3_2) {sym_name = "buf60"} : memref<64x64xbf16, 2 : i32> 
    %buf59 = aie.buffer(%tile_3_2) {sym_name = "buf59"} : memref<64x64xbf16, 2 : i32> 
    %buf58 = aie.buffer(%tile_3_2) {sym_name = "buf58"} : memref<64x64xbf16, 2 : i32> 
    %buf57 = aie.buffer(%tile_3_2) {sym_name = "buf57"} : memref<64x64xbf16, 2 : i32> 
    %buf56 = aie.buffer(%tile_3_2) {sym_name = "buf56"} : memref<64x1xbf16, 2 : i32> 
    %buf55 = aie.buffer(%tile_3_2) {sym_name = "buf55"} : memref<64x1xbf16, 2 : i32> 
    %buf54 = aie.buffer(%tile_3_2) {sym_name = "buf54"} : memref<64x64xbf16, 2 : i32> 
    %buf53 = aie.buffer(%tile_3_2) {sym_name = "buf53"} : memref<64x1xbf16, 2 : i32> 
    %buf52 = aie.buffer(%tile_3_2) {sym_name = "buf52"} : memref<64x1xbf16, 2 : i32> 
    %buf51 = aie.buffer(%tile_3_2) {sym_name = "buf51"} : memref<64x1xbf16, 2 : i32> 
    %buf50 = aie.buffer(%tile_3_2) {sym_name = "buf50"} : memref<64x1xbf16, 2 : i32> 
    %buf49 = aie.buffer(%tile_3_2) {sym_name = "buf49"} : memref<64x1xbf16, 2 : i32> 
    %buf48 = aie.buffer(%tile_3_2) {sym_name = "buf48"} : memref<64x1xbf16, 2 : i32> 
    %buf47 = aie.buffer(%tile_2_2) {sym_name = "buf47"} : memref<64x1xbf16, 2 : i32> 
    %buf46 = aie.buffer(%tile_2_2) {sym_name = "buf46"} : memref<64x1xbf16, 2 : i32> 
    %buf45 = aie.buffer(%tile_2_2) {sym_name = "buf45"} : memref<64x64xbf16, 2 : i32> 
    %buf44 = aie.buffer(%tile_2_2) {sym_name = "buf44"} : memref<64x64xbf16, 2 : i32> 
    %buf43 = aie.buffer(%tile_2_2) {sym_name = "buf43"} : memref<64x64xbf16, 2 : i32> 
    %buf42 = aie.buffer(%tile_2_2) {sym_name = "buf42"} : memref<64x64xbf16, 2 : i32> 
    %buf41 = aie.buffer(%tile_2_2) {sym_name = "buf41"} : memref<64x64xbf16, 2 : i32> 
    %buf40 = aie.buffer(%tile_2_2) {sym_name = "buf40"} : memref<64x1xbf16, 2 : i32> 
    %buf39 = aie.buffer(%tile_2_2) {sym_name = "buf39"} : memref<64x1xbf16, 2 : i32> 
    %buf38 = aie.buffer(%tile_2_2) {sym_name = "buf38"} : memref<64x64xbf16, 2 : i32> 
    %buf37 = aie.buffer(%tile_2_2) {sym_name = "buf37"} : memref<64x1xbf16, 2 : i32> 
    %buf36 = aie.buffer(%tile_2_2) {sym_name = "buf36"} : memref<64x1xbf16, 2 : i32> 
    %buf35 = aie.buffer(%tile_2_2) {sym_name = "buf35"} : memref<64x1xbf16, 2 : i32> 
    %buf34 = aie.buffer(%tile_2_2) {sym_name = "buf34"} : memref<64x1xbf16, 2 : i32> 
    %buf33 = aie.buffer(%tile_2_2) {sym_name = "buf33"} : memref<64x1xbf16, 2 : i32> 
    %buf32 = aie.buffer(%tile_2_2) {sym_name = "buf32"} : memref<64x1xbf16, 2 : i32> 
    %buf31 = aie.buffer(%tile_1_2) {sym_name = "buf31"} : memref<64x1xbf16, 2 : i32> 
    %buf30 = aie.buffer(%tile_1_2) {sym_name = "buf30"} : memref<64x1xbf16, 2 : i32> 
    %buf29 = aie.buffer(%tile_1_2) {sym_name = "buf29"} : memref<64x64xbf16, 2 : i32> 
    %buf28 = aie.buffer(%tile_1_2) {sym_name = "buf28"} : memref<64x64xbf16, 2 : i32> 
    %buf27 = aie.buffer(%tile_1_2) {sym_name = "buf27"} : memref<64x64xbf16, 2 : i32> 
    %buf26 = aie.buffer(%tile_1_2) {sym_name = "buf26"} : memref<64x64xbf16, 2 : i32> 
    %buf25 = aie.buffer(%tile_1_2) {sym_name = "buf25"} : memref<64x64xbf16, 2 : i32> 
    %buf24 = aie.buffer(%tile_1_2) {sym_name = "buf24"} : memref<64x1xbf16, 2 : i32> 
    %buf23 = aie.buffer(%tile_1_2) {sym_name = "buf23"} : memref<64x1xbf16, 2 : i32> 
    %buf22 = aie.buffer(%tile_1_2) {sym_name = "buf22"} : memref<64x64xbf16, 2 : i32> 
    %buf21 = aie.buffer(%tile_1_2) {sym_name = "buf21"} : memref<64x1xbf16, 2 : i32> 
    %buf20 = aie.buffer(%tile_1_2) {sym_name = "buf20"} : memref<64x1xbf16, 2 : i32> 
    %buf19 = aie.buffer(%tile_1_2) {sym_name = "buf19"} : memref<64x1xbf16, 2 : i32> 
    %buf18 = aie.buffer(%tile_1_2) {sym_name = "buf18"} : memref<64x1xbf16, 2 : i32> 
    %buf17 = aie.buffer(%tile_1_2) {sym_name = "buf17"} : memref<64x1xbf16, 2 : i32> 
    %buf16 = aie.buffer(%tile_1_2) {sym_name = "buf16"} : memref<64x1xbf16, 2 : i32> 
    %buf15 = aie.buffer(%tile_0_2) {sym_name = "buf15"} : memref<64x1xbf16, 2 : i32> 
    %buf14 = aie.buffer(%tile_0_2) {sym_name = "buf14"} : memref<64x1xbf16, 2 : i32> 
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
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2x256x64xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2x256x64xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2x256x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<2x256x64xbf16>
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_74, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf222 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_75, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf223 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_73, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf225) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf227) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf226) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf222, %buf221) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf224 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf224 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf221, %buf222, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_73, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf224 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf226, %buf220, %buf219) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf219, %buf225) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf224 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf223, %buf225) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf227, %buf219, %buf220) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf220, %buf227) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf225 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_81 = memref.collapse_shape %buf227 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_71, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf213 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf214 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_70, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf216) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf218) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf217) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf213, %buf212) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf215 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf215 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf212, %buf213, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_70, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf215 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf217, %buf211, %buf210) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf210, %buf216) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf215 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf214, %buf216) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf218, %buf210, %buf211) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf211, %buf218) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf216 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf217 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_81 = memref.collapse_shape %buf218 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_68, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf204 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_69, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf205 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_67, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf207) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf209) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf208) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf204, %buf203) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf203, %buf204, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf208, %buf202, %buf201) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf201, %buf207) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf205, %buf207) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf209, %buf201, %buf202) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf202, %buf209) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf207 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf208 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_81 = memref.collapse_shape %buf209 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_65, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf195 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_66, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf196 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_64, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf198) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf200) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf199) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf195, %buf194) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf197 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf197 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf194, %buf195, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_64, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf197 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf199, %buf193, %buf192) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf192, %buf198) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf197 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf196, %buf198) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf200, %buf192, %buf193) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf193, %buf200) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf198 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf199 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_81 = memref.collapse_shape %buf200 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf186 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_63, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf187 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_61, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf189) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf191) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf190) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf186, %buf185) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf188 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf188 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf185, %buf186, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_61, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf188 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf190, %buf184, %buf183) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf183, %buf189) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf188 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf187, %buf189) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf191, %buf183, %buf184) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf184, %buf191) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf182 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf181 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf180 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf190, %buf179) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf181, %buf190) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf181, %buf190, %buf178) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf179, %buf190, %buf177) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf178, %buf182) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf177, %buf189) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf189, %buf182) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf176) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf180, %buf178, %buf176) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf191, %buf177, %buf176) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf176, %buf180) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf182 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf190 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf180 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_59, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf170 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf171 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_58, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf173) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf175) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf174) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf170, %buf169) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf172 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf172 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf169, %buf170, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_58, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf172 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf174, %buf168, %buf167) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf167, %buf173) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf172 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf171, %buf173) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf175, %buf167, %buf168) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf168, %buf175) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf166 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf165 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf164 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf174, %buf163) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf165, %buf174) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf165, %buf174, %buf162) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf163, %buf174, %buf161) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf162, %buf166) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf161, %buf173) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf173, %buf166) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf160) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf164, %buf162, %buf160) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf175, %buf161, %buf160) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf160, %buf164) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf166 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf174 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf164 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_56, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf154 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf155 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf157) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf159) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf158) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf154, %buf153) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf156 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf156 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf153, %buf154, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf156 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf158, %buf152, %buf151) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf151, %buf157) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf156 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf155, %buf157) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf159, %buf151, %buf152) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf152, %buf159) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf150 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf149 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf148 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf158, %buf147) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf149, %buf158) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf149, %buf158, %buf146) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf147, %buf158, %buf145) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf146, %buf150) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf145, %buf157) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf157, %buf150) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf144) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf148, %buf146, %buf144) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf159, %buf145, %buf144) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf144, %buf148) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf150 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf158 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf148 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf138 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf139 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_52, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf141) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf143) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf142) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf138, %buf137) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf140 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf140 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf137, %buf138, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf140 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf142, %buf136, %buf135) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf135, %buf141) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf140 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf139, %buf141) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf143, %buf135, %buf136) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf136, %buf143) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf134 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf133 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf132 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf142, %buf131) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf133, %buf142) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf133, %buf142, %buf130) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf131, %buf142, %buf129) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf130, %buf134) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf129, %buf141) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf141, %buf134) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf128) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf132, %buf130, %buf128) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf143, %buf129, %buf128) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf128, %buf132) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf134 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf142 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf132 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf122 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_51, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf123 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_49, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf125) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf127) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf126) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf122, %buf121) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf124 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf124 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf121, %buf122, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf124 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf126, %buf120, %buf119) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf119, %buf125) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf124 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf123, %buf125) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf127, %buf119, %buf120) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf120, %buf127) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf118 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf117 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf116 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf126, %buf115) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf117, %buf126) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf117, %buf126, %buf114) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf115, %buf126, %buf113) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf114, %buf118) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf113, %buf125) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf125, %buf118) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf112) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf116, %buf114, %buf112) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf127, %buf113, %buf112) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf112, %buf116) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf118 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf126 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf116 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf106 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf107 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf109) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf111) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf110) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf106, %buf105) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf108 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf108 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf105, %buf106, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf108 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf110, %buf104, %buf103) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf103, %buf109) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf108 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf107, %buf109) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf111, %buf103, %buf104) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf104, %buf111) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf102 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf101 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf100 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf110, %buf99) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf101, %buf110) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf101, %buf110, %buf98) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf99, %buf110, %buf97) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf98, %buf102) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf97, %buf109) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf109, %buf102) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf96) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf100, %buf98, %buf96) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf111, %buf97, %buf96) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf96, %buf100) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf102 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf110 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf100 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_44, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf90 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf91 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_43, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf93) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf95) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf94) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf90, %buf89) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf92 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf92 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf89, %buf90, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf92 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf94, %buf88, %buf87) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf87, %buf93) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf92 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf91, %buf93) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf95, %buf87, %buf88) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf88, %buf95) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf86 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf85 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf84 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf94, %buf83) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf85, %buf94) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf85, %buf94, %buf82) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf83, %buf94, %buf81) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf82, %buf86) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf81, %buf93) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf93, %buf86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf80) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf84, %buf82, %buf80) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf95, %buf81, %buf80) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf80, %buf84) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf86 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf94 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf84 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf75 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_40, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf77) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf79) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf78) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf74, %buf73) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf76 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf76 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf73, %buf74, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_40, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf76 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf78, %buf72, %buf71) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf71, %buf77) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf76 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf75, %buf77) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf79, %buf71, %buf72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf72, %buf79) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf70 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf69 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf68 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf78, %buf67) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf69, %buf78) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf69, %buf78, %buf66) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf67, %buf78, %buf65) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf66, %buf70) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf65, %buf77) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf77, %buf70) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf64) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf68, %buf66, %buf64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf79, %buf65, %buf64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf64, %buf68) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf70 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf78 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf68 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf54 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf58 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf59 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_35, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_2_38, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf61) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf63) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf62) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf58, %buf57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf57, %buf58, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf62, %buf56, %buf55) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf55, %buf61) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf59, %buf61) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf63, %buf55, %buf56) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf56, %buf63) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf54 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf53 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf52 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf62, %buf51) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf53, %buf62) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf53, %buf62, %buf50) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf51, %buf62, %buf49) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf50, %buf54) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf49, %buf61) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf61, %buf54) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf48) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf52, %buf50, %buf48) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf63, %buf49, %buf48) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf48, %buf52) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf52, %buf54) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_39, Release, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf38 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf42 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf43 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_30, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_2_33, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf45) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf47) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf46) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf42, %buf41) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf41, %buf42, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf46, %buf40, %buf39) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf39, %buf45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf43, %buf45) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf47, %buf39, %buf40) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf40, %buf47) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf38 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf37 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf36 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf46, %buf35) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf37, %buf46) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf37, %buf46, %buf34) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf35, %buf46, %buf33) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf34, %buf38) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf33, %buf45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf45, %buf38) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf32) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf36, %buf34, %buf32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf47, %buf33, %buf32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf32, %buf36) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf36, %buf38) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_34, Release, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf22 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf26 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_28, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf29) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf31) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf30) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf26, %buf25) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf28 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf28 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf25, %buf26, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf28 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf30, %buf24, %buf23) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf23, %buf29) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf28 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf27, %buf29) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf31, %buf23, %buf24) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf24, %buf31) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf22 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf21 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf20 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf30, %buf19) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf21, %buf30) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf21, %buf30, %buf18) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf19, %buf30, %buf17) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf18, %buf22) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf17, %buf29) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf29, %buf22) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf16) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf20, %buf18, %buf16) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf31, %buf17, %buf16) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf16, %buf20) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf20, %buf22) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_29, Release, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2, Release, 1)
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
      aie.dma_bd(%buf10 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf11 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_20, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_23, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf13) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf15) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf14) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf10, %buf9) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf12 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf12 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf9, %buf10, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf12 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf14, %buf8, %buf7) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf7, %buf13) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf12 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf11, %buf13) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf15, %buf7, %buf8) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf8, %buf15) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf6 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf5 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf4 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf14, %buf3) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf5, %buf14) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf5, %buf14, %buf2) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf3, %buf14, %buf1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf2, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf1, %buf13) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf13, %buf6) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf4, %buf2, %buf0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf15, %buf1, %buf0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf0, %buf4) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf4, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_24, Release, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
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
      aie.dma_bd(%buf231 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf239 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_16, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf235 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf239 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_17, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf235 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_15, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_0_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf231 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_19, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf238 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_11, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf234 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf238 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_12, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf234 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_10, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_1_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_14, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf229 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf237 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_6, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf233 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf237 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_7, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf233 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_2_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf229 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf236 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf232 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf236 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf232 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_3_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
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
    %buf479 = aie.buffer(%mem_tile_0_1) {sym_name = "buf479"} : memref<64x64xbf16, 1 : i32> 
    %buf478 = aie.buffer(%mem_tile_1_1) {sym_name = "buf478"} : memref<64x64xbf16, 1 : i32> 
    %buf477 = aie.buffer(%mem_tile_2_1) {sym_name = "buf477"} : memref<64x64xbf16, 1 : i32> 
    %buf476 = aie.buffer(%mem_tile_3_1) {sym_name = "buf476"} : memref<64x64xbf16, 1 : i32> 
    %buf475 = aie.buffer(%mem_tile_0_1) {sym_name = "buf475"} : memref<64x64xbf16, 1 : i32> 
    %buf474 = aie.buffer(%mem_tile_1_1) {sym_name = "buf474"} : memref<64x64xbf16, 1 : i32> 
    %buf473 = aie.buffer(%mem_tile_2_1) {sym_name = "buf473"} : memref<64x64xbf16, 1 : i32> 
    %buf472 = aie.buffer(%mem_tile_3_1) {sym_name = "buf472"} : memref<64x64xbf16, 1 : i32> 
    %buf471 = aie.buffer(%mem_tile_0_1) {sym_name = "buf471"} : memref<64x64xbf16, 1 : i32> 
    %buf470 = aie.buffer(%mem_tile_1_1) {sym_name = "buf470"} : memref<64x64xbf16, 1 : i32> 
    %buf469 = aie.buffer(%mem_tile_2_1) {sym_name = "buf469"} : memref<64x64xbf16, 1 : i32> 
    %buf468 = aie.buffer(%mem_tile_3_1) {sym_name = "buf468"} : memref<64x64xbf16, 1 : i32> 
    %buf467 = aie.buffer(%tile_3_5) {sym_name = "buf467"} : memref<64x1xbf16, 2 : i32> 
    %buf466 = aie.buffer(%tile_3_5) {sym_name = "buf466"} : memref<64x1xbf16, 2 : i32> 
    %buf465 = aie.buffer(%tile_3_5) {sym_name = "buf465"} : memref<64x64xbf16, 2 : i32> 
    %buf464 = aie.buffer(%tile_3_5) {sym_name = "buf464"} : memref<64x64xbf16, 2 : i32> 
    %buf463 = aie.buffer(%tile_3_5) {sym_name = "buf463"} : memref<64x64xbf16, 2 : i32> 
    %buf462 = aie.buffer(%tile_3_5) {sym_name = "buf462"} : memref<64x64xbf16, 2 : i32> 
    %buf461 = aie.buffer(%tile_3_5) {sym_name = "buf461"} : memref<64x64xbf16, 2 : i32> 
    %buf460 = aie.buffer(%tile_3_5) {sym_name = "buf460"} : memref<64x1xbf16, 2 : i32> 
    %buf459 = aie.buffer(%tile_3_5) {sym_name = "buf459"} : memref<64x1xbf16, 2 : i32> 
    %buf458 = aie.buffer(%tile_2_5) {sym_name = "buf458"} : memref<64x1xbf16, 2 : i32> 
    %buf457 = aie.buffer(%tile_2_5) {sym_name = "buf457"} : memref<64x1xbf16, 2 : i32> 
    %buf456 = aie.buffer(%tile_2_5) {sym_name = "buf456"} : memref<64x64xbf16, 2 : i32> 
    %buf455 = aie.buffer(%tile_2_5) {sym_name = "buf455"} : memref<64x64xbf16, 2 : i32> 
    %buf454 = aie.buffer(%tile_2_5) {sym_name = "buf454"} : memref<64x64xbf16, 2 : i32> 
    %buf453 = aie.buffer(%tile_2_5) {sym_name = "buf453"} : memref<64x64xbf16, 2 : i32> 
    %buf452 = aie.buffer(%tile_2_5) {sym_name = "buf452"} : memref<64x64xbf16, 2 : i32> 
    %buf451 = aie.buffer(%tile_2_5) {sym_name = "buf451"} : memref<64x1xbf16, 2 : i32> 
    %buf450 = aie.buffer(%tile_2_5) {sym_name = "buf450"} : memref<64x1xbf16, 2 : i32> 
    %buf449 = aie.buffer(%tile_1_5) {sym_name = "buf449"} : memref<64x1xbf16, 2 : i32> 
    %buf448 = aie.buffer(%tile_1_5) {sym_name = "buf448"} : memref<64x1xbf16, 2 : i32> 
    %buf447 = aie.buffer(%tile_1_5) {sym_name = "buf447"} : memref<64x64xbf16, 2 : i32> 
    %buf446 = aie.buffer(%tile_1_5) {sym_name = "buf446"} : memref<64x64xbf16, 2 : i32> 
    %buf445 = aie.buffer(%tile_1_5) {sym_name = "buf445"} : memref<64x64xbf16, 2 : i32> 
    %buf444 = aie.buffer(%tile_1_5) {sym_name = "buf444"} : memref<64x64xbf16, 2 : i32> 
    %buf443 = aie.buffer(%tile_1_5) {sym_name = "buf443"} : memref<64x64xbf16, 2 : i32> 
    %buf442 = aie.buffer(%tile_1_5) {sym_name = "buf442"} : memref<64x1xbf16, 2 : i32> 
    %buf441 = aie.buffer(%tile_1_5) {sym_name = "buf441"} : memref<64x1xbf16, 2 : i32> 
    %buf440 = aie.buffer(%tile_0_5) {sym_name = "buf440"} : memref<64x1xbf16, 2 : i32> 
    %buf439 = aie.buffer(%tile_0_5) {sym_name = "buf439"} : memref<64x1xbf16, 2 : i32> 
    %buf438 = aie.buffer(%tile_0_5) {sym_name = "buf438"} : memref<64x64xbf16, 2 : i32> 
    %buf437 = aie.buffer(%tile_0_5) {sym_name = "buf437"} : memref<64x64xbf16, 2 : i32> 
    %buf436 = aie.buffer(%tile_0_5) {sym_name = "buf436"} : memref<64x64xbf16, 2 : i32> 
    %buf435 = aie.buffer(%tile_0_5) {sym_name = "buf435"} : memref<64x64xbf16, 2 : i32> 
    %buf434 = aie.buffer(%tile_0_5) {sym_name = "buf434"} : memref<64x64xbf16, 2 : i32> 
    %buf433 = aie.buffer(%tile_0_5) {sym_name = "buf433"} : memref<64x1xbf16, 2 : i32> 
    %buf432 = aie.buffer(%tile_0_5) {sym_name = "buf432"} : memref<64x1xbf16, 2 : i32> 
    %buf431 = aie.buffer(%tile_3_4) {sym_name = "buf431"} : memref<64x1xbf16, 2 : i32> 
    %buf430 = aie.buffer(%tile_3_4) {sym_name = "buf430"} : memref<64x1xbf16, 2 : i32> 
    %buf429 = aie.buffer(%tile_3_4) {sym_name = "buf429"} : memref<64x64xbf16, 2 : i32> 
    %buf428 = aie.buffer(%tile_3_4) {sym_name = "buf428"} : memref<64x64xbf16, 2 : i32> 
    %buf427 = aie.buffer(%tile_3_4) {sym_name = "buf427"} : memref<64x64xbf16, 2 : i32> 
    %buf426 = aie.buffer(%tile_3_4) {sym_name = "buf426"} : memref<64x64xbf16, 2 : i32> 
    %buf425 = aie.buffer(%tile_3_4) {sym_name = "buf425"} : memref<64x64xbf16, 2 : i32> 
    %buf424 = aie.buffer(%tile_3_4) {sym_name = "buf424"} : memref<64x1xbf16, 2 : i32> 
    %buf423 = aie.buffer(%tile_3_4) {sym_name = "buf423"} : memref<64x1xbf16, 2 : i32> 
    %buf422 = aie.buffer(%tile_3_4) {sym_name = "buf422"} : memref<64x64xbf16, 2 : i32> 
    %buf421 = aie.buffer(%tile_3_4) {sym_name = "buf421"} : memref<64x1xbf16, 2 : i32> 
    %buf420 = aie.buffer(%tile_3_4) {sym_name = "buf420"} : memref<64x1xbf16, 2 : i32> 
    %buf419 = aie.buffer(%tile_3_4) {sym_name = "buf419"} : memref<64x1xbf16, 2 : i32> 
    %buf418 = aie.buffer(%tile_3_4) {sym_name = "buf418"} : memref<64x1xbf16, 2 : i32> 
    %buf417 = aie.buffer(%tile_3_4) {sym_name = "buf417"} : memref<64x1xbf16, 2 : i32> 
    %buf416 = aie.buffer(%tile_3_4) {sym_name = "buf416"} : memref<64x1xbf16, 2 : i32> 
    %buf415 = aie.buffer(%tile_2_4) {sym_name = "buf415"} : memref<64x1xbf16, 2 : i32> 
    %buf414 = aie.buffer(%tile_2_4) {sym_name = "buf414"} : memref<64x1xbf16, 2 : i32> 
    %buf413 = aie.buffer(%tile_2_4) {sym_name = "buf413"} : memref<64x64xbf16, 2 : i32> 
    %buf412 = aie.buffer(%tile_2_4) {sym_name = "buf412"} : memref<64x64xbf16, 2 : i32> 
    %buf411 = aie.buffer(%tile_2_4) {sym_name = "buf411"} : memref<64x64xbf16, 2 : i32> 
    %buf410 = aie.buffer(%tile_2_4) {sym_name = "buf410"} : memref<64x64xbf16, 2 : i32> 
    %buf409 = aie.buffer(%tile_2_4) {sym_name = "buf409"} : memref<64x64xbf16, 2 : i32> 
    %buf408 = aie.buffer(%tile_2_4) {sym_name = "buf408"} : memref<64x1xbf16, 2 : i32> 
    %buf407 = aie.buffer(%tile_2_4) {sym_name = "buf407"} : memref<64x1xbf16, 2 : i32> 
    %buf406 = aie.buffer(%tile_2_4) {sym_name = "buf406"} : memref<64x64xbf16, 2 : i32> 
    %buf405 = aie.buffer(%tile_2_4) {sym_name = "buf405"} : memref<64x1xbf16, 2 : i32> 
    %buf404 = aie.buffer(%tile_2_4) {sym_name = "buf404"} : memref<64x1xbf16, 2 : i32> 
    %buf403 = aie.buffer(%tile_2_4) {sym_name = "buf403"} : memref<64x1xbf16, 2 : i32> 
    %buf402 = aie.buffer(%tile_2_4) {sym_name = "buf402"} : memref<64x1xbf16, 2 : i32> 
    %buf401 = aie.buffer(%tile_2_4) {sym_name = "buf401"} : memref<64x1xbf16, 2 : i32> 
    %buf400 = aie.buffer(%tile_2_4) {sym_name = "buf400"} : memref<64x1xbf16, 2 : i32> 
    %buf399 = aie.buffer(%tile_1_4) {sym_name = "buf399"} : memref<64x1xbf16, 2 : i32> 
    %buf398 = aie.buffer(%tile_1_4) {sym_name = "buf398"} : memref<64x1xbf16, 2 : i32> 
    %buf397 = aie.buffer(%tile_1_4) {sym_name = "buf397"} : memref<64x64xbf16, 2 : i32> 
    %buf396 = aie.buffer(%tile_1_4) {sym_name = "buf396"} : memref<64x64xbf16, 2 : i32> 
    %buf395 = aie.buffer(%tile_1_4) {sym_name = "buf395"} : memref<64x64xbf16, 2 : i32> 
    %buf394 = aie.buffer(%tile_1_4) {sym_name = "buf394"} : memref<64x64xbf16, 2 : i32> 
    %buf393 = aie.buffer(%tile_1_4) {sym_name = "buf393"} : memref<64x64xbf16, 2 : i32> 
    %buf392 = aie.buffer(%tile_1_4) {sym_name = "buf392"} : memref<64x1xbf16, 2 : i32> 
    %buf391 = aie.buffer(%tile_1_4) {sym_name = "buf391"} : memref<64x1xbf16, 2 : i32> 
    %buf390 = aie.buffer(%tile_1_4) {sym_name = "buf390"} : memref<64x64xbf16, 2 : i32> 
    %buf389 = aie.buffer(%tile_1_4) {sym_name = "buf389"} : memref<64x1xbf16, 2 : i32> 
    %buf388 = aie.buffer(%tile_1_4) {sym_name = "buf388"} : memref<64x1xbf16, 2 : i32> 
    %buf387 = aie.buffer(%tile_1_4) {sym_name = "buf387"} : memref<64x1xbf16, 2 : i32> 
    %buf386 = aie.buffer(%tile_1_4) {sym_name = "buf386"} : memref<64x1xbf16, 2 : i32> 
    %buf385 = aie.buffer(%tile_1_4) {sym_name = "buf385"} : memref<64x1xbf16, 2 : i32> 
    %buf384 = aie.buffer(%tile_1_4) {sym_name = "buf384"} : memref<64x1xbf16, 2 : i32> 
    %buf383 = aie.buffer(%tile_0_4) {sym_name = "buf383"} : memref<64x1xbf16, 2 : i32> 
    %buf382 = aie.buffer(%tile_0_4) {sym_name = "buf382"} : memref<64x1xbf16, 2 : i32> 
    %buf381 = aie.buffer(%tile_0_4) {sym_name = "buf381"} : memref<64x64xbf16, 2 : i32> 
    %buf380 = aie.buffer(%tile_0_4) {sym_name = "buf380"} : memref<64x64xbf16, 2 : i32> 
    %buf379 = aie.buffer(%tile_0_4) {sym_name = "buf379"} : memref<64x64xbf16, 2 : i32> 
    %buf378 = aie.buffer(%tile_0_4) {sym_name = "buf378"} : memref<64x64xbf16, 2 : i32> 
    %buf377 = aie.buffer(%tile_0_4) {sym_name = "buf377"} : memref<64x64xbf16, 2 : i32> 
    %buf376 = aie.buffer(%tile_0_4) {sym_name = "buf376"} : memref<64x1xbf16, 2 : i32> 
    %buf375 = aie.buffer(%tile_0_4) {sym_name = "buf375"} : memref<64x1xbf16, 2 : i32> 
    %buf374 = aie.buffer(%tile_0_4) {sym_name = "buf374"} : memref<64x64xbf16, 2 : i32> 
    %buf373 = aie.buffer(%tile_0_4) {sym_name = "buf373"} : memref<64x1xbf16, 2 : i32> 
    %buf372 = aie.buffer(%tile_0_4) {sym_name = "buf372"} : memref<64x1xbf16, 2 : i32> 
    %buf371 = aie.buffer(%tile_0_4) {sym_name = "buf371"} : memref<64x1xbf16, 2 : i32> 
    %buf370 = aie.buffer(%tile_0_4) {sym_name = "buf370"} : memref<64x1xbf16, 2 : i32> 
    %buf369 = aie.buffer(%tile_0_4) {sym_name = "buf369"} : memref<64x1xbf16, 2 : i32> 
    %buf368 = aie.buffer(%tile_0_4) {sym_name = "buf368"} : memref<64x1xbf16, 2 : i32> 
    %buf367 = aie.buffer(%tile_3_3) {sym_name = "buf367"} : memref<64x1xbf16, 2 : i32> 
    %buf366 = aie.buffer(%tile_3_3) {sym_name = "buf366"} : memref<64x1xbf16, 2 : i32> 
    %buf365 = aie.buffer(%tile_3_3) {sym_name = "buf365"} : memref<64x64xbf16, 2 : i32> 
    %buf364 = aie.buffer(%tile_3_3) {sym_name = "buf364"} : memref<64x64xbf16, 2 : i32> 
    %buf363 = aie.buffer(%tile_3_3) {sym_name = "buf363"} : memref<64x64xbf16, 2 : i32> 
    %buf362 = aie.buffer(%tile_3_3) {sym_name = "buf362"} : memref<64x64xbf16, 2 : i32> 
    %buf361 = aie.buffer(%tile_3_3) {sym_name = "buf361"} : memref<64x64xbf16, 2 : i32> 
    %buf360 = aie.buffer(%tile_3_3) {sym_name = "buf360"} : memref<64x1xbf16, 2 : i32> 
    %buf359 = aie.buffer(%tile_3_3) {sym_name = "buf359"} : memref<64x1xbf16, 2 : i32> 
    %buf358 = aie.buffer(%tile_3_3) {sym_name = "buf358"} : memref<64x64xbf16, 2 : i32> 
    %buf357 = aie.buffer(%tile_3_3) {sym_name = "buf357"} : memref<64x1xbf16, 2 : i32> 
    %buf356 = aie.buffer(%tile_3_3) {sym_name = "buf356"} : memref<64x1xbf16, 2 : i32> 
    %buf355 = aie.buffer(%tile_3_3) {sym_name = "buf355"} : memref<64x1xbf16, 2 : i32> 
    %buf354 = aie.buffer(%tile_3_3) {sym_name = "buf354"} : memref<64x1xbf16, 2 : i32> 
    %buf353 = aie.buffer(%tile_3_3) {sym_name = "buf353"} : memref<64x1xbf16, 2 : i32> 
    %buf352 = aie.buffer(%tile_3_3) {sym_name = "buf352"} : memref<64x1xbf16, 2 : i32> 
    %buf351 = aie.buffer(%tile_2_3) {sym_name = "buf351"} : memref<64x1xbf16, 2 : i32> 
    %buf350 = aie.buffer(%tile_2_3) {sym_name = "buf350"} : memref<64x1xbf16, 2 : i32> 
    %buf349 = aie.buffer(%tile_2_3) {sym_name = "buf349"} : memref<64x64xbf16, 2 : i32> 
    %buf348 = aie.buffer(%tile_2_3) {sym_name = "buf348"} : memref<64x64xbf16, 2 : i32> 
    %buf347 = aie.buffer(%tile_2_3) {sym_name = "buf347"} : memref<64x64xbf16, 2 : i32> 
    %buf346 = aie.buffer(%tile_2_3) {sym_name = "buf346"} : memref<64x64xbf16, 2 : i32> 
    %buf345 = aie.buffer(%tile_2_3) {sym_name = "buf345"} : memref<64x64xbf16, 2 : i32> 
    %buf344 = aie.buffer(%tile_2_3) {sym_name = "buf344"} : memref<64x1xbf16, 2 : i32> 
    %buf343 = aie.buffer(%tile_2_3) {sym_name = "buf343"} : memref<64x1xbf16, 2 : i32> 
    %buf342 = aie.buffer(%tile_2_3) {sym_name = "buf342"} : memref<64x64xbf16, 2 : i32> 
    %buf341 = aie.buffer(%tile_2_3) {sym_name = "buf341"} : memref<64x1xbf16, 2 : i32> 
    %buf340 = aie.buffer(%tile_2_3) {sym_name = "buf340"} : memref<64x1xbf16, 2 : i32> 
    %buf339 = aie.buffer(%tile_2_3) {sym_name = "buf339"} : memref<64x1xbf16, 2 : i32> 
    %buf338 = aie.buffer(%tile_2_3) {sym_name = "buf338"} : memref<64x1xbf16, 2 : i32> 
    %buf337 = aie.buffer(%tile_2_3) {sym_name = "buf337"} : memref<64x1xbf16, 2 : i32> 
    %buf336 = aie.buffer(%tile_2_3) {sym_name = "buf336"} : memref<64x1xbf16, 2 : i32> 
    %buf335 = aie.buffer(%tile_1_3) {sym_name = "buf335"} : memref<64x1xbf16, 2 : i32> 
    %buf334 = aie.buffer(%tile_1_3) {sym_name = "buf334"} : memref<64x1xbf16, 2 : i32> 
    %buf333 = aie.buffer(%tile_1_3) {sym_name = "buf333"} : memref<64x64xbf16, 2 : i32> 
    %buf332 = aie.buffer(%tile_1_3) {sym_name = "buf332"} : memref<64x64xbf16, 2 : i32> 
    %buf331 = aie.buffer(%tile_1_3) {sym_name = "buf331"} : memref<64x64xbf16, 2 : i32> 
    %buf330 = aie.buffer(%tile_1_3) {sym_name = "buf330"} : memref<64x64xbf16, 2 : i32> 
    %buf329 = aie.buffer(%tile_1_3) {sym_name = "buf329"} : memref<64x64xbf16, 2 : i32> 
    %buf328 = aie.buffer(%tile_1_3) {sym_name = "buf328"} : memref<64x1xbf16, 2 : i32> 
    %buf327 = aie.buffer(%tile_1_3) {sym_name = "buf327"} : memref<64x1xbf16, 2 : i32> 
    %buf326 = aie.buffer(%tile_1_3) {sym_name = "buf326"} : memref<64x64xbf16, 2 : i32> 
    %buf325 = aie.buffer(%tile_1_3) {sym_name = "buf325"} : memref<64x1xbf16, 2 : i32> 
    %buf324 = aie.buffer(%tile_1_3) {sym_name = "buf324"} : memref<64x1xbf16, 2 : i32> 
    %buf323 = aie.buffer(%tile_1_3) {sym_name = "buf323"} : memref<64x1xbf16, 2 : i32> 
    %buf322 = aie.buffer(%tile_1_3) {sym_name = "buf322"} : memref<64x1xbf16, 2 : i32> 
    %buf321 = aie.buffer(%tile_1_3) {sym_name = "buf321"} : memref<64x1xbf16, 2 : i32> 
    %buf320 = aie.buffer(%tile_1_3) {sym_name = "buf320"} : memref<64x1xbf16, 2 : i32> 
    %buf319 = aie.buffer(%tile_0_3) {sym_name = "buf319"} : memref<64x1xbf16, 2 : i32> 
    %buf318 = aie.buffer(%tile_0_3) {sym_name = "buf318"} : memref<64x1xbf16, 2 : i32> 
    %buf317 = aie.buffer(%tile_0_3) {sym_name = "buf317"} : memref<64x64xbf16, 2 : i32> 
    %buf316 = aie.buffer(%tile_0_3) {sym_name = "buf316"} : memref<64x64xbf16, 2 : i32> 
    %buf315 = aie.buffer(%tile_0_3) {sym_name = "buf315"} : memref<64x64xbf16, 2 : i32> 
    %buf314 = aie.buffer(%tile_0_3) {sym_name = "buf314"} : memref<64x64xbf16, 2 : i32> 
    %buf313 = aie.buffer(%tile_0_3) {sym_name = "buf313"} : memref<64x64xbf16, 2 : i32> 
    %buf312 = aie.buffer(%tile_0_3) {sym_name = "buf312"} : memref<64x1xbf16, 2 : i32> 
    %buf311 = aie.buffer(%tile_0_3) {sym_name = "buf311"} : memref<64x1xbf16, 2 : i32> 
    %buf310 = aie.buffer(%tile_0_3) {sym_name = "buf310"} : memref<64x64xbf16, 2 : i32> 
    %buf309 = aie.buffer(%tile_0_3) {sym_name = "buf309"} : memref<64x1xbf16, 2 : i32> 
    %buf308 = aie.buffer(%tile_0_3) {sym_name = "buf308"} : memref<64x1xbf16, 2 : i32> 
    %buf307 = aie.buffer(%tile_0_3) {sym_name = "buf307"} : memref<64x1xbf16, 2 : i32> 
    %buf306 = aie.buffer(%tile_0_3) {sym_name = "buf306"} : memref<64x1xbf16, 2 : i32> 
    %buf305 = aie.buffer(%tile_0_3) {sym_name = "buf305"} : memref<64x1xbf16, 2 : i32> 
    %buf304 = aie.buffer(%tile_0_3) {sym_name = "buf304"} : memref<64x1xbf16, 2 : i32> 
    %buf303 = aie.buffer(%tile_3_2) {sym_name = "buf303"} : memref<64x1xbf16, 2 : i32> 
    %buf302 = aie.buffer(%tile_3_2) {sym_name = "buf302"} : memref<64x1xbf16, 2 : i32> 
    %buf301 = aie.buffer(%tile_3_2) {sym_name = "buf301"} : memref<64x64xbf16, 2 : i32> 
    %buf300 = aie.buffer(%tile_3_2) {sym_name = "buf300"} : memref<64x64xbf16, 2 : i32> 
    %buf299 = aie.buffer(%tile_3_2) {sym_name = "buf299"} : memref<64x64xbf16, 2 : i32> 
    %buf298 = aie.buffer(%tile_3_2) {sym_name = "buf298"} : memref<64x64xbf16, 2 : i32> 
    %buf297 = aie.buffer(%tile_3_2) {sym_name = "buf297"} : memref<64x64xbf16, 2 : i32> 
    %buf296 = aie.buffer(%tile_3_2) {sym_name = "buf296"} : memref<64x1xbf16, 2 : i32> 
    %buf295 = aie.buffer(%tile_3_2) {sym_name = "buf295"} : memref<64x1xbf16, 2 : i32> 
    %buf294 = aie.buffer(%tile_3_2) {sym_name = "buf294"} : memref<64x64xbf16, 2 : i32> 
    %buf293 = aie.buffer(%tile_3_2) {sym_name = "buf293"} : memref<64x1xbf16, 2 : i32> 
    %buf292 = aie.buffer(%tile_3_2) {sym_name = "buf292"} : memref<64x1xbf16, 2 : i32> 
    %buf291 = aie.buffer(%tile_3_2) {sym_name = "buf291"} : memref<64x1xbf16, 2 : i32> 
    %buf290 = aie.buffer(%tile_3_2) {sym_name = "buf290"} : memref<64x1xbf16, 2 : i32> 
    %buf289 = aie.buffer(%tile_3_2) {sym_name = "buf289"} : memref<64x1xbf16, 2 : i32> 
    %buf288 = aie.buffer(%tile_3_2) {sym_name = "buf288"} : memref<64x1xbf16, 2 : i32> 
    %buf287 = aie.buffer(%tile_2_2) {sym_name = "buf287"} : memref<64x1xbf16, 2 : i32> 
    %buf286 = aie.buffer(%tile_2_2) {sym_name = "buf286"} : memref<64x1xbf16, 2 : i32> 
    %buf285 = aie.buffer(%tile_2_2) {sym_name = "buf285"} : memref<64x64xbf16, 2 : i32> 
    %buf284 = aie.buffer(%tile_2_2) {sym_name = "buf284"} : memref<64x64xbf16, 2 : i32> 
    %buf283 = aie.buffer(%tile_2_2) {sym_name = "buf283"} : memref<64x64xbf16, 2 : i32> 
    %buf282 = aie.buffer(%tile_2_2) {sym_name = "buf282"} : memref<64x64xbf16, 2 : i32> 
    %buf281 = aie.buffer(%tile_2_2) {sym_name = "buf281"} : memref<64x64xbf16, 2 : i32> 
    %buf280 = aie.buffer(%tile_2_2) {sym_name = "buf280"} : memref<64x1xbf16, 2 : i32> 
    %buf279 = aie.buffer(%tile_2_2) {sym_name = "buf279"} : memref<64x1xbf16, 2 : i32> 
    %buf278 = aie.buffer(%tile_2_2) {sym_name = "buf278"} : memref<64x64xbf16, 2 : i32> 
    %buf277 = aie.buffer(%tile_2_2) {sym_name = "buf277"} : memref<64x1xbf16, 2 : i32> 
    %buf276 = aie.buffer(%tile_2_2) {sym_name = "buf276"} : memref<64x1xbf16, 2 : i32> 
    %buf275 = aie.buffer(%tile_2_2) {sym_name = "buf275"} : memref<64x1xbf16, 2 : i32> 
    %buf274 = aie.buffer(%tile_2_2) {sym_name = "buf274"} : memref<64x1xbf16, 2 : i32> 
    %buf273 = aie.buffer(%tile_2_2) {sym_name = "buf273"} : memref<64x1xbf16, 2 : i32> 
    %buf272 = aie.buffer(%tile_2_2) {sym_name = "buf272"} : memref<64x1xbf16, 2 : i32> 
    %buf271 = aie.buffer(%tile_1_2) {sym_name = "buf271"} : memref<64x1xbf16, 2 : i32> 
    %buf270 = aie.buffer(%tile_1_2) {sym_name = "buf270"} : memref<64x1xbf16, 2 : i32> 
    %buf269 = aie.buffer(%tile_1_2) {sym_name = "buf269"} : memref<64x64xbf16, 2 : i32> 
    %buf268 = aie.buffer(%tile_1_2) {sym_name = "buf268"} : memref<64x64xbf16, 2 : i32> 
    %buf267 = aie.buffer(%tile_1_2) {sym_name = "buf267"} : memref<64x64xbf16, 2 : i32> 
    %buf266 = aie.buffer(%tile_1_2) {sym_name = "buf266"} : memref<64x64xbf16, 2 : i32> 
    %buf265 = aie.buffer(%tile_1_2) {sym_name = "buf265"} : memref<64x64xbf16, 2 : i32> 
    %buf264 = aie.buffer(%tile_1_2) {sym_name = "buf264"} : memref<64x1xbf16, 2 : i32> 
    %buf263 = aie.buffer(%tile_1_2) {sym_name = "buf263"} : memref<64x1xbf16, 2 : i32> 
    %buf262 = aie.buffer(%tile_1_2) {sym_name = "buf262"} : memref<64x64xbf16, 2 : i32> 
    %buf261 = aie.buffer(%tile_1_2) {sym_name = "buf261"} : memref<64x1xbf16, 2 : i32> 
    %buf260 = aie.buffer(%tile_1_2) {sym_name = "buf260"} : memref<64x1xbf16, 2 : i32> 
    %buf259 = aie.buffer(%tile_1_2) {sym_name = "buf259"} : memref<64x1xbf16, 2 : i32> 
    %buf258 = aie.buffer(%tile_1_2) {sym_name = "buf258"} : memref<64x1xbf16, 2 : i32> 
    %buf257 = aie.buffer(%tile_1_2) {sym_name = "buf257"} : memref<64x1xbf16, 2 : i32> 
    %buf256 = aie.buffer(%tile_1_2) {sym_name = "buf256"} : memref<64x1xbf16, 2 : i32> 
    %buf255 = aie.buffer(%tile_0_2) {sym_name = "buf255"} : memref<64x1xbf16, 2 : i32> 
    %buf254 = aie.buffer(%tile_0_2) {sym_name = "buf254"} : memref<64x1xbf16, 2 : i32> 
    %buf253 = aie.buffer(%tile_0_2) {sym_name = "buf253"} : memref<64x64xbf16, 2 : i32> 
    %buf252 = aie.buffer(%tile_0_2) {sym_name = "buf252"} : memref<64x64xbf16, 2 : i32> 
    %buf251 = aie.buffer(%tile_0_2) {sym_name = "buf251"} : memref<64x64xbf16, 2 : i32> 
    %buf250 = aie.buffer(%tile_0_2) {sym_name = "buf250"} : memref<64x64xbf16, 2 : i32> 
    %buf249 = aie.buffer(%tile_0_2) {sym_name = "buf249"} : memref<64x64xbf16, 2 : i32> 
    %buf248 = aie.buffer(%tile_0_2) {sym_name = "buf248"} : memref<64x1xbf16, 2 : i32> 
    %buf247 = aie.buffer(%tile_0_2) {sym_name = "buf247"} : memref<64x1xbf16, 2 : i32> 
    %buf246 = aie.buffer(%tile_0_2) {sym_name = "buf246"} : memref<64x64xbf16, 2 : i32> 
    %buf245 = aie.buffer(%tile_0_2) {sym_name = "buf245"} : memref<64x1xbf16, 2 : i32> 
    %buf244 = aie.buffer(%tile_0_2) {sym_name = "buf244"} : memref<64x1xbf16, 2 : i32> 
    %buf243 = aie.buffer(%tile_0_2) {sym_name = "buf243"} : memref<64x1xbf16, 2 : i32> 
    %buf242 = aie.buffer(%tile_0_2) {sym_name = "buf242"} : memref<64x1xbf16, 2 : i32> 
    %buf241 = aie.buffer(%tile_0_2) {sym_name = "buf241"} : memref<64x1xbf16, 2 : i32> 
    %buf240 = aie.buffer(%tile_0_2) {sym_name = "buf240"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2x256x64xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2x256x64xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2x256x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<2x256x64xbf16>
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_74, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf462 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_75, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf463 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_73, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf465) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf467) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf466) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf462, %buf461) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf464 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf464 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf461, %buf462, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_73, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf464 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf466, %buf460, %buf459) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf459, %buf465) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf464 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf463, %buf465) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf467, %buf459, %buf460) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf460, %buf467) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf465 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf466 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_81 = memref.collapse_shape %buf467 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_71, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf453 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf454 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_70, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf456) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf458) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf457) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf453, %buf452) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf455 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf455 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf452, %buf453, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_70, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf455 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf457, %buf451, %buf450) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf450, %buf456) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf455 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf454, %buf456) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf458, %buf450, %buf451) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf451, %buf458) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf456 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf457 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_81 = memref.collapse_shape %buf458 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_68, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf444 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_69, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf445 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_67, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf447) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf449) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf448) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf444, %buf443) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf446 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf446 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf443, %buf444, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf446 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf448, %buf442, %buf441) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf441, %buf447) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf446 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf445, %buf447) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf449, %buf441, %buf442) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf442, %buf449) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf447 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf448 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_81 = memref.collapse_shape %buf449 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_65, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf435 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_66, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf436 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_64, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf438) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf440) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf439) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf435, %buf434) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf437 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf437 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf434, %buf435, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_64, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf437 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf439, %buf433, %buf432) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf432, %buf438) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf437 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf436, %buf438) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf440, %buf432, %buf433) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf433, %buf440) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf438 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_80 = memref.collapse_shape %buf439 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_81 = memref.collapse_shape %buf440 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf426 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_63, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf427 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_61, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf429) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf431) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf430) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf426, %buf425) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf428 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf428 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf425, %buf426, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_61, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf428 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf430, %buf424, %buf423) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf423, %buf429) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf428 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf427, %buf429) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf431, %buf423, %buf424) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf424, %buf431) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf422 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf421 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf420 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf430, %buf419) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf421, %buf430) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf421, %buf430, %buf418) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf419, %buf430, %buf417) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf418, %buf422) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf417, %buf429) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf429, %buf422) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf416) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf420, %buf418, %buf416) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf431, %buf417, %buf416) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf416, %buf420) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf422 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf430 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf420 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_59, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf410 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf411 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_58, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf413) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf415) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf414) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf410, %buf409) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf412 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf412 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf409, %buf410, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_58, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf412 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf414, %buf408, %buf407) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf407, %buf413) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf412 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf411, %buf413) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf415, %buf407, %buf408) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf408, %buf415) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf406 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf405 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf404 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf414, %buf403) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf405, %buf414) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf405, %buf414, %buf402) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf403, %buf414, %buf401) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf402, %buf406) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf401, %buf413) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf413, %buf406) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf400) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf404, %buf402, %buf400) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf415, %buf401, %buf400) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf400, %buf404) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf406 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf414 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf404 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_56, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf394 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf395 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf397) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf399) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf398) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf394, %buf393) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf396 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf396 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf393, %buf394, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf396 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf398, %buf392, %buf391) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf391, %buf397) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf396 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf395, %buf397) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf399, %buf391, %buf392) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf392, %buf399) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf390 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf389 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf388 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf398, %buf387) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf389, %buf398) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf389, %buf398, %buf386) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf387, %buf398, %buf385) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf386, %buf390) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf385, %buf397) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf397, %buf390) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf384) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf388, %buf386, %buf384) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf399, %buf385, %buf384) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf384, %buf388) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf390 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf398 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf388 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf378 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf379 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_52, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf381) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf383) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf382) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf378, %buf377) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf380 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf380 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf377, %buf378, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf380 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf382, %buf376, %buf375) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf375, %buf381) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf380 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf379, %buf381) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf383, %buf375, %buf376) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf376, %buf383) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf374 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf372 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf382, %buf371) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf373, %buf382) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf373, %buf382, %buf370) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf371, %buf382, %buf369) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf370, %buf374) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf369, %buf381) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf381, %buf374) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf368) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf372, %buf370, %buf368) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf383, %buf369, %buf368) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf368, %buf372) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf374 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf382 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf372 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf362 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_51, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf363 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_49, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf365) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf367) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf366) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf362, %buf361) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf364 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf364 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf361, %buf362, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf364 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf366, %buf360, %buf359) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf359, %buf365) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf364 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf363, %buf365) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf367, %buf359, %buf360) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf360, %buf367) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf358 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf357 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf356 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf366, %buf355) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf357, %buf366) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf357, %buf366, %buf354) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf355, %buf366, %buf353) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf354, %buf358) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf353, %buf365) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf365, %buf358) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf352) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf356, %buf354, %buf352) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf367, %buf353, %buf352) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf352, %buf356) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf358 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf366 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf356 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf346 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf347 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf349) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf351) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf350) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf346, %buf345) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf348 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf348 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf345, %buf346, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf348 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf350, %buf344, %buf343) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf343, %buf349) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf348 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf347, %buf349) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf351, %buf343, %buf344) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf344, %buf351) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf342 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf341 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf340 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf350, %buf339) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf341, %buf350) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf341, %buf350, %buf338) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf339, %buf350, %buf337) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf338, %buf342) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf337, %buf349) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf349, %buf342) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf336) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf340, %buf338, %buf336) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf351, %buf337, %buf336) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf336, %buf340) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf342 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf350 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf340 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_44, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf330 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf331 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_43, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf333) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf335) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf334) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf330, %buf329) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf332 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf332 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf329, %buf330, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf332 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf334, %buf328, %buf327) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf327, %buf333) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf332 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf331, %buf333) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf335, %buf327, %buf328) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf328, %buf335) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf326 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf325 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf324 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf334, %buf323) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf325, %buf334) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf325, %buf334, %buf322) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf323, %buf334, %buf321) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf322, %buf326) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf321, %buf333) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf333, %buf326) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf320) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf324, %buf322, %buf320) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf335, %buf321, %buf320) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf320, %buf324) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf326 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf334 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf324 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf314 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf315 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_40, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf317) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf319) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf318) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf314, %buf313) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf316 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf316 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf313, %buf314, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_40, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf316 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf318, %buf312, %buf311) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf311, %buf317) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf316 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf315, %buf317) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf319, %buf311, %buf312) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf312, %buf319) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf310 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf309 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf308 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf318, %buf307) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf309, %buf318) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf309, %buf318, %buf306) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf307, %buf318, %buf305) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf306, %buf310) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf305, %buf317) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf317, %buf310) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf304) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf308, %buf306, %buf304) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf319, %buf305, %buf304) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf304, %buf308) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf310 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_82[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_83 = memref.collapse_shape %buf318 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf308 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf294 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf299 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_35, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_2_38, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf301) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf303) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf302) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf298, %buf297) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf300 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf300 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf297, %buf298, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf300 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf302, %buf296, %buf295) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf295, %buf301) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf300 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf299, %buf301) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf303, %buf295, %buf296) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf296, %buf303) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf294 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf293 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf292 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf302, %buf291) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf293, %buf302) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf293, %buf302, %buf290) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf291, %buf302, %buf289) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf290, %buf294) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf289, %buf301) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf301, %buf294) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf288) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf292, %buf290, %buf288) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf303, %buf289, %buf288) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf288, %buf292) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf292, %buf294) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_39, Release, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf278 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf282 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf283 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_30, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_2_33, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf285) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf287) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf286) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf282, %buf281) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf284 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf284 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf281, %buf282, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf284 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf286, %buf280, %buf279) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf279, %buf285) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf284 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf283, %buf285) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf287, %buf279, %buf280) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf280, %buf287) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf278 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf277 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf276 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf286, %buf275) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf277, %buf286) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf277, %buf286, %buf274) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf275, %buf286, %buf273) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf274, %buf278) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf273, %buf285) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf285, %buf278) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf272) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf276, %buf274, %buf272) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf287, %buf273, %buf272) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf272, %buf276) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf276, %buf278) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_34, Release, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf262 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf266 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_27, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf267 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_25, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_28, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf269) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf271) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf270) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf266, %buf265) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf268 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf268 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf265, %buf266, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf268 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf270, %buf264, %buf263) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf263, %buf269) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf268 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf267, %buf269) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf271, %buf263, %buf264) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf264, %buf271) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf262 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf261 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf260 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf270, %buf259) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf261, %buf270) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf261, %buf270, %buf258) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf259, %buf270, %buf257) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf258, %buf262) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf257, %buf269) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf269, %buf262) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf256) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf260, %buf258, %buf256) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf271, %buf257, %buf256) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf256, %buf260) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf260, %buf262) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_29, Release, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf246 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_23, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf250 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf251 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_20, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_23, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf253) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf255) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf254) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf250, %buf249) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf252 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf252 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf249, %buf250, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      %collapse_shape_77 = memref.collapse_shape %buf252 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf254, %buf248, %buf247) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf247, %buf253) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf252 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf251, %buf253) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf255, %buf247, %buf248) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf248, %buf255) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_79 = memref.collapse_shape %buf246 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_79[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_80 = memref.collapse_shape %buf245 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_80[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_81 = memref.collapse_shape %buf244 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_81[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf254, %buf243) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf245, %buf254) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf245, %buf254, %buf242) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf243, %buf254, %buf241) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf242, %buf246) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf241, %buf253) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf253, %buf246) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf240) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf244, %buf242, %buf240) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf255, %buf241, %buf240) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf240, %buf244) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf244, %buf246) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_24, Release, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
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
      aie.dma_bd(%buf471 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf479 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_16, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf475 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf479 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_17, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf475 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_15, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_0_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf471 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_19, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf470 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf478 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_11, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf474 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf478 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_12, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf474 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_10, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_1_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf470 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_14, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf469 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf477 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_6, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf473 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf477 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_7, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf473 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_2_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf469 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf468 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb11
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf476 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf472 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf476 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf472 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %5 = aie.dma_start(S2MM, 2, ^bb12, ^bb2)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_3_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf468 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
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
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 33 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 36 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 39 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 42 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 45 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 48 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 51 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 54 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 57 : i64, location = 0 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 60 : i64, location = 1 : i64, row = -1 : i64}, {channel = 3 : i64, col = 2 : i64, id = 63 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 66 : i64, location = 3 : i64, row = -1 : i64}], sym_name = "attn_seg_0_0"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 33 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 36 : i64, location = 0 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 39 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 1 : i64, id = 42 : i64, location = 1 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 45 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 48 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 51 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 3 : i64, id = 54 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 57 : i64, location = 0 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 60 : i64, location = 1 : i64, row = -1 : i64}, {channel = 3 : i64, col = 2 : i64, id = 63 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 66 : i64, location = 3 : i64, row = -1 : i64}], sym_name = "attn_seg_1_0"}{
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
  func.func @attention_bf16(%arg0: memref<2x256x64xbf16>, %arg1: memref<2x256x64xbf16>, %arg2: memref<2x256x64xbf16>, %arg3: memref<2x256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16>, memref<2x256x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QKIn_0[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_0]) {id = 1 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %3 = air.channel.put async  @QKIn_1[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_0]) {id = 2 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %4 = air.channel.put async  @QKIn_2[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_0]) {id = 3 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %5 = air.channel.put async  @QKIn_3[%c0] (%arg8[%c0, %1] [%c256, %c64] [%c64, %c1_0]) {id = 4 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %6 = affine.apply #map1()[%arg5]
      %7 = air.channel.put async  @QKIn_0[%c0] (%arg9[%c0, %6] [%c64, %c64] [%c64, %c1_0]) {id = 5 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %8 = affine.apply #map2()[%arg5]
      %9 = air.channel.put async  @QKIn_1[%c0] (%arg9[%c0, %8] [%c64, %c64] [%c64, %c1_0]) {id = 6 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %10 = affine.apply #map3()[%arg5]
      %11 = air.channel.put async  @QKIn_2[%c0] (%arg9[%c0, %10] [%c64, %c64] [%c64, %c1_0]) {id = 7 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %12 = affine.apply #map4()[%arg5]
      %13 = air.channel.put async  @QKIn_3[%c0] (%arg9[%c0, %12] [%c64, %c64] [%c64, %c1_0]) {id = 8 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %14 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %6] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 9 : i32, metadataArray = [{base = "air_VIn_0_0_0", index = 0 : i32}, {base = "air_VIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %15 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %8] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 10 : i32, metadataArray = [{base = "air_VIn_1_0_0", index = 0 : i32}, {base = "air_VIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %16 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %10] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 11 : i32, metadataArray = [{base = "air_VIn_2_0_0", index = 0 : i32}, {base = "air_VIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %17 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %12] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 12 : i32, metadataArray = [{base = "air_VIn_3_0_0", index = 0 : i32}, {base = "air_VIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %18 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %1] [%c64, %c64] [%c64, %c1_0]) {id = 13 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %19 = air.channel.get async  @channel_0[%c1_0, %c0] (%arg11[%c64, %1] [%c64, %c64] [%c64, %c1_0]) {id = 14 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %20 = air.channel.get async  @channel_0[%c2, %c0] (%arg11[%c128, %1] [%c64, %c64] [%c64, %c1_0]) {id = 15 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %21 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %1] [%c64, %c64] [%c64, %c1_0]) {id = 16 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %22 = affine.apply #map5()[%arg5, %arg4]
      %23 = air.channel.put async  @QKIn_0[%c1_0] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_0]) {id = 17 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %24 = air.channel.put async  @QKIn_1[%c1_0] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_0]) {id = 18 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %25 = air.channel.put async  @QKIn_2[%c1_0] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_0]) {id = 19 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %26 = air.channel.put async  @QKIn_3[%c1_0] (%arg8[%c0, %22] [%c256, %c64] [%c64, %c1_0]) {id = 20 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %27 = affine.apply #map6()[%arg5]
      %28 = air.channel.put async  @QKIn_0[%c1_0] (%arg9[%c0, %27] [%c64, %c64] [%c64, %c1_0]) {id = 21 : i32, metadataArray = [{base = "air_QKIn_0_0_0", index = 0 : i32}, {base = "air_QKIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %29 = affine.apply #map7()[%arg5]
      %30 = air.channel.put async  @QKIn_1[%c1_0] (%arg9[%c0, %29] [%c64, %c64] [%c64, %c1_0]) {id = 22 : i32, metadataArray = [{base = "air_QKIn_1_0_0", index = 0 : i32}, {base = "air_QKIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %31 = affine.apply #map8()[%arg5]
      %32 = air.channel.put async  @QKIn_2[%c1_0] (%arg9[%c0, %31] [%c64, %c64] [%c64, %c1_0]) {id = 23 : i32, metadataArray = [{base = "air_QKIn_2_0_0", index = 0 : i32}, {base = "air_QKIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %33 = affine.apply #map9()[%arg5]
      %34 = air.channel.put async  @QKIn_3[%c1_0] (%arg9[%c0, %33] [%c64, %c64] [%c64, %c1_0]) {id = 24 : i32, metadataArray = [{base = "air_QKIn_3_0_0", index = 0 : i32}, {base = "air_QKIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %35 = air.channel.put async  @VIn_0[%c1_0] (%arg10[%c0, %c0, %27] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 25 : i32, metadataArray = [{base = "air_VIn_0_0_0", index = 0 : i32}, {base = "air_VIn_0_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %36 = air.channel.put async  @VIn_1[%c1_0] (%arg10[%c0, %c0, %29] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 26 : i32, metadataArray = [{base = "air_VIn_1_0_0", index = 0 : i32}, {base = "air_VIn_1_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %37 = air.channel.put async  @VIn_2[%c1_0] (%arg10[%c0, %c0, %31] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 27 : i32, metadataArray = [{base = "air_VIn_2_0_0", index = 0 : i32}, {base = "air_VIn_2_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %38 = air.channel.put async  @VIn_3[%c1_0] (%arg10[%c0, %c0, %33] [%c1_0, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 28 : i32, metadataArray = [{base = "air_VIn_3_0_0", index = 0 : i32}, {base = "air_VIn_3_1_0_0", index = 1 : i32}]} : (memref<2x256x64xbf16>)
      %39 = air.channel.get async  @channel_0[%c0, %c1_0] (%arg11[%c0, %22] [%c64, %c64] [%c64, %c1_0]) {id = 29 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %40 = air.channel.get async  @channel_0[%c1_0, %c1_0] (%arg11[%c64, %22] [%c64, %c64] [%c64, %c1_0]) {id = 30 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %41 = air.channel.get async  @channel_0[%c2, %c1_0] (%arg11[%c128, %22] [%c64, %c64] [%c64, %c1_0]) {id = 31 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %42 = air.channel.get async  @channel_0[%c3, %c1_0] (%arg11[%c192, %22] [%c64, %c64] [%c64, %c1_0]) {id = 32 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<2x256x64xbf16>)
      %43 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c3_1 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c64_3 = arith.constant 64 : index
        %c8 = arith.constant 8 : index
        %c1_4 = arith.constant 1 : index
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
        %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_22, %results_23 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_24, %results_25 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_26, %results_27 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %44 = scf.for %arg16 = %c0_5 to %c4 step %c1_4 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %74 = air.channel.get async [%arg17]  @QKIn_0[%arg12] (%results[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %75 = arith.cmpi eq, %arg12, %c0_5 : index
          %76 = scf.if %75 -> (!air.async.token) {
            %77 = air.channel.put async [%74]  @QK2L1_0_0[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %77 : !air.async.token
          } else {
            %77 = air.channel.put async [%74]  @QK2L1_0_1[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %77 : !air.async.token
          }
          scf.yield %76 : !air.async.token
        }
        %45 = air.channel.get async [%44]  @QKIn_0[%arg12] (%results[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
        %46 = arith.cmpi eq, %arg12, %c0_5 : index
        %47 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%45]  @QK2L1_0_0[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%45]  @QK2L1_0_1[%c0_5, %c0_5, %c0_5] (%results[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %48 = scf.for %arg16 = %c0_5 to %c4 step %c1_4 iter_args(%arg17 = %async_token_6) -> (!air.async.token) {
          %74 = air.channel.get async [%arg17]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %75 = scf.if %46 -> (!air.async.token) {
            %76 = air.channel.put async [%74]  @QK2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          } else {
            %76 = air.channel.put async [%74]  @QK2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          }
          scf.yield %75 : !air.async.token
        }
        %49 = air.channel.get async [%48]  @QKIn_1[%arg12] (%results_7[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
        %50 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%49]  @QK2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%49]  @QK2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_7[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %51 = scf.for %arg16 = %c0_5 to %c4 step %c1_4 iter_args(%arg17 = %async_token_8) -> (!air.async.token) {
          %74 = air.channel.get async [%arg17]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
          %75 = scf.if %46 -> (!air.async.token) {
            %76 = air.channel.put async [%74]  @QK2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          } else {
            %76 = air.channel.put async [%74]  @QK2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          }
          scf.yield %75 : !air.async.token
        }
        %52 = air.channel.get async [%51]  @QKIn_2[%arg12] (%results_9[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
        %53 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%52]  @QK2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%52]  @QK2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_9[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %54 = scf.for %arg16 = %c0_5 to %c4 step %c1_4 iter_args(%arg17 = %async_token_10) -> (!air.async.token) {
          %74 = air.channel.get async [%arg17]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
          %75 = scf.if %46 -> (!air.async.token) {
            %76 = air.channel.put async [%74]  @QK2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          } else {
            %76 = air.channel.put async [%74]  @QK2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 53 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %76 : !air.async.token
          }
          scf.yield %75 : !air.async.token
        }
        %55 = air.channel.get async [%54]  @QKIn_3[%arg12] (%results_11[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 1 : i32>)
        %56 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%55]  @QK2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 55 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%55]  @QK2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_11[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 56 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %57 = air.channel.get async [%async_token_12]  @VIn_0[%arg12] (%results_13[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 1 : i32>)
        %58 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%57]  @V2L1_0_0[%c0_5, %c0_5, %c0_5] (%results_13[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 58 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%57]  @V2L1_0_1[%c0_5, %c0_5, %c0_5] (%results_13[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 59 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %59 = air.channel.get async [%async_token_14]  @VIn_1[%arg12] (%results_15[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 1 : i32>)
        %60 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%59]  @V2L1_1_0[%c0_5, %c0_5, %c0_5] (%results_15[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 61 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%59]  @V2L1_1_1[%c0_5, %c0_5, %c0_5] (%results_15[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 62 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %61 = air.channel.get async [%async_token_16]  @VIn_2[%arg12] (%results_17[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 1 : i32>)
        %62 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%61]  @V2L1_2_0[%c0_5, %c0_5, %c0_5] (%results_17[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 64 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%61]  @V2L1_2_1[%c0_5, %c0_5, %c0_5] (%results_17[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 65 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %63 = air.channel.get async [%async_token_18]  @VIn_3[%arg12] (%results_19[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 1 : i32>)
        %64 = scf.if %46 -> (!air.async.token) {
          %74 = air.channel.put async [%63]  @V2L1_3_0[%c0_5, %c0_5, %c0_5] (%results_19[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 67 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        } else {
          %74 = air.channel.put async [%63]  @V2L1_3_1[%c0_5, %c0_5, %c0_5] (%results_19[%c0_5, %c0_5, %c0_5] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_4]) {id = 68 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %74 : !air.async.token
        }
        %65 = air.channel.get async [%async_token_20]  @Gp2L2[%c0_5, %c0_5] (%results_21[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 1 : i32>)
        %66 = air.channel.get async [%async_token_22]  @Gp2L2[%c1_4, %c0_5] (%results_23[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 1 : i32>)
        %67 = air.channel.get async [%async_token_24]  @Gp2L2[%c2_2, %c0_5] (%results_25[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 1 : i32>)
        %68 = air.channel.get async [%async_token_26]  @Gp2L2[%c3_1, %c0_5] (%results_27[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 1 : i32>)
        %69 = air.channel.put async [%65]  @channel_0[%c0_5, %arg12] (%results_21[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 1 : i32>)
        %70 = air.channel.put async [%66]  @channel_0[%c1_4, %arg12] (%results_23[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 1 : i32>)
        %71 = air.channel.put async [%67]  @channel_0[%c2_2, %arg12] (%results_25[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 1 : i32>)
        %72 = air.channel.put async [%68]  @channel_0[%c3_1, %arg12] (%results_27[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 1 : i32>)
        %73 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c64_40 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c1_41 = arith.constant 1 : index
          %c0_42 = arith.constant 0 : index
          %c8_43 = arith.constant 8 : index
          %c512 = arith.constant 512 : index
          %async_token_44, %results_45 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_46, %results_47 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_48, %results_49 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_50, %results_51 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_52, %results_53 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_54, %results_55 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_56, %results_57 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_58 = air.execute [%async_token_48] {
            func.call @zero_fill_gp_bf16(%results_49) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_59 = air.execute [%async_token_44] {
            func.call @zero_fill_sp_bf16(%results_45) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_60 = air.execute [%async_token_46] {
            func.call @neg_inf_fill_up_bf16(%results_47) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %74]  @QK2L1_1_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %74]  @QK2L1_1_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %75]  @QK2L1_2_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %75]  @QK2L1_2_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %77 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %76]  @QK2L1_3_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %76]  @QK2L1_3_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %78 = arith.index_cast %arg16 : index to i32
          %79 = arith.cmpi eq, %78, %c0_i32 : i32
          scf.if %79 {
            %async_token_81 = air.execute [%async_token_54, %async_token_56, %77] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %80 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %81 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %80]  @QK2L1_1_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %80]  @QK2L1_1_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %82 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %81]  @QK2L1_2_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %81]  @QK2L1_2_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %83 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %82]  @QK2L1_3_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %82]  @QK2L1_3_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %84 = arith.cmpi eq, %78, %c1_i32 : i32
          scf.if %84 {
            %async_token_81 = air.execute [%async_token_54, %async_token_56, %83] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %85 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %86 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %85]  @QK2L1_1_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %85]  @QK2L1_1_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %87 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %86]  @QK2L1_2_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %86]  @QK2L1_2_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %88 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %87]  @QK2L1_3_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %87]  @QK2L1_3_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %89 = arith.cmpi eq, %78, %c2_i32 : i32
          scf.if %89 {
            %async_token_81 = air.execute [%async_token_54, %async_token_56, %88] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %90 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 102 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %91 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %90]  @QK2L1_1_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 103 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %90]  @QK2L1_1_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %92 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %91]  @QK2L1_2_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 105 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %91]  @QK2L1_2_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 106 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %93 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %92]  @QK2L1_3_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %92]  @QK2L1_3_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 108 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %94 = arith.cmpi eq, %78, %c3_i32 : i32
          scf.if %94 {
            %async_token_81 = air.execute [%async_token_54, %async_token_56, %93] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %async_token_61 = air.execute [%async_token_50] {
            %collapse_shape = memref.collapse_shape %results_51 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          %95 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 109 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %96 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %95]  @QK2L1_1_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 111 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %95]  @QK2L1_1_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 112 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %97 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %96]  @QK2L1_2_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %96]  @QK2L1_2_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 114 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %98 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_54, %97]  @QK2L1_3_0[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 115 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_54, %97]  @QK2L1_3_1[%c0_42, %arg17, %arg16] (%results_55[] [] []) {id = 116 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %async_token_62 = air.execute [%async_token_61, %98, %async_token_56, %async_token_54] {
            %collapse_shape = memref.collapse_shape %results_51 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%results_57, %results_55, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %99 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_52]  @V2L1_0_0[%c0_42, %arg17, %arg16] (%results_53[] [] []) {id = 117 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_52]  @V2L1_0_1[%c0_42, %arg17, %arg16] (%results_53[] [] []) {id = 118 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %100 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_52, %99]  @V2L1_1_0[%c0_42, %arg17, %arg16] (%results_53[] [] []) {id = 119 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_52, %99]  @V2L1_1_1[%c0_42, %arg17, %arg16] (%results_53[] [] []) {id = 120 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %101 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_52, %100]  @V2L1_2_0[%c0_42, %arg17, %arg16] (%results_53[] [] []) {id = 121 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_52, %100]  @V2L1_2_1[%c0_42, %arg17, %arg16] (%results_53[] [] []) {id = 122 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %102 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.cmpi eq, %arg20, %c0_42 : index
            %105 = scf.if %104 -> (!air.async.token) {
              %106 = air.channel.get async [%async_token_52, %101]  @V2L1_3_0[%c0_42, %arg17, %arg16] (%results_53[] [] []) {id = 123 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            } else {
              %106 = air.channel.get async [%async_token_52, %101]  @V2L1_3_1[%c0_42, %arg17, %arg16] (%results_53[] [] []) {id = 124 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %106 : !air.async.token
            }
            affine.yield %105 : !air.async.token
          } else {
            %104 = air.wait_all async 
            affine.yield %104 : !air.async.token
          }
          %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_67 = air.execute [%async_token_60, %async_token_62, %async_token_63, %async_token_65] {
            %collapse_shape = memref.collapse_shape %results_51 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %results_47, %results_64, %results_66) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_68 = air.execute [%async_token_58, %async_token_67] {
            func.call @mul_r_gp(%results_66, %results_49) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_69 = air.execute [%102, %async_token_68, %async_token_52, %async_token_50] {
            %collapse_shape = memref.collapse_shape %results_51 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %results_53, %results_49) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_70 = air.execute [%async_token_59, %async_token_68] {
            func.call @accum_sp_r_s(%results_45, %results_66, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_71 = air.execute [%async_token_70] {
            func.call @vector_copy_32elems(%c0_i32, %results_64, %results_45) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_72 = air.execute [%async_token_71] {
            memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_73 = air.execute [%async_token_70] {
            memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
          }
          %103 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %104 = arith.subi %arg17, %c1_41 : index
            %105 = air.channel.put async [%async_token_48, %async_token_69]  @cascade_gp[%arg16, %104] (%results_49[] [] []) {id = 125 : i32} : (memref<64x64xbf16, 2 : i32>)
            %106 = air.channel.put async [%async_token_60]  @cascade_up[%arg16, %104] (%results_47[] [] []) {id = 126 : i32} : (memref<64x1xbf16, 2 : i32>)
            %107 = air.channel.put async [%async_token_44, %async_token_71]  @cascade_sp[%arg16, %104] (%results_45[] [] []) {id = 127 : i32} : (memref<64x1xbf16, 2 : i32>)
            %108 = air.wait_all async [%105, %106, %107] 
            affine.yield %108 : !air.async.token
          } else {
            %104 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %async_token_81, %results_82 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_83, %results_84 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_85, %results_86 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %106 = air.channel.get async [%async_token_81]  @cascade_gp[%arg16, %arg17] (%results_82[] [] []) {id = 128 : i32} : (memref<64x64xbf16, 2 : i32>)
              %107 = air.channel.get async [%async_token_83]  @cascade_up[%arg16, %arg17] (%results_84[] [] []) {id = 129 : i32} : (memref<64x1xbf16, 2 : i32>)
              %108 = air.channel.get async [%async_token_85]  @cascade_sp[%arg16, %arg17] (%results_86[] [] []) {id = 130 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_87, %results_88 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_89 = air.execute [%async_token_60, %async_token_87] {
                func.call @vector_copy_32elems(%c0_i32, %results_47, %results_88) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_90 = air.execute [%107, %async_token_89] {
                func.call @maximum_up_u_bf16(%results_84, %results_47) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_91, %results_92 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_93 = air.execute [%async_token_90, %async_token_91] {
                func.call @exp_up_minus_u(%results_84, %results_47, %results_92) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_94, %results_95 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_96 = air.execute [%async_token_93, %async_token_94] {
                func.call @exp_up_minus_u(%results_88, %results_47, %results_95) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_97 = air.execute [%async_token_93, %106] {
                func.call @mul_r_gp(%results_92, %results_82) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_98 = air.execute [%async_token_69, %async_token_96] {
                func.call @mul_r_gp(%results_95, %results_49) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_99 = air.execute [%async_token_97, %async_token_98] {
                func.call @add_gp_g(%results_49, %results_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_100, %results_101 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_102 = air.execute [%async_token_100] {
                func.call @zero_fill_sp_bf16(%results_101) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_103 = air.execute [%async_token_102, %async_token_97, %108] {
                func.call @accum_sp_r_s(%results_86, %results_92, %results_101) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_104 = air.execute [%async_token_71, %async_token_98, %async_token_103] {
                func.call @accum_sp_r_s(%results_45, %results_95, %results_101) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_105 = air.execute [%async_token_104] {
                func.call @vector_copy_32elems(%c0_i32, %results_101, %results_86) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %109 = arith.subi %arg17, %c1_41 : index
              %110 = air.channel.put async [%async_token_99]  @cascade_gp[%arg16, %109] (%results_82[] [] []) {id = 131 : i32} : (memref<64x64xbf16, 2 : i32>)
              %111 = air.channel.put async [%async_token_46, %async_token_96]  @cascade_up[%arg16, %109] (%results_47[] [] []) {id = 132 : i32} : (memref<64x1xbf16, 2 : i32>)
              %112 = air.channel.put async [%async_token_105]  @cascade_sp[%arg16, %109] (%results_86[] [] []) {id = 133 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_106 = air.execute [%110] {
                memref.dealloc %results_82 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_107 = air.execute [%async_token_93] {
                memref.dealloc %results_84 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_108 = air.execute [%112] {
                memref.dealloc %results_86 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_109 = air.execute [%async_token_96] {
                memref.dealloc %results_88 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_110 = air.execute [%async_token_103] {
                memref.dealloc %results_92 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_111 = air.execute [%async_token_104] {
                memref.dealloc %results_95 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_105] {
                memref.dealloc %results_101 : memref<64x1xbf16, 2 : i32>
              }
              %113 = air.wait_all async [%110, %111, %112] 
              affine.yield %113 : !air.async.token
            } else {
              %async_token_81, %results_82 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_83, %results_84 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_85, %results_86 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %106 = air.channel.get async [%async_token_81]  @cascade_gp[%arg16, %arg17] (%results_82[] [] []) {id = 134 : i32} : (memref<64x64xbf16, 2 : i32>)
              %107 = air.channel.get async [%async_token_83]  @cascade_up[%arg16, %arg17] (%results_84[] [] []) {id = 135 : i32} : (memref<64x1xbf16, 2 : i32>)
              %108 = air.channel.get async [%async_token_85]  @cascade_sp[%arg16, %arg17] (%results_86[] [] []) {id = 136 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_87, %results_88 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_89 = air.execute [%async_token_60, %async_token_87] {
                func.call @vector_copy_32elems(%c0_i32, %results_47, %results_88) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_90 = air.execute [%107, %async_token_89] {
                func.call @maximum_up_u_bf16(%results_84, %results_47) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_91, %results_92 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_93 = air.execute [%async_token_90, %async_token_91] {
                func.call @exp_up_minus_u(%results_84, %results_47, %results_92) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_94, %results_95 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_96 = air.execute [%async_token_93, %async_token_94] {
                func.call @exp_up_minus_u(%results_88, %results_47, %results_95) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_97 = air.execute [%async_token_93, %106] {
                func.call @mul_r_gp(%results_92, %results_82) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_98 = air.execute [%async_token_69, %async_token_96] {
                func.call @mul_r_gp(%results_95, %results_49) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_99 = air.execute [%async_token_97, %async_token_98] {
                func.call @add_gp_g(%results_49, %results_82) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_100, %results_101 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_102 = air.execute [%async_token_100] {
                func.call @zero_fill_sp_bf16(%results_101) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_103 = air.execute [%async_token_102, %async_token_97, %108] {
                func.call @accum_sp_r_s(%results_86, %results_92, %results_101) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_104 = air.execute [%async_token_71, %async_token_98, %async_token_103] {
                func.call @accum_sp_r_s(%results_45, %results_95, %results_101) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_105 = air.execute [%async_token_104] {
                func.call @vector_copy_32elems(%c0_i32, %results_101, %results_86) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_106 = air.execute [%async_token_105, %async_token_99] {
                func.call @div_gp_sp(%results_86, %results_82) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %109 = air.channel.put async [%async_token_106]  @Gp2L2[%arg16, %c0_42] (%results_82[%c0_42, %c0_42, %c0_42] [%c64_40, %c8_43, %c8_43] [%c8_43, %c512, %c1_41]) {id = 137 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_107 = air.execute [%109] {
                memref.dealloc %results_82 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_108 = air.execute [%async_token_93] {
                memref.dealloc %results_84 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_109 = air.execute [%async_token_106] {
                memref.dealloc %results_86 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_110 = air.execute [%async_token_96] {
                memref.dealloc %results_88 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_111 = air.execute [%async_token_103] {
                memref.dealloc %results_92 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_104] {
                memref.dealloc %results_95 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_113 = air.execute [%async_token_105] {
                memref.dealloc %results_101 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %109 : !air.async.token
            }
            %105 = air.wait_all async [%95, %96, %97, %99, %100, %101, %async_token_69, %async_token_71] 
            affine.yield %105 : !air.async.token
          }
          %async_token_74 = air.execute [%async_token_62] {
            memref.dealloc %results_57 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_75 = air.execute [%74, %75, %76, %77, %80, %81, %82, %83, %85, %86, %87, %88, %90, %91, %92, %93, %95, %96, %97, %async_token_62] {
            memref.dealloc %results_55 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_76 = air.execute [%99, %100, %101, %async_token_69] {
            memref.dealloc %results_53 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_77 = air.execute [%async_token_67, %async_token_69] {
            memref.dealloc %results_51 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_78 = air.execute [%async_token_69, %103] {
            memref.dealloc %results_49 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_79 = air.execute [%async_token_67, %103] {
            memref.dealloc %results_47 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_80 = air.execute [%async_token_71, %103] {
            memref.dealloc %results_45 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_28 = air.execute [%47] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_29 = air.execute [%58] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_30 = air.execute [%50] {
          memref.dealloc %results_7 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_31 = air.execute [%60] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_32 = air.execute [%53] {
          memref.dealloc %results_9 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_33 = air.execute [%62] {
          memref.dealloc %results_17 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_34 = air.execute [%56] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_35 = air.execute [%64] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_36 = air.execute [%72] {
          memref.dealloc %results_27 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_37 = air.execute [%71] {
          memref.dealloc %results_25 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_38 = air.execute [%70] {
          memref.dealloc %results_23 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_39 = air.execute [%69] {
          memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
        }
        air.wait_all [%73, %async_token_28, %async_token_29, %async_token_30, %async_token_31, %async_token_32, %async_token_33, %async_token_34, %async_token_35, %async_token_36, %async_token_37, %async_token_38, %async_token_39]  {air.segment_end}
      }
    }
    return
  }
}
