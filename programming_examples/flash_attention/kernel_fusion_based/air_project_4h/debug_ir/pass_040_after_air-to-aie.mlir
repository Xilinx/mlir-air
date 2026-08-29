#loop_annotation = #llvm.loop_annotation<mustProgress = true>
#map = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 262144)>
#map2 = affine_map<()[s0] -> (s0 * 262144 + 32768)>
#map3 = affine_map<()[s0] -> (s0 * 262144 + 65536)>
#map4 = affine_map<()[s0] -> (s0 * 262144 + 98304)>
#map5 = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 16384 + 131072)>
#map6 = affine_map<()[s0] -> (s0 * 262144 + 131072)>
#map7 = affine_map<()[s0] -> (s0 * 262144 + 163840)>
#map8 = affine_map<()[s0] -> (s0 * 262144 + 196608)>
#map9 = affine_map<()[s0] -> (s0 * 262144 + 229376)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set6 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set7 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
  aie.device(npu2_4col) @attn_seg_0_0 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
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
    %lock_3_1 = aie.lock(%mem_tile_3_1, 3) {init = 1 : i32}
    %lock_3_1_0 = aie.lock(%mem_tile_3_1, 2) {init = 0 : i32}
    %lock_3_1_1 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_2 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 3) {init = 1 : i32}
    %lock_2_1_3 = aie.lock(%mem_tile_2_1, 2) {init = 0 : i32}
    %lock_2_1_4 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_5 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 3) {init = 1 : i32}
    %lock_1_1_6 = aie.lock(%mem_tile_1_1, 2) {init = 0 : i32}
    %lock_1_1_7 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_8 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_9 = aie.lock(%mem_tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_10 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_11 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_12 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_13 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_14 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_15 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_16 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_17 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_18 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_19 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_20 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_21 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_22 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_23 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_24 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_25 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_26 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_27 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_28 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_29 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_30 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_31 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_32 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_33 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_34 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_35 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_36 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_37 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_38 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_39 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_40 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 3) {init = 1 : i32}
    %lock_3_3_41 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_42 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_43 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_44 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_45 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_46 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 3) {init = 1 : i32}
    %lock_1_4_47 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_48 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_49 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 3) {init = 1 : i32}
    %lock_2_4_50 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_51 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_52 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 3) {init = 1 : i32}
    %lock_3_4_53 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_54 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_55 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_56 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_57 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_58 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 3) {init = 1 : i32}
    %lock_1_5_59 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_60 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_61 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 3) {init = 1 : i32}
    %lock_2_5_62 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_63 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_64 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 3) {init = 1 : i32}
    %lock_3_5_65 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_66 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_67 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
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
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<12x2048x64xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<12x2048x64xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<12x2048x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<12x2048x64xbf16>
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_66, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf224 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_67, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf222 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_65, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_68 = arith.constant 1 : index
      %c8_69 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf225) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf227) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf226) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_66, Release, 1)
      aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_66, Release, 1)
      aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_66, Release, 1)
      aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf224, %buf223) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_66, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_69 step %c1_68 {
        %collapse_shape_73 = memref.collapse_shape %buf221 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_5_65, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf221 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf223, %buf224, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_66, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf221 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf226, %buf220, %buf219) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf219, %buf225) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf221 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf222, %buf225) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf227, %buf219, %buf220) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf220, %buf227) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf225 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_71 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_72 = memref.collapse_shape %buf227 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf215 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_64, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf213 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_62, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_68 = arith.constant 1 : index
      %c8_69 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf216) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf218) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf217) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_63, Release, 1)
      aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_63, Release, 1)
      aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf215, %buf214) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_63, Release, 1)
      aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_63, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_69 step %c1_68 {
        %collapse_shape_73 = memref.collapse_shape %buf212 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_5_62, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf212 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf214, %buf215, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_63, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf212 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf217, %buf211, %buf210) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf210, %buf216) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf212 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf213, %buf216) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf218, %buf210, %buf211) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf211, %buf218) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf216 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_71 = memref.collapse_shape %buf217 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_72 = memref.collapse_shape %buf218 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf206 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_61, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf204 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_59, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf207) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf209) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf208) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_60, Release, 1)
      aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf206, %buf205) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_60, Release, 1)
      aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_60, Release, 1)
      aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_60, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_73 = memref.collapse_shape %buf203 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_5_59, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf203 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf205, %buf206, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_60, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf203 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf208, %buf202, %buf201) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf201, %buf207) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf203 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf204, %buf207) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf209, %buf201, %buf202) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf202, %buf209) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf207 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_71 = memref.collapse_shape %buf208 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_72 = memref.collapse_shape %buf209 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_57, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf197 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_58, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf195 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_56, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_68 = arith.constant 1 : index
      %c8_69 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf198) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf200) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf199) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf197, %buf196) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_57, Release, 1)
      aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_57, Release, 1)
      aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_57, Release, 1)
      aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_57, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_69 step %c1_68 {
        %collapse_shape_73 = memref.collapse_shape %buf194 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_5_56, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf194 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf196, %buf197, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_57, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf194 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf199, %buf193, %buf192) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf192, %buf198) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf194 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf195, %buf198) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf200, %buf192, %buf193) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf193, %buf200) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf198 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_71 = memref.collapse_shape %buf199 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_72 = memref.collapse_shape %buf200 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_54, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf188 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_55, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf186 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_53, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_68 = arith.constant 1 : index
      %c8_69 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf189) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf191) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf190) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_54, Release, 1)
      aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_54, Release, 1)
      aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_54, Release, 1)
      aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf188, %buf187) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_54, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_69 step %c1_68 {
        %collapse_shape_76 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_4_53, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf187, %buf188, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_54, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf190, %buf184, %buf183) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf183, %buf189) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf186, %buf189) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf191, %buf183, %buf184) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf184, %buf191) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf182 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf181 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf180 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      %collapse_shape_73 = memref.collapse_shape %buf182 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf190 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf180 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_51, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf172 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_52, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf170 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_50, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_68 = arith.constant 1 : index
      %c8_69 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf173) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf175) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf174) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_51, Release, 1)
      aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_51, Release, 1)
      aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf172, %buf171) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_51, Release, 1)
      aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_51, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_69 step %c1_68 {
        %collapse_shape_76 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_4_50, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf171, %buf172, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_51, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf174, %buf168, %buf167) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf167, %buf173) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf170, %buf173) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf175, %buf167, %buf168) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf168, %buf175) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf166 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf165 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf164 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      %collapse_shape_73 = memref.collapse_shape %buf166 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf174 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf164 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf156 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_49, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf154 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_47, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf157) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf159) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf158) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_48, Release, 1)
      aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf156, %buf155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_48, Release, 1)
      aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_48, Release, 1)
      aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_48, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_4_47, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf155, %buf156, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_48, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf158, %buf152, %buf151) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf151, %buf157) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf154, %buf157) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf159, %buf151, %buf152) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf152, %buf159) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf150 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf149 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf148 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      %collapse_shape_73 = memref.collapse_shape %buf150 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf158 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf148 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf140 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_46, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf138 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_44, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_68 = arith.constant 1 : index
      %c8_69 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf141) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf143) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf142) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf140, %buf139) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_45, Release, 1)
      aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_45, Release, 1)
      aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_45, Release, 1)
      aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_45, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_69 step %c1_68 {
        %collapse_shape_76 = memref.collapse_shape %buf137 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_4_44, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf137 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf139, %buf140, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_45, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf137 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf142, %buf136, %buf135) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf135, %buf141) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf137 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf138, %buf141) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf143, %buf135, %buf136) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf136, %buf143) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf134 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf133 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf132 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      %collapse_shape_73 = memref.collapse_shape %buf134 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf142 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf132 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_42, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf124 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_43, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf122 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_41, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf125) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf127) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf126) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_42, Release, 1)
      aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_42, Release, 1)
      aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_42, Release, 1)
      aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf124, %buf123) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_42, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf121 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3_41, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf121 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf123, %buf124, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_42, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf121 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf126, %buf120, %buf119) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf119, %buf125) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf121 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf122, %buf125) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf127, %buf119, %buf120) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf120, %buf127) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf118 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf117 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf116 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      %collapse_shape_73 = memref.collapse_shape %buf118 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf126 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf116 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf108 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_40, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf106 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_38, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf109) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf111) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf110) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_39, Release, 1)
      aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_39, Release, 1)
      aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf108, %buf107) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_39, Release, 1)
      aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_39, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf105 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3_38, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf105 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf107, %buf108, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_39, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf105 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf110, %buf104, %buf103) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf103, %buf109) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf105 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf106, %buf109) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf111, %buf103, %buf104) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf104, %buf111) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf102 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf101 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf100 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      %collapse_shape_73 = memref.collapse_shape %buf102 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf110 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf100 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf92 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_37, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf90 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_35, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf93) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf95) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf94) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_36, Release, 1)
      aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf92, %buf91) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_36, Release, 1)
      aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_36, Release, 1)
      aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_36, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3_35, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf91, %buf92, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_36, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf94, %buf88, %buf87) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf87, %buf93) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf90, %buf93) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf95, %buf87, %buf88) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf88, %buf95) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf86 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf85 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf84 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      %collapse_shape_73 = memref.collapse_shape %buf86 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf94 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf84 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_32, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_69 = arith.constant 1 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf77) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf79) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf78) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf76, %buf75) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_33, Release, 1)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_33, Release, 1)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_33, Release, 1)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_33, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_68 step %c1_69 {
        %collapse_shape_76 = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3_32, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf75, %buf76, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_33, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf78, %buf72, %buf71) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf71, %buf77) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf74, %buf77) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf79, %buf71, %buf72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf72, %buf79) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf70 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf69 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf68 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      %collapse_shape_73 = memref.collapse_shape %buf70 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf78 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf68 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf54 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_30, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf60 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_29, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf58 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_68 = arith.constant 1 : index
      %c8_69 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_2_30, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf61) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf63) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf62) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_28, Release, 1)
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_28, Release, 1)
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_28, Release, 1)
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf60, %buf59) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_28, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_69 step %c1_68 {
        %collapse_shape_73 = memref.collapse_shape %buf57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf59, %buf60, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_28, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf62, %buf56, %buf55) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf55, %buf61) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf58, %buf61) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf63, %buf55, %buf56) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf56, %buf63) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf54 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf53 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf52 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_3_2_31, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf38 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_25, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_23, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf44 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_24, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf42 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_22, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_68 = arith.constant 1 : index
      %c8_69 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_2_25, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf45) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf47) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf46) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_23, Release, 1)
      aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_23, Release, 1)
      aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf44, %buf43) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_23, Release, 1)
      aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_23, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_69 step %c1_68 {
        %collapse_shape_73 = memref.collapse_shape %buf41 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_22, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf41 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf43, %buf44, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_23, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf41 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf46, %buf40, %buf39) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf39, %buf45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf41 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf42, %buf45) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf47, %buf39, %buf40) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf40, %buf47) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf38 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf37 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf36 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_2_2_26, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf22 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_20, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf28 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_19, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf26 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_17, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_20, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf29) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf31) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf30) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_18, Release, 1)
      aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf28, %buf27) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_18, Release, 1)
      aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_18, Release, 1)
      aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_18, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_73 = memref.collapse_shape %buf25 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_17, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf25 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf27, %buf28, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_18, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf25 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf30, %buf24, %buf23) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf23, %buf29) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf25 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf26, %buf29) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf31, %buf23, %buf24) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf24, %buf31) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf22 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf21 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf20 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_1_2_21, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_15, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_14, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_12, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c1_68 = arith.constant 1 : index
      %c8_69 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_15, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf13) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf15) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf14) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf12, %buf11) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_13, Release, 1)
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_13, Release, 1)
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_13, Release, 1)
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_13, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_69 step %c1_68 {
        %collapse_shape_73 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_12, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf11, %buf12, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_13, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf14, %buf8, %buf7) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf7, %buf13) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf10, %buf13) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf15, %buf7, %buf8) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf8, %buf15) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf6 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf5 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf4 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_0_2_16, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
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
    air.channel @QK2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @QK2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @QK2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @QK2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
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
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_2_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_3_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_0_5, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_2_5, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_3_5, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 1)
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
      aie.use_lock(%lock_0_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf235 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf231 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf231 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_9, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf235 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_11, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf234 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_7, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_6, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf234 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_8, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf233 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf229 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf229 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf233 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf232 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf232 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @air_channel_0_0_0_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_0_0_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_0_0_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_0_0_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_QK2L1_0_0_0_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_QK2L1_0_1_0_0(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_QK2L1_0_2_0_0(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_QK2L1_0_3_0_0(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_0_0_0(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_1_0_0(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_2_0_0(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_3_0_0(%shim_noc_tile_3_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 0 : i64, segment_unroll_y = 0 : i64}
  aie.device(npu2_4col) @attn_seg_1_0 {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
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
    %lock_3_1 = aie.lock(%mem_tile_3_1, 3) {init = 1 : i32}
    %lock_3_1_0 = aie.lock(%mem_tile_3_1, 2) {init = 0 : i32}
    %lock_3_1_1 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_2 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 3) {init = 1 : i32}
    %lock_2_1_3 = aie.lock(%mem_tile_2_1, 2) {init = 0 : i32}
    %lock_2_1_4 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_5 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 3) {init = 1 : i32}
    %lock_1_1_6 = aie.lock(%mem_tile_1_1, 2) {init = 0 : i32}
    %lock_1_1_7 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_8 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_9 = aie.lock(%mem_tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_10 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_11 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_12 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_13 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_14 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_15 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_16 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_17 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_18 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_19 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_20 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_21 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_22 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_23 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_24 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_25 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_26 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_27 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_28 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_29 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_30 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_31 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_32 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_33 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_34 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_35 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_36 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_37 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_38 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_39 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_40 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 3) {init = 1 : i32}
    %lock_3_3_41 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_42 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_43 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_44 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_45 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_46 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 3) {init = 1 : i32}
    %lock_1_4_47 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_48 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_49 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 3) {init = 1 : i32}
    %lock_2_4_50 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_51 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_52 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 3) {init = 1 : i32}
    %lock_3_4_53 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_54 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_55 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_56 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_57 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_58 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 3) {init = 1 : i32}
    %lock_1_5_59 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_60 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_61 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 3) {init = 1 : i32}
    %lock_2_5_62 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_63 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_64 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 3) {init = 1 : i32}
    %lock_3_5_65 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_66 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_67 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf471 = aie.buffer(%mem_tile_0_1) {sym_name = "buf471"} : memref<64x64xbf16, 1 : i32> 
    %buf470 = aie.buffer(%mem_tile_1_1) {sym_name = "buf470"} : memref<64x64xbf16, 1 : i32> 
    %buf469 = aie.buffer(%mem_tile_2_1) {sym_name = "buf469"} : memref<64x64xbf16, 1 : i32> 
    %buf468 = aie.buffer(%mem_tile_3_1) {sym_name = "buf468"} : memref<64x64xbf16, 1 : i32> 
    %buf467 = aie.buffer(%mem_tile_0_1) {sym_name = "buf467"} : memref<64x64xbf16, 1 : i32> 
    %buf466 = aie.buffer(%mem_tile_1_1) {sym_name = "buf466"} : memref<64x64xbf16, 1 : i32> 
    %buf465 = aie.buffer(%mem_tile_2_1) {sym_name = "buf465"} : memref<64x64xbf16, 1 : i32> 
    %buf464 = aie.buffer(%mem_tile_3_1) {sym_name = "buf464"} : memref<64x64xbf16, 1 : i32> 
    %buf463 = aie.buffer(%tile_3_5) {sym_name = "buf463"} : memref<64x1xbf16, 2 : i32> 
    %buf462 = aie.buffer(%tile_3_5) {sym_name = "buf462"} : memref<64x1xbf16, 2 : i32> 
    %buf461 = aie.buffer(%tile_3_5) {sym_name = "buf461"} : memref<64x64xbf16, 2 : i32> 
    %buf460 = aie.buffer(%tile_3_5) {sym_name = "buf460"} : memref<64x64xbf16, 2 : i32> 
    %buf459 = aie.buffer(%tile_3_5) {sym_name = "buf459"} : memref<64x64xbf16, 2 : i32> 
    %buf458 = aie.buffer(%tile_3_5) {sym_name = "buf458"} : memref<64x64xbf16, 2 : i32> 
    %buf457 = aie.buffer(%tile_3_5) {sym_name = "buf457"} : memref<64x64xbf16, 2 : i32> 
    %buf456 = aie.buffer(%tile_3_5) {sym_name = "buf456"} : memref<64x1xbf16, 2 : i32> 
    %buf455 = aie.buffer(%tile_3_5) {sym_name = "buf455"} : memref<64x1xbf16, 2 : i32> 
    %buf454 = aie.buffer(%tile_2_5) {sym_name = "buf454"} : memref<64x1xbf16, 2 : i32> 
    %buf453 = aie.buffer(%tile_2_5) {sym_name = "buf453"} : memref<64x1xbf16, 2 : i32> 
    %buf452 = aie.buffer(%tile_2_5) {sym_name = "buf452"} : memref<64x64xbf16, 2 : i32> 
    %buf451 = aie.buffer(%tile_2_5) {sym_name = "buf451"} : memref<64x64xbf16, 2 : i32> 
    %buf450 = aie.buffer(%tile_2_5) {sym_name = "buf450"} : memref<64x64xbf16, 2 : i32> 
    %buf449 = aie.buffer(%tile_2_5) {sym_name = "buf449"} : memref<64x64xbf16, 2 : i32> 
    %buf448 = aie.buffer(%tile_2_5) {sym_name = "buf448"} : memref<64x64xbf16, 2 : i32> 
    %buf447 = aie.buffer(%tile_2_5) {sym_name = "buf447"} : memref<64x1xbf16, 2 : i32> 
    %buf446 = aie.buffer(%tile_2_5) {sym_name = "buf446"} : memref<64x1xbf16, 2 : i32> 
    %buf445 = aie.buffer(%tile_1_5) {sym_name = "buf445"} : memref<64x1xbf16, 2 : i32> 
    %buf444 = aie.buffer(%tile_1_5) {sym_name = "buf444"} : memref<64x1xbf16, 2 : i32> 
    %buf443 = aie.buffer(%tile_1_5) {sym_name = "buf443"} : memref<64x64xbf16, 2 : i32> 
    %buf442 = aie.buffer(%tile_1_5) {sym_name = "buf442"} : memref<64x64xbf16, 2 : i32> 
    %buf441 = aie.buffer(%tile_1_5) {sym_name = "buf441"} : memref<64x64xbf16, 2 : i32> 
    %buf440 = aie.buffer(%tile_1_5) {sym_name = "buf440"} : memref<64x64xbf16, 2 : i32> 
    %buf439 = aie.buffer(%tile_1_5) {sym_name = "buf439"} : memref<64x64xbf16, 2 : i32> 
    %buf438 = aie.buffer(%tile_1_5) {sym_name = "buf438"} : memref<64x1xbf16, 2 : i32> 
    %buf437 = aie.buffer(%tile_1_5) {sym_name = "buf437"} : memref<64x1xbf16, 2 : i32> 
    %buf436 = aie.buffer(%tile_0_5) {sym_name = "buf436"} : memref<64x1xbf16, 2 : i32> 
    %buf435 = aie.buffer(%tile_0_5) {sym_name = "buf435"} : memref<64x1xbf16, 2 : i32> 
    %buf434 = aie.buffer(%tile_0_5) {sym_name = "buf434"} : memref<64x64xbf16, 2 : i32> 
    %buf433 = aie.buffer(%tile_0_5) {sym_name = "buf433"} : memref<64x64xbf16, 2 : i32> 
    %buf432 = aie.buffer(%tile_0_5) {sym_name = "buf432"} : memref<64x64xbf16, 2 : i32> 
    %buf431 = aie.buffer(%tile_0_5) {sym_name = "buf431"} : memref<64x64xbf16, 2 : i32> 
    %buf430 = aie.buffer(%tile_0_5) {sym_name = "buf430"} : memref<64x64xbf16, 2 : i32> 
    %buf429 = aie.buffer(%tile_0_5) {sym_name = "buf429"} : memref<64x1xbf16, 2 : i32> 
    %buf428 = aie.buffer(%tile_0_5) {sym_name = "buf428"} : memref<64x1xbf16, 2 : i32> 
    %buf427 = aie.buffer(%tile_3_4) {sym_name = "buf427"} : memref<64x1xbf16, 2 : i32> 
    %buf426 = aie.buffer(%tile_3_4) {sym_name = "buf426"} : memref<64x1xbf16, 2 : i32> 
    %buf425 = aie.buffer(%tile_3_4) {sym_name = "buf425"} : memref<64x64xbf16, 2 : i32> 
    %buf424 = aie.buffer(%tile_3_4) {sym_name = "buf424"} : memref<64x64xbf16, 2 : i32> 
    %buf423 = aie.buffer(%tile_3_4) {sym_name = "buf423"} : memref<64x64xbf16, 2 : i32> 
    %buf422 = aie.buffer(%tile_3_4) {sym_name = "buf422"} : memref<64x64xbf16, 2 : i32> 
    %buf421 = aie.buffer(%tile_3_4) {sym_name = "buf421"} : memref<64x64xbf16, 2 : i32> 
    %buf420 = aie.buffer(%tile_3_4) {sym_name = "buf420"} : memref<64x1xbf16, 2 : i32> 
    %buf419 = aie.buffer(%tile_3_4) {sym_name = "buf419"} : memref<64x1xbf16, 2 : i32> 
    %buf418 = aie.buffer(%tile_3_4) {sym_name = "buf418"} : memref<64x64xbf16, 2 : i32> 
    %buf417 = aie.buffer(%tile_3_4) {sym_name = "buf417"} : memref<64x1xbf16, 2 : i32> 
    %buf416 = aie.buffer(%tile_3_4) {sym_name = "buf416"} : memref<64x1xbf16, 2 : i32> 
    %buf415 = aie.buffer(%tile_3_4) {sym_name = "buf415"} : memref<64x1xbf16, 2 : i32> 
    %buf414 = aie.buffer(%tile_3_4) {sym_name = "buf414"} : memref<64x1xbf16, 2 : i32> 
    %buf413 = aie.buffer(%tile_3_4) {sym_name = "buf413"} : memref<64x1xbf16, 2 : i32> 
    %buf412 = aie.buffer(%tile_3_4) {sym_name = "buf412"} : memref<64x1xbf16, 2 : i32> 
    %buf411 = aie.buffer(%tile_2_4) {sym_name = "buf411"} : memref<64x1xbf16, 2 : i32> 
    %buf410 = aie.buffer(%tile_2_4) {sym_name = "buf410"} : memref<64x1xbf16, 2 : i32> 
    %buf409 = aie.buffer(%tile_2_4) {sym_name = "buf409"} : memref<64x64xbf16, 2 : i32> 
    %buf408 = aie.buffer(%tile_2_4) {sym_name = "buf408"} : memref<64x64xbf16, 2 : i32> 
    %buf407 = aie.buffer(%tile_2_4) {sym_name = "buf407"} : memref<64x64xbf16, 2 : i32> 
    %buf406 = aie.buffer(%tile_2_4) {sym_name = "buf406"} : memref<64x64xbf16, 2 : i32> 
    %buf405 = aie.buffer(%tile_2_4) {sym_name = "buf405"} : memref<64x64xbf16, 2 : i32> 
    %buf404 = aie.buffer(%tile_2_4) {sym_name = "buf404"} : memref<64x1xbf16, 2 : i32> 
    %buf403 = aie.buffer(%tile_2_4) {sym_name = "buf403"} : memref<64x1xbf16, 2 : i32> 
    %buf402 = aie.buffer(%tile_2_4) {sym_name = "buf402"} : memref<64x64xbf16, 2 : i32> 
    %buf401 = aie.buffer(%tile_2_4) {sym_name = "buf401"} : memref<64x1xbf16, 2 : i32> 
    %buf400 = aie.buffer(%tile_2_4) {sym_name = "buf400"} : memref<64x1xbf16, 2 : i32> 
    %buf399 = aie.buffer(%tile_2_4) {sym_name = "buf399"} : memref<64x1xbf16, 2 : i32> 
    %buf398 = aie.buffer(%tile_2_4) {sym_name = "buf398"} : memref<64x1xbf16, 2 : i32> 
    %buf397 = aie.buffer(%tile_2_4) {sym_name = "buf397"} : memref<64x1xbf16, 2 : i32> 
    %buf396 = aie.buffer(%tile_2_4) {sym_name = "buf396"} : memref<64x1xbf16, 2 : i32> 
    %buf395 = aie.buffer(%tile_1_4) {sym_name = "buf395"} : memref<64x1xbf16, 2 : i32> 
    %buf394 = aie.buffer(%tile_1_4) {sym_name = "buf394"} : memref<64x1xbf16, 2 : i32> 
    %buf393 = aie.buffer(%tile_1_4) {sym_name = "buf393"} : memref<64x64xbf16, 2 : i32> 
    %buf392 = aie.buffer(%tile_1_4) {sym_name = "buf392"} : memref<64x64xbf16, 2 : i32> 
    %buf391 = aie.buffer(%tile_1_4) {sym_name = "buf391"} : memref<64x64xbf16, 2 : i32> 
    %buf390 = aie.buffer(%tile_1_4) {sym_name = "buf390"} : memref<64x64xbf16, 2 : i32> 
    %buf389 = aie.buffer(%tile_1_4) {sym_name = "buf389"} : memref<64x64xbf16, 2 : i32> 
    %buf388 = aie.buffer(%tile_1_4) {sym_name = "buf388"} : memref<64x1xbf16, 2 : i32> 
    %buf387 = aie.buffer(%tile_1_4) {sym_name = "buf387"} : memref<64x1xbf16, 2 : i32> 
    %buf386 = aie.buffer(%tile_1_4) {sym_name = "buf386"} : memref<64x64xbf16, 2 : i32> 
    %buf385 = aie.buffer(%tile_1_4) {sym_name = "buf385"} : memref<64x1xbf16, 2 : i32> 
    %buf384 = aie.buffer(%tile_1_4) {sym_name = "buf384"} : memref<64x1xbf16, 2 : i32> 
    %buf383 = aie.buffer(%tile_1_4) {sym_name = "buf383"} : memref<64x1xbf16, 2 : i32> 
    %buf382 = aie.buffer(%tile_1_4) {sym_name = "buf382"} : memref<64x1xbf16, 2 : i32> 
    %buf381 = aie.buffer(%tile_1_4) {sym_name = "buf381"} : memref<64x1xbf16, 2 : i32> 
    %buf380 = aie.buffer(%tile_1_4) {sym_name = "buf380"} : memref<64x1xbf16, 2 : i32> 
    %buf379 = aie.buffer(%tile_0_4) {sym_name = "buf379"} : memref<64x1xbf16, 2 : i32> 
    %buf378 = aie.buffer(%tile_0_4) {sym_name = "buf378"} : memref<64x1xbf16, 2 : i32> 
    %buf377 = aie.buffer(%tile_0_4) {sym_name = "buf377"} : memref<64x64xbf16, 2 : i32> 
    %buf376 = aie.buffer(%tile_0_4) {sym_name = "buf376"} : memref<64x64xbf16, 2 : i32> 
    %buf375 = aie.buffer(%tile_0_4) {sym_name = "buf375"} : memref<64x64xbf16, 2 : i32> 
    %buf374 = aie.buffer(%tile_0_4) {sym_name = "buf374"} : memref<64x64xbf16, 2 : i32> 
    %buf373 = aie.buffer(%tile_0_4) {sym_name = "buf373"} : memref<64x64xbf16, 2 : i32> 
    %buf372 = aie.buffer(%tile_0_4) {sym_name = "buf372"} : memref<64x1xbf16, 2 : i32> 
    %buf371 = aie.buffer(%tile_0_4) {sym_name = "buf371"} : memref<64x1xbf16, 2 : i32> 
    %buf370 = aie.buffer(%tile_0_4) {sym_name = "buf370"} : memref<64x64xbf16, 2 : i32> 
    %buf369 = aie.buffer(%tile_0_4) {sym_name = "buf369"} : memref<64x1xbf16, 2 : i32> 
    %buf368 = aie.buffer(%tile_0_4) {sym_name = "buf368"} : memref<64x1xbf16, 2 : i32> 
    %buf367 = aie.buffer(%tile_0_4) {sym_name = "buf367"} : memref<64x1xbf16, 2 : i32> 
    %buf366 = aie.buffer(%tile_0_4) {sym_name = "buf366"} : memref<64x1xbf16, 2 : i32> 
    %buf365 = aie.buffer(%tile_0_4) {sym_name = "buf365"} : memref<64x1xbf16, 2 : i32> 
    %buf364 = aie.buffer(%tile_0_4) {sym_name = "buf364"} : memref<64x1xbf16, 2 : i32> 
    %buf363 = aie.buffer(%tile_3_3) {sym_name = "buf363"} : memref<64x1xbf16, 2 : i32> 
    %buf362 = aie.buffer(%tile_3_3) {sym_name = "buf362"} : memref<64x1xbf16, 2 : i32> 
    %buf361 = aie.buffer(%tile_3_3) {sym_name = "buf361"} : memref<64x64xbf16, 2 : i32> 
    %buf360 = aie.buffer(%tile_3_3) {sym_name = "buf360"} : memref<64x64xbf16, 2 : i32> 
    %buf359 = aie.buffer(%tile_3_3) {sym_name = "buf359"} : memref<64x64xbf16, 2 : i32> 
    %buf358 = aie.buffer(%tile_3_3) {sym_name = "buf358"} : memref<64x64xbf16, 2 : i32> 
    %buf357 = aie.buffer(%tile_3_3) {sym_name = "buf357"} : memref<64x64xbf16, 2 : i32> 
    %buf356 = aie.buffer(%tile_3_3) {sym_name = "buf356"} : memref<64x1xbf16, 2 : i32> 
    %buf355 = aie.buffer(%tile_3_3) {sym_name = "buf355"} : memref<64x1xbf16, 2 : i32> 
    %buf354 = aie.buffer(%tile_3_3) {sym_name = "buf354"} : memref<64x64xbf16, 2 : i32> 
    %buf353 = aie.buffer(%tile_3_3) {sym_name = "buf353"} : memref<64x1xbf16, 2 : i32> 
    %buf352 = aie.buffer(%tile_3_3) {sym_name = "buf352"} : memref<64x1xbf16, 2 : i32> 
    %buf351 = aie.buffer(%tile_3_3) {sym_name = "buf351"} : memref<64x1xbf16, 2 : i32> 
    %buf350 = aie.buffer(%tile_3_3) {sym_name = "buf350"} : memref<64x1xbf16, 2 : i32> 
    %buf349 = aie.buffer(%tile_3_3) {sym_name = "buf349"} : memref<64x1xbf16, 2 : i32> 
    %buf348 = aie.buffer(%tile_3_3) {sym_name = "buf348"} : memref<64x1xbf16, 2 : i32> 
    %buf347 = aie.buffer(%tile_2_3) {sym_name = "buf347"} : memref<64x1xbf16, 2 : i32> 
    %buf346 = aie.buffer(%tile_2_3) {sym_name = "buf346"} : memref<64x1xbf16, 2 : i32> 
    %buf345 = aie.buffer(%tile_2_3) {sym_name = "buf345"} : memref<64x64xbf16, 2 : i32> 
    %buf344 = aie.buffer(%tile_2_3) {sym_name = "buf344"} : memref<64x64xbf16, 2 : i32> 
    %buf343 = aie.buffer(%tile_2_3) {sym_name = "buf343"} : memref<64x64xbf16, 2 : i32> 
    %buf342 = aie.buffer(%tile_2_3) {sym_name = "buf342"} : memref<64x64xbf16, 2 : i32> 
    %buf341 = aie.buffer(%tile_2_3) {sym_name = "buf341"} : memref<64x64xbf16, 2 : i32> 
    %buf340 = aie.buffer(%tile_2_3) {sym_name = "buf340"} : memref<64x1xbf16, 2 : i32> 
    %buf339 = aie.buffer(%tile_2_3) {sym_name = "buf339"} : memref<64x1xbf16, 2 : i32> 
    %buf338 = aie.buffer(%tile_2_3) {sym_name = "buf338"} : memref<64x64xbf16, 2 : i32> 
    %buf337 = aie.buffer(%tile_2_3) {sym_name = "buf337"} : memref<64x1xbf16, 2 : i32> 
    %buf336 = aie.buffer(%tile_2_3) {sym_name = "buf336"} : memref<64x1xbf16, 2 : i32> 
    %buf335 = aie.buffer(%tile_2_3) {sym_name = "buf335"} : memref<64x1xbf16, 2 : i32> 
    %buf334 = aie.buffer(%tile_2_3) {sym_name = "buf334"} : memref<64x1xbf16, 2 : i32> 
    %buf333 = aie.buffer(%tile_2_3) {sym_name = "buf333"} : memref<64x1xbf16, 2 : i32> 
    %buf332 = aie.buffer(%tile_2_3) {sym_name = "buf332"} : memref<64x1xbf16, 2 : i32> 
    %buf331 = aie.buffer(%tile_1_3) {sym_name = "buf331"} : memref<64x1xbf16, 2 : i32> 
    %buf330 = aie.buffer(%tile_1_3) {sym_name = "buf330"} : memref<64x1xbf16, 2 : i32> 
    %buf329 = aie.buffer(%tile_1_3) {sym_name = "buf329"} : memref<64x64xbf16, 2 : i32> 
    %buf328 = aie.buffer(%tile_1_3) {sym_name = "buf328"} : memref<64x64xbf16, 2 : i32> 
    %buf327 = aie.buffer(%tile_1_3) {sym_name = "buf327"} : memref<64x64xbf16, 2 : i32> 
    %buf326 = aie.buffer(%tile_1_3) {sym_name = "buf326"} : memref<64x64xbf16, 2 : i32> 
    %buf325 = aie.buffer(%tile_1_3) {sym_name = "buf325"} : memref<64x64xbf16, 2 : i32> 
    %buf324 = aie.buffer(%tile_1_3) {sym_name = "buf324"} : memref<64x1xbf16, 2 : i32> 
    %buf323 = aie.buffer(%tile_1_3) {sym_name = "buf323"} : memref<64x1xbf16, 2 : i32> 
    %buf322 = aie.buffer(%tile_1_3) {sym_name = "buf322"} : memref<64x64xbf16, 2 : i32> 
    %buf321 = aie.buffer(%tile_1_3) {sym_name = "buf321"} : memref<64x1xbf16, 2 : i32> 
    %buf320 = aie.buffer(%tile_1_3) {sym_name = "buf320"} : memref<64x1xbf16, 2 : i32> 
    %buf319 = aie.buffer(%tile_1_3) {sym_name = "buf319"} : memref<64x1xbf16, 2 : i32> 
    %buf318 = aie.buffer(%tile_1_3) {sym_name = "buf318"} : memref<64x1xbf16, 2 : i32> 
    %buf317 = aie.buffer(%tile_1_3) {sym_name = "buf317"} : memref<64x1xbf16, 2 : i32> 
    %buf316 = aie.buffer(%tile_1_3) {sym_name = "buf316"} : memref<64x1xbf16, 2 : i32> 
    %buf315 = aie.buffer(%tile_0_3) {sym_name = "buf315"} : memref<64x1xbf16, 2 : i32> 
    %buf314 = aie.buffer(%tile_0_3) {sym_name = "buf314"} : memref<64x1xbf16, 2 : i32> 
    %buf313 = aie.buffer(%tile_0_3) {sym_name = "buf313"} : memref<64x64xbf16, 2 : i32> 
    %buf312 = aie.buffer(%tile_0_3) {sym_name = "buf312"} : memref<64x64xbf16, 2 : i32> 
    %buf311 = aie.buffer(%tile_0_3) {sym_name = "buf311"} : memref<64x64xbf16, 2 : i32> 
    %buf310 = aie.buffer(%tile_0_3) {sym_name = "buf310"} : memref<64x64xbf16, 2 : i32> 
    %buf309 = aie.buffer(%tile_0_3) {sym_name = "buf309"} : memref<64x64xbf16, 2 : i32> 
    %buf308 = aie.buffer(%tile_0_3) {sym_name = "buf308"} : memref<64x1xbf16, 2 : i32> 
    %buf307 = aie.buffer(%tile_0_3) {sym_name = "buf307"} : memref<64x1xbf16, 2 : i32> 
    %buf306 = aie.buffer(%tile_0_3) {sym_name = "buf306"} : memref<64x64xbf16, 2 : i32> 
    %buf305 = aie.buffer(%tile_0_3) {sym_name = "buf305"} : memref<64x1xbf16, 2 : i32> 
    %buf304 = aie.buffer(%tile_0_3) {sym_name = "buf304"} : memref<64x1xbf16, 2 : i32> 
    %buf303 = aie.buffer(%tile_0_3) {sym_name = "buf303"} : memref<64x1xbf16, 2 : i32> 
    %buf302 = aie.buffer(%tile_0_3) {sym_name = "buf302"} : memref<64x1xbf16, 2 : i32> 
    %buf301 = aie.buffer(%tile_0_3) {sym_name = "buf301"} : memref<64x1xbf16, 2 : i32> 
    %buf300 = aie.buffer(%tile_0_3) {sym_name = "buf300"} : memref<64x1xbf16, 2 : i32> 
    %buf299 = aie.buffer(%tile_3_2) {sym_name = "buf299"} : memref<64x1xbf16, 2 : i32> 
    %buf298 = aie.buffer(%tile_3_2) {sym_name = "buf298"} : memref<64x1xbf16, 2 : i32> 
    %buf297 = aie.buffer(%tile_3_2) {sym_name = "buf297"} : memref<64x64xbf16, 2 : i32> 
    %buf296 = aie.buffer(%tile_3_2) {sym_name = "buf296"} : memref<64x64xbf16, 2 : i32> 
    %buf295 = aie.buffer(%tile_3_2) {sym_name = "buf295"} : memref<64x64xbf16, 2 : i32> 
    %buf294 = aie.buffer(%tile_3_2) {sym_name = "buf294"} : memref<64x64xbf16, 2 : i32> 
    %buf293 = aie.buffer(%tile_3_2) {sym_name = "buf293"} : memref<64x64xbf16, 2 : i32> 
    %buf292 = aie.buffer(%tile_3_2) {sym_name = "buf292"} : memref<64x1xbf16, 2 : i32> 
    %buf291 = aie.buffer(%tile_3_2) {sym_name = "buf291"} : memref<64x1xbf16, 2 : i32> 
    %buf290 = aie.buffer(%tile_3_2) {sym_name = "buf290"} : memref<64x64xbf16, 2 : i32> 
    %buf289 = aie.buffer(%tile_3_2) {sym_name = "buf289"} : memref<64x1xbf16, 2 : i32> 
    %buf288 = aie.buffer(%tile_3_2) {sym_name = "buf288"} : memref<64x1xbf16, 2 : i32> 
    %buf287 = aie.buffer(%tile_3_2) {sym_name = "buf287"} : memref<64x1xbf16, 2 : i32> 
    %buf286 = aie.buffer(%tile_3_2) {sym_name = "buf286"} : memref<64x1xbf16, 2 : i32> 
    %buf285 = aie.buffer(%tile_3_2) {sym_name = "buf285"} : memref<64x1xbf16, 2 : i32> 
    %buf284 = aie.buffer(%tile_3_2) {sym_name = "buf284"} : memref<64x1xbf16, 2 : i32> 
    %buf283 = aie.buffer(%tile_2_2) {sym_name = "buf283"} : memref<64x1xbf16, 2 : i32> 
    %buf282 = aie.buffer(%tile_2_2) {sym_name = "buf282"} : memref<64x1xbf16, 2 : i32> 
    %buf281 = aie.buffer(%tile_2_2) {sym_name = "buf281"} : memref<64x64xbf16, 2 : i32> 
    %buf280 = aie.buffer(%tile_2_2) {sym_name = "buf280"} : memref<64x64xbf16, 2 : i32> 
    %buf279 = aie.buffer(%tile_2_2) {sym_name = "buf279"} : memref<64x64xbf16, 2 : i32> 
    %buf278 = aie.buffer(%tile_2_2) {sym_name = "buf278"} : memref<64x64xbf16, 2 : i32> 
    %buf277 = aie.buffer(%tile_2_2) {sym_name = "buf277"} : memref<64x64xbf16, 2 : i32> 
    %buf276 = aie.buffer(%tile_2_2) {sym_name = "buf276"} : memref<64x1xbf16, 2 : i32> 
    %buf275 = aie.buffer(%tile_2_2) {sym_name = "buf275"} : memref<64x1xbf16, 2 : i32> 
    %buf274 = aie.buffer(%tile_2_2) {sym_name = "buf274"} : memref<64x64xbf16, 2 : i32> 
    %buf273 = aie.buffer(%tile_2_2) {sym_name = "buf273"} : memref<64x1xbf16, 2 : i32> 
    %buf272 = aie.buffer(%tile_2_2) {sym_name = "buf272"} : memref<64x1xbf16, 2 : i32> 
    %buf271 = aie.buffer(%tile_2_2) {sym_name = "buf271"} : memref<64x1xbf16, 2 : i32> 
    %buf270 = aie.buffer(%tile_2_2) {sym_name = "buf270"} : memref<64x1xbf16, 2 : i32> 
    %buf269 = aie.buffer(%tile_2_2) {sym_name = "buf269"} : memref<64x1xbf16, 2 : i32> 
    %buf268 = aie.buffer(%tile_2_2) {sym_name = "buf268"} : memref<64x1xbf16, 2 : i32> 
    %buf267 = aie.buffer(%tile_1_2) {sym_name = "buf267"} : memref<64x1xbf16, 2 : i32> 
    %buf266 = aie.buffer(%tile_1_2) {sym_name = "buf266"} : memref<64x1xbf16, 2 : i32> 
    %buf265 = aie.buffer(%tile_1_2) {sym_name = "buf265"} : memref<64x64xbf16, 2 : i32> 
    %buf264 = aie.buffer(%tile_1_2) {sym_name = "buf264"} : memref<64x64xbf16, 2 : i32> 
    %buf263 = aie.buffer(%tile_1_2) {sym_name = "buf263"} : memref<64x64xbf16, 2 : i32> 
    %buf262 = aie.buffer(%tile_1_2) {sym_name = "buf262"} : memref<64x64xbf16, 2 : i32> 
    %buf261 = aie.buffer(%tile_1_2) {sym_name = "buf261"} : memref<64x64xbf16, 2 : i32> 
    %buf260 = aie.buffer(%tile_1_2) {sym_name = "buf260"} : memref<64x1xbf16, 2 : i32> 
    %buf259 = aie.buffer(%tile_1_2) {sym_name = "buf259"} : memref<64x1xbf16, 2 : i32> 
    %buf258 = aie.buffer(%tile_1_2) {sym_name = "buf258"} : memref<64x64xbf16, 2 : i32> 
    %buf257 = aie.buffer(%tile_1_2) {sym_name = "buf257"} : memref<64x1xbf16, 2 : i32> 
    %buf256 = aie.buffer(%tile_1_2) {sym_name = "buf256"} : memref<64x1xbf16, 2 : i32> 
    %buf255 = aie.buffer(%tile_1_2) {sym_name = "buf255"} : memref<64x1xbf16, 2 : i32> 
    %buf254 = aie.buffer(%tile_1_2) {sym_name = "buf254"} : memref<64x1xbf16, 2 : i32> 
    %buf253 = aie.buffer(%tile_1_2) {sym_name = "buf253"} : memref<64x1xbf16, 2 : i32> 
    %buf252 = aie.buffer(%tile_1_2) {sym_name = "buf252"} : memref<64x1xbf16, 2 : i32> 
    %buf251 = aie.buffer(%tile_0_2) {sym_name = "buf251"} : memref<64x1xbf16, 2 : i32> 
    %buf250 = aie.buffer(%tile_0_2) {sym_name = "buf250"} : memref<64x1xbf16, 2 : i32> 
    %buf249 = aie.buffer(%tile_0_2) {sym_name = "buf249"} : memref<64x64xbf16, 2 : i32> 
    %buf248 = aie.buffer(%tile_0_2) {sym_name = "buf248"} : memref<64x64xbf16, 2 : i32> 
    %buf247 = aie.buffer(%tile_0_2) {sym_name = "buf247"} : memref<64x64xbf16, 2 : i32> 
    %buf246 = aie.buffer(%tile_0_2) {sym_name = "buf246"} : memref<64x64xbf16, 2 : i32> 
    %buf245 = aie.buffer(%tile_0_2) {sym_name = "buf245"} : memref<64x64xbf16, 2 : i32> 
    %buf244 = aie.buffer(%tile_0_2) {sym_name = "buf244"} : memref<64x1xbf16, 2 : i32> 
    %buf243 = aie.buffer(%tile_0_2) {sym_name = "buf243"} : memref<64x1xbf16, 2 : i32> 
    %buf242 = aie.buffer(%tile_0_2) {sym_name = "buf242"} : memref<64x64xbf16, 2 : i32> 
    %buf241 = aie.buffer(%tile_0_2) {sym_name = "buf241"} : memref<64x1xbf16, 2 : i32> 
    %buf240 = aie.buffer(%tile_0_2) {sym_name = "buf240"} : memref<64x1xbf16, 2 : i32> 
    %buf239 = aie.buffer(%tile_0_2) {sym_name = "buf239"} : memref<64x1xbf16, 2 : i32> 
    %buf238 = aie.buffer(%tile_0_2) {sym_name = "buf238"} : memref<64x1xbf16, 2 : i32> 
    %buf237 = aie.buffer(%tile_0_2) {sym_name = "buf237"} : memref<64x1xbf16, 2 : i32> 
    %buf236 = aie.buffer(%tile_0_2) {sym_name = "buf236"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<12x2048x64xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<12x2048x64xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<12x2048x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<12x2048x64xbf16>
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    scf.for %arg0 = %c0 to %c8 step %c1 {
    } {loop_annotation = #loop_annotation}
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_66, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf460 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_67, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf458 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_65, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf461) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf463) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf462) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_66, Release, 1)
      aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_66, Release, 1)
      aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_66, Release, 1)
      aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf460, %buf459) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_66, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_73 = memref.collapse_shape %buf457 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_67, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_5_65, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf457 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf459, %buf460, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_66, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf457 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf462, %buf456, %buf455) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf455, %buf461) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf457 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf458, %buf461) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf463, %buf455, %buf456) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf456, %buf463) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf461 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_71 = memref.collapse_shape %buf462 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_72 = memref.collapse_shape %buf463 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf451 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_64, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf449 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_62, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf452) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf454) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf453) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_63, Release, 1)
      aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_63, Release, 1)
      aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf451, %buf450) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_63, Release, 1)
      aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_63, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_73 = memref.collapse_shape %buf448 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_64, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_5_62, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf448 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf450, %buf451, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_63, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf448 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf453, %buf447, %buf446) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf446, %buf452) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf448 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf449, %buf452) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf454, %buf446, %buf447) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf447, %buf454) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf452 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_71 = memref.collapse_shape %buf453 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_72 = memref.collapse_shape %buf454 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf442 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_61, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf440 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_59, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf443) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf445) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf444) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_60, Release, 1)
      aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf442, %buf441) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_60, Release, 1)
      aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_60, Release, 1)
      aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_60, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_73 = memref.collapse_shape %buf439 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_61, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_5_59, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf439 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf441, %buf442, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_60, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf439 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf444, %buf438, %buf437) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf437, %buf443) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf439 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf440, %buf443) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf445, %buf437, %buf438) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf438, %buf445) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf443 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_71 = memref.collapse_shape %buf444 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_72 = memref.collapse_shape %buf445 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_57, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf433 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_58, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf431 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_56, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_69 = arith.constant 1 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf434) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf436) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf435) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf433, %buf432) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_57, Release, 1)
      aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_57, Release, 1)
      aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_57, Release, 1)
      aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_57, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_68 step %c1_69 {
        %collapse_shape_73 = memref.collapse_shape %buf430 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_58, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_5_56, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf430 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf432, %buf433, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_57, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf430 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf435, %buf429, %buf428) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf428, %buf434) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf430 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf431, %buf434) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf436, %buf428, %buf429) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf429, %buf436) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf434 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_71 = memref.collapse_shape %buf435 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_72 = memref.collapse_shape %buf436 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_54, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf424 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_55, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf422 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_53, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf425) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf427) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf426) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_54, Release, 1)
      aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_54, Release, 1)
      aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_54, Release, 1)
      aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf424, %buf423) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_54, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf421 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_55, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_4_53, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf421 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf423, %buf424, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_54, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf421 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf426, %buf420, %buf419) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf419, %buf425) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf421 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf422, %buf425) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf427, %buf419, %buf420) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf420, %buf427) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf418 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf417 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf416 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf426, %buf415) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf417, %buf426) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf417, %buf426, %buf414) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf415, %buf426, %buf413) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf414, %buf418) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf413, %buf425) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf425, %buf418) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf412) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf416, %buf414, %buf412) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf427, %buf413, %buf412) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf412, %buf416) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_73 = memref.collapse_shape %buf418 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf426 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf416 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_51, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf408 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_52, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf406 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_50, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf409) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf411) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf410) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_51, Release, 1)
      aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_51, Release, 1)
      aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf408, %buf407) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_51, Release, 1)
      aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_51, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf405 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_52, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_4_50, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf405 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf407, %buf408, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_51, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf405 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf410, %buf404, %buf403) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf403, %buf409) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf405 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf406, %buf409) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf411, %buf403, %buf404) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf404, %buf411) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf402 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf401 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf400 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf410, %buf399) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf401, %buf410) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf401, %buf410, %buf398) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf399, %buf410, %buf397) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf398, %buf402) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf397, %buf409) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf409, %buf402) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf396) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf400, %buf398, %buf396) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf411, %buf397, %buf396) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf396, %buf400) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_73 = memref.collapse_shape %buf402 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf410 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf400 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf392 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_49, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf390 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_47, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf393) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf395) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf394) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_48, Release, 1)
      aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf392, %buf391) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_48, Release, 1)
      aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_48, Release, 1)
      aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_48, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf389 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_49, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_4_47, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf389 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf391, %buf392, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_48, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf389 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf394, %buf388, %buf387) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf387, %buf393) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf389 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf390, %buf393) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf395, %buf387, %buf388) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf388, %buf395) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf386 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf385 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf384 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf394, %buf383) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf385, %buf394) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf385, %buf394, %buf382) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf383, %buf394, %buf381) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf382, %buf386) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf381, %buf393) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf393, %buf386) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf380) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf384, %buf382, %buf380) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf395, %buf381, %buf380) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf380, %buf384) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_73 = memref.collapse_shape %buf386 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf394 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf384 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf376 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_46, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf374 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_44, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_69 = arith.constant 1 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf377) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf379) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf378) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf376, %buf375) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_45, Release, 1)
      aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_45, Release, 1)
      aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_45, Release, 1)
      aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_45, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_68 step %c1_69 {
        %collapse_shape_76 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_4_44, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf375, %buf376, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_45, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf378, %buf372, %buf371) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf371, %buf377) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf374, %buf377) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf379, %buf371, %buf372) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf372, %buf379) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf370 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf369 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf368 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf378, %buf367) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf369, %buf378) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf369, %buf378, %buf366) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf367, %buf378, %buf365) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf366, %buf370) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf365, %buf377) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf377, %buf370) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf364) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf368, %buf366, %buf364) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf379, %buf365, %buf364) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf364, %buf368) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_73 = memref.collapse_shape %buf370 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf378 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf368 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_42, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf360 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_43, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf358 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_41, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf361) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf363) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf362) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_42, Release, 1)
      aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_42, Release, 1)
      aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_42, Release, 1)
      aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf360, %buf359) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_42, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf357 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_43, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3_41, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf357 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf359, %buf360, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_42, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf357 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf362, %buf356, %buf355) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf355, %buf361) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf357 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf358, %buf361) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf363, %buf355, %buf356) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf356, %buf363) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf354 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf353 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf352 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf362, %buf351) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf353, %buf362) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf353, %buf362, %buf350) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf351, %buf362, %buf349) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf350, %buf354) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf349, %buf361) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf361, %buf354) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf348) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf352, %buf350, %buf348) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf363, %buf349, %buf348) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf348, %buf352) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_73 = memref.collapse_shape %buf354 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf362 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf352 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf344 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_40, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf342 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_38, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf345) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf347) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf346) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_39, Release, 1)
      aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_39, Release, 1)
      aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf344, %buf343) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_39, Release, 1)
      aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_39, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf341 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_40, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3_38, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf341 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf343, %buf344, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_39, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf341 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf346, %buf340, %buf339) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf339, %buf345) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf341 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf342, %buf345) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf347, %buf339, %buf340) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf340, %buf347) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf338 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf337 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf336 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf346, %buf335) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf337, %buf346) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf337, %buf346, %buf334) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf335, %buf346, %buf333) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf334, %buf338) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf333, %buf345) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf345, %buf338) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf332) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf336, %buf334, %buf332) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf347, %buf333, %buf332) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf332, %buf336) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_73 = memref.collapse_shape %buf338 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf346 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf336 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf328 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_37, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf326 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_35, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf329) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf331) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf330) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_36, Release, 1)
      aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf328, %buf327) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_36, Release, 1)
      aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_36, Release, 1)
      aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_36, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_76 = memref.collapse_shape %buf325 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_37, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3_35, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf325 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf327, %buf328, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_36, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf325 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf330, %buf324, %buf323) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf323, %buf329) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf325 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf326, %buf329) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf331, %buf323, %buf324) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf324, %buf331) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf322 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf321 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf320 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf330, %buf319) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf321, %buf330) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf321, %buf330, %buf318) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf319, %buf330, %buf317) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf318, %buf322) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf317, %buf329) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf329, %buf322) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf316) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf320, %buf318, %buf316) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf331, %buf317, %buf316) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf316, %buf320) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_73 = memref.collapse_shape %buf322 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf330 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf320 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_69], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf312 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf310 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_32, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_69 = arith.constant 1 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf313) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf315) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf314) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf312, %buf311) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_33, Release, 1)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_33, Release, 1)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_33, Release, 1)
      aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_33, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_68 step %c1_69 {
        %collapse_shape_76 = memref.collapse_shape %buf309 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_76) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3_32, AcquireGreaterEqual, 1)
        %collapse_shape_77 = memref.collapse_shape %buf309 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf311, %buf312, %collapse_shape_77) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_33, Release, 1)
        %collapse_shape_78 = memref.collapse_shape %buf309 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_78, %buf314, %buf308, %buf307) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf307, %buf313) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_79 = memref.collapse_shape %buf309 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_79, %buf310, %buf313) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf315, %buf307, %buf308) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf308, %buf315) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf306 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf305 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf304 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf314, %buf303) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf305, %buf314) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf305, %buf314, %buf302) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf303, %buf314, %buf301) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf302, %buf306) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf301, %buf313) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf313, %buf306) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf300) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf304, %buf302, %buf300) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf315, %buf301, %buf300) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf300, %buf304) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_73 = memref.collapse_shape %buf306 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf314 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf304 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0_70], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf290 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_30, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf296 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_29, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf294 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c1_69 = arith.constant 1 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_2_30, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf297) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf299) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf298) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_28, Release, 1)
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_28, Release, 1)
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_28, Release, 1)
      aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf296, %buf295) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_28, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_68 step %c1_69 {
        %collapse_shape_73 = memref.collapse_shape %buf293 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf293 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf295, %buf296, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_28, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf293 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf298, %buf292, %buf291) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf291, %buf297) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf293 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf294, %buf297) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf299, %buf291, %buf292) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf292, %buf299) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf290 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf289 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf288 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf298, %buf287) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf289, %buf298) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf289, %buf298, %buf286) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf287, %buf298, %buf285) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf286, %buf290) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf285, %buf297) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf297, %buf290) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf284) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf288, %buf286, %buf284) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf299, %buf285, %buf284) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf284, %buf288) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf288, %buf290) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_31, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf274 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_25, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_23, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf280 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_24, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf278 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_22, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c1_69 = arith.constant 1 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_2_25, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf281) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf283) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf282) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_23, Release, 1)
      aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_23, Release, 1)
      aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf280, %buf279) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_23, Release, 1)
      aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_23, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_68 step %c1_69 {
        %collapse_shape_73 = memref.collapse_shape %buf277 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_22, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf277 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf279, %buf280, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_23, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf277 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf282, %buf276, %buf275) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf275, %buf281) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf277 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf278, %buf281) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf283, %buf275, %buf276) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf276, %buf283) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf274 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf273 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf272 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf282, %buf271) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf273, %buf282) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf273, %buf282, %buf270) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf271, %buf282, %buf269) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf270, %buf274) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf269, %buf281) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf281, %buf274) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf268) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf272, %buf270, %buf268) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf283, %buf269, %buf268) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf268, %buf272) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf272, %buf274) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_26, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf258 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_20, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf264 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_19, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf262 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_17, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0_69 = arith.constant 0 : index
      %c1_70 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_20, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf265) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf267) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf266) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_18, Release, 1)
      aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf264, %buf263) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_18, Release, 1)
      aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_18, Release, 1)
      aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_18, Release, 1)
      scf.for %arg0 = %c0_69 to %c8_68 step %c1_70 {
        %collapse_shape_73 = memref.collapse_shape %buf261 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_17, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf261 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf263, %buf264, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_18, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf261 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf266, %buf260, %buf259) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf259, %buf265) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf261 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf262, %buf265) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf267, %buf259, %buf260) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf260, %buf267) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf258 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf257 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf256 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_69 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_69] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf266, %buf255) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf257, %buf266) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf257, %buf266, %buf254) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf255, %buf266, %buf253) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf254, %buf258) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf253, %buf265) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf265, %buf258) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf252) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf256, %buf254, %buf252) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf267, %buf253, %buf252) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf252, %buf256) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf256, %buf258) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_21, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf242 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_15, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf248 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_14, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf246 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_12, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c8_68 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c1_69 = arith.constant 1 : index
      %c0_70 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_15, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%buf249) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf251) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf250) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf248, %buf247) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_13, Release, 1)
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_13, Release, 1)
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_13, Release, 1)
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_13, Release, 1)
      scf.for %arg0 = %c0_70 to %c8_68 step %c1_69 {
        %collapse_shape_73 = memref.collapse_shape %buf245 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_73) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_12, AcquireGreaterEqual, 1)
        %collapse_shape_74 = memref.collapse_shape %buf245 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_a_b_bf16(%buf247, %buf248, %collapse_shape_74) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_13, Release, 1)
        %collapse_shape_75 = memref.collapse_shape %buf245 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @fused_softmax(%collapse_shape_75, %buf250, %buf244, %buf243) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf243, %buf249) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        %collapse_shape_76 = memref.collapse_shape %buf245 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @matmul_g_b_bf16(%collapse_shape_76, %buf246, %buf249) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf251, %buf243, %buf244) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf244, %buf251) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf242 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_71 = memref.collapse_shape %buf241 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_71[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_72 = memref.collapse_shape %buf240 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0_70 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_72[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0_70] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf250, %buf239) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf241, %buf250) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf241, %buf250, %buf238) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf239, %buf250, %buf237) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf238, %buf242) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf237, %buf249) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf249, %buf242) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf236) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf240, %buf238, %buf236) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf251, %buf237, %buf236) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf236, %buf240) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf240, %buf242) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_16, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
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
    air.channel @QK2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @QK2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @QK2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
    air.channel @QK2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
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
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_2_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_3_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_0_5, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_2_5, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_3_5, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 1)
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
      aie.use_lock(%lock_0_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf471 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf467 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf467 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_9, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf471 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_11, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf470 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_7, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf466 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf466 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_6, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf470 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_8, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf469 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf465 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf465 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf469 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf468 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf464 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf464 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf468 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @air_channel_0_1_0_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_VIn_0_1_0_0(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_1_1_0_0(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_2_1_0_0(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_3_1_0_0(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @air_QK2L1_1_0_1_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_QK2L1_1_1_1_0(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_QK2L1_1_2_1_0(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_QK2L1_1_3_1_0(%shim_noc_tile_1_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 1 : i64, segment_unroll_y = 0 : i64}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 33 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 36 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 39 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 42 : i64, location = 3 : i64, row = -1 : i64}], sym_name = "attn_seg_0_0"}{
      airrt.herd_metadata {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 53 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 61 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 69 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 77 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 85 : i64, location = 0 : i64, row = 0 : i64}, {channel = 3 : i64, col = 0 : i64, id = 54 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 62 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 70 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 78 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 86 : i64, location = 0 : i64, row = 1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 55 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 63 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 71 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 79 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 87 : i64, location = 1 : i64, row = 2 : i64}, {channel = 3 : i64, col = 0 : i64, id = 56 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 64 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 72 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 80 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 88 : i64, location = 1 : i64, row = 3 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 33 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 36 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 39 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 42 : i64, location = 3 : i64, row = -1 : i64}], sym_name = "attn_seg_1_0"}{
      airrt.herd_metadata {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 57 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 65 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 73 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 81 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 89 : i64, location = 0 : i64, row = 0 : i64}, {channel = 3 : i64, col = 0 : i64, id = 58 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 66 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 74 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 82 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 90 : i64, location = 0 : i64, row = 1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 59 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 67 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 75 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 83 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 91 : i64, location = 1 : i64, row = 2 : i64}, {channel = 3 : i64, col = 0 : i64, id = 60 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 68 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 76 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 84 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 92 : i64, location = 1 : i64, row = 3 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
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
  air.channel @QK2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QK2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
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
  func.func @attention_bf16(%arg0: memref<12x2048x64xbf16>, %arg1: memref<12x2048x64xbf16>, %arg2: memref<12x2048x64xbf16>, %arg3: memref<12x2048x64xbf16>) {
    %c8 = arith.constant 8 : index
    %c6 = arith.constant 6 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c8, %arg7=%c6) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<12x2048x64xbf16>, memref<12x2048x64xbf16>, memref<12x2048x64xbf16>, memref<12x2048x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8_0 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QK2L1_0_0[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 1 : i32, metadataArray = [{base = "air_QK2L1_0_0_0_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %3 = air.channel.put async  @QK2L1_0_1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 2 : i32, metadataArray = [{base = "air_QK2L1_0_1_0_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %4 = air.channel.put async  @QK2L1_0_2[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 3 : i32, metadataArray = [{base = "air_QK2L1_0_2_0_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %5 = air.channel.put async  @QK2L1_0_3[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 4 : i32, metadataArray = [{base = "air_QK2L1_0_3_0_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %6 = affine.apply #map1()[%arg5]
      %7 = air.channel.put async  @QK2L1_0_0[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %6] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 5 : i32, metadataArray = [{base = "air_QK2L1_0_0_0_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %8 = affine.apply #map2()[%arg5]
      %9 = air.channel.put async  @QK2L1_0_1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %8] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 6 : i32, metadataArray = [{base = "air_QK2L1_0_1_0_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %10 = affine.apply #map3()[%arg5]
      %11 = air.channel.put async  @QK2L1_0_2[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %10] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 7 : i32, metadataArray = [{base = "air_QK2L1_0_2_0_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %12 = affine.apply #map4()[%arg5]
      %13 = air.channel.put async  @QK2L1_0_3[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %12] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 8 : i32, metadataArray = [{base = "air_QK2L1_0_3_0_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %14 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %6] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 9 : i32, metadataArray = [{base = "air_VIn_0_0_0", index = 0 : i32}, {base = "air_VIn_0_1_0_0", index = 1 : i32}]} : (memref<12x2048x64xbf16>)
      %15 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %8] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 10 : i32, metadataArray = [{base = "air_VIn_1_0_0", index = 0 : i32}, {base = "air_VIn_1_1_0_0", index = 1 : i32}]} : (memref<12x2048x64xbf16>)
      %16 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %10] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 11 : i32, metadataArray = [{base = "air_VIn_2_0_0", index = 0 : i32}, {base = "air_VIn_2_1_0_0", index = 1 : i32}]} : (memref<12x2048x64xbf16>)
      %17 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %12] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 12 : i32, metadataArray = [{base = "air_VIn_3_0_0", index = 0 : i32}, {base = "air_VIn_3_1_0_0", index = 1 : i32}]} : (memref<12x2048x64xbf16>)
      %18 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %1] [%c64, %c64] [%c64, %c1]) {id = 13 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<12x2048x64xbf16>)
      %19 = air.channel.get async  @channel_0[%c1, %c0] (%arg11[%c64, %1] [%c64, %c64] [%c64, %c1]) {id = 14 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<12x2048x64xbf16>)
      %20 = air.channel.get async  @channel_0[%c2, %c0] (%arg11[%c128, %1] [%c64, %c64] [%c64, %c1]) {id = 15 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<12x2048x64xbf16>)
      %21 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %1] [%c64, %c64] [%c64, %c1]) {id = 16 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<12x2048x64xbf16>)
      %22 = affine.apply #map5()[%arg5, %arg4]
      %23 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 17 : i32, metadataArray = [{base = "air_QK2L1_1_0_1_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %24 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 18 : i32, metadataArray = [{base = "air_QK2L1_1_1_1_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %25 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 19 : i32, metadataArray = [{base = "air_QK2L1_1_2_1_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %26 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 20 : i32, metadataArray = [{base = "air_QK2L1_1_3_1_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %27 = affine.apply #map6()[%arg5]
      %28 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %27] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 21 : i32, metadataArray = [{base = "air_QK2L1_1_0_1_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %29 = affine.apply #map7()[%arg5]
      %30 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %29] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 22 : i32, metadataArray = [{base = "air_QK2L1_1_1_1_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %31 = affine.apply #map8()[%arg5]
      %32 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %31] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 23 : i32, metadataArray = [{base = "air_QK2L1_1_2_1_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %33 = affine.apply #map9()[%arg5]
      %34 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %33] [%c8_0, %c8_0, %c8_0, %c8_0, %c8_0] [%c4096, %c8_0, %c512, %c64, %c1]) {id = 24 : i32, metadataArray = [{base = "air_QK2L1_1_3_1_0", index = 0 : i32}]} : (memref<12x2048x64xbf16>)
      %35 = air.channel.put async  @VIn_0[%c1] (%arg10[%c0, %c0, %27] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 25 : i32, metadataArray = [{base = "air_VIn_0_0_0", index = 0 : i32}, {base = "air_VIn_0_1_0_0", index = 1 : i32}]} : (memref<12x2048x64xbf16>)
      %36 = air.channel.put async  @VIn_1[%c1] (%arg10[%c0, %c0, %29] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 26 : i32, metadataArray = [{base = "air_VIn_1_0_0", index = 0 : i32}, {base = "air_VIn_1_1_0_0", index = 1 : i32}]} : (memref<12x2048x64xbf16>)
      %37 = air.channel.put async  @VIn_2[%c1] (%arg10[%c0, %c0, %31] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 27 : i32, metadataArray = [{base = "air_VIn_2_0_0", index = 0 : i32}, {base = "air_VIn_2_1_0_0", index = 1 : i32}]} : (memref<12x2048x64xbf16>)
      %38 = air.channel.put async  @VIn_3[%c1] (%arg10[%c0, %c0, %33] [%c8_0, %c64, %c64] [%c4096, %c64, %c1]) {id = 28 : i32, metadataArray = [{base = "air_VIn_3_0_0", index = 0 : i32}, {base = "air_VIn_3_1_0_0", index = 1 : i32}]} : (memref<12x2048x64xbf16>)
      %39 = air.channel.get async  @channel_0[%c0, %c1] (%arg11[%c0, %22] [%c64, %c64] [%c64, %c1]) {id = 29 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<12x2048x64xbf16>)
      %40 = air.channel.get async  @channel_0[%c1, %c1] (%arg11[%c64, %22] [%c64, %c64] [%c64, %c1]) {id = 30 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<12x2048x64xbf16>)
      %41 = air.channel.get async  @channel_0[%c2, %c1] (%arg11[%c128, %22] [%c64, %c64] [%c64, %c1]) {id = 31 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<12x2048x64xbf16>)
      %42 = air.channel.get async  @channel_0[%c3, %c1] (%arg11[%c192, %22] [%c64, %c64] [%c64, %c1]) {id = 32 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_1_0_0", index = 4 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_1_0_1", index = 5 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_1_0_2", index = 6 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_3", index = 7 : i32}]} : (memref<12x2048x64xbf16>)
      %43 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1) attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c3_1 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c64_3 = arith.constant 64 : index
        %c1_4 = arith.constant 1 : index
        %c8_5 = arith.constant 8 : index
        %c0_6 = arith.constant 0 : index
        %c4_7 = arith.constant 4 : index
        %44 = air.wait_all async 
        %45 = air.wait_all async 
        %46 = air.wait_all async 
        %47 = air.wait_all async 
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
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
        %48 = scf.for %arg16 = %c0_6 to %c8_5 step %c1_4 iter_args(%arg17 = %44) -> (!air.async.token) {
          %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %61 = air.channel.get async [%async_token_18, %arg17]  @VIn_0[%arg12] (%results_19[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_6 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @V2L1_0_0[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @V2L1_0_1[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          %async_token_20 = air.execute [%63, %61] {
            memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %63 : !air.async.token
        }
        %49 = scf.for %arg16 = %c0_6 to %c8_5 step %c1_4 iter_args(%arg17 = %45) -> (!air.async.token) {
          %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %61 = air.channel.get async [%async_token_18, %arg17]  @VIn_1[%arg12] (%results_19[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_6 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @V2L1_1_0[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @V2L1_1_1[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          %async_token_20 = air.execute [%63, %61] {
            memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %63 : !air.async.token
        }
        %50 = scf.for %arg16 = %c0_6 to %c8_5 step %c1_4 iter_args(%arg17 = %46) -> (!air.async.token) {
          %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %61 = air.channel.get async [%async_token_18, %arg17]  @VIn_2[%arg12] (%results_19[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_6 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @V2L1_2_0[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @V2L1_2_1[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          %async_token_20 = air.execute [%63, %61] {
            memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %63 : !air.async.token
        }
        %51 = scf.for %arg16 = %c0_6 to %c8_5 step %c1_4 iter_args(%arg17 = %47) -> (!air.async.token) {
          %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %61 = air.channel.get async [%async_token_18, %arg17]  @VIn_3[%arg12] (%results_19[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_6 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @V2L1_3_0[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @V2L1_3_1[%c0_6, %c0_6, %c0_6] (%results_19[%c0_6, %c0_6, %c0_6] [%c8_5, %c64_3, %c8_5] [%c8_5, %c64_3, %c1_4]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          %async_token_20 = air.execute [%63, %61] {
            memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %63 : !air.async.token
        }
        %52 = air.channel.get async [%async_token]  @Gp2L2[%c0_6, %c0_6] (%results[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
        %53 = air.channel.get async [%async_token_8]  @Gp2L2[%c1_4, %c0_6] (%results_9[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
        %54 = air.channel.get async [%async_token_10]  @Gp2L2[%c2_2, %c0_6] (%results_11[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
        %55 = air.channel.get async [%async_token_12]  @Gp2L2[%c3_1, %c0_6] (%results_13[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
        %56 = air.channel.put async [%52]  @channel_0[%c0_6, %arg12] (%results[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
        %57 = air.channel.put async [%53]  @channel_0[%c1_4, %arg12] (%results_9[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
        %58 = air.channel.put async [%54]  @channel_0[%c2_2, %arg12] (%results_11[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
        %59 = air.channel.put async [%55]  @channel_0[%c3_1, %arg12] (%results_13[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
        %60 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_7, %arg19=%c4_7) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c64_18 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c8_19 = arith.constant 8 : index
          %c0_20 = arith.constant 0 : index
          %c1_21 = arith.constant 1 : index
          %c512_22 = arith.constant 512 : index
          %async_token_23, %results_24 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_25, %results_26 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_27, %results_28 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_31, %results_32 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_33 = air.execute [%async_token_27] {
            func.call @zero_fill_gp_bf16(%results_28) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_34 = air.execute [%async_token_23] {
            func.call @zero_fill_sp_bf16(%results_24) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_35 = air.execute [%async_token_25] {
            func.call @neg_inf_fill_up_bf16(%results_26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %61 = arith.cmpi eq, %arg20, %c0_20 : index
          scf.if %61 {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          } else {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          }
          %62 = arith.index_cast %arg16 : index to i32
          %63 = arith.cmpi eq, %62, %c0_i32 : i32
          scf.if %63 {
            %async_token_41 = air.execute [%async_token_29, %async_token_31] {
              func.call @copy_tile(%results_30, %results_32) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %61 {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          } else {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          }
          %64 = arith.cmpi eq, %62, %c1_i32 : i32
          scf.if %64 {
            %async_token_41 = air.execute [%async_token_29, %async_token_31] {
              func.call @copy_tile(%results_30, %results_32) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %61 {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          } else {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          }
          %65 = arith.cmpi eq, %62, %c2_i32 : i32
          scf.if %65 {
            %async_token_41 = air.execute [%async_token_29, %async_token_31] {
              func.call @copy_tile(%results_30, %results_32) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %61 {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          } else {
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_29]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_29]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                } else {
                  %73 = air.channel.get async [%async_token_29]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %73 : !air.async.token
                }
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
          }
          %66 = arith.cmpi eq, %62, %c3_i32 : i32
          scf.if %66 {
            %async_token_41 = air.execute [%async_token_29, %async_token_31] {
              func.call @copy_tile(%results_30, %results_32) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %67 = air.wait_all async [%async_token_29, %async_token_31, %async_token_33, %async_token_34, %async_token_35] 
          %68 = scf.for %arg21 = %c0_20 to %c8_19 step %c1_21 iter_args(%arg22 = %67) -> (!air.async.token) {
            %async_token_41, %results_42 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_43, %results_44 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_45 = air.execute [%async_token_43, %arg22] {
              %collapse_shape = memref.collapse_shape %results_44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %61 {
              %75 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%arg22]  @QK2L1_0_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%arg22]  @QK2L1_0_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %78 = air.channel.get async [%arg22]  @QK2L1_0_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %78 : !air.async.token
                  } else {
                    %78 = air.channel.get async [%arg22]  @QK2L1_0_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %78 : !air.async.token
                  }
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
            } else {
              %75 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%arg22]  @QK2L1_1_0[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%arg22]  @QK2L1_1_1[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %78 = air.channel.get async [%arg22]  @QK2L1_1_2[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %78 : !air.async.token
                  } else {
                    %78 = air.channel.get async [%arg22]  @QK2L1_1_3[%c0_20, %c0_20, %arg16] (%results_30[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %78 : !air.async.token
                  }
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
            }
            %70 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %75 = scf.if %61 -> (!air.async.token) {
                %76 = air.channel.get async [%async_token_41]  @V2L1_0_0[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              } else {
                %76 = air.channel.get async [%async_token_41]  @V2L1_0_1[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            } else {
              %75 = air.wait_all async 
              affine.yield %75 : !air.async.token
            }
            %71 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %75 = scf.if %61 -> (!air.async.token) {
                %76 = air.channel.get async [%async_token_41, %arg22, %70]  @V2L1_1_0[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              } else {
                %76 = air.channel.get async [%async_token_41, %arg22, %70]  @V2L1_1_1[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            } else {
              %75 = air.wait_all async 
              affine.yield %75 : !air.async.token
            }
            %72 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %75 = scf.if %61 -> (!air.async.token) {
                %76 = air.channel.get async [%async_token_41, %arg22, %71]  @V2L1_2_0[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              } else {
                %76 = air.channel.get async [%async_token_41, %arg22, %71]  @V2L1_2_1[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            } else {
              %75 = air.wait_all async 
              affine.yield %75 : !air.async.token
            }
            %73 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %75 = scf.if %61 -> (!air.async.token) {
                %76 = air.channel.get async [%async_token_41, %arg22, %72]  @V2L1_3_0[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              } else {
                %76 = air.channel.get async [%async_token_41, %arg22, %72]  @V2L1_3_1[%c0_20, %arg17, %arg16] (%results_42[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
                scf.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            } else {
              %75 = air.wait_all async 
              affine.yield %75 : !air.async.token
            }
            %async_token_46 = air.execute [%async_token_45] {
              %collapse_shape = memref.collapse_shape %results_44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_32, %results_30, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_47, %results_48 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_49, %results_50 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_51 = air.execute [%async_token_46, %async_token_47, %async_token_49] {
              %collapse_shape = memref.collapse_shape %results_44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_26, %results_48, %results_50) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_52 = air.execute [%async_token_51] {
              func.call @mul_r_gp(%results_50, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_53 = air.execute [%73, %async_token_52, %async_token_41, %async_token_43] {
              %collapse_shape = memref.collapse_shape %results_44 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_42, %results_28) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_54 = air.execute [%async_token_52] {
              func.call @accum_sp_r_s(%results_24, %results_50, %results_48) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_55 = air.execute [%async_token_54] {
              func.call @vector_copy_32elems(%c0_i32, %results_48, %results_24) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_56 = air.execute [%async_token_55] {
              memref.dealloc %results_48 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_57 = air.execute [%async_token_54] {
              memref.dealloc %results_50 : memref<64x1xbf16, 2 : i32>
            }
            %74 = air.wait_all async [%70, %71, %72, %async_token_53, %async_token_55] 
            %async_token_58 = air.execute [%async_token_51, %async_token_53] {
              memref.dealloc %results_44 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_59 = air.execute [%70, %71, %72, %async_token_53] {
              memref.dealloc %results_42 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %74 : !air.async.token
          }
          %69 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %70 = arith.subi %arg17, %c1_21 : index
            %71 = air.channel.put async [%async_token_27, %68]  @cascade_gp[%arg16, %70] (%results_28[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
            %72 = air.channel.put async [%async_token_25, %68]  @cascade_up[%arg16, %70] (%results_26[] [] []) {id = 102 : i32} : (memref<64x1xbf16, 2 : i32>)
            %73 = air.channel.put async [%async_token_23, %68]  @cascade_sp[%arg16, %70] (%results_24[] [] []) {id = 103 : i32} : (memref<64x1xbf16, 2 : i32>)
            %74 = air.wait_all async [%71, %72, %73] 
            affine.yield %74 : !air.async.token
          } else {
            %70 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_41, %results_42 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_43, %results_44 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_45, %results_46 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %71 = air.channel.get async [%async_token_41]  @cascade_gp[%arg16, %arg17] (%results_42[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
              %72 = air.channel.get async [%async_token_43]  @cascade_up[%arg16, %arg17] (%results_44[] [] []) {id = 105 : i32} : (memref<64x1xbf16, 2 : i32>)
              %73 = air.channel.get async [%async_token_45]  @cascade_sp[%arg16, %arg17] (%results_46[] [] []) {id = 106 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_47, %results_48 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_49 = air.execute [%async_token_25, %async_token_47, %68] {
                func.call @vector_copy_32elems(%c0_i32, %results_26, %results_48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_50 = air.execute [%72, %async_token_49] {
                func.call @maximum_up_u_bf16(%results_44, %results_26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_51, %results_52 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_53 = air.execute [%async_token_50, %async_token_51] {
                func.call @exp_up_minus_u(%results_44, %results_26, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56 = air.execute [%async_token_53, %async_token_54] {
                func.call @exp_up_minus_u(%results_48, %results_26, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_57 = air.execute [%async_token_53, %71] {
                func.call @mul_r_gp(%results_52, %results_42) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_58 = air.execute [%async_token_27, %async_token_56] {
                func.call @mul_r_gp(%results_55, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_59 = air.execute [%async_token_57, %async_token_58] {
                func.call @add_gp_g(%results_28, %results_42) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_62 = air.execute [%async_token_60] {
                func.call @zero_fill_sp_bf16(%results_61) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_63 = air.execute [%async_token_62, %async_token_57, %73] {
                func.call @accum_sp_r_s(%results_46, %results_52, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_64 = air.execute [%async_token_23, %async_token_63, %async_token_58] {
                func.call @accum_sp_r_s(%results_24, %results_55, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65 = air.execute [%async_token_64] {
                func.call @vector_copy_32elems(%c0_i32, %results_61, %results_46) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %74 = arith.subi %arg17, %c1_21 : index
              %75 = air.channel.put async [%async_token_59]  @cascade_gp[%arg16, %74] (%results_42[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              %76 = air.channel.put async [%async_token_25, %async_token_56]  @cascade_up[%arg16, %74] (%results_26[] [] []) {id = 108 : i32} : (memref<64x1xbf16, 2 : i32>)
              %77 = air.channel.put async [%async_token_65]  @cascade_sp[%arg16, %74] (%results_46[] [] []) {id = 109 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_66 = air.execute [%75] {
                memref.dealloc %results_42 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_67 = air.execute [%async_token_53] {
                memref.dealloc %results_44 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_68 = air.execute [%77] {
                memref.dealloc %results_46 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_56] {
                memref.dealloc %results_48 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%async_token_63] {
                memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%async_token_64] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_65] {
                memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
              }
              %78 = air.wait_all async [%75, %76, %77] 
              affine.yield %78 : !air.async.token
            } else {
              %async_token_41, %results_42 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_43, %results_44 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_45, %results_46 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %71 = air.channel.get async [%async_token_41]  @cascade_gp[%arg16, %arg17] (%results_42[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              %72 = air.channel.get async [%async_token_43]  @cascade_up[%arg16, %arg17] (%results_44[] [] []) {id = 111 : i32} : (memref<64x1xbf16, 2 : i32>)
              %73 = air.channel.get async [%async_token_45]  @cascade_sp[%arg16, %arg17] (%results_46[] [] []) {id = 112 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_47, %results_48 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_49 = air.execute [%async_token_25, %async_token_47, %68] {
                func.call @vector_copy_32elems(%c0_i32, %results_26, %results_48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_50 = air.execute [%72, %async_token_49] {
                func.call @maximum_up_u_bf16(%results_44, %results_26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_51, %results_52 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_53 = air.execute [%async_token_50, %async_token_51] {
                func.call @exp_up_minus_u(%results_44, %results_26, %results_52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56 = air.execute [%async_token_53, %async_token_54] {
                func.call @exp_up_minus_u(%results_48, %results_26, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_57 = air.execute [%async_token_53, %71] {
                func.call @mul_r_gp(%results_52, %results_42) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_58 = air.execute [%async_token_27, %async_token_56] {
                func.call @mul_r_gp(%results_55, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_59 = air.execute [%async_token_57, %async_token_58] {
                func.call @add_gp_g(%results_28, %results_42) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_62 = air.execute [%async_token_60] {
                func.call @zero_fill_sp_bf16(%results_61) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_63 = air.execute [%async_token_62, %async_token_57, %73] {
                func.call @accum_sp_r_s(%results_46, %results_52, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_64 = air.execute [%async_token_23, %async_token_63, %async_token_58] {
                func.call @accum_sp_r_s(%results_24, %results_55, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65 = air.execute [%async_token_64] {
                func.call @vector_copy_32elems(%c0_i32, %results_61, %results_46) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_65, %async_token_59] {
                func.call @div_gp_sp(%results_46, %results_42) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %74 = air.channel.put async [%async_token_66]  @Gp2L2[%arg16, %c0_20] (%results_42[%c0_20, %c0_20, %c0_20] [%c64_18, %c8_19, %c8_19] [%c8_19, %c512_22, %c1_21]) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_67 = air.execute [%74] {
                memref.dealloc %results_42 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_68 = air.execute [%async_token_53] {
                memref.dealloc %results_44 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_66] {
                memref.dealloc %results_46 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%async_token_56] {
                memref.dealloc %results_48 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%async_token_63] {
                memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_64] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_65] {
                memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %74 : !air.async.token
            }
            affine.yield %68 : !air.async.token
          }
          %async_token_36 = air.execute [%68] {
            memref.dealloc %results_32 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_37 = air.execute [%68] {
            memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_38 = air.execute [%69, %68, %async_token_33] {
            memref.dealloc %results_28 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_39 = air.execute [%69, %68, %async_token_35] {
            memref.dealloc %results_26 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_40 = air.execute [%69, %68, %async_token_34] {
            memref.dealloc %results_24 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_14 = air.execute [%59] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_15 = air.execute [%58] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_16 = air.execute [%57] {
          memref.dealloc %results_9 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17 = air.execute [%56] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        air.wait_all [%48, %49, %50, %51, %60, %async_token_14, %async_token_15, %async_token_16, %async_token_17]  {air.segment_end}
      }
    }
    return
  }
}
